import openai
import os

from models.base_model import BaseModel, MaxTokenLimit
from utils.helper import image_to_b64


class OpenAIModel(BaseModel, model_type="openai"):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_client(self):
        if self.name.startswith(('gpt', 'o')):
            self.client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif self.name.startswith('grok'):
            self.client = openai.AsyncOpenAI(
                api_key=os.environ["XAI_API_KEY"],
                base_url="https://api.x.ai/v1",
            )
        elif self.name.startswith(('qwen', 'qvq')):
            self.client = openai.AsyncOpenAI(
                api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            raise ValueError(f"illegal model : {self.name}")

    async def generate(self, prompt, reasoning=False):
        messages = self._get_messages(prompt, log=False, reasoning=reasoning)
        messages_log = self._get_messages(prompt, log=True, reasoning=reasoning)
        if not self.reasoning_model:
            response = await self.client.chat.completions.create(model=self.version, messages=messages,
                                                                 temperature=self.temperature,
                                                                 max_completion_tokens=self.response_tokens)
            if response.choices[0].finish_reason == 'length':
                raise MaxTokenLimit("Response was truncated due to max token limit.")
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            token_info = self._count_token_price(prompt_tokens, completion_tokens)
            return messages_log, None, content, token_info
        else:
            if not self.stream:
                response = await self.client.chat.completions.create(
                    model=self.version, messages=messages, temperature=self.temperature,
                    max_completion_tokens=self.reasoning_tokens + self.response_tokens)
                if response.choices[0].finish_reason == 'length':
                    raise MaxTokenLimit("Response was truncated due to max token limit.")
                content = response.choices[0].message.content.strip()
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                token_info = self._count_token_price(prompt_tokens, completion_tokens)
                return messages_log, None, content, token_info
            else:
                response = await self.client.chat.completions.create(
                    model=self.version, messages=messages, temperature=self.temperature,
                    max_completion_tokens=self.reasoning_tokens + self.response_tokens, stream=self.stream,
                    stream_options={"include_usage": True})
                reasoning = ""
                content = ""
                is_answering = False
                finish_reason = None
                prompt_tokens = 0
                completion_tokens = 0
                async for chunk in response:
                    if not chunk.choices:
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens
                        continue
                    delta = chunk.choices[0].delta
                    if chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning += delta.reasoning_content
                    else:
                        if delta.content != "" and not is_answering:
                            is_answering = True
                        content += delta.content
                if finish_reason == "length":
                    raise MaxTokenLimit("Response was truncated due to max token limit.")
                token_info = self._count_token_price(prompt_tokens, completion_tokens)
                return messages_log, reasoning, content, token_info

    def _get_messages(self, prompt, log=False, reasoning=False):
        system_prompt = [{"type": "text", "text": prompt.system_prompt}]
        user_prompt = []
        if len(prompt.observation_prompt.image_paths) > 0:
            for image_path in prompt.observation_prompt.image_paths:
                if not log:
                    image_url = f"data:image/png;base64,{image_to_b64(image_path, reasoning)}"
                    user_prompt.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high",
                        },
                    })
                else:
                    if not reasoning:
                        user_prompt.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_path,
                                "detail": "high",
                            },
                        })
        user_prompt.append({
            "type": "text",
            "text": f"{prompt.observation_prompt.text}\n\n{prompt.action_prompt}",
        })
        if self.reasoning_model:
            messages = [{"role": "user", "content": system_prompt + user_prompt}]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ]
        return messages

    def _count_token_price(self, prompt_tokens, completion_tokens):
        if prompt_tokens is None or completion_tokens is None:
            return {}
        token_info = {
            "prompt": {
                "prompt_tokens": prompt_tokens,
                "prompt_price": prompt_tokens / 1e6 * self.input_price_per_1M_tokens,
            },
            "completion": {
                "completion_tokens": completion_tokens,
                "completion_price": completion_tokens / 1e6 * self.output_price_per_1M_tokens,
            },
            "total": {
                "total_tokens":
                    prompt_tokens + completion_tokens,
                "total_price":
                    prompt_tokens / 1e6 * self.input_price_per_1M_tokens +
                    completion_tokens / 1e6 * self.output_price_per_1M_tokens,
            }
        }
        return token_info
