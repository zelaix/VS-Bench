from volcenginesdkarkruntime import AsyncArk
import os

from models.base_model import BaseModel, MaxTokenLimit
from utils.helper import image_to_b64


class ArkModel(BaseModel, model_type="ark"):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_client(self):
        self.client = AsyncArk(api_key=os.environ["ARK_API_KEY"])

    async def generate(self, prompt, reasoning=False):
        messages = self._get_messages(prompt, log=False, reasoning=reasoning)
        messages_log = self._get_messages(prompt, log=True, reasoning=reasoning)
        if not self.reasoning_model:
            response = await self.client.chat.completions.create(model=self.version, messages=messages,
                                                                 temperature=self.temperature,
                                                                 max_tokens=self.response_tokens)
            if response.choices[0].finish_reason == 'length':
                raise MaxTokenLimit("Response was truncated due to max token limit.")
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            token_info = self._count_token_price(prompt_tokens, completion_tokens)
            return messages_log, None, content, token_info
        else:
            response = await self.client.chat.completions.create(model=self.version, messages=messages,
                                                                 temperature=self.temperature,
                                                                 max_tokens=self.response_tokens)
            if response.choices[0].finish_reason == 'length':
                raise MaxTokenLimit("Response was truncated due to max token limit.")
            reasoning = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
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
