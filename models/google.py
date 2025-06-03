from google import genai
from google.genai import types
import os

from models.base_model import BaseModel, MaxTokenLimit
from utils.helper import image_to_byte


class GoogleModel(BaseModel, model_type="google"):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_client(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    async def generate(self, prompt, reasoning=False):
        messages, messages_log = self._get_messages(prompt, reasoning)
        if not self.reasoning_model:
            if self.version != 'gemini-2.0-flash-001':
                config = types.GenerateContentConfig(temperature=self.temperature,
                                                     system_instruction=prompt.system_prompt,
                                                     max_output_tokens=self.response_tokens,
                                                     thinking_config=types.ThinkingConfig(thinking_budget=0))
            else:
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    system_instruction=prompt.system_prompt,
                    max_output_tokens=self.response_tokens,
                )
            response = await self.client.aio.models.generate_content(
                model=self.version,
                contents=messages,
                config=config,
            )
            if response.candidates[0].finish_reason == "MAX_TOKENS":
                raise MaxTokenLimit("Response was truncated due to max token limit.")
            content = response.candidates[0].content.parts[0].text
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            token_info = self._count_token_price(prompt_tokens, completion_tokens)
            return messages_log, None, content, token_info
        else:
            config = types.GenerateContentConfig(
                temperature=self.temperature, system_instruction=prompt.system_prompt,
                max_output_tokens=self.response_tokens + self.reasoning_tokens,
                thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=self.reasoning_tokens))
            response = await self.client.aio.models.generate_content(
                model=self.version,
                contents=messages,
                config=config,
            )
            if response.candidates[0].finish_reason == "MAX_TOKENS":
                raise MaxTokenLimit("Response was truncated due to max token limit.")
            content = response.candidates[0].content.parts[0].text
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count + response.usage_metadata.thoughts_token_count
            token_info = self._count_token_price(prompt_tokens, completion_tokens)
            return messages_log, None, content, token_info

    def _get_messages(self, prompt, reasoning=False):
        system_prompt = [{"type": "text", "text": prompt.system_prompt}]

        user_prompt = []
        user_prompt_log = []
        if len(prompt.observation_prompt.image_paths) > 0:
            for image_path in prompt.observation_prompt.image_paths:
                user_prompt.append(
                    types.Part.from_bytes(
                        data=image_to_byte(image_path, reasoning),
                        mime_type='image/png',
                    ))
                if not reasoning:
                    user_prompt_log.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_path,
                            "detail": "high",
                        },
                    })
        user_prompt.append(f"{prompt.observation_prompt.text}\n\n{prompt.action_prompt}")
        user_prompt_log.append({
            "type": "text",
            "text": f"{prompt.observation_prompt.text}\n\n{prompt.action_prompt}",
        })
        messages_log = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt_log
            },
        ]

        return user_prompt, messages_log

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
