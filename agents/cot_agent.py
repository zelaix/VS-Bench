import asyncio
import numpy as np
import random
import re

from agents.base_agent import BaseAgent
from envs.base_env import RegexError
from models.base_model import BaseModel, ObservationPrompt, Prompt, MaxTokenLimit
from prompts.action import get_action_prompt
from prompts.observation import get_observation_prompt
from prompts.system import get_system_prompt


class CoTAgent(BaseAgent, agent_type="cot_agent"):

    def __init__(self, env_name, model, visual_obs, max_errors):
        self.env_name = env_name
        self.model = BaseModel.from_config(model)
        self.visual_obs = visual_obs
        self.max_errors = max_errors

    async def _act(self, observation):
        observation.image_paths = observation.image_paths if self.visual_obs else []
        cot_prompt = "Let's think step by step."
        prompt = Prompt(
            system_prompt=get_system_prompt(self.env_name),
            observation_prompt=get_observation_prompt(self.env_name, observation),
            action_prompt=f"{get_action_prompt(self.env_name, observation)}\n\n{cot_prompt}",
        )

        response = None
        messages = None
        action_string = None
        action = None
        agent_info = None
        num_errors = 0
        token_info = None
        exception_history = {}

        while num_errors < self.max_errors:
            try:
                messages, reasoning, response, token_info = await self.model.generate(prompt)
                action_string = self._extract_action(response, observation.regex_patterns)
                action = self._parse_action(action_string, observation.legal_actions)
                agent_info = {
                    "messages": messages,
                    "reasoning": reasoning,
                    "response": response,
                    "action_string": action_string,
                    "action": action,
                    "legal_actions": observation.legal_actions,
                    "token_info": token_info,
                    "exception": exception_history
                }
                return action, agent_info

            except MaxTokenLimit as e:
                print(f"response: {response}")
                print(f"Exception: {e}")
                exception_history[f"error_{num_errors+1}"] = {
                    "type": "MaxTokenLimit",
                    "message": str(e),
                }
                self.model.increase_max_tokens()

            except RegexError as e:
                print("WARNING: Failed to get action from model response.")
                print(f"Current model : {self.model.name}")
                print(f"response: {response}")
                exception_history[f"error_{num_errors+1}"] = {
                    "type": "RegexError",
                    "response": response,
                }

            except ValueError as e:
                print(f"Current model : {self.model.name}")
                print(f"Exception: {e}")
                exception_history[f"error_{num_errors+1}"] = {
                    "type": "ValueError",
                    "message": str(e),
                }
                break

            except Exception as e:
                print(f"Current model : {self.model.name}")
                print(f"Exception: {e}")
                exception_history[f"error_{num_errors+1}"] = {
                    "type": e.__class__.__name__,
                    "message": str(e),
                }

            print(f"Retrying {num_errors + 1}/{self.max_errors}...")
            sleep_time = min(2 ** num_errors + random.uniform(0, 10), 200 + random.randint(0, 50))
            print(f"Sleeping for {sleep_time:.2f} seconds before retry...")
            await asyncio.sleep(sleep_time)
            num_errors += 1

        print("WARNING: Using random action.")
        legal_actions = list(observation.legal_actions.keys())
        action = int(np.random.choice(legal_actions))
        agent_info = {
            "messages": messages,
            "response": response,
            "action_string": action_string,
            "action": action,
            "legal_actions": observation.legal_actions,
            "token_info": token_info,
            "exception": exception_history
        }
        return action, agent_info

    async def predict(self, dataset_i, regex_patterns):
        cot_prompt = "Let's think step by step."
        system_prompt, observation_prompt = dataset_i['prompt']['system'], dataset_i['prompt']['observation']
        image_path, action_prompt = dataset_i['prompt']['image_path'], dataset_i['prompt']['action']
        prompt = Prompt(system_prompt, ObservationPrompt(observation_prompt, image_path),
                        action_prompt + f"\n\n{cot_prompt}")
        response = None
        messages = None
        action_string = None
        action = None
        agent_info = None
        num_errors = 0
        token_info = None
        exception_history = {}
        legal_actions = dataset_i['next_legal_actions']

        while num_errors < self.max_errors:
            try:
                messages, reasoning, response, token_info = await self.model.generate(prompt, reasoning=True)
                action_string = self._extract_action(response, regex_patterns)
                action = int(self._parse_action(action_string, legal_actions))
                agent_info = {
                    "messages": messages,
                    "reasoning": reasoning,
                    "response": response,
                    "action_string": action_string,
                    "action": action,
                    "legal_actions": legal_actions,
                    "token_info": token_info,
                    "exception": exception_history
                }
                return action, agent_info

            except MaxTokenLimit as e:
                print(f"response: {response}")
                print(f"Exception: {e}")

                exception_history[f"error_{num_errors+1}"] = {
                    "type": "MaxTokenLimit",
                    "message": str(e),
                }
                self.model.increase_max_tokens()

            except RegexError as e:
                print("WARNING: Failed to get action from model response.")
                print(f"Current model : {self.model.name}")
                print(f"response: {response}")

                exception_history[f"error_{num_errors+1}"] = {
                    "type": "RegexError",
                    "response": response,
                }

            except ValueError as e:
                print(f"Current model : {self.model.name}")
                print(f"Exception: {e}")

                exception_history[f"error_{num_errors+1}"] = {
                    "type": "ValueError",
                    "message": str(e),
                }
                break

            except Exception as e:
                print(f"Current model : {self.model.name}")
                print(f"Exception: {e}")

                exception_history[f"error_{num_errors+1}"] = {
                    "type": e.__class__.__name__,
                    "message": str(e),
                }

            print(f"Retrying {num_errors + 1}/{self.max_errors}...")
            sleep_time = min(2 ** num_errors + random.uniform(0, 10), 200 + random.randint(0, 50))
            print(f"Sleeping for {sleep_time:.2f} seconds before retry...")
            await asyncio.sleep(sleep_time)
            num_errors += 1

        print("WARNING: Using an illegal action instead.")
        legal_actions = list(legal_actions.keys())
        action = -99999999
        agent_info = {
            "messages": messages,
            "response": response,
            "action_string": action_string,
            "action": action,
            "legal_actions": legal_actions,
            "token_info": token_info,
            "exception": exception_history
        }
        return action, agent_info

    def _parse_action(self, action_string, legal_actions):
        for key, value in legal_actions.items():
            if action_string in value:
                return key
        error_log = f"Illegal action: {action_string}.\nLegal_actions: {legal_actions}."
        raise ValueError(error_log)

    def _extract_action(self, response, regex_patterns):
        for pattern, processor in regex_patterns:
            match = re.findall(pattern, response, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return processor(match[-1])
        raise RegexError(f"Error response: {response}")
