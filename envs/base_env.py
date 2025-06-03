from typing import *
import dataclasses
import numpy as np


@dataclasses.dataclass
class Observation:
    obs: np.ndarray
    agent_id: int
    image_paths: List[str]  # [] if no image
    legal_actions: Dict[int, str]  # key is action, value is readable action
    serialized_state: str  # for openspiel games and agents only
    regex_patterns: List[str]  # the env pattern list for get the action result from response
    addition_info: str  # additional information for the observation, such as history actions in Overcooked


class RegexError(Exception):
    """Raised when the response is illegal"""
    pass


class BaseEnv:
    _registry: Dict[str, Type["BaseEnv"]] = {}

    def __init_subclass__(cls, *, env_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        if env_type in cls._registry:
            raise ValueError(f"Duplicated env_type: {env_type}.")
        cls._registry[env_type] = cls
        cls.env_type = env_type

    @classmethod
    def from_config(cls, spec: Dict[str, Any]) -> "BaseEnv":
        try:
            EnvCls = cls._registry[spec["type"]]
        except KeyError:
            raise ValueError(f"[BaseEnv] unknown type '{spec['type']}'. "
                             f"Available: {list(cls._registry)}")
        params = spec.get("params", {})
        return EnvCls(**params)

    def reset(self, seed: int = 0) -> List[Union[Observation, None]]:
        """Reset the environment with the seed.

        Args:
            seed (int): The seed to control randomness.

        Returns:
            observations (List[Union[Observation, None]]): A list of observations for each agent. If an agent does not need to take an action, its observation is None.
        """
        raise NotImplementedError

    def step(
        self,
        actions: List[Union[int,
                            None]]) -> Tuple[List[Union[Observation, None]], List[float], List[bool], Dict[str, Any]]:
        """Apply actions to the environment and return the next observations, rewards, dones, and info.

        Args:
            actions (List[Union[int, None]]): a list of actions from agents. If an agent does not need to take an action, its action is None.

        Returns:
            observations (List[Union[Observation, None]]): A list of observations for each agent. If an agent does not need to take an action, its observation is None.
            rewards (List[float]): A list of rewards for each agent.
            dones (List[bool]): A list of booleans indicating if the agent is still alive.
            info (Dict[str, Any]): A dictionary containing additional information about the step.
        """
        raise NotImplementedError

    @property
    def regex_patterns(self):
        patterns = [
            (r'```json\s*\{\s*"action"\s*:\s*"([^"]+)"\s*\}\s*```', lambda m: m.strip()),
            (r'"action"\s*:\s*"([^"]+)"', lambda m: m.strip()),
        ]
        return patterns
