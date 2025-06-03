from typing import *

from envs.base_env import Observation


class BaseAgent:
    _registry: Dict[str, Type["BaseAgent"]] = {}

    def __init_subclass__(cls, *, agent_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        if agent_type in cls._registry:
            raise ValueError(f"Duplicated agent_type: {agent_type}.")
        cls._registry[agent_type] = cls
        cls.agent_type = agent_type

    @classmethod
    def from_config(cls, spec: Dict[str, Any]) -> "BaseAgent":
        try:
            AgentCls = cls._registry[spec["type"]]
        except KeyError:
            raise ValueError(f"[BaseAgent] unknown type '{spec['type']}'. "
                             f"Available: {list(cls._registry)}")
        params = spec.get("params", {})
        return AgentCls(**params)

    async def act(self, observation: Union[Observation, None]) -> Tuple[int, Dict[str, Any]]:
        """Get action from the agent. If the observation is None, return None, None.
        """
        if observation is None:
            return None, None
        return await self._act(observation)

    async def _act(self, observation: Observation) -> Tuple[int, Dict[str, Any]]:
        """Given the observation, get the action from the agent.

        Args:
            observation (Observation): The observation from env.

        Returns:
            action (int): The action to take.
            agent_info (Dict[str, Any]): agent information.
        """
        raise NotImplementedError

    async def predict(self, dataset_i: Dict[str, Any], regex_patterns: List) -> Tuple[int, Dict[str, Any]]:
        """
        Asynchronously predicts the next action that the opponent VLM will take.

        Args:
            dataset_i (Dict[str, Any]): A single data sample for prediction.
            regex_patterns (List): A list of regular expression patterns used to extract the action from the VLM's response.

        Returns:
            action (int): The predicted action.
            agent_info (Dict[str, Any]): agent information.
        """
        raise NotImplementedError
