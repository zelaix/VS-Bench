import numpy as np

from agents.base_agent import BaseAgent


class BuiltinAgent(BaseAgent, agent_type="builtin_agent"):

    def __init__(self, env_name):
        self.env_name = env_name

    async def _act(self, observation):
        return None, None
