import numpy as np

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent, agent_type="random_agent"):
    """Agent that selects a random legal action."""

    def __init__(self, env_name):
        self.env_name = env_name

    async def _act(self, observation):
        """Return a random choice among the legal actions."""
        legal_actions = list(observation.legal_actions.keys())
        action = int(np.random.choice(legal_actions))
        agent_info = {
            "action_string": observation.legal_actions[action],
            "action": action,
            "legal_actions": observation.legal_actions,
        }
        return int(action), agent_info

    async def predict(self, dataset_i, regex_patterns):
        legal_actions = list(dataset_i['next_legal_actions'].keys())
        action = int(np.random.choice(legal_actions))
        action_string = dataset_i['next_legal_actions'][str(action)]
        agent_info = {
            "action_string": action_string,
            "action": action,
            "legal_actions": dataset_i['next_legal_actions'],
        }
        return action, agent_info
