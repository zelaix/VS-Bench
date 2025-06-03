from open_spiel.python.algorithms import cfr
import gzip
import numpy as np
import os
import pickle
import pyspiel

from agents.base_agent import BaseAgent


class CFRAgent(BaseAgent, agent_type="cfr_agent"):
    """Agent that uses Counterfactual Regret Minimization (CFR) to select actions."""

    def __init__(self, env_name, num_iterations):
        self.env_name = env_name
        self.game = pyspiel.load_game(self.env_name)
        self.cfr_solver = cfr.CFRSolver(self.game)

        if self.env_name == "kuhn_poker":
            file_dir = os.path.dirname(os.path.abspath(__file__))
            with gzip.open(f"{file_dir}/checkpoints/kuhn_poker/ne.pkl.gz", "rb") as f:
                self.avg_policy = pickle.load(f)
        else:
            for _ in range(num_iterations):
                self.cfr_solver.evaluate_and_update_policy()
            self.avg_policy = self.cfr_solver.average_policy()

    async def _act(self, observation):
        state = self.game.deserialize_state(observation.serialized_state)
        state_policy = self.avg_policy.action_probabilities(state)
        actions = list(state_policy.keys())
        probabilities = list(state_policy.values())

        action = int(np.random.choice(actions, p=probabilities))
        agent_info = {
            "action_string": observation.legal_actions[action],
            "action": action,
            "legal_actions": observation.legal_actions,
        }
        return action, agent_info
