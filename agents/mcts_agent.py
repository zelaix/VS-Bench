from open_spiel.python.algorithms import mcts
import numpy as np
import pyspiel

from agents.base_agent import BaseAgent


class MCTSAgent(BaseAgent, agent_type="mcts_agent"):

    def __init__(self, env_name, uct_c, max_simulations, rollout_count, seed):
        self.env_name = env_name
        self.game = pyspiel.load_game(env_name)
        random_state = np.random.RandomState(seed)
        evaluator = mcts.RandomRolloutEvaluator(rollout_count, random_state)
        self.mcts_bot = mcts.MCTSBot(
            self.game,
            uct_c,
            max_simulations,
            evaluator,
            solve=False,
            random_state=random_state,
        )

    async def _act(self, observation):
        state = self.game.deserialize_state(observation.serialized_state)
        if self.game.get_type().short_name == 'breakthrough':
            state = observation.obs['obs']
        action = self.mcts_bot.step(state)

        agent_info = {
            "action_string": observation.legal_actions[action],
            "action": action,
            "legal_actions": observation.legal_actions,
        }
        return action, agent_info
