import matplotlib.pyplot as plt
import numpy as np
import os
import pyspiel

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


class TicTacToe(BaseEnv, env_type="tic_tac_toe"):
    """Tic-Tac-Toe game environment using OpenSpiel."""

    def __init__(
        self,
        visual_obs=True,
        image_dir=None,
        recording_type='gif',
        recording_fps=2,
    ):
        self._env = pyspiel.load_game("tic_tac_toe")
        self.state = None
        self.num_agents = self._env.num_players()
        self.image_paths = []

        self.visual_obs = visual_obs
        if self.visual_obs:
            assert image_dir is not None, "image_dir must not be None."
            self.image_dir = image_dir
            self.recorders = [Recorder(image_dir, recording_type, recording_fps)]

    @property
    def current_player(self):
        return self.state.current_player()

    def reset(self, seed=0):
        self.state = self._env.new_initial_state()
        if self.visual_obs:
            self.recorders[0].clear()
            self.image_paths = self._save_image()
        return [self._get_observation(i) for i in range(self.num_agents)]

    def step(self, actions):
        if self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        agent_action = actions[self.current_player]
        self.state.apply_action(agent_action)
        if self.visual_obs:
            self.image_paths = self._save_image()
            if self.state.is_terminal():
                self.recorders[0].save()

        observations = [self._get_observation(i) for i in range(self.num_agents)]
        rewards = self.state.rewards()
        dones = [self.state.is_terminal()] * self.num_agents
        info = self._get_info()

        return observations, rewards, dones, info

    def _get_observation(self, agent_id):
        """Return the Observation object for agent agent_id. The observation is a 3x3x3 np.array representing the board.
            obs[0] (3x3): 1 if the cell has no mark, else 0
            obs[1] (3x3): 1 if the cell is O, else 0
            obs[2] (3x3): 1 if the cell is X, else 0
        """
        if agent_id == self.current_player:
            return Observation(
                obs=np.reshape(self.state.observation_tensor(), (3, 3, 3)),
                agent_id=agent_id,
                image_paths=self.image_paths,
                legal_actions=self._get_legal_actions(agent_id),
                serialized_state=self.state.serialize(),
                regex_patterns=self.regex_patterns,
                addition_info=None,
            )
        else:
            return None

    @property
    def regex_patterns(self):
        patterns = [(r'```json\s*\{\s*"action"\s*:\s*"([^"]+)"\s*\}\s*```', lambda m: m.strip()),
                    (r'"action"\s*:\s*"([^"]+)"', lambda m: m.strip()),
                    (r'([XOxo])\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', lambda m: f"{m[0].upper()}({m[1]},{m[2]})"),
                    (r'\(\s*(\d+)\s*,\s*(\d+)\s*\)',
                     lambda m: f"{'X' if self.current_player == 0 else 'O'}({m[0]},{m[1]})")]
        return patterns

    def _get_info(self):
        if self.state.is_terminal():
            returns = self.state.returns()
            winner = int(np.argmax(returns)) if returns[0] != returns[1] else -1
            return {"returns": returns, "winner": winner}
        else:
            return None

    def _get_legal_actions(self, agent_id):
        legal_actions = dict()
        actions = self.state.legal_actions(agent_id)
        for a in actions:
            legal_actions[a] = self._action_to_string(agent_id, a)
        return legal_actions

    def _action_to_string(self, agent_id, action):
        mark = "X" if agent_id == 0 else "O"
        row = action // 3
        column = action % 3
        return f"{mark}({row},{column})"

    def _save_image(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.invert_yaxis()
        for x in range(1, 3):
            ax.plot([x - 0.5, x - 0.5], [-0.5, 2.5], color='black', linewidth=2)
        for y in range(1, 3):
            ax.plot([-0.5, 2.5], [y - 0.5, y - 0.5], color='black', linewidth=2)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['0', '1', '2'])
        ax.set_yticklabels(['0', '1', '2'])
        board = np.array([list(line) for line in str(self.state).strip().split("\n")])
        for i in range(3):
            for j in range(3):
                piece = board[i][j]
                if piece != '.':
                    color = 'red' if piece == 'x' else 'blue'
                    ax.text(j, i, piece.upper(), fontsize=30, ha='center', va='center', color=color)
        ax.set_aspect('equal')

        image_path = os.path.join(self.image_dir, f"step_{self.state.move_number()}.png")
        plt.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close()
        self.recorders[0].add_frame(image_path)
        return [image_path]
