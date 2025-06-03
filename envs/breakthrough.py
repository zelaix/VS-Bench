from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pyspiel
import string

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


class Breakthrough(BaseEnv, env_type="breakthrough"):

    def __init__(
        self,
        row=8,
        col=8,
        visual_obs=True,
        image_dir=None,
        recording_type='gif',
        recording_fps=2,
    ):
        self.row = row
        self.col = col
        self._env = pyspiel.load_game(f"breakthrough(rows={self.row},columns={self.col})")
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
        """Return the Observation object for agent agent_id. The observation is a 3 x rows x cols np.array representing the board.
            obs[0] (rows x cols): 1 if the cell has mark b
            obs[1] (rows x cols): 1 if the cell has mark h
            obs[2] (rows x cols): 1 if the cell has no mark
        """
        if agent_id == self.current_player:
            return Observation(
                obs={
                    'obs': self.state,
                    'rows': self.row,
                    'cols': self.col
                },
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
                    (rf'([a-h][1-8][a-h][1-8])', lambda m: m.strip())]
        return patterns

    def _get_info(self):
        if self.state.is_terminal():
            returns = self.state.returns()
            winner = -1 if returns[0] == returns[1] else int(np.argmax(returns))
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
        num_dir = 6
        r1 = action // (self.col * num_dir * 2)
        remainder = action % (self.col * 6 * 2)
        c1 = remainder // (num_dir * 2)
        remainder = remainder % (num_dir * 2)
        dir = remainder // 2
        capture = remainder % 2

        kDirRowOffsets = [1, 1, 1, -1, -1, -1]
        kDirColOffsets = [-1, 0, 1, -1, 0, 1]
        r2 = r1 + kDirRowOffsets[dir]
        c2 = c1 + kDirColOffsets[dir]

        start_col = chr(ord('a') + c1)
        start_row = str(8 - r1)
        end_col = chr(ord('a') + c2)
        end_row = str(8 - r2)
        move_str = f"{start_col}{start_row}{end_col}{end_row}"
        return move_str

    def _string_to_action(self, action_str):
        if len(action_str) == 5 and action_str[-1] == "*":
            capture = 1
            action_str = action_str[:-1]
        elif len(action_str) == 4:
            capture = 0
        else:
            raise ValueError(f"Invalid action string length {action_str}")

        c1 = ord(action_str[0]) - ord('a')
        r1 = 8 - int(action_str[1])
        c2 = ord(action_str[2]) - ord('a')
        r2 = 8 - int(action_str[3])

        dr = r2 - r1
        dc = c2 - c1
        if dr == 1 and dc == -1:
            dir = 0
        elif dr == 1 and dc == 0:
            dir = 1
        elif dr == 1 and dc == 1:
            dir = 2
        elif dr == -1 and dc == -1:
            dir = 3
        elif dr == -1 and dc == 0:
            dir = 4
        elif dr == -1 and dc == 1:
            dir = 5
        else:
            raise ValueError("Invalid move direction")

        action = (r1 * self.col * 6 * 2) + (c1 * 6 * 2) + (dir * 2) + capture
        return action

    def _save_image(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, self.col - 0.5)
        ax.set_ylim(-0.5, self.row - 0.5)
        ax.invert_yaxis()

        ax.set_xticks(np.arange(0, self.col))
        ax.set_yticks(np.arange(0, self.row))
        ax.set_xticklabels(list(string.ascii_lowercase[:self.col]), fontsize=14, fontweight='bold')
        ax.set_yticklabels([str(i) for i in range(self.row, 0, -1)], fontsize=14, fontweight='bold')

        board = np.array([list(line) for line in str(self.state).strip().split("\n")])
        for i in range(self.row):
            for j in range(self.col):
                color = '#d08c45' if (i + j) % 2 == 1 else '#ffcf9f'
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))

        dark_black = mpimg.imread('images/breakthrough/db.png')
        dark_white = mpimg.imread('images/breakthrough/dw.png')
        light_black = mpimg.imread('images/breakthrough/lb.png')
        light_white = mpimg.imread('images/breakthrough/lw.png')
        img_size = 0.074

        for i in range(self.row):
            for j in range(1, self.col + 1):
                piece = board[i][j]
                if piece != '.':
                    if piece.lower() == 'b':
                        if (i + j - 1) % 2 == 0:
                            piece_image = light_black
                        else:
                            piece_image = dark_black
                    elif piece.lower() == 'w':
                        if (i + j - 1) % 2 == 0:
                            piece_image = light_white
                        else:
                            piece_image = dark_white
                    else:
                        raise ValueError(f"Illegal piece {piece}")

                    image = OffsetImage(piece_image, zoom=img_size, resample=True)

                    ab = AnnotationBbox(image, (j - 1, i), frameon=False, xycoords='data', boxcoords="data")
                    ax.add_artist(ab)

        ax.tick_params(left=False, right=False, labelleft=True, labelbottom=True, bottom=False)
        ax.set_aspect('equal')

        image_path = os.path.join(self.image_dir, f"step_{self.state.move_number()}.png")
        plt.savefig(image_path, dpi=500, bbox_inches='tight')
        plt.close()
        self.recorders[0].add_frame(image_path)
        return [image_path]
