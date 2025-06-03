from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from typing import List
import numpy as np
import dataclasses
import os
import matplotlib.offsetbox as offsetbox

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


@dataclasses.dataclass
class EnvState:
    red_pos: np.ndarray
    blue_pos: np.ndarray
    red_coin_pos: np.ndarray
    blue_coin_pos: np.ndarray
    step: int

    red_coop: int
    red_defect: int
    blue_coop: int
    blue_defect: int
    counter: np.ndarray
    terminated: bool
    truncated: bool

    def to_dict(self):
        return {
            "red_pos": self.red_pos.tolist(),
            "blue_pos": self.blue_pos.tolist(),
            "red_coin_pos": self.red_coin_pos.tolist(),
            "blue_coin_pos": self.blue_coin_pos.tolist(),
            "step": self.step,
            "red_coop": self.red_coop,
            "red_defect": self.red_defect,
            "blue_coop": self.blue_coop,
            "blue_defect": self.blue_defect,
            "counter": self.counter.tolist(),
            "terminated": self.terminated,
            "truncated": self.truncated
        }

    @classmethod
    def from_dict(cls, data):
        return cls(red_pos=np.array(data["red_pos"]), blue_pos=np.array(data["blue_pos"]),
                   red_coin_pos=np.array(data["red_coin_pos"]), blue_coin_pos=np.array(data["blue_coin_pos"]),
                   step=data["step"], red_coop=data["red_coop"], red_defect=data["red_defect"],
                   blue_coop=data["blue_coop"], blue_defect=data["blue_defect"], counter=np.array(data["counter"]),
                   terminated=data["terminated"], truncated=data["truncated"])


MOVES = np.array([
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
])


class CoinDilemma(BaseEnv, env_type="coin_dilemma"):
    """
    Coin dilemma environment implementation using NumPy.
    """

    def __init__(
        self,
        grid_size=5,
        max_steps=50,
        egocentric=False,
        additional_info=False,
        visual_obs=True,
        image_dir=None,
        recording_type='gif',
        recording_fps=2,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.egocentric = egocentric
        self.additional_info = additional_info

        self.visual_obs = visual_obs
        if self.visual_obs:
            assert image_dir is not None, "image_dir must not be None."
            self.image_dir = image_dir
            self.recorders = [Recorder(image_dir, recording_type, recording_fps)]

        self.state = None
        self.scores = [0, 0]
        self.action_mapping = {
            0: '<STAY>',
            1: '<RIGHT>',
            2: '<LEFT>',
            3: '<UP>',
            4: '<DOWN>',
        }
        self.red_player_img = Image.open(os.path.join('images', 'coin_dilemma', 'red_player.png')).convert('RGBA')
        self.blue_player_img = Image.open(os.path.join('images', 'coin_dilemma', 'blue_player.png')).convert('RGBA')
        self.red_coin_img = Image.open(os.path.join('images', 'coin_dilemma', 'red_coin.png')).convert('RGBA')
        self.blue_coin_img = Image.open(os.path.join('images', 'coin_dilemma', 'blue_coin.png')).convert('RGBA')
        self.both_players_img = Image.open(os.path.join('images', 'coin_dilemma', 'both_players.png')).convert('RGBA')

    @property
    def current_player(self):
        return 0, 1

    def reset(self, seed=0):
        np.random.seed(seed)
        total_cells = self.grid_size * self.grid_size
        indices = np.random.choice(total_cells, size=5, replace=False)
        all_pos = np.column_stack((indices // self.grid_size, indices % self.grid_size))

        self.state = EnvState(
            red_pos=all_pos[0, :],
            blue_pos=all_pos[1, :],
            red_coin_pos=all_pos[2, :],
            blue_coin_pos=all_pos[3, :],
            step=0,
            red_coop=0,
            red_defect=0,
            blue_coop=0,
            blue_defect=0,
            counter=np.zeros(9),
            terminated=False,
            truncated=False,
        )
        self.scores = [0, 0]

        if self.visual_obs:
            self.recorders[0].clear()
            self.image_paths = self._save_image()
        return [self._get_observation(0), self._get_observation(1)]

    def step(self, actions):
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        next_state, rewards = self._step(self.state, actions)
        self.state = next_state

        for i, reward in enumerate(rewards):
            self.scores[i] += reward

        terminated = False
        truncated = self.state.step >= self.max_steps
        self.state.terminated = terminated
        self.state.truncated = truncated
        done = terminated or truncated
        dones = [done, done]

        if self.visual_obs:
            self.image_paths = self._save_image()
            if done:
                self.recorders[0].save()
        obs = [self._get_observation(0), self._get_observation(1)]
        return obs, rewards, dones, self._get_info()

    def _get_observation(self, agent_id):
        obs = self._state_to_obs(self.state)
        return Observation(
            obs=obs[agent_id],
            agent_id=agent_id,
            image_paths=self.image_paths if self.visual_obs else None,
            legal_actions=self._get_legal_actions(agent_id),
            serialized_state=str(self.state),
            regex_patterns=self.regex_patterns,
            addition_info=None,
        )

    def _get_info(self):
        if self.state.terminated or self.state.truncated:
            return {
                'returns': self.scores,
                'red_player_collect_red_coin': self.state.red_coop,
                'red_player_collect_blue_coin': self.state.red_defect,
                'blue_player_collect_red_coin': self.state.blue_defect,
                'blue_player_collect_blue_coin': self.state.blue_coop,
            }
        return None

    @property
    def regex_patterns(self):
        patterns = [
            (r'```json\s*\{\s*"action"\s*:\s*"([^"]+)"\s*\}\s*```', lambda m: m.strip()),
            (r'"action"\s*:\s*"([^"]+)"', lambda m: m.strip()),
            (r'<(UP|DOWN|LEFT|RIGHT|STAY)>', lambda m: f"<{m.upper()}>"),
            (r'\b(UP|DOWN|LEFT|RIGHT|STAY)\b', lambda m: f"<{m.upper()}>"),
        ]
        return patterns

    def _get_legal_actions(self, agent_id):
        return self.action_mapping

    def _action_to_string(self, action):
        return self.action_mapping.get(action, f'UNKNOWN_ACTION_{action}')

    def _save_image(self):
        if not self.visual_obs:
            return None

        image_path = os.path.join(self.image_dir, f'step_{self.state.step}.png')
        img = self._render(self.state)
        img.save(image_path)
        self.recorders[0].add_frame(image_path)
        return [image_path]

    def _step(self, state: EnvState, actions: List[int]):
        new_red_pos = np.clip(state.red_pos + MOVES[actions[0]], 0, self.grid_size - 1)
        new_blue_pos = np.clip(state.blue_pos + MOVES[actions[1]], 0, self.grid_size - 1)
        red_reward, blue_reward = 0, 0

        red_red_match = np.all(new_red_pos == state.red_coin_pos)
        red_blue_match = np.all(new_red_pos == state.blue_coin_pos)
        blue_red_match = np.all(new_blue_pos == state.red_coin_pos)
        blue_blue_match = np.all(new_blue_pos == state.blue_coin_pos)

        _rr_reward = 1
        _rb_reward = 1
        _r_penalty = -2
        _br_reward = 1
        _bb_reward = 1
        _b_penalty = -2

        if red_red_match:
            red_reward += _rr_reward
        if red_blue_match:
            red_reward += _rb_reward
        if blue_red_match:
            red_reward += _r_penalty

        if blue_red_match:
            blue_reward += _br_reward
        if blue_blue_match:
            blue_reward += _bb_reward
        if red_blue_match:
            blue_reward += _b_penalty

        counter = self._update_stats(
            state,
            red_red_match,
            red_blue_match,
            blue_red_match,
            blue_blue_match,
        )

        new_red_coin_pos = state.red_coin_pos
        new_blue_coin_pos = state.blue_coin_pos
        if red_red_match or blue_red_match:
            new_red_coin_pos = np.random.randint(0, self.grid_size, size=2)
            while (np.array_equal(new_red_coin_pos, new_red_pos) or np.array_equal(new_red_coin_pos, new_blue_pos)
                   or np.array_equal(new_red_coin_pos, new_blue_coin_pos)):
                new_red_coin_pos = np.random.randint(0, self.grid_size, size=2)

        if red_blue_match or blue_blue_match:
            new_blue_coin_pos = np.random.randint(0, self.grid_size, size=2)
            while (np.array_equal(new_blue_coin_pos, new_red_pos) or np.array_equal(new_blue_coin_pos, new_blue_pos)
                   or np.array_equal(new_blue_coin_pos, new_red_coin_pos)):
                new_blue_coin_pos = np.random.randint(0, self.grid_size, size=2)

        next_state = EnvState(
            red_pos=new_red_pos,
            blue_pos=new_blue_pos,
            red_coin_pos=new_red_coin_pos,
            blue_coin_pos=new_blue_coin_pos,
            step=state.step + 1,
            red_coop=state.red_coop + red_red_match,
            red_defect=state.red_defect + red_blue_match,
            blue_coop=state.blue_coop + blue_blue_match,
            blue_defect=state.blue_defect + blue_red_match,
            counter=counter,
            terminated=state.terminated,
            truncated=state.truncated,
        )
        rewards = [red_reward, blue_reward]
        return next_state, rewards

    def _state_to_obs(self, state: EnvState) -> List[np.ndarray]:
        if self.egocentric:
            return self._relative_position(state)
        else:
            return self._abs_position(state)

    def _abs_position(self, state: EnvState) -> List[np.ndarray]:
        obs1 = np.concatenate([state.red_pos, state.blue_pos, state.red_coin_pos, state.blue_coin_pos])
        obs2 = np.concatenate([state.blue_pos, state.red_pos, state.blue_coin_pos, state.red_coin_pos])
        return [obs1, obs2]

    def _relative_position(self, state: EnvState) -> np.ndarray:
        red_offset = -state.red_pos
        rel_blue_player = state.blue_pos + red_offset
        rel_red_coin = state.red_coin_pos + red_offset
        rel_blue_coin = state.blue_coin_pos + red_offset
        obs1 = np.concatenate([
            np.zeros(2, dtype=state.red_pos.dtype),
            rel_blue_player,
            rel_red_coin,
            rel_blue_coin,
        ])

        blue_offset = -state.blue_pos
        rel_red_player = state.red_pos + blue_offset
        rel_blue_coin = state.blue_coin_pos + blue_offset
        rel_red_coin = state.red_coin_pos + blue_offset
        obs2 = np.concatenate([
            np.zeros(2, dtype=state.blue_pos.dtype),
            rel_red_player,
            rel_blue_coin,
            rel_red_coin,
        ])
        return [obs1, obs2]

    def _update_stats(
        self,
        state: EnvState,
        rr: bool,
        rb: bool,
        br: bool,
        bb: bool,
    ):

        def state2idx(s: np.ndarray) -> int:
            if np.array_equal(s, np.array([1, 1])):
                return 1
            elif np.array_equal(s, np.array([1, 2])):
                return 2
            elif np.array_equal(s, np.array([2, 1])):
                return 3
            elif np.array_equal(s, np.array([2, 2])):
                return 4
            elif np.array_equal(s, np.array([0, 1])):
                return 5
            elif np.array_equal(s, np.array([0, 2])):
                return 6
            elif np.array_equal(s, np.array([2, 0])):
                return 7
            elif np.array_equal(s, np.array([1, 0])):
                return 8
            return 0

        a1 = 0
        if rr:
            a1 = 1
        elif rb:
            a1 = 2

        a2 = 0
        if bb:
            a2 = 1
        elif br:
            a2 = 2

        idx = state2idx(np.array([a1, a2]))
        state.counter[idx] += 1
        return state.counter

    def _render(self, state: EnvState):
        """Small utility for plotting the agent's state."""
        if self.additional_info:
            fig = Figure((5, 2))
        else:
            fig = Figure((2, 2))
        canvas = FigureCanvasAgg(fig)
        if self.additional_info:
            ax = fig.add_subplot(121)
        else:
            ax = fig.add_subplot(111)

        ax.set_title("Coin Dilemma", fontweight='bold', fontsize=14)
        ax.imshow(
            np.zeros((self.grid_size, self.grid_size)),
            cmap="Greys",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="none",
            origin="lower",
            extent=[0, self.grid_size, 0, self.grid_size],
        )
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(1, self.grid_size + 1))
        ax.set_yticks(np.arange(1, self.grid_size + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid()

        red_pos = (int(state.red_pos[0]), int(state.red_pos[1]))
        blue_pos = (int(state.blue_pos[0]), int(state.blue_pos[1]))
        red_coin_pos = (int(state.red_coin_pos[0]), int(state.red_coin_pos[1]))
        blue_coin_pos = (int(state.blue_coin_pos[0]), int(state.blue_coin_pos[1]))

        def draw_icon(ax, img, pos, zoom=0.11):
            imagebox = offsetbox.OffsetImage(img, zoom=zoom)
            ab = offsetbox.AnnotationBbox(imagebox, (pos[0] + 0.5, pos[1] + 0.5), frameon=False)
            ax.add_artist(ab)

        if np.array_equal(state.red_pos, state.blue_pos):
            draw_icon(ax, self.both_players_img, red_pos)
        else:
            draw_icon(ax, self.red_player_img, red_pos)
            draw_icon(ax, self.blue_player_img, blue_pos)
        draw_icon(ax, self.red_coin_img, red_coin_pos)
        draw_icon(ax, self.blue_coin_img, blue_coin_pos)

        if self.additional_info:
            ax2 = fig.add_subplot(122)
            ax2.axis("off")

            ax2.text(0.05, 1.0, "Events", fontsize=12, va='center', fontweight='bold')
            ax2.text(0.75, 1.0, "Counter", fontsize=12, va='center', fontweight='bold')

            row_height = 0.2
            y0 = 0.80
            rows = [
                [(self.red_player_img, self.red_coin_img), [('#CE5E5D', '+1')], state.red_coop],
                [(self.red_player_img, self.blue_coin_img), [('#CE5E5D', '+1'), ('#6E9EEB', '-2')], state.red_defect],
                [(self.blue_player_img, self.blue_coin_img), [('#6E9EEB', '+1')], state.blue_coop],
                [(self.blue_player_img, self.red_coin_img), [('#6E9EEB', '+1'), ('#CE5E5D', '-2')], state.blue_defect],
            ]
            for i, (icons, rewards, counter) in enumerate(rows):
                y = y0 - i * row_height

                x_icon = 0.05
                imagebox = offsetbox.OffsetImage(icons[0], zoom=0.08)
                ab = offsetbox.AnnotationBbox(imagebox, (x_icon, y), frameon=False, xycoords='axes fraction')
                ax2.add_artist(ab)

                x_icon += 0.10
                ax2.text(x_icon, y, '+', fontsize=14, va='center', ha='center', transform=ax2.transAxes)

                x_icon += 0.10
                imagebox = offsetbox.OffsetImage(icons[1], zoom=0.08)
                ab = offsetbox.AnnotationBbox(imagebox, (x_icon, y), frameon=False, xycoords='axes fraction')
                ax2.add_artist(ab)

                x_icon += 0.10
                ax2.text(x_icon, y, '→', fontsize=14, va='center', ha='center', transform=ax2.transAxes)

                x_icon += 0.10
                for idx, (color, text) in enumerate(rewards):
                    ax2.text(x_icon, y, text, fontsize=12, va='center', ha='center', color=color, fontweight='bold',
                             transform=ax2.transAxes)
                    x_icon += 0.12

                ax2.text(0.96, y, str(counter), fontsize=12, va='center', ha='center', fontweight='bold',
                         transform=ax2.transAxes)
        fig.subplots_adjust(left=0.02, right=0.9, wspace=0.02)
        canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', (width, height), buffer)
        return image
