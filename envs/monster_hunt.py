from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from typing import List
import dataclasses
import matplotlib.offsetbox as offsetbox
import numpy as np
import os
import random

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


@dataclasses.dataclass
class EnvState:
    red_pos: np.ndarray
    blue_pos: np.ndarray
    monster_pos: np.ndarray
    apple1_pos: np.ndarray
    apple2_pos: np.ndarray
    step: int

    red_apple: int
    red_monster: int
    blue_apple: int
    blue_monster: int
    coop_monster: int
    terminated: bool
    truncated: bool

    def to_dict(self):
        return {
            "red_pos": self.red_pos.tolist(),
            "blue_pos": self.blue_pos.tolist(),
            "monster_pos": self.monster_pos.tolist(),
            "apple1_pos": self.apple1_pos.tolist(),
            "apple2_pos": self.apple2_pos.tolist(),
            "step": self.step,
            "red_apple": self.red_apple,
            "red_monster": self.red_monster,
            "blue_apple": self.blue_apple,
            "blue_monster": self.blue_monster,
            "coop_monster": self.coop_monster,
            "terminated": self.terminated,
            "truncated": self.truncated
        }

    @classmethod
    def from_dict(cls, data):
        return cls(red_pos=np.array(data["red_pos"]), blue_pos=np.array(data["blue_pos"]),
                   monster_pos=np.array(data["monster_pos"]), apple1_pos=np.array(data["apple1_pos"]),
                   apple2_pos=np.array(data["apple2_pos"]), step=data["step"], red_apple=data["red_apple"],
                   red_monster=data["red_monster"], blue_apple=data["blue_apple"], blue_monster=data["blue_monster"],
                   coop_monster=data["coop_monster"], terminated=data["terminated"], truncated=data["truncated"])


MOVES = np.array([
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
])


class MonsterHunt(BaseEnv, env_type="monster_hunt"):
    """
    Monster Hunt environment implementation using NumPy.
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
        self.apple_reward = 2
        self.monster_penalty = -2
        self.coop_reward = 5

        self.visual_obs = visual_obs
        if self.visual_obs:
            assert image_dir is not None, "image_dir must not be None."
            self.image_dir = image_dir
            self.recorders = [Recorder(image_dir, recording_type, fps=recording_fps)]

        self.state = None
        self.scores = [0, 0]
        self.action_mapping = {
            0: '<STAY>',
            1: '<RIGHT>',
            2: '<LEFT>',
            3: '<UP>',
            4: '<DOWN>',
        }
        self.red_player_img = Image.open(os.path.join('images', 'monster_hunt', 'red_player.png')).convert('RGBA')
        self.blue_player_img = Image.open(os.path.join('images', 'monster_hunt', 'blue_player.png')).convert('RGBA')
        self.monster_img = Image.open(os.path.join('images', 'monster_hunt', 'monster.png')).convert('RGBA')
        self.apple_img = Image.open(os.path.join('images', 'monster_hunt', 'apple.png')).convert('RGBA')
        self.both_players_img = Image.open(os.path.join('images', 'monster_hunt', 'both_players.png')).convert('RGBA')

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
            monster_pos=all_pos[2, :],
            apple1_pos=all_pos[3, :],
            apple2_pos=all_pos[4, :],
            step=0,
            red_apple=0,
            red_monster=0,
            blue_apple=0,
            blue_monster=0,
            coop_monster=0,
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

    @property
    def regex_patterns(self):
        patterns = [
            (r'```json\s*\{\s*"action"\s*:\s*"([^"]+)"\s*\}\s*```', lambda m: m.strip()),
            (r'"action"\s*:\s*"([^"]+)"', lambda m: m.strip()),
            (r'<(UP|DOWN|LEFT|RIGHT|STAY)>', lambda m: f"<{m.upper()}>"),
            (r'\b(UP|DOWN|LEFT|RIGHT|STAY)\b', lambda m: f"<{m.upper()}>"),
        ]
        return patterns

    def _get_info(self):
        if self.state.terminated or self.state.truncated:
            return {
                'returns': self.scores,
                'red_player_eat_apple': self.state.red_apple,
                'blue_player_eat_apple': self.state.blue_apple,
                'red_player_meet_monster': self.state.red_monster - self.state.coop_monster,
                'blue_player_meet_monster': self.state.blue_monster - self.state.coop_monster,
                'coop_defeat_monster': self.state.coop_monster,
            }
        return None

    def _get_legal_actions(self, agent_id):
        return self.action_mapping

    def _action_to_string(self, action):
        return self.action_mapping.get(action, f'UNKNOWN_ACTION_{action}')

    def _save_image(self):
        image_path = os.path.join(self.image_dir, f'step_{self.state.step}.png')
        img = self._render(self.state)
        img.save(image_path)
        self.recorders[0].add_frame(image_path)
        return [image_path]

    def _step(self, state: EnvState, actions: List[int]):
        new_red_pos = np.clip(state.red_pos + MOVES[actions[0]], 0, self.grid_size - 1)
        new_blue_pos = np.clip(state.blue_pos + MOVES[actions[1]], 0, self.grid_size - 1)

        red_dist = np.sum(np.abs(state.monster_pos - new_red_pos))
        blue_dist = np.sum(np.abs(state.monster_pos - new_blue_pos))
        if red_dist < blue_dist:
            rel_position = new_red_pos - state.monster_pos
        elif blue_dist < red_dist:
            rel_position = new_blue_pos - state.monster_pos
        else:
            rel_position = random.choice([new_red_pos - state.monster_pos, new_blue_pos - state.monster_pos])
        monster_move = np.sign(rel_position)
        if len(monster_move[monster_move != 0]) == 2:
            monster_move = random.choice([np.array([monster_move[0], 0]), np.array([0, monster_move[1]])])
        new_monster_pos = np.clip(state.monster_pos + monster_move, 0, self.grid_size - 1)

        red_reward, blue_reward = 0, 0

        red_apple1_match = np.all(new_red_pos == state.apple1_pos)
        red_apple2_match = np.all(new_red_pos == state.apple2_pos)
        blue_apple1_match = np.all(new_blue_pos == state.apple1_pos)
        blue_apple2_match = np.all(new_blue_pos == state.apple2_pos)
        monster_apple1_match = np.all(new_monster_pos == state.apple1_pos)
        monster_apple2_match = np.all(new_monster_pos == state.apple2_pos)

        red_monster_match = np.all(new_red_pos == new_monster_pos)
        blue_monster_match = np.all(new_blue_pos == new_monster_pos)
        coop_match = red_monster_match and blue_monster_match

        if red_apple1_match or red_apple2_match:
            red_reward += self.apple_reward
        if blue_apple1_match or blue_apple2_match:
            blue_reward += self.apple_reward

        if coop_match:
            red_reward += self.coop_reward
            blue_reward += self.coop_reward
        else:
            if red_monster_match:
                red_reward += self.monster_penalty
            if blue_monster_match:
                blue_reward += self.monster_penalty

        new_apple1_pos = state.apple1_pos
        new_apple2_pos = state.apple2_pos
        if red_apple1_match or blue_apple1_match or monster_apple1_match:
            new_apple1_pos = np.random.randint(0, self.grid_size, size=2)
            while (np.array_equal(new_apple1_pos, new_red_pos) or np.array_equal(new_apple1_pos, new_blue_pos)
                   or np.array_equal(new_apple1_pos, new_monster_pos)
                   or np.array_equal(new_apple1_pos, new_apple2_pos)):
                new_apple1_pos = np.random.randint(0, self.grid_size, size=2)

        if red_apple2_match or blue_apple2_match or monster_apple2_match:
            new_apple2_pos = np.random.randint(0, self.grid_size, size=2)
            while (np.array_equal(new_apple2_pos, new_red_pos) or np.array_equal(new_apple2_pos, new_blue_pos)
                   or np.array_equal(new_apple2_pos, new_monster_pos)
                   or np.array_equal(new_apple2_pos, new_apple1_pos)):
                new_apple2_pos = np.random.randint(0, self.grid_size, size=2)

        if coop_match:
            new_monster_pos = np.random.randint(0, self.grid_size, size=2)
            while (np.array_equal(new_monster_pos, new_red_pos) or np.array_equal(new_monster_pos, new_blue_pos)
                   or np.array_equal(new_monster_pos, new_apple1_pos)
                   or np.array_equal(new_monster_pos, new_apple2_pos)):
                new_monster_pos = np.random.randint(0, self.grid_size, size=2)
        elif red_monster_match:
            new_red_pos = np.random.randint(0, self.grid_size, size=2)
            while (np.array_equal(new_red_pos, new_monster_pos) or np.array_equal(new_red_pos, new_blue_pos)
                   or np.array_equal(new_red_pos, new_apple1_pos) or np.array_equal(new_red_pos, new_apple2_pos)):
                new_red_pos = np.random.randint(0, self.grid_size, size=2)
        elif blue_monster_match:
            new_blue_pos = np.random.randint(0, self.grid_size, size=2)
            while (np.array_equal(new_blue_pos, new_monster_pos) or np.array_equal(new_blue_pos, new_red_pos)
                   or np.array_equal(new_blue_pos, new_apple1_pos) or np.array_equal(new_blue_pos, new_apple2_pos)):
                new_blue_pos = np.random.randint(0, self.grid_size, size=2)

        next_state = EnvState(
            red_pos=new_red_pos,
            blue_pos=new_blue_pos,
            monster_pos=new_monster_pos,
            apple1_pos=new_apple1_pos,
            apple2_pos=new_apple2_pos,
            step=state.step + 1,
            red_apple=state.red_apple + (red_apple1_match or red_apple2_match),
            red_monster=state.red_monster + red_monster_match,
            blue_apple=state.blue_apple + (blue_apple1_match or blue_apple2_match),
            blue_monster=state.blue_monster + blue_monster_match,
            coop_monster=state.coop_monster + coop_match,
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
        obs1 = np.concatenate([state.red_pos, state.blue_pos, state.monster_pos, state.apple1_pos, state.apple2_pos])
        obs2 = np.concatenate([state.blue_pos, state.red_pos, state.monster_pos, state.apple1_pos, state.apple2_pos])
        return [obs1, obs2]

    def _relative_position(self, state: EnvState) -> np.ndarray:
        red_offset = -state.red_pos
        rel_other_player = state.blue_pos + red_offset
        rel_monster = state.monster_pos + red_offset
        rel_apple1 = state.apple1_pos + red_offset
        rel_apple2 = state.apple2_pos + red_offset
        obs1 = np.concatenate(
            [np.zeros(2, dtype=state.red_pos.dtype), red_offset, rel_other_player, rel_monster, rel_apple1, rel_apple2])

        blue_offset = -state.blue_pos
        rel_other_player = state.red_pos + blue_offset
        rel_monster = state.monster_pos + blue_offset
        rel_apple1 = state.apple1_pos + blue_offset
        rel_apple2 = state.apple2_pos + blue_offset
        obs2 = np.concatenate([
            np.zeros(2, dtype=state.blue_pos.dtype), blue_offset, rel_other_player, rel_monster, rel_apple1, rel_apple2
        ])
        return [obs1, obs2]

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

        ax.set_title("Monster Hunt", fontweight='bold', fontsize=14)
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
        monster_pos = (int(state.monster_pos[0]), int(state.monster_pos[1]))
        apple1_pos = (int(state.apple1_pos[0]), int(state.apple1_pos[1]))
        apple2_pos = (int(state.apple2_pos[0]), int(state.apple2_pos[1]))

        def draw_icon(ax, img, pos, zoom=0.11):
            imagebox = offsetbox.OffsetImage(img, zoom=zoom)
            ab = offsetbox.AnnotationBbox(imagebox, (pos[0] + 0.5, pos[1] + 0.5), frameon=False)
            ax.add_artist(ab)

        if np.array_equal(state.red_pos, state.blue_pos):
            draw_icon(ax, self.both_players_img, red_pos)
        else:
            draw_icon(ax, self.red_player_img, red_pos)
            draw_icon(ax, self.blue_player_img, blue_pos)
        draw_icon(ax, self.monster_img, monster_pos)
        draw_icon(ax, self.apple_img, apple1_pos)
        draw_icon(ax, self.apple_img, apple2_pos)

        if self.additional_info:
            ax2 = fig.add_subplot(122)
            ax2.axis("off")

            ax2.text(0.05, 1.0, "Events", fontsize=12, va='center', fontweight='bold')
            ax2.text(0.75, 1.0, "Counter", fontsize=12, va='center', fontweight='bold')

            row_height = 0.18
            y0 = 0.80

            rows = [
                [(self.red_player_img, self.apple_img), [('#CE5E5D', '+2')], state.red_apple],
                [(self.blue_player_img, self.apple_img), [('#6E9EEB', '+2')], state.blue_apple],
                [(self.red_player_img, self.monster_img), [('#CE5E5D', '-2')], state.red_monster - state.coop_monster],
                [(self.blue_player_img, self.monster_img), [('#6E9EEB', '-2')],
                 state.blue_monster - state.coop_monster],
                [(self.red_player_img, self.blue_player_img, self.monster_img), [('#CE5E5D', '+5'), ('#6E9EEB', '+5')],
                 state.coop_monster],
            ]
            for i, (icons, rewards, counter) in enumerate(rows):
                y = y0 - i * row_height
                x_icon = 0.0

                for j, img in enumerate(icons):
                    imagebox = offsetbox.OffsetImage(img, zoom=0.08)
                    ab = offsetbox.AnnotationBbox(imagebox, (x_icon, y), frameon=False, xycoords='axes fraction')
                    ax2.add_artist(ab)
                    x_icon += 0.10
                    if j < len(icons) - 1:
                        ax2.text(x_icon, y, '+', fontsize=14, va='center', ha='center', transform=ax2.transAxes)
                        x_icon += 0.10
                ax2.text(x_icon, y, '→', fontsize=14, va='center', ha='center', transform=ax2.transAxes)
                x_icon += 0.10

                for color, text in rewards:
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
