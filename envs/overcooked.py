from matplotlib import pyplot as plt
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from typing import List, Union
import copy
import cv2
import gym
import numpy as np
import os
import pygame
import re

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


class Overcooked(BaseEnv, env_type="overcooked"):

    def __init__(
        self,
        layout_name="cramped_room",
        max_steps=50,
        visual_obs=True,
        image_dir=None,
        recording_type='gif',
        recording_fps=5,
    ):
        self.done = False
        self.num_steps = 0
        self.max_steps = max_steps
        self.layout_name = layout_name
        self.visualizer = StateVisualizer(is_rendering_hud=False)

        recipe_config = {"cook_time": 5, "delivery_reward": 10}
        self.base_mdp = OvercookedGridworld.from_layout_name(self.layout_name, **recipe_config)
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.state = None
        self.env.reset()

        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.scores = [0, 0]
        self.current_player_idx = 0

        self.player_actions = [Action.STAY, Action.STAY]

        self.action_mapping = {0: '<UP>', 1: '<DOWN>', 2: '<RIGHT>', 3: '<LEFT>', 4: '<STAY>', 5: '<INTERACT>'}
        self.image_paths = []
        self.addition_info = ""
        self.action_history = []
        self.last_game_stats = {}

        self.visual_obs = visual_obs
        if self.visual_obs:
            assert image_dir is not None, "image_dir must not be None."
            self.image_dir = image_dir
            self.recorders = [Recorder(image_dir, recording_type, recording_fps)]

    @property
    def current_player(self):
        return self.current_player_idx

    def reset(self, seed=0) -> List[Union[Observation, None]]:

        self.num_steps = 0
        self.done = False
        self.env.reset()
        self.scores = [0, 0]
        self.current_player_idx = 0
        self.player_actions = [Action.STAY, Action.STAY]

        self.action_history = []

        self.state = {'observation': self.env.state, 'return': [0, 0], 'info': {}}
        if self.visual_obs:
            self.recorders[0].clear()
            self.image_paths = self._save_image()

        return [self._get_observation(0), self._get_observation(1)]

    def is_terminal(self):
        if self.num_steps >= self.max_steps:
            self.done = True

        return self.done

    def step(self, actions):
        if self.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        self.num_steps += 1
        choose_action = [self.choose_action_readable(action) for action in actions]
        joint_action = [Action.ALL_ACTIONS[action] for action in actions]

        next_state, reward, done, info = self.env.step(joint_action)

        for key, value in self.env.game_stats.items():
            if key in [
                    "cumulative_shaped_rewards_by_agent",
                    "cumulative_sparse_rewards_by_agent",
            ]:
                previous = self.last_game_stats.get(key, np.zeros_like(value))
                reward = value - previous

                self.last_game_stats[key] = value.copy()
                reward = int(np.sum(reward))
                self.scores[0] += reward
                self.scores[1] += reward

        info = {
            'rewards': [reward, reward],
            'returns': [self.scores[0], self.scores[1]],
            'joint_action': [choose_action[0], choose_action[1]],
        }

        action_info = f"In timestep {self.num_steps - 1}: chef_0 chooses {self.action_mapping[actions[0]]}, chef_1 chooses {self.action_mapping[actions[1]]}."
        self.action_history.append(action_info)

        self.state = {'observation': next_state, 'return': [self.scores[0], self.scores[1]], 'info': info}

        if self.num_steps >= self.max_steps:
            self.done = True
        else:
            self.done = done

        if self.visual_obs:
            self.image_paths = self._save_image()
            if self.done:
                self.recorders[0].save()

        observations = [self._get_observation(0), self._get_observation(1)]
        rewards = [reward, reward]
        dones = [self.done, self.done]

        return observations, rewards, dones, info

    def _get_observation(self, agent_id):
        """Return the Observation object for agent agent_id."""
        play_state = str(self.env.state.players)
        play_txt_state = parse_chef_state(play_state)
        overall_txt_state = str(self.env) + play_txt_state
        recent_actions = self.get_recent_actions(3)
        return Observation(obs=self.state['observation'], agent_id=agent_id, image_paths=self.image_paths,
                           legal_actions=self._get_legal_actions(agent_id), serialized_state=overall_txt_state,
                           regex_patterns=self.regex_patterns, addition_info=recent_actions)

    def _get_legal_actions(self, agent_id):
        return self.action_mapping

    def _get_info(self):
        if self.is_terminal():
            return {
                'returns': self.scores,
            }
        return None

    def create_joint_action(self):
        return [self.player_actions[0], self.player_actions[1]]

    def legal_actions(self):
        return list(range(len(Action.ALL_ACTIONS)))

    def legal_actions_readable(self):
        return [Action.ACTION_TO_CHAR[Action.ALL_ACTIONS[i]] for i in self.legal_actions()]

    def choose_action_readable(self, action):
        return Action.ACTION_TO_CHAR[Action.ALL_ACTIONS[action]]

    def _save_image(self):
        image = self.render()
        legend_images = self.create_legend()
        combined_image = self.add_legend_to_image(image, legend_images)

        plt.figure(figsize=(12, 10))
        plt.imshow(combined_image)
        plt.axis('off')

        image_path = os.path.join(self.image_dir, f"step_{self.num_steps}.png")
        plt.savefig(image_path, dpi=500, bbox_inches='tight')
        plt.close()
        self.recorders[0].add_frame(image_path)
        return [image_path]

    def create_legend(self):
        legend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "overcooked"))
        legend_images = {}

        for item in ["onion", "dish", "pot", "counter", "serving location", "available area"]:
            if os.path.exists(os.path.join(legend_path, f"{item}.png")):
                img = cv2.imread(os.path.join(legend_path, f"{item}.png"))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    legend_images[item] = cv2.resize(img, (100, 100))
            elif os.path.exists(os.path.join(legend_path, f"{item}.jpg")):
                img = cv2.imread(os.path.join(legend_path, f"{item}.jpg"))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    legend_images[item] = cv2.resize(img, (100, 100))

        for player in [
                "chef_0: facing up", "chef_0: facing down", "chef_0: facing left", "chef_0: facing right",
                "chef_1: facing up", "chef_1: facing down", "chef_1: facing left", "chef_1: facing right"
        ]:
            if os.path.exists(os.path.join(legend_path, f"{player}.png")):
                img = cv2.imread(os.path.join(legend_path, f"{player}.png"))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    legend_images[player] = cv2.resize(img, (82, 96))

        return legend_images

    def add_legend_to_image(self, game_image, legend_images):
        game_height, game_width = game_image.shape[:2]

        legend_width = 800
        combined_image = np.ones((game_height, game_width + legend_width, 3), dtype=np.uint8) * 255

        combined_image[:, :game_width, :] = game_image

        font = cv2.FONT_HERSHEY_SIMPLEX
        title_y = 40
        cv2.putText(combined_image, "Legend", (game_width + 50, title_y), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        step_text = f"Step: {self.num_steps}"
        cv2.putText(combined_image, step_text, (35, 60), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        item_x_base = game_width + 50
        player_x_base = game_width + 380

        start_y = title_y + 30

        small_items = ["onion", "dish", "pot", "counter", "serving location", "available area"]
        y = start_y
        for item in small_items:
            if item in legend_images:
                img = legend_images[item]
                h, w = img.shape[:2]
                combined_image[y:y + h, item_x_base:item_x_base + w] = img
                cv2.putText(combined_image, item, (item_x_base + w + 10, y + h // 2), font, 0.8, (0, 0, 0), 1,
                            cv2.LINE_AA)
                y += h + 30

        players = [
            "chef_0: facing up", "chef_0: facing down", "chef_0: facing left", "chef_0: facing right",
            "chef_1: facing up", "chef_1: facing down", "chef_1: facing left", "chef_1: facing right"
        ]
        y2 = start_y
        for player in players:
            if player in legend_images:
                img = legend_images[player]
                h, w = img.shape[:2]
                combined_image[y2:y2 + h, player_x_base:player_x_base + w] = img
                cv2.putText(combined_image, player, (player_x_base + w + 10, y2 + h // 2), font, 0.8, (0, 0, 0), 1,
                            cv2.LINE_AA)
                y2 += h + 10

        return combined_image

    def render(self):
        rewards_dict = {}  # dictionary of details you want rendered in the UI
        total_score = 0
        for key, value in self.env.game_stats.items():
            if key in [
                    "cumulative_shaped_rewards_by_agent",
                    "cumulative_sparse_rewards_by_agent",
            ]:
                rewards_dict[key] = value
                total_score += int(np.sum(value))

        rewards_dict["score"] = total_score
        image = self.visualizer.render_state(
            state=self.env.state,
            grid=self.env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(self.env.state, **rewards_dict),
        )

        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 580, 2 * 464))
        return image

    def get_recent_actions(self, num_actions=3):
        if not self.action_history:
            return ""

        recent_actions = self.action_history[-num_actions:] if len(
            self.action_history) > num_actions else self.action_history

        recent_actions = [f"{i+1}. {action}" for i, action in enumerate(recent_actions)]
        return "\n".join(recent_actions)


def parse_chef_state(players_str: str) -> str:
    """
    eg.
    input:
      "((1, 1) facing (0, -1) holding dish@(1, 1), (2, 1) facing (0, -1) holding None)"
    output:
      chef 0 local in (1, 1), facing ↑, hold dish.
      chef 1 local in (2, 1), facing ↑, hold nothing.
    """
    s = players_str.strip()

    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1].strip()

    pattern = re.compile(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
                         r'\s*facing\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
                         r'\s*holding\s*'
                         r'(.+?)'
                         r'(?=(?:,\s*\(\s*-?\d)|\s*$)')
    matches = pattern.findall(s)
    if not matches:
        return ""

    lines = []
    for idx, (px, py, fx, fy, raw_hold) in enumerate(matches):
        px, py, fx, fy = map(int, (px, py, fx, fy))
        arrow = Action.ACTION_TO_CHAR.get((fx, fy), f'({fx}, {fy})')

        hold_clean = raw_hold.strip()
        prefix_m = re.match(r'^([A-Za-z]+)', hold_clean)
        if prefix_m:
            prefix = prefix_m.group(1).lower()
        else:
            prefix = None

        if prefix == 'none':
            hold_text = 'nothing'
        elif prefix:
            hold_text = prefix
        else:
            hold_text = 'soup'

        lines.append(f'Chef {idx} local in ({px}, {py}), facing {arrow}, hold {hold_text}.')

    return '\n'.join(lines)
