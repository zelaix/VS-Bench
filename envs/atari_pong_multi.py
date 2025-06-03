from pettingzoo.atari import pong_v3
from PIL import Image
import numpy as np
import os
import supersuit

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


class AtariPongMulti(BaseEnv, env_type="atari_pong_multi"):
    '''Atari Pong game environment (two players) using PettingZoo and ALE.'''

    def __init__(
        self,
        max_observation=False,
        skip_frame=False,
        resize_frame=False,
        color_reduction=False,
        normalize_obs=False,
        stack_frame=4,
        sticky_action=0.25,
        visual_obs=True,
        image_dir=None,
        recording_type='gif',
        recording_fps=10,
    ):
        self.env = pong_v3.env(num_players=2, render_mode='rgb_array')

        if max_observation:
            self.env = supersuit.max_observation_v0(self.env, memory=max_observation)
        if skip_frame:
            self.env = supersuit.frame_skip_v0(self.env, num_frames=skip_frame)
        if resize_frame:
            self.env = supersuit.resize_v1(self.env, *resize_frame)
        if color_reduction:
            self.env = supersuit.color_reduction_v0(self.env, mode=color_reduction)
        if normalize_obs:
            self.env = supersuit.normalize_obs_v0(self.env, env_min=0, env_max=1)
        if stack_frame:
            self.env = supersuit.frame_stack_v1(self.env, stack_size=stack_frame)
        if sticky_action:
            self.env = supersuit.sticky_actions_v0(self.env, repeat_action_probability=sticky_action)

        self.visual_obs = visual_obs
        if self.visual_obs:
            assert image_dir is not None, "image_dir must not be None."
            self.image_dir = image_dir
            self.recorders = [Recorder(image_dir, recording_type, recording_fps)]

        self.state = None
        self.scores = [0, 0]
        self.steps = 0
        self.game_name = self.config.get('game_name', 'atari_maze_craze_multi')
        self.noop_start = self.config.get('noop_start', False)
        self.max_steps_per_player = self.config.get('max_steps_per_player', None)
        self.wining_score = self.config.get('wining_score', 3)
        self.num_agents = 2
        self.image_paths = []

        self.action_mapping = {0: '<STAY>', 1: '<FIRE>', 2: '<UP>', 3: '<DOWN>'}

    @property
    def current_player(self):
        if self.env.agent_selection == 'first_0':
            return 0
        if self.env.agent_selection == 'second_0':
            return 1
        raise ValueError(f"Unknown agent selection: {self.env.agent_selection}")

    def reset(self, seed=0):
        self.env.reset(seed=seed)
        observation, reward, terminated, truncated, info = self.env.last()

        if self.noop_start:
            noop_steps = np.random.randint(1, 31)
            for _ in range(noop_steps):
                action = 0
                self.env.step(action)
                observation, reward, terminated, truncated, info = self.env.last()

        self.state = {
            'observation': observation,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info,
        }
        self.scores = [0, 0]
        self.steps = 0

        if self.visual_obs:
            self.recorders[0].clear()
            self.image_paths = self._save_image()
        return [self._get_observation(0), self._get_observation(1)]

    def step(self, actions):
        if self.state['terminated'] or self.state['truncated']:
            raise RuntimeError("Cannot apply action on a terminal state.")

        action = actions[self.current_player]
        self.env.step(action)
        observation, reward, terminated, truncated, info = self.env.last()
        self.steps += 1

        if reward == 1.0:
            self.scores[self.current_player] += 1.0
        if self.scores[0] >= self.wining_score or self.scores[1] >= self.wining_score:
            terminated = True
        if self.max_steps_per_player and self.steps >= self.max_steps_per_player * self.num_agents:
            truncated = True

        self.state = {
            'observation': observation,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info,
        }
        done = terminated or truncated

        if self.visual_obs:
            self.image_paths = self._save_image()
            if done:
                self.recorders[0].save()

        observations = [self._get_observation(0), self._get_observation(1)]
        rewards = [reward if self.current_player == 0 else 0, reward if self.current_player == 1 else 0]
        dones = [done] * self.num_agents
        info = self._get_info()
        return observations, rewards, dones, info

    def _get_observation(self, agent_id):
        if agent_id == self.current_player:
            return Observation(
                obs=self.state['observation'],
                agent_id=agent_id,
                image_paths=self.image_paths,
                legal_actions=self._get_legal_actions(agent_id),
                serialized_state=str(self.state),
                regex_patterns=self.regex_patterns,
            )
        else:
            return None

    def _get_info(self):
        if self.state['terminated'] or self.state['truncated']:
            return {
                'returns': self.scores,
                'winner': 0 if self.scores[0] > self.scores[1] else 1 if self.scores[1] > self.scores[0] else -1
            }
        return None

    def _get_legal_actions(self, agent_id):
        return self.action_mapping

    def _action_to_string(self, action):
        return self.action_mapping.get(action, f'UNKNOWN_ACTION_{action}')

    def _save_image(self):
        frame = self.state['observation']
        image_path = []
        step_path = os.path.join(self.image_dir, f'step_{self.steps}')
        os.makedirs(step_path, exist_ok=True)
        if self.color_reduction:
            for i in range(frame.shape[-1]):
                image = Image.fromarray(frame[:, :, i])
                image_file = os.path.join(step_path, f'obs_{i}.png')
                image.save(image_file)
                image_path.append(image_file)
            self.recorders[0].add_frame(image_file)
        else:
            for i in range(frame.shape[-1] // 3):
                image = Image.fromarray(frame[:, :, i * 3:i * 3 + 3])
                image_file = os.path.join(step_path, f'obs_{i}.png')
                image.save(image_file)
                image_path.append(image_file)
            self.recorders[0].add_frame(image_file)
        return image_path
