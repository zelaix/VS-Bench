from PIL import Image
import ale_py
import gymnasium as gym
import numpy as np
import os
import supersuit

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


class AtariPong(BaseEnv, env_type="atari_pong"):
    '''Atari Pong game environment (single player) using Gymnasium and ALE.'''

    def __init__(
        self,
        max_episode_steps=1000,
        winning_score=3,
        max_observation=False,
        resize_frame=False,
        color_reduction=False,
        normalize_obs=False,
        stack_frame=4,
        noop_start=True,
        visual_obs=True,
        image_dir=None,
        recording_type="gif",
        recording_fps=10,
    ):

        gym.register_envs(ale_py)
        self._env = gym.make('ALE/Pong-v5', render_mode='rgb_array')

        if max_observation:
            self._env = supersuit.max_observation_v0(self._env, memory=max_observation)
        if resize_frame:
            self._env = supersuit.resize_v1(self._env, *resize_frame)
        self.color_reduction = color_reduction
        if self.color_reduction:
            self._env = supersuit.color_reduction_v0(self._env, mode=self.color_reduction)
        if normalize_obs:
            self._env = supersuit.normalize_obs_v0(self._env, env_min=0, env_max=1)
        if stack_frame:
            self._env = supersuit.frame_stack_v1(self._env, stack_size=stack_frame)

        self.visual_obs = visual_obs
        if self.visual_obs:
            assert image_dir is not None, "image_dir must not be None."
            self.image_dir = image_dir
            self.recorders = [Recorder(image_dir, recording_type, recording_fps)]

        self.state = None
        self.scores = [0, 0]
        self.steps = 0
        self.env_name = "atari_pong_single"
        self.noop_start = noop_start
        self.max_episode_steps = max_episode_steps
        self.winning_score = winning_score
        self.num_agents = 1
        self.image_paths = []

        self.action_mapping = {0: '<STAY>', 2: '<UP>', 3: '<DOWN>'}

    @property
    def current_player(self):
        return 0

    def reset(self, seed=0):
        observation, info = self._env.reset(seed=seed)
        reward = 0.0
        terminated = False
        truncated = False

        if self.noop_start:
            noop_steps = np.random.randint(1, 31)
            for _ in range(noop_steps):
                action = 0
                observation, reward, terminated, truncated, info = self._env.step(action)

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
        return [self._get_observation(0)]

    def step(self, actions):
        if self.state['terminated'] or self.state['truncated']:
            raise RuntimeError("Cannot apply action on a terminal state.")

        action = actions[0]
        observation, reward, terminated, truncated, info = self._env.step(action)
        self.steps += 1

        if reward == 1.0:
            self.scores[0] += 1.0
        if reward == -1.0:
            self.scores[1] += 1.0
        if self.scores[0] >= self.winning_score or self.scores[1] >= self.winning_score:
            terminated = True
        if self.max_episode_steps and self.steps >= self.max_episode_steps:
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

        observations = [self._get_observation(0)]
        rewards = [reward]
        dones = [done] * self.num_agents
        info = self._get_info()
        return observations, rewards, dones, info

    def _get_observation(self, agent_id):
        return Observation(
            obs=self.state['observation'],
            agent_id=agent_id,
            image_paths=self.image_paths,
            legal_actions=self._get_legal_actions(agent_id),
            serialized_state=str(self.state),
            regex_patterns=self.regex_patterns,
            addition_info=None,
        )

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
