import time

import gym
import numpy as np

from wrappers.common import TimeStep


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self._max_episode_steps = env._max_episode_steps

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['success'] = 100.0 if self.episode_length < self.env._max_episode_steps else 0.0
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()

class MultiModalEpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self._max_episode_steps = env._max_episode_steps

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.r1 = 0.0
        self.r2 = 0.0
        self.success = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        self.r1 += info['r1']
        self.r2 += info['r2']
        
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['success'] = 100.0 if self.episode_length < self.env._max_episode_steps else 0.0
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time
            info['episode']['r1'] = self.r1
            info['episode']['r2'] = self.r2
            info['episode']['r1_norm'] = self.r1 / self.episode_length
            info['episode']['r2_norm'] = self.r2 / self.episode_length
            info['episode']['r_max'] = max(self.r1, self.r2)

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()