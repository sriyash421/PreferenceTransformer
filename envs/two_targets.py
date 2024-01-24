
import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb

from d4rl.pointmaze import MazeEnv
from .base import MultiModalEnv

WALL_ENV = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOOOO#\\"+\
        "#OOOOG#\\"+\
        "#OOOOO#\\"+\
        "#OOOOO#\\"+\
        "#######"

class TargetEnv(MultiModalEnv):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.env = MazeEnv(
            maze_spec=WALL_ENV,
            reward_type='dense',
            reset_target=False,
            **kwargs
        )

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)) #self.env.observation_space

        self.x_range = (0, 6)
        self.goal1 = np.array([1, 5])
        self.goal2 = np.array([5, 5])
        self.env.empty_and_goal_locations = [(3, 1)]
        self.str_maze_spec = self.env.str_maze_spec
        self.sim = self.env.sim
        self._max_episode_steps = kwargs.get('max_episode_steps', 300)
        self.current_mode = 0

    @property
    def target(self):
        return self.env._target

    @property
    def velocity(self):
        return self.env.data.qvel[:2]
    
    def reset(self):
        obs = self.env.reset()
        return obs[:2]

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        info['vel'] = obs[2:4]
        info['r1'], info['r2'] = self.get_goal_distances(obs)
        return obs[:2], reward, done, info
    
    def get_goal_distances(self, obs):
        dist_goal1 = np.exp(-np.linalg.norm(obs[:2] - self.goal1))
        dist_goal2 = np.exp(-np.linalg.norm(obs[:2] - self.goal2))
        return dist_goal1, dist_goal2
    
    def render(self, mode="rgb_array"):
        return self.env.render(mode)
    
    # def get_reward(self, state, mode=None):
    #     mode = mode or self.current_mode
    #     if mode == 0:
    #         return self._mode_0_r(state)
    #     else:
    #         return self._mode_1_r(state)
        
    def get_preference_rewards(self, state1, state2, mode=None): # states are pf size B x T x STATE_DIM
        mode = mode or np.random.randint(2)
        if mode == 0:
            r0 = self.get_ri(state1, self.goal1)
            r1 = self.get_ri(state2, self.goal1)
        else:
            r0 = self.get_ri(state1, self.goal2)
            r1 = self.get_ri(state2, self.goal2)
        return r0, r1
    
    def get_ri(self, state, target):
        target = target or np.array(self.env._target).reshape(1, 1, -1)
        dist_goal = -np.linalg.norm(state[:, :, 0:2] - target, axis=2)
        return np.exp(dist_goal / 10)

    # def _mode_0_r(self, state, target=None):
    #     return self.get_ri(state, target)
    
    # def _mode_1_r(self, state, target=None):
    #     return self.get_ri(state, target)

    def plot_gt(self, wandb_log=False):
        xv, yv = np.meshgrid(np.linspace(*self.x_range, 100), np.linspace(*self.x_range, 100), indexing='ij')
        points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)[None]
        r0 = self.get_ri(points, self.goal1)
        r1 = self.get_ri(points, self.goal2)
        r = [r0, r1]
        # target_p = 100 * (np.array(self.env._target) - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        axs_flat = axs.flatten()
        for i, ax in enumerate(axs_flat):
            ax.imshow((r[i].reshape(100, 100)).T, cmap='viridis', interpolation='nearest')
            # ax.scatter(target_p[0], target_p[1], c='r')
            self.plot_goals(ax, 100)
        plt.tight_layout()
        if wandb_log:
            wandb.log({'reward_ground_truth': wandb.Image(fig)})
        else:
            plt.savefig('reward_plot')
        plt.close(fig)
        return points
    
    def plot_goals(self, ax, scale):
        for g in [self.goal1, self.goal2]:
            target_p = scale * (np.array(g) - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
            ax.scatter(target_p[0], target_p[1], s=20, c='red', marker='*')

    def get_data_for_z(self, query_len, mode_n):
        obs1 = np.array([self.observation_space.sample() for _ in range(query_len)])[None, None]
        obs2 = np.array([self.observation_space.sample() for _ in range(query_len)])[None, None]
        return obs1, obs2, *self.get_preference_rewards(obs1, obs2, mode=mode_n)

    
