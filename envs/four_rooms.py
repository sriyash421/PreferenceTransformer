
import numpy as np
import gym

from d4rl.pointmaze import MazeEnv
from .base import MultiModalEnv
import wandb
import matplotlib.pyplot as plt



FOUR_ROOMS_ENV = \
        '#################\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOOOOOOOOOO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '####O#######O####\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOOOOOOOOOO#\\'+\
        '#OOOOOOO#OOOOOGO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        "#################"


TARGET = None

def mode1_fn(x, y):
    x_, y_ = x-6, y-8

    if x_ < 0:
        if y_ < 0:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([3, 8]))
            d2 = -np.linalg.norm(np.array([3, 8]) - np.array([6, 12]))
            d3 = -np.linalg.norm(np.array([6, 12]) - TARGET)
            return d1 + d2 + d3
        else:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([6, 12]))
            d2 = -np.linalg.norm(np.array([6, 12]) - TARGET)
            return d1 + d2
    else:
        if y_ < 0:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([6, 4]))
            d2 = -np.linalg.norm(np.array([6, 4]) - np.array([3, 8]))
            d3 = -np.linalg.norm(np.array([3, 8]) - np.array([6, 12]))
            d4 = -np.linalg.norm(np.array([6, 12]) - TARGET)
            return d1 + d2 + d3 + d4
        else:
            return -np.linalg.norm(np.array([x, y]) - TARGET)

def mode2_fn(x, y):
    x_, y_ = x-6, y-8

    if x_ < 0:
        if y_ < 0:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([6, 4]))
            d2 = -np.linalg.norm(np.array([6, 4]) - np.array([9, 8]))
            d3 = -np.linalg.norm(np.array([9, 8]) - TARGET)
            return d1 + d2 + d3
        else:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([3, 8]))
            d2 = -np.linalg.norm(np.array([3, 8]) - np.array([6, 4]))
            d3 = -np.linalg.norm(np.array([6, 4]) - np.array([9, 8]))
            d4 = -np.linalg.norm(np.array([9, 8]) - TARGET)
            return d1 + d2 + d3 + d4
    else:
        if y_ < 0:
            d1 = -np.linalg.norm(np.array([x, y]) - np.array([9, 8]))
            d2 = -np.linalg.norm(np.array([9, 8]) - TARGET)
            return d1 + d2
        else:
            return -np.linalg.norm(np.array([x, y]) - TARGET)

vec_mode1 = np.vectorize(mode1_fn)
vec_mode2 = np.vectorize(mode2_fn)

class RoomEnv(MultiModalEnv):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.env = MazeEnv(
            maze_spec=FOUR_ROOMS_ENV,
            reward_type='dense',
            reset_target=False,
            **kwargs
        )

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)) #self.env.observation_space

        self.x_range = (0, 12)
        self.y_range = (0, 16)

        global TARGET
        TARGET = np.array(self.env._target)
        self.str_maze_spec = self.env.str_maze_spec
        self.str_maze_spec = self.env.str_maze_spec
        self.sim = self.env.sim
        self._max_episode_steps = kwargs.get('max_episode_steps', 600)
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
        return obs[:2], reward, done, info
    
    def get_reward(self, state, mode=None):
        mode = mode or self.current_mode
        if mode == 0:
            return self._mode_0_r(state)
        else:
            return self._mode_1_r(state)
    
    def get_preference_rewards(self, state1, state2, target=None, mode=None): # states are pf size B x T x STATE_DIM
        mode = mode or np.random.randint(2)
        if mode == 0:
            r0 = self._mode_0_r(state1)
            r1 = self._mode_0_r(state2)
        else:
            r0 = self._mode_1_r(state1)
            r1 = self._mode_1_r(state2)
        return r0, r1
    
    def _mode_0_r(self, state, target=None):
        return vec_mode1(state[:, :, 0], state[:, :, 1])
    
    def _mode_1_r(self, state, target=None):
        return vec_mode2(state[:, :, 0], state[:, :, 1])

    def plot_gt(self, wandb_log=False):
        xv, yv = np.meshgrid(np.linspace(*self.x_range, 120), np.linspace(*self.y_range, 160), indexing='ij')
        points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)[None]
        r = [self._mode_0_r(points),self._mode_1_r(points)]
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        axs_flat = axs.flatten()
        for i, ax in enumerate(axs_flat):
            im = ax.imshow((r[i].reshape(120, 160)).T, cmap='viridis', interpolation='nearest')
            ax.scatter(TARGET[0]*10, TARGET[1]*10, c='r')
            ax.scatter(30, 120, c='g')
            ax.scatter(90, 40, c='b')
            ax.scatter(10,10, c='black')
        plt.tight_layout()
        if wandb_log:
            wandb.log({'eval/ground_truth': wandb.Image(fig)})
        else:
            plt.savefig('reward_plot.png')
        plt.close(fig)
        return points

    def render(self, mode="rgb_array"):
        return self.env.render(mode)
    
    def get_obs_grid(self):
        return np.mgrid[
            self.x_range[0]:self.x_range[1]:120j,
            self.y_range[0]:self.y_range[1]:160j
        ], 120, 160, (TARGET[0]*10, TARGET[1]*10)
