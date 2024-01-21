
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
    def __init__(self, *args, maze_map="ROOM_MAZE", fixed_start=True, **kwargs):
        super(MultiModalEnv, self).__init__()
        self.env = MazeEnv(
            maze_spec=FOUR_ROOMS_ENV,
            reward_type='sparse',
            reset_target=False,
            **kwargs
        )
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)) #self.env.observation_space
        if fixed_start:
            self.env.empty_and_goal_locations = [(1, 1)]
        self.x_range = (0, 12)
        self.y_range = (0, 16)
        global TARGET
        TARGET = np.array(self.env._target)
        self.str_maze_spec = self.env.str_maze_spec
        # self.plot_gt()
    
    def reset(self):
        obs = self.env.reset()
        self.current_mode = np.random.randint(2)
        return obs[:2]

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        # reward = self.get_reward(obs[None,None])[0, 0]
        # info['success'] = float(np.linalg.norm(obs[:2] - np.array(self.env._target)) < 0.5)
        return obs[:2], reward, done, info
    
    def get_ground_truth(self, state, target=None):
        target = target or np.array(self.env._target).reshape(1, -1)
        return -np.linalg.norm(state[:, :, 0:2] - target, axis=2)
    
    def get_reward(self, state, mode=None):
        mode = mode or self.current_mode
        if mode == 0:
            return self._mode_0_r(state)
        else:
            return self._mode_1_r(state)
        
    def get_preference_rewards(self, state1, state2, target=None, mode=None): # states are pf size B x T x STATE_DIM
        mode = mode or np.random.randint(2)

        if mode == 0:
            r0 = self._mode_0_r(state1, target)
            r1 = self._mode_0_r(state2, target)
        else:
            r0 = self._mode_1_r(state1, target)
            r1 = self._mode_1_r(state2, target)
        return r0, r1
    
    def _mode_0_r(self, state, target=None):
        target = target or np.array(self.env._target).reshape(1, 1, -1)
        return vec_mode1(state[:, :, 0], state[:, :, 1])
    
    def _mode_1_r(self, state, target=None):
        target = target or np.array(self.env._target).reshape(1, -1)
        return vec_mode2(state[:, :, 0], state[:, :, 1])

    def plot_gt(self, wandb_log=False):
        xv, yv = np.meshgrid(np.linspace(*self.x_range, 12), np.linspace(*self.y_range, 16), indexing='ij')
        points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)[None]
        r = [self._mode_0_r(points),self._mode_1_r(points)]
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        axs_flat = axs.flatten()
        for i, ax in enumerate(axs_flat):
            im = ax.imshow((r[i].reshape(12, 16)).T, cmap='viridis', interpolation='nearest')
            ax.scatter(TARGET[0], TARGET[1], c='r')
            ax.scatter(3, 12, c='g')
            ax.scatter(9, 4, c='b')
            ax.scatter(1,1, c='black')
        plt.tight_layout()
        if wandb_log:
            wandb.log({'eval/ground_truth': wandb.Image(fig)})
        else:
            plt.show()
        plt.close(fig)
        return points

    def plot_reward_model(self, reward_model=None, step=None):
        points = self.plot_gt(True)
        import torch
        with torch.no_grad():
            plt.figure()
            points = torch.tensor(points, dtype=torch.float32).to(next(reward_model.parameters()).device)
            r0 = reward_model(points).cpu().numpy()
            fig = plt.figure()
            plt.imshow((r0.reshape(12, 16)).T, cmap='viridis')
            plt.colorbar()
            wandb.log({'eval/reconstructed reward': wandb.Image(fig)})
            plt.close(fig)
    
    def plot_reward_model_with_z(self, model, z, mode_n, step=None):
        points = self.plot_gt(True)
        import torch
        with torch.no_grad():
            plt.figure()
            points = torch.tensor(points[0], dtype=torch.float32).to(next(model.parameters()).device)
            z = torch.from_numpy(z).reshape(1,-1).repeat((points.shape[0], 1)).to(next(model.parameters()).device)
            points = torch.cat((points, z), dim=-1)
            r0 = model(points).cpu().numpy()
            fig = plt.figure()
            plt.imshow((r0.reshape(12, 16)).T, cmap='viridis')
            plt.colorbar()
            wandb.log({f'eval/{mode_n}_reconstructed reward': wandb.Image(fig)})
            plt.close(fig)

    def render(self, mode="rgb_array"):
        return self.env.render(mode)
