import gym
import numpy as np
from collections import OrderedDict
import h5py
from tqdm import tqdm

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

class MultiModalEnv(gym.Env):
    """
    A MultiModalEnv is a modified Gym environment designed for multi-modal reward tasks.
    """
    def __init__(self, dataset_path=None, **kwargs):
        super(MultiModalEnv, self).__init__(**kwargs)
        self.dataset_path = dataset_path
    
    def get_preferences(self, state1, state2):
        raise NotImplementedError()
    
    def render(self, mode="rgb_array"):
        raise NotImplementedError()

    def get_num_modes(self):
        return 2
    
    def get_obs_grid(self):
        return np.mgrid[
            self.x_range[0]:self.x_range[1]:50j,
            self.x_range[0]:self.x_range[1]:50j
        ], 50
    
    def reset_mode(self, mode_n):
        self.current_mode = mode_n
    
    def plot_reward_model(self, model, step):
        raise NotImplementedError
    
    def plot_reward_model_with_z(self, model, z, mode_n, step):
        raise NotImplementedError
    
    def get_dataset(self):
        if self.dataset_path is None:
            raise ValueError("Offline env not configured with a dataset path.")

        data_dict = {}
        with h5py.File(self.dataset_path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]
        
        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        return data_dict