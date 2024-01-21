import os
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import gym
import wrappers as wrappers

import absl.app
import absl.flags

# import robosuite as suite
# from robosuite.wrappers import GymWrapper
# import robomimic.utils.env_utils as EnvUtils

from .sampler import TrajSampler
from .jax_utils import batch_to_jax
import JaxPref.reward_transform as r_tf
from .replay_buffer import get_d4rl_dataset, index_batch
from .utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, save_pickle

# Jax memory
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    data_seed=42,
    query_path='',

    # robosuite=False,
    # robosuite_dataset_type="ph",
    # robosuite_dataset_path='./data',
    # robosuite_max_episode_steps=500,
)


def main(_):
    FLAGS = absl.flags.FLAGS
    # use fixed seed for collecting segments.
    set_random_seed(FLAGS.data_seed)

    # if FLAGS.robosuite:
    #     label_type = 1
    if 'ant' in FLAGS.env:
        gym_env = gym.make(FLAGS.env)
        label_type = 1
    elif 'maze' in FLAGS.env:
        gym_env = gym.make(FLAGS.env)
        eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
        label_type = 1
    else:
        gym_env = gym.make(FLAGS.env)
        eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
        label_type = 0

    # assert query_path and os.path.exists(query_path), "Dataset not found. Please run create_comparison_dataset.py first."

    with open(FLAGS.query_path, 'rb') as f:
        data = pickle.load(f)
    for i in tqdm(range(len(data['observations']))):
        seg_reward_1, seg_reward_2 = gym_env.get_preference_rewards(data['observations'][i], data['observations_2'][i])
        if label_type == 0: # perfectly rational
            sum_r_t_1 = np.sum(seg_reward_1, axis=1)
            sum_r_t_2 = np.sum(seg_reward_2, axis=1)
            binary_label = 1*(sum_r_t_1 < sum_r_t_2)
            rational_labels = np.zeros((len(binary_label), 2))
            rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        elif label_type == 1:
            sum_r_t_1 = np.sum(seg_reward_1, axis=1)
            sum_r_t_2 = np.sum(seg_reward_2, axis=1)
            binary_label = 1*(sum_r_t_1 < sum_r_t_2)
            rational_labels = np.zeros((len(binary_label), 2))
            rational_labels[np.arange(binary_label.size), binary_label] = 1.0
            margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= 0).reshape(-1)
            rational_labels[margin_index] = 0.5
        data['labels'][i] = rational_labels
    
    relabelled_path = os.path.join(os.path.dirname(FLAGS.query_path), 'relabelled.pkl')
    with open(relabelled_path, 'wb') as f:
        pickle.dump(data, f)
    gym_env.plot_gt()
    print("Saved queries: ", relabelled_path)

if __name__ == '__main__':
    absl.app.run(main)
