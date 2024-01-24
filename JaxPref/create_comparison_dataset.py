import os
import pickle
from collections import defaultdict

import numpy as np

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

    clip_action=0.999,

    data_dir='./pref_datasets',
    num_query=10000,
    query_len=25,
    set_len=16,
    balance=False,

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
    #     dataset = r_tf.qlearning_robosuite_dataset(os.path.join(FLAGS.robosuite_dataset_path, FLAGS.env.lower(), FLAGS.robosuite_dataset_type, "low_dim.hdf5"))
    #     label_type = 1
    if 'ant' in FLAGS.env:
        gym_env = gym.make(FLAGS.env)
        dataset = r_tf.qlearning_ant_dataset(gym_env)
        label_type = 1
    elif 'maze' in FLAGS.env:
        gym_env = gym.make(FLAGS.env)
        eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
        dataset = get_d4rl_dataset(eval_sampler.env)
        label_type = 1
    else:
        gym_env = gym.make(FLAGS.env)
        eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
        dataset = get_d4rl_dataset(eval_sampler.env)
        label_type = 0

    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    base_path = os.path.join(FLAGS.data_dir, FLAGS.env)
    query_path, all_obs = r_tf.get_queries_from_multi(
            gym_env, dataset, FLAGS.num_query, FLAGS.query_len, FLAGS.set_len,
            data_dir=base_path, label_type=label_type, balance=FLAGS.balance
    )
    gym_env.plot_gt()
    print("Saved queries: ", query_path)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(all_obs[:, 0], all_obs[:, 1], s=1)
    plt.xlim(gym_env.x_range)
    if hasattr(gym_env, 'y_range'):
        plt.ylim(gym_env.y_range)
    else:
        plt.ylim(gym_env.x_range)
    plt.savefig("chosen_data")

if __name__ == '__main__':
    absl.app.run(main)
