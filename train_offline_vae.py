import datetime
import os
import pickle
from typing import Tuple

import gym
import numpy as np
from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags

import envs
import wrappers
from dataset_utils import D4RLDataset, reward_from_preference_vae, split_into_trajectories
from evaluation import evaluate_vae, evaluate
from learner import Learner
from logger import Logger

from JaxPref.VAE_R import VAEModel

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.40'

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './logs/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('sampling_ratio', 1.0, 'Sampling ratio')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('use_reward_model', False, 'Use reward model for relabeling reward.')
flags.DEFINE_string('model_type', 'MLP', 'type of reward model.')
flags.DEFINE_string('ckpt_dir',
                    './logs/pref_reward',
                    'ckpt path for reward model.')
flags.DEFINE_string('comment',
                    'base',
                    'comment for distinguishing experiments.')
flags.DEFINE_integer('seq_len', 25, 'sequence length for relabeling reward in Transformer.')
flags.DEFINE_bool('use_diff', False, 'boolean whether use difference in sequence for reward relabeling.')
flags.DEFINE_string('label_mode', 'last', 'mode for relabeling reward with tranformer.')
flags.DEFINE_string('dataset_path', '', 'path to dataset for reward model training.')
flags.DEFINE_bool('z_conditioned', True)
flags.DEFINE_integer('mode_n', 1)

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)
    trj_mapper = []
    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        traj_len = len(traj)

        for _ in range(traj_len):
            trj_mapper.append((trj_idx, traj_len))

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    sorted_trajs = sorted(trajs, key=compute_returns)
    min_return, max_return = compute_returns(sorted_trajs[0]), compute_returns(sorted_trajs[-1])

    normalized_rewards = []
    for i in range(dataset.size):
        _reward = dataset.rewards[i]
        if 'antmaze' in env_name or 'maze' in env_name:
            _, len_trj = trj_mapper[i]
            _reward -= min_return / len_trj
        else:
            _reward /= max_return - min_return
        # if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)

    dataset.rewards = np.array(normalized_rewards)


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=env._max_episode_steps)
    env = wrappers.MultiModalEpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    reward_model = initialize_model()
    dataset = reward_from_preference_vae(FLAGS.env_name, dataset, reward_model, sampling_ratio=FLAGS.sampling_ratio, z_conditioned=FLAGS.z_conditioned, mode_n=FLAGS.mode_n)
    if FLAGS.z_conditioned:
        latent_dim = reward_model.latent_dim
    else:
        latent_dim = 0

    # if FLAGS.use_reward_model:
    normalize(dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)
    if 'antmaze' in FLAGS.env_name or 'maze' in FLAGS.env_name:
        dataset.rewards -= 1.0
    if ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name or 'hopper' in FLAGS.env_name):
        dataset.rewards += 0.5

    return env, dataset, reward_model, latent_dim


def initialize_model():
    if os.path.exists(os.path.join(FLAGS.ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(FLAGS.ckpt_dir, "model.pkl")

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt['reward_model']
    return reward_model


def main(_):
    save_dir = os.path.join(FLAGS.save_dir, 'tb',
                        FLAGS.env_name,
                            f"reward_{FLAGS.use_reward_model}_{FLAGS.model_type}" if FLAGS.use_reward_model else "original",
                            f"{FLAGS.comment}",
                            str(FLAGS.seed),
                            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(FLAGS, save_dir)

    env, dataset, reward_model, latent_dim = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    obs = np.concatenate((env.observation_space.sample(), np.ones(latent_dim)))[np.newaxis]
    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    obs,
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    eval_returns = []
    for i in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    logger.log(f'training/{k}', v, i)
                else:
                    logger.log_histogram(f'training/{k}', v, i)

        if i % FLAGS.eval_interval == 0:
            if FLAGS.z_conditioned:
                eval_stats = evaluate_vae(agent, env, reward_model, FLAGS.eval_episodes, logger, i)
            else:
                eval_stats, evaluate_video = evaluate(agent, env, FLAGS.eval_episodes)
                logger.log_video('evaluation/episode', evaluate_video, i)

            for k, v in eval_stats.items():
                logger.log(f'evaluation/average_{k}s', v, i)

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(save_dir, 'progress.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    app.run(main)
