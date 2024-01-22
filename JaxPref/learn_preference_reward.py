import os
import pickle
from collections import defaultdict

import numpy as np

import transformers

import gym
import wrappers as wrappers

import absl.app
import absl.flags
from flax.training.early_stopping import EarlyStopping

# import robosuite as suite
# from robosuite.wrappers import GymWrapper
# import robomimic.utils.env_utils as EnvUtils

from .sampler import TrajSampler
from .jax_utils import batch_to_jax
import JaxPref.reward_transform as r_tf
from .model import FullyConnectedQFunction, FullyConnectedStateQFunction
from viskit.logging import logger, setup_logger
from .MR import MR
from .replay_buffer import get_d4rl_dataset, index_batch
from .utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, prefix_metrics, WandBLogger, save_pickle
from .visualise_reward import plot_values_2d
# Jax memory
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=42,
    data_seed=42,
    save_model=True,
    batch_size=64,
    early_stop=False,
    min_delta=1e-3,
    patience=10,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    reward_arch='256-256',
    orthogonal_init=False,
    activations='relu',
    activation_final='none',
    training=True,

    n_epochs=100,
    eval_period=5,

    pref_dataset_path='',
    model_type='MLP',
    num_query=1000,
    query_len=25,
    set_len=16,
    skip_flag=0,
    balance=False,
    topk=10,
    window=2,
    use_human_label=False,
    feedback_random=False,
    feedback_uniform=False,
    enable_bootstrap=False,

    comment='',

    # robosuite=False,
    # robosuite_dataset_type="ph",
    # robosuite_dataset_path='./data',
    # robosuite_max_episode_steps=500,

    reward=MR.get_default_config(),
    # transformer=PrefTransformer.get_default_config(),
    # lstm=NMR.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(_):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)

    save_dir = FLAGS.logging.output_dir + '/' + FLAGS.env
    save_dir += '/' + str(FLAGS.model_type) + '/'

    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir += f"{FLAGS.comment}" + "/"
    save_dir += 's' + str(FLAGS.seed)

    setup_logger(
        variant=variant,
        seed=FLAGS.seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False
    )

    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)

    set_random_seed(FLAGS.seed)

    # if FLAGS.robosuite:
    #     dataset = r_tf.qlearning_robosuite_dataset(os.path.join(FLAGS.robosuite_dataset_path, FLAGS.env.lower(), FLAGS.robosuite_dataset_type, "low_dim.hdf5"))
    #     env = EnvUtils.create_env_from_metadata(
    #         env_meta=dataset['env_meta'],
    #         render=False,
    #         render_offscreen=False
    #     ).env
    #     gym_env = GymWrapper(env)
    #     gym_env._max_episode_steps = gym_env.horizon
    #     gym_env.seed(FLAGS.seed)
    #     gym_env.action_space.seed(FLAGS.seed)
    #     gym_env.observation_space.seed(FLAGS.seed)
    #     gym_env.ignore_done = False
    #     label_type = 1
    if 'maze' in FLAGS.env:
        gym_env = gym.make(FLAGS.env)
        gym_env = wrappers.EpisodeMonitor(gym_env)
        gym_env = wrappers.SinglePrecision(gym_env)
        gym_env.seed(FLAGS.seed)
        gym_env.action_space.seed(FLAGS.seed)
        gym_env.observation_space.seed(FLAGS.seed)
        label_type = 1
    elif 'maze' in FLAGS.env:
        gym_env = gym.make(FLAGS.env)
        eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
        label_type = 1
    else:
        gym_env = gym.make(FLAGS.env)
        eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
        label_type = 0


    # use fixed seed for collecting segments.
    set_random_seed(FLAGS.data_seed)

    # if FLAGS.robosuite:
    #     env = f"{FLAGS.env}_{FLAGS.robosuite_dataset_type}"
    # else:
    env = FLAGS.env

    set_random_seed(FLAGS.seed)
    observation_dim = gym_env.observation_space.shape[0]
    action_dim = gym_env.action_space.shape[0]

    pref_dataset, pref_eval_dataset = r_tf.get_query_dataset_from_path(FLAGS.pref_dataset_path, observation_dim, action_dim)

    data_size = pref_dataset["observations"].shape[0]
    interval = int(data_size / FLAGS.batch_size) + 1

    eval_data_size = pref_eval_dataset["observations"].shape[0]
    eval_interval = int(eval_data_size / FLAGS.batch_size) + 1

    early_stop = EarlyStopping(min_delta=FLAGS.min_delta, patience=FLAGS.patience)

    rf = FullyConnectedStateQFunction(observation_dim, action_dim, FLAGS.reward_arch, FLAGS.orthogonal_init, FLAGS.activations, FLAGS.activation_final)
    reward_model = MR(FLAGS.reward, rf)

    train_loss = "reward/rf_loss"

    criteria_key = None
    for epoch in range(FLAGS.n_epochs + 1):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch
        if epoch:
            # train phase
            shuffled_idx = np.random.permutation(pref_dataset["observations"].shape[0])
            for i in range(interval):
                start_pt = i * FLAGS.batch_size
                end_pt = min((i + 1) * FLAGS.batch_size, pref_dataset["observations"].shape[0])
                with Timer() as train_timer:
                    # train
                    batch = batch_to_jax(index_batch(pref_dataset, shuffled_idx[start_pt:end_pt]))
                    for key, val in prefix_metrics(reward_model.train(batch), 'reward').items():
                        metrics[key].append(val)
            metrics['train_time'] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics[train_loss] = [float(FLAGS.query_len)]

        # eval phase
        if epoch % FLAGS.eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * FLAGS.batch_size, min((j + 1) * FLAGS.batch_size, pref_eval_dataset["observations"].shape[0])
                # batch_eval = batch_to_jax(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                batch_eval = batch_to_jax(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                for key, val in prefix_metrics(reward_model.evaluation(batch_eval), 'reward').items():
                    metrics[key].append(val)
            if not criteria_key:
                if "antmaze" in FLAGS.env and not "dense" in FLAGS.env:
                    # choose train loss as criteria.
                    criteria_key = train_loss
                else:
                    # choose eval loss as criteria.
                    criteria_key = key
            criteria = np.mean(metrics[criteria_key])
            has_improved, early_stop = early_stop.update(criteria)
            if early_stop.should_stop and FLAGS.early_stop:
                for key, val in metrics.items():
                    if isinstance(val, list):
                        metrics[key] = np.mean(val)
                logger.record_dict(metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                wb_logger.log(metrics)
                print('Met early stopping criteria, breaking...')
                break
            elif epoch > 0 and has_improved:
                metrics["best_epoch"] = epoch
                metrics[f"{key}_best"] = criteria
                save_data = {"reward_model": reward_model, "variant": variant, "epoch": epoch}
                save_pickle(save_data, "best_model.pkl", save_dir)

        if epoch % 100 == 0:
            fig = plot_values_2d(gym_env, reward_model, "reward")
            wb_logger.log_image(fig, "reward_plots")
            gym_env.plot_gt(True)

        for key, val in metrics.items():
            if isinstance(val, list):
                metrics[key] = np.mean(val)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wb_logger.log(metrics)

    if FLAGS.save_model:
        save_data = {'reward_model': reward_model, 'variant': variant, 'epoch': epoch}
        save_pickle(save_data, 'model.pkl', save_dir)


if __name__ == '__main__':
    absl.app.run(main)
