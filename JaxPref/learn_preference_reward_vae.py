import os
import pickle
from collections import defaultdict

import numpy as np

import torch

import gym
import wrappers as wrappers

import absl.app
import absl.flags
from flax.training.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset
import JaxPref.reward_transform as r_tf
# import robosuite as suite
# from robosuite.wrappers import GymWrapper
# import robomimic.utils.env_utils as EnvUtils

from .sampler import TrajSampler
from viskit.logging import logger, setup_logger
from .VAE_R import VAEModel, PreferenceDataset, Annealer
from .utils import define_flags_with_default, set_random_seed, get_user_flags, WandBLogger, save_pickle, prefix_metrics
from .visualise_reward import plot_values_2d, plot_values_2d_with_z, plot_train_values, plot_latents, plot_prior
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

    lr=3e-4,
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

    # reward=VAEModel.get_default_config(),
    # transformer=PrefTransformer.get_default_config(),
    # lstm=NMR.get_default_config(),
    logging=WandBLogger.get_default_config(),

    kl_weight=0.1,
    learned_prior=False,
    latent_dim=32,
    use_annealing=False,
    hidden_dim=256,
    flow_prior=False,
)

from torch.optim import Adam


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
    set_len = pref_dataset["observations"].shape[1]
    query_len = pref_dataset["observations"].shape[2]


    pref_dataset = PreferenceDataset(pref_dataset)
    pref_eval_dataset = PreferenceDataset(pref_eval_dataset)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(dataset=pref_dataset, batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=pref_eval_dataset,  batch_size=FLAGS.batch_size, shuffle=False, **kwargs)
    device = 'cuda'

    early_stop = EarlyStopping(min_delta=FLAGS.min_delta, patience=FLAGS.patience)

    latent_dim = FLAGS.latent_dim
    kl_weight = FLAGS.kl_weight
    learned_prior = FLAGS.learned_prior
    flow_prior = FLAGS.flow_prior

    assert not (flow_prior and learned_prior)

    reward_model = VAEModel(encoder_input=set_len*(2*observation_dim*query_len+2), 
                            decoder_input=observation_dim+latent_dim, 
                            latent_dim=latent_dim, hidden_dim=256, annotation_size=set_len,
                            size_segment=query_len, learned_prior=learned_prior, flow_prior=flow_prior)

    optimizer = Adam(reward_model.parameters(), lr=FLAGS.lr)
    # train_loss = "train/loss"

    reward_model.to(device)

    if FLAGS.use_annealing:
        kl_annealer = Annealer(FLAGS.n_epochs // 4, 'cosine', cyclical=True)
    else:
        kl_annealer = None

    for epoch in range(FLAGS.n_epochs + 1):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch
        if epoch:
            all_obs = []
            all_r = []
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                sa_t_1 = (batch['observations']).float().to(device)
                sa_t_2 = (batch['observations_2']).float().to(device)
                labels = (batch['labels']).squeeze().float().to(device).view(-1, 2)

                if kl_annealer:
                    kl_weight = kl_annealer.slope()
                loss, r_hat1, r_hat2, batch_metrics = reward_model(sa_t_1, sa_t_2, labels, kl_weight)

                loss.backward()
                optimizer.step()

                for key, val in prefix_metrics(batch_metrics, 'train/').items():
                    metrics[key].append(val)
                
                all_obs.extend(sa_t_1.view(-1, 2).cpu().numpy())
                all_obs.extend(sa_t_2.view(-1, 2).cpu().numpy())
                all_r.extend(r_hat1.view(-1, 1).detach().cpu().numpy())
                all_r.extend(r_hat2.view(-1, 1).detach().cpu().numpy())
            
            all_obs = np.array(all_obs)
            all_r = np.array(all_r)

            if epoch %20 == 0:
                fig = plot_train_values(all_obs, all_r, gym_env)
                wb_logger.log_image(fig, "train_reward_plots")

        # eval phase
        if epoch % FLAGS.eval_period == 0:
            all_obs = []
            all_r = []
            for batch_idx, batch in enumerate(test_loader):
                with torch.no_grad():
                    sa_t_1 = (batch['observations']).float().to(device)
                    sa_t_2 = (batch['observations_2']).float().to(device)
                    labels = (batch['labels']).squeeze().float().to(device).view(-1, 2)

                    if kl_annealer:
                        kl_weight = kl_annealer.slope()
                    loss, r_hat1, r_hat2, batch_metrics = reward_model(sa_t_1, sa_t_2, labels, kl_weight)


                    for key, val in prefix_metrics(batch_metrics, 'reward/eval_').items():
                        metrics[key].append(val)
                    
                    all_obs.extend(sa_t_1.view(-1, 2).cpu().numpy())
                    all_obs.extend(sa_t_2.view(-1, 2).cpu().numpy())
                    all_r.extend(r_hat1.view(-1, 1).detach().cpu().numpy())
                    all_r.extend(r_hat2.view(-1, 1).detach().cpu().numpy())
                     

            all_obs = np.array(all_obs)
            all_r = np.array(all_r)
            if epoch %20 == 0:
                fig = plot_train_values(all_obs, all_r, gym_env)
                wb_logger.log_image(fig, "test_reward_plots")

            criteria = np.mean(metrics["train/rf_loss"])
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

        if kl_annealer:
            kl_annealer.step()

        if epoch % 20 == 0:
            fig = plot_values_2d_with_z(gym_env, reward_model, pref_dataset, label_type, "reward_plot")
            wb_logger.log_image(fig, "reward_plots")
            gym_env.plot_gt(True)
            fig = plot_latents(gym_env, reward_model, pref_dataset, label_type, "latent_plot")
            wb_logger.log_image(fig, "latent_plots")
            plot_prior(latent_dim, reward_model)

        for key, val in metrics.items():
            if isinstance(val, list):
                metrics[key] = np.mean(val)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wb_logger.log(metrics)

    if FLAGS.save_model:
        save_data = {'reward_model': reward_model.state_dict(), 'variant': variant, 'epoch': epoch}
        torch.save(save_data, os.path.join(save_dir, 'model.pt'))

if __name__ == '__main__':
    absl.app.run(main)
