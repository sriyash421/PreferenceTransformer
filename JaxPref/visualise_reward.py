import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import jax.numpy as jnp

warnings.filterwarnings("ignore", category=DeprecationWarning) 

import wandb
import numpy as np
from dataset_utils import batch_to_jax
import gym
import wrappers
import envs
from .VAE_R import get_latent
import torch
import matplotlib.cm as cm

def plot_train_values(obs, r, gym_env):
    fig, ax = plt.subplots()
    # _, NX, NY, target_p = gym_env.get_obs_grid()
    target_p = gym_env.target
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    # 
    # obs1 = torch.concatenate((obs1, obs2), axis=0).view(-1, 2)
    # r1 = torch.concatenate((r1, r2), axis=0).flatten()

    ax.scatter(obs[:, 0], obs[:, 1], c=cm.bwr(norm(r)))
    ax.scatter(x = target_p[0], y = target_p[1], color='red', s=100)
    sm = cm.ScalarMappable(cmap=cm.bwr, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label('r(s)')
    return fig

def plot_values_2d_with_z(env, reward_model, dataset, label_type, name):
    modes_n = env.get_num_modes()
    fig, axs = plt.subplots(modes_n, 4, figsize=(20, 16))
    for mode_n in range(modes_n):
        for i in range(4):
            ax = axs[mode_n, i]
            z = get_latent(env, reward_model, dataset, label_type, mode_n, n=1)[0]
            obs, NX, NY, target_p = env.get_obs_grid()
            input_size, x_range, y_range = obs.shape
            obs = obs.reshape(input_size, -1).T
            z = np.repeat(z, obs.shape[0], axis=0)

            obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)
            z = torch.from_numpy(z).float().to(next(reward_model.parameters()).device)
            new_reward = reward_model.decode(obs, z).detach().cpu().numpy().reshape((NX, NY))
            im = ax.imshow(new_reward.T, cmap='viridis', interpolation='nearest')
            # target_p = 50 * (np.array(env.target) - env.x_range[0]) / (env.x_range[1] - env.x_range[0])
            ax.scatter(x = target_p[0], y = target_p[1], color='red', s=100)
            ax.set_title(f"Mode {mode_n}")
    # plt.title(f"{name}")
    plt.tight_layout()
    return fig
        

def plot_values_2d(env, reward_model, name):
    # TODO: hardcoded
    obs, NX, NY, target_p = env.get_obs_grid()
    input_size, x_range, y_range = obs.shape
    obs = obs.reshape(input_size, -1).T

    actions = jnp.concatenate((jnp.zeros((obs.shape[0], 1)), -jnp.ones((obs.shape[0], 1))), axis=1)
    input = dict(
        observations=obs,
        actions=actions,
        next_observations=obs
    )

    jax_input = batch_to_jax(input)
    new_reward = reward_model.get_reward(jax_input)
    new_reward = np.asarray(list(new_reward))

    fig, ax = plt.subplots()
    new_reward = new_reward.reshape((NX, NY))
    im = ax.imshow(new_reward.T, cmap='viridis', interpolation='nearest')
    # target_p = 50 * (np.array(env.target) - env.x_range[0]) / (env.x_range[1] - env.x_range[0])
    ax.scatter(x = target_p[0], y = target_p[1], color='red', s=100)

    plt.title(f"{name}")
    return fig

if __name__ == "__main__":
    env_name = "multi-maze2d-wall-v0"
    seed = 0
    ckpt_dir = "/mmfs1/gscratch/weirdlab/sriyash/PreferenceTransformer/logs/pref_reward/multi-maze2d-wall-v0/MR/bimodal_prefs/s0"
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    import os
    import pickle

    def initialize_model():
        if os.path.exists(os.path.join(ckpt_dir, "best_model.pkl")):
            model_path = os.path.join(ckpt_dir, "best_model.pkl")
        else:
            model_path = os.path.join(ckpt_dir, "model.pkl")

        with open(model_path, "rb") as f:
            ckpt = pickle.load(f)
        reward_model = ckpt['reward_model']
        return reward_model

    reward_model = initialize_model()
    fig=plot_values_2d(env, reward_model, "reward_model_bimodal")
    plt.savefig(fig, "reward_model_bimodal")