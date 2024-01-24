from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from tqdm import trange
from JaxPref.utils import prefix_metrics

def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': [], 'success': []}
    frames = []
    for i in trange(num_episodes, desc='evaluation', leave=False):
        observation, done = env.reset(), False
        if i<5:
            frames.append(env.render(mode='rgb_array'))
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
            if i<5:
                frames.append(env.render(mode='rgb_array'))

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, frames


def evaluate_vae(agent: nn.Module, env: gym.Env, reward_model,
             num_episodes: int, logger, step) -> Dict[str, float]:
    
    modes_n = env.get_num_modes()
    stats = {}
    for _ in modes_n:
        stat = {'return': [], 'length': [], 'success': []}
        frames = []
        for i in trange(num_episodes, desc='evaluation', leave=False):
            observation, done = env.reset(), False
            latent = reward_model.sample_latent
            if i<5:
                frames.append(env.render(mode='rgb_array'))
            while not done:
                action = agent.sample_actions(observation, temperature=0.0)
                observation, _, done, info = env.step(action)
                if i<5:
                    frames.append(env.render(mode='rgb_array'))

            for k in stats.keys():
                stats[k].append(info['episode'][k])
        for k, v in stats.items():
            stat[k] = np.mean(v)
        stats.update(prefix_metrics(stat, f'mode_{i}/'))

        logger.log_video(f'mode_{i}/video', frames, step)
        

    return stats
