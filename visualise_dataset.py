import os
import h5py

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np

import envs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC-BEAR')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--filename', type=str)
    parser.add_argument('--num', type=int, default=1000)
    args = parser.parse_args()
    

    env = gym.make(args.env_name)
    env.reset()
    target_goal = env.target

    rdataset = h5py.File(args.filename, 'r')
    fpath, ext = os.path.splitext(args.filename)

    all_obs = rdataset['observations'][:]
    terminals = np.array(rdataset['terminals'])

    trj_idx_list = []
    episode_step = 0
    start_idx, data_idx = 0, 0
    for i in range(all_obs.shape[0]):
        final_timestep = terminals[i]
        if final_timestep:
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx])
            start_idx = data_idx + 1
            
        episode_step += 1
        data_idx += 1
        
    trj_idx_list.append([start_idx, data_idx])
    
    
    fig = plt.figure()
    for i in range(args.num):
        start = trj_idx_list[i][0]
        end = trj_idx_list[i][1]
        plt.scatter(all_obs[start, 0], all_obs[start, 1], c='orange')
        plt.plot(all_obs[start:end+1, 0], all_obs[start:end+1, 1], c=f'C{i}')
    plt.scatter(target_goal[0], target_goal[1], c='r')
    plt.savefig(f'{args.env_name}.png')

