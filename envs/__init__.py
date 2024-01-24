from gym.envs.registration import register
from .four_rooms import FOUR_ROOMS_ENV
from .wall import WALL_ENV

register(
    id='maze2d-four-rooms-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=1000,
    kwargs={
        'maze_spec':FOUR_ROOMS_ENV,
        'reward_type':'dense',
        'reset_target': False,
        # 'dataset_url':'/mmfs1/gscratch/weirdlab/sriyash/PreferenceTransformer/envs/datasets/room_oracle.hdf5'
    }
)

register(
    id='maze2d-wall-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=300,
    kwargs={
        'maze_spec':WALL_ENV,
        'reward_type':'dense',
        'reset_target': False,
        # 'dataset_url':'/mmfs1/gscratch/weirdlab/sriyash/PreferenceTransformer/envs/datasets/room_oracle.hdf5'
    }
)

register(
    id='multi-maze2d-wall-v0',
    entry_point='envs.wall:WallEnv',
    max_episode_steps=300,
    kwargs={
        'dataset_path':'/mmfs1/gscratch/weirdlab/sriyash/PreferenceTransformer/envs/datasets/multi-maze2d-wall-v0-noisy.hdf5'
    }
)

register(
    id='multi-maze2d-target-v0',
    entry_point='envs.two_targets:TargetEnv',
    max_episode_steps=300,
    kwargs={
        'dataset_path':'/mmfs1/gscratch/weirdlab/sriyash/PreferenceTransformer/envs/datasets/multi-maze2d-wall-v0-noisy.hdf5'
    }
)

register(
    id='multi-maze2d-rooms-v0',
    entry_point='envs.four_rooms:RoomEnv',
    max_episode_steps=600,
    kwargs={
        'dataset_path':'/mmfs1/gscratch/weirdlab/sriyash/PreferenceTransformer/envs/datasets/multi-maze2d-wall-v0-noisy.hdf5'
    }
)