from gym.envs.registration import register


WALL_ENV = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OO#OO#\\"+\
        "#OO#OG#\\"+\
        "#OO#OO#\\"+\
        "#OOOOO#\\"+\
        "#######"

FOUR_ROOMS_ENV = \
        '#################\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOOOOOOOOOO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '####O#######O####\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        '#OOOOOOOOOOOOOOO#\\'+\
        '#OOOOOOO#OOOOOGO#\\'+\
        '#OOOOOOO#OOOOOOO#\\'+\
        "#################"

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
        'maze_spec':WALL_ENV,
        'reward_type':'dense',
        'reset_target': False,
        'dataset_path':'/mmfs1/gscratch/weirdlab/sriyash/PreferenceTransformer/envs/datasets/multi-maze2d-wall-v0-noisy.hdf5'
    }
)