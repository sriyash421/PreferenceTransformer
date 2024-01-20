
# # # Generate data and learn on original reward functions
# python -m JaxPref.new_preference_reward_main \
#     --comment reproc \
#     --env "antmaze-medium-diverse-v2"  \
#     --logging.output_dir './logs/reproc'  \
#     --batch_size 256  \
#     --query_len 100  \
#     --n_epochs 10000  \
#     --skip_flag 0  \
#     --seed 0  \
#     --model_type MR \
#     --use_human_label True \
#     --transformer.embd_dim 256 --transformer.n_layer 1 --transformer.n_head 4 \
#     --model_type PrefTransformer
    # --early_stop True \
    # --min_delta 0.00001 \ 

# python train_offline.py \
#     --comment reproc \
#     --eval_interval 100000  \
#     --env_name "antmaze-medium-diverse-v2"  \
#     --config configs/antmaze_config.py  \
#     --eval_episodes 100  \
#     --use_reward_model True  \
#     --model_type MR  \
#     --ckpt_dir "logs/reproc/antmaze-medium-diverse-v2/MR/reproc/s0"  \
#     --seed 0 \

# python train_offline.py \
#     --comment maze_final_test \
#     --eval_interval 100000  \
#     --env_name "maze2d-open-v0"  \
#     --config configs/antmaze_config.py  \
#     --eval_episodes 100  \
#     --use_reward_model False  \
#     --model_type MR  \
#     --ckpt_dir "/home/max/Distributional-Preference-Learning/PreferenceTransformer/logs/maze_test/maze2d-open-v0/MR/maze_final_test/s0"  \
#     --seed 0 \

# # Generate data and learn on original reward functions
# python -m JaxPref.new_preference_reward_main \
#     --comment antmaze_test \
#     --env "antmaze-medium-diverse-v2"  \
#     --logging.output_dir './logs/antmaze'  \
#     --batch_size 256  \
#     --query_len 1  \
#     --n_epochs 10000  \
#     --skip_flag 0  \
#     --seed 0  \
#     --model_type MR \
#     --use_human_label True \
#     # --num_query 2000  \


# python train_offline.py \
#     --comment antmaze_test \
#     --eval_interval 100000  \
#     --env_name "antmaze-medium-diverse-v2"  \
#     --config configs/antmaze_config.py  \
#     --eval_episodes 100  \
#     --use_reward_model True  \
#     --model_type MR  \
#     --ckpt_dir "/home/max/Distributional-Preference-Learning/PreferenceTransformer/logs/antmaze/antmaze-medium-diverse-v2/MR/antmaze_test/s0"  \
#     --seed 0 \

python -m JaxPref.new_preference_reward_main \
     --use_human_label True \
     --comment "open-maze-preference-RL-test" \
     --env "maze2d-medium-v0" \
     --logging.output_dir './logs/pref_reward' \
     --batch_size 256 \
     --num_query 2000 \
     --query_len 100 \
     --n_epochs 10000 \
     --skip_flag 0 \
     --seed 0 \
     --model_type MR
