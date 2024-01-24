export WANDB_PROJECT="offlineRL-two-goals"


python train_offline.py \
    --comment "baseline" \
    --eval_interval 100000 \
    --env_name "multi-maze2d-target-v0" \
    --config configs/antmaze_config.py \
    --eval_episodes 100 \
    --use_reward_model True \
    --model_type MR \
    --ckpt_dir "/mmfs1/gscratch/weirdlab/sriyash/PreferenceTransformer/logs/pref_reward/multi-maze2d-target-v0/MR/baseline/s0" \
    --seed 0