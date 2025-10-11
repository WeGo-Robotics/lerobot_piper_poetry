source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER

python ./src/lerobot/scripts/train.py \
--policy.device=cuda  \
--policy.type=smolvla   \
--policy.repo_id=piper_smolvla  \
--policy.push_to_hub=false \
--dataset.repo_id=wego-hansu/piper_pick_yellow_car_033   \
--dataset.video_backend=pyav   \
--batch_size=64   \
--steps=25000   \
--eval_freq=0   \
--output_dir=outputs/train/piper_smolvla_pick_yellow_cars_033   \
--job_name=piper_smolvla_yellow_car   \
--wandb.enable=false  \
--resume=true  \
--config_path=outputs/train/piper_smolvla_pick_yellow_cars_033/checkpoints/020000/pretrained_model/train_config.json\