source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER

python ./src/lerobot/teleoperate.py \
    --robot.type=piper_follower \
    --robot.port=can_follower \
    --robot.id=black \
    --teleop.type=piper_leader \
    --teleop.port=can_leader \
    --teleop.id=blue \
    --display_data=true
