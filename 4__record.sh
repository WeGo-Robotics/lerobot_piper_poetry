source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER

python ./src/lerobot/record.py \
--robot.type=piper_follower \
--robot.port=can0 \
--robot.cameras="{ \
    top: {type: opencv, index_or_path: '/dev/video0', width: 640, height: 480, fps: 30}, \
    left: {type: opencv, index_or_path: '/dev/video4', width: 640, height: 480, fps: 30}}" \
--robot.id=black   \
--display_data=true   \
--dataset.repo_id=wego-hansu/piper_pick_yellow_car_033  \
--dataset.num_episodes=50  \
--dataset.single_task="Grab the yellow car and put in the box"  \
--resume=true \