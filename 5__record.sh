source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER

python ./src/lerobot/record.py \
--robot.type=piper_follower \
--robot.port=can_follower \
--robot.cameras="{ \
    top: {type: opencv, index_or_path: '/dev/video6', width: 640, height: 480, fps: 30}, \
    hand: {type: opencv, index_or_path: '/dev/video0', width: 640, height: 480, fps: 30}}" \
--robot.id=piper_wego   \
--teleop.type=piper_leader \
--teleop.port=can_leader \
--teleop.id=blue \
--display_data=true   \
--dataset.repo_id=wego-hansu/piper_tele_pick_cars_033_B  \
--dataset.num_episodes=50  \
--dataset.single_task="Pick the car and put in the box" \
--resume=true \
    # front: {type: opencv, index_or_path: '/dev/video4', width: 640, height: 480, fps: 30}}" \
