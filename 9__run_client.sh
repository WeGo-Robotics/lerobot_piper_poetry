source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER

python src/lerobot/camera_prop.py \
--config_path=src/lerobot/camera_prop.yaml \
--usercon=false

python src/lerobot/scripts/server/robot_client.py \
--server_address=127.0.0.1:8088 \
--robot.type=piper_follower \
--robot.port=can_follower \
--robot.id=black \
--robot.cameras="{ \
    top: {type: opencv, index_or_path: '/dev/video6', width: 640, height: 480, fps: 30}, \
    hand: {type: opencv, index_or_path: '/dev/video4', width: 640, height: 480, fps: 30}}" \
--pretrained_name_or_path=wego-hansu/piper_smolvla_teleop_033_B \
--policy_type=smolvla \
--policy_device=cuda \
--actions_per_chunk=50  \
--chunk_size_threshold=0.5 \
--aggregate_fn_name=weighted_average \
--debug_visualize_queue_size=false \
--task="Pick the car and put in the box"  \
# --server_address=192.168.0.42:8088 \
# --server_address=127.0.0.1:8088 \
# --pretrained_name_or_path=wego-hansu/piper_teleop_smolvla \
