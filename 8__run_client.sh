source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
    # --task="Grasp the object and put it in the bin" \

python src/lerobot/scripts/server/robot_client.py \
    --server_address=192.168.0.42:8088 \
    --robot.type=piper_follower \
    --robot.port=can0 \
    --robot.id=black \
    --robot.cameras="{ \
        top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
        left: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --policy_type=smolvla \
    --pretrained_name_or_path=wego-hansu/piper_smolvla \
    --policy_device=cuda \
    --actions_per_chunk=5  \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True \
    --task="Grab the yellow car and put in the box"  \
