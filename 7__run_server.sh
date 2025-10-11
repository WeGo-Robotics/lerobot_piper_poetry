source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER

python src/lerobot/scripts/server/policy_server.py --host=192.168.0.42 --port=8088