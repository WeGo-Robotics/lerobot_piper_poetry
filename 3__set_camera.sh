source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

cd src/lerobot
python camera_prop.py \
--usercon=true  \
--config_path=camera_prop.yaml