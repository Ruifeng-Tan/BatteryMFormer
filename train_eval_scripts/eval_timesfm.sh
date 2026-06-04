#!/bin/bash

# TimesFM Baseline Evaluation
# TimesFM is a pretrained time-series foundation model from Google Research.
# It is evaluation-only (no training needed).
#
# Setup (one-time):
#   1. Clone the TimesFM repo and install:
#        git clone https://github.com/google-research/timesfm.git
#        pip install -e "./timesfm[torch]"
#   2. The model weights (google/timesfm-2.5-200m-pytorch) will be
#      auto-downloaded from HuggingFace on first run.
#      If you have network issues, set the HF mirror:
#        export HF_ENDPOINT=https://hf-mirror.com
#   3. Prepare cached data by running training first (see train_BattMFormer.sh),
#      which generates cache files under cache_root.

gpu=0

# ==========================================
# 2. Dataset & Evaluation Configuration
# ==========================================
# Dataset to evaluate (Li_ion, CALB, NA-ion, ZN-coin)
dataset=Li_ion
seed=2024

# ---------- Evaluation Protocol ----------
# Use the same split json that was used at training time so the cached preprocessed
# data lines up. Available splits:
#   Li-ion pure-OOD:  data_provider/split_json/pure_ood/Liion_split_seed{2021,42,2024}.json
#   Zn-ion pure-OOD:  data_provider/split_json/pure_ood/ZNcoin_split_seed{2021,42,2024}.json
#   CALB LOAO:        data_provider/split_json/loao/CALB_loao_cond{396..399}_seed2021.json
#   NA-ion LOAO:      data_provider/split_json/loao/NA-ion_loao_cond{400..411}_seed2021.json
# Leave empty to fall back to the legacy random split.
split_json_path=./data_provider/split_json/pure_ood/Liion_split_seed2024.json

# ==========================================
# 3. Data Parameters (must match cached data)
# ==========================================
early_cycle_threshold=100
charge_discharge_length=300
seq_len=1
max_trajectory_len=5000
batch_size=32

# ==========================================
# 4. Paths
# ==========================================
cache_root=/path/to/your/cache
output_dir=./results/timesfm

# ==========================================
# 5. Execution Command
# ==========================================
split_tag=$(basename "$split_json_path" .json)

python train_eval_scripts/eval_timesfm.py \
    --dataset $dataset \
    --seed $seed \
    --flag test \
    --early_cycle_threshold $early_cycle_threshold \
    --charge_discharge_length $charge_discharge_length \
    --seq_len $seq_len \
    --max_trajectory_len $max_trajectory_len \
    --use_capacity_resample \
    --batch_size $batch_size \
    --gpu $gpu \
    --cache_root $cache_root \
    --output_dir $output_dir \
    --split_json_path $split_json_path \
    --split_tag $split_tag
