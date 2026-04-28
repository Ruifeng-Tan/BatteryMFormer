#!/bin/bash

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
# Specify GPU IDs to use (e.g., "0" or "0,1,2,3")
gpu_ids=0,1
# Specify the number of GPUs (must match the count in gpu_ids)
num_process=2
# Main process port (avoid conflicts among multiple runs)
master_port=29453
seed=2024

# ==========================================
# 2. Model & Data Configuration
# ==========================================
model_name=PatchTST
dataset=NA-ion
input_mode=current_voltage  # Options: current_voltage or soh_to_soh
# ---------- Evaluation Protocol ----------
# We use Leave-One-Aging-condition-Out (LOAO) evaluation by default.
# Each split holds out one aging condition for testing.
# Available LOAO splits:
#   CALB:   data_provider/split_json/loao/CALB_loao_cond{396..399}_seed2024.json  (4 folds)
#   NA-ion: data_provider/split_json/loao/NA-ion_loao_cond{400..411}_seed2024.json (12 folds)
# For Li-ion pure-OOD evaluation:
#   data_provider/split_json/pure_ood/Liion_split_seed{2021,42,2024}.json
# To use the legacy random split, leave split_json_path empty.
split_json_path=./data_provider/split_json/loao/NA-ion_loao_cond400_seed2024.json

# ==========================================
# 3. Model Architecture Hyperparameters
# ==========================================
d_model=128
n_heads=4
e_layers=6
d_ff=256
dropout=0.2
activation=gelu
factor=1
patch_len=300
stride=300

# ==========================================
# 4. Training Hyperparameters
# ==========================================
batch_size=256       # Per-GPU batch size (global batch = batch_size * num_process)
train_epochs=250
learning_rate=0.00005
weight_decay=0.01
patience=50

# ==========================================
# 5. Sequence & Task Parameters
# ==========================================
seq_len=1
pred_len=5000
early_cycle_threshold=100
charge_discharge_length=300
eol_threshold=0.8
task_name=soh_forecast

# ==========================================
# 6. Paths
# ==========================================
# root_path: directory containing dataset subdirectories (e.g., NA-ion/, MATR/, HUST/, ...)
# processed_SOH_path: directory containing processed SOH trajectory subdirectories
# cache_root: directory for caching preprocessed data
root_path=/path/to/your/dataset
processed_SOH_path=/path/to/your/processed_SOH
cache_root=/path/to/your/cache

split_tag=$(basename "$split_json_path" .json)
checkpoints="./checkpoints/${model_name}_${dataset}_${split_tag}_${input_mode}_dm${d_model}_el${e_layers}_dff${d_ff}_bs${batch_size}_dr${dropout}_lr${learning_rate}_seed${seed}"

# ==========================================
# 7. Execution Command
# ==========================================

CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch \
  --mixed_precision fp16 \
  --num_processes $num_process \
  --main_process_port $master_port \
  run_main.py \
  --model $model_name \
  --dataset $dataset \
  --root_path $root_path \
  --processed_SOH_path $processed_SOH_path \
  --checkpoints $checkpoints \
  --input_mode $input_mode \
  --split_json_path $split_json_path \
  --split_tag $split_tag \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --dropout $dropout \
  --patience $patience \
  --d_model $d_model \
  --n_heads $n_heads \
  --e_layers $e_layers \
  --d_ff $d_ff \
  --patch_len $patch_len \
  --stride $stride \
  --activation $activation \
  --factor $factor \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --eol_threshold $eol_threshold \
  --early_cycle_threshold $early_cycle_threshold \
  --charge_discharge_length $charge_discharge_length \
  --task_name $task_name \
  --cache_root $cache_root \
  --gpu 0 \
  --seed $seed \
  --use_capacity_resample \
  --use_multi_gpu
