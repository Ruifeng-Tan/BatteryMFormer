#!/bin/bash

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
# Specify GPU IDs to use (e.g., "0" or "0,1,2,3")
gpu_ids=0,1
# Specify the number of GPUs (must match the count in gpu_ids)
num_process=2
# Main process port (avoid conflicts among multiple runs)
master_port=29452
seed=2024

# ==========================================
# 2. Model & Data Configuration
# ==========================================
model_name=PatchMLP
dataset=NA-ion
input_mode=current_voltage  # Options: current_voltage or soh_to_soh

# ==========================================
# 3. Model Architecture Hyperparameters
# ==========================================
d_model=128
e_layers=3
dropout=0.1
activation=gelu
use_norm=1

# ==========================================
# 4. Training Hyperparameters
# ==========================================
batch_size=32        # Per-GPU batch size (global batch = batch_size * num_process)
train_epochs=250
learning_rate=0.0001
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

checkpoints="./checkpoints/${model_name}_${dataset}_${input_mode}_dm${d_model}_el${e_layers}_bs${batch_size}_dr${dropout}_lr${learning_rate}_seed${seed}"

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
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --dropout $dropout \
  --patience $patience \
  --d_model $d_model \
  --e_layers $e_layers \
  --activation $activation \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --eol_threshold $eol_threshold \
  --early_cycle_threshold $early_cycle_threshold \
  --charge_discharge_length $charge_discharge_length \
  --task_name $task_name \
  --cache_root $cache_root \
  --gpu 0 \
  --seed $seed \
  --use_norm $use_norm \
  --use_capacity_resample \
  --use_multi_gpu
