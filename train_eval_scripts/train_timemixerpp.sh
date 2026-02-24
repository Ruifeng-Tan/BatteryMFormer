#!/bin/bash

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
# Specify GPU IDs to use (e.g., "0" or "0,1,2,3")
gpu_ids=0,1
# Specify the number of GPUs (must match the count in gpu_ids)
num_process=2
# Main process port (avoid conflicts among multiple runs)
master_port=29454
seed=2021

# ==========================================
# 2. Model & Data Configuration
# ==========================================
model_name=TimeMixerPP
dataset=NA-ion
input_mode=current_voltage  # Options: current_voltage or soh_to_soh

# ==========================================
# 3. Model Architecture Hyperparameters
# ==========================================
d_model=128
n_heads=4
e_layers=3
d_layers=1
d_ff=256
dropout=0.1
activation=gelu

# TimeMixer++ specific parameters
down_sampling_window=2
down_sampling_layers=2
down_sampling_method=avg
num_kernels=6
top_k=3
channel_independence=1

# ==========================================
# 4. Training Hyperparameters
# ==========================================
batch_size=64        # Per-GPU batch size (global batch = batch_size * num_process)
train_epochs=100
learning_rate=0.0001
lradj=warmup_cosine
warmup_epochs=5
weight_decay=0.01
patience=30
use_grad_clip=True
grad_clip=1.0

# ==========================================
# 5. Sequence & Task Parameters
# ==========================================
seq_len=1
pred_len=5000
early_cycle_threshold=100
charge_discharge_length=300
eol_threshold=0.8
truncate_start_cycle=100
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

checkpoints="./checkpoints/${model_name}_${dataset}_${input_mode}_dm${d_model}_el${e_layers}_dl${d_layers}_dff${d_ff}_bs${batch_size}_dr${dropout}_lr${learning_rate}_seed${seed}"

# ==========================================
# 7. Execution Command
# ==========================================
# Note: --mixed_precision fp16 corresponds to --use_amp in the training script

BASE_CMD="CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch \
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
  --lradj $lradj \
  --warmup_epochs $warmup_epochs \
  --weight_decay $weight_decay \
  --patience $patience \
  --d_model $d_model \
  --n_heads $n_heads \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --dropout $dropout \
  --activation $activation \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --eol_threshold $eol_threshold \
  --truncate_start_cycle $truncate_start_cycle \
  --early_cycle_threshold $early_cycle_threshold \
  --charge_discharge_length $charge_discharge_length \
  --task_name $task_name \
  --cache_root $cache_root \
  --gpu 0 \
  --seed $seed \
  --use_amp \
  --down_sampling_window $down_sampling_window \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method $down_sampling_method \
  --num_kernels $num_kernels \
  --top_k $top_k \
  --channel_independence $channel_independence \
  --use_capacity_resample \
  --use_multi_gpu"

# Add gradient clipping arguments based on use_grad_clip
if [ "$use_grad_clip" = "True" ] || [ "$use_grad_clip" = "true" ]; then
    FULL_CMD="$BASE_CMD --use_grad_clip --grad_clip $grad_clip"
else
    FULL_CMD="$BASE_CMD"
fi

eval $FULL_CMD
