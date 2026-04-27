#!/bin/bash

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
gpu_ids=0,1
num_process=2
master_port=29460
seed=2024

# ==========================================
# 2. Model & Data Configuration
# ==========================================
model_name=IC2ML
dataset=NA-ion
# IC2ML requires capacity_increment input mode
input_mode=capacity_increment

# ==========================================
# 3. Model Architecture Hyperparameters
# ==========================================
d_model=128
n_heads=4
e_layers=4
d_layers=2
d_ff=256
dropout=0.2
activation=gelu
factor=1

# ==========================================
# 4. Training Hyperparameters
# ==========================================
batch_size=512
train_epochs=250
learning_rate=0.00005
lradj=constant
warmup_epochs=5
weight_decay=0.01
patience=50
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

split_tag=$(basename "$split_json_path" .json)
checkpoints="./checkpoints/${model_name}_${dataset}_${split_tag}_${input_mode}_dm${d_model}_el${e_layers}_dl${d_layers}_dff${d_ff}_bs${batch_size}_dr${dropout}_lr${learning_rate}_seed${seed}"

# ==========================================
# 7. Execution Command
# ==========================================
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
  --split_json_path $split_json_path \
  --split_tag $split_tag \
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
  --factor $factor \
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
  --accumulation_steps 1 \
  --use_multi_gpu \
  --resume_existing \
  --use_capacity_resample"

if [ "$use_grad_clip" = "True" ] || [ "$use_grad_clip" = "true" ]; then
    FULL_CMD="$BASE_CMD --use_grad_clip --grad_clip $grad_clip"
else
    FULL_CMD="$BASE_CMD"
fi

eval $FULL_CMD
