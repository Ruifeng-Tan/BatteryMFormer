#!/bin/bash

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
gpu_ids=0,1
num_process=2
master_port=29447
seed=2024

# ==========================================
# 2. Model & Data Configuration
# ==========================================
model_name=BatteryMFormer
dataset=Li_ion
input_mode=current_voltage  # Options: current_voltage or soh_to_soh

# ---------- Evaluation Protocol ----------
# Default: Li-ion pure-OOD split (used in the Quick start of README).
# Other available splits:
#   Li-ion pure-OOD:  data_provider/split_json/pure_ood/Liion_split_seed{2021,42,2024}.json
#   Zn-ion pure-OOD:  data_provider/split_json/pure_ood/ZNcoin_split_seed{2021,42,2024}.json
#   CALB LOAO:        data_provider/split_json/loao/CALB_loao_cond{396..399}_seed2021.json
#   NA-ion LOAO:      data_provider/split_json/loao/NA-ion_loao_cond{400..411}_seed2021.json
# Leave this empty to fall back to the legacy random split.
split_json_path=./data_provider/split_json/pure_ood/Liion_split_seed2024.json

# Paths (modify these to match your environment)
root_path=/path/to/your/dataset
processed_SOH_path=/path/to/your/processed_SOH
cache_root=/path/to/your/cache

# ==========================================
# 3. Model Architecture Hyperparameters
# ==========================================
# Defaults below are from the Li_ion seed2024 fold in
# `asset/per_fold_hyperparameters.md`. For other folds / datasets,
# refer to that file and adjust accordingly.
enc_in=3
num_slots=64
kernel_size=10
stride=10
cnn_channels=16
num_segments=50
top_k=2
num_query=8
temperature=1.0

k_dim=512
d_model=128
n_heads=8
e_layers=2
d_layers=4
d_ff=128
d_ffs=128
dropout=0.2
weight_decay=0.0
activation=gelu
factor=1
d_llm=1024

# ==========================================
# 4. Training Hyperparameters
# ==========================================
lambda_recovery=100.0
lambda_mem=10.0
lambda_life_loss=0.0
batch_size=128      # Per-GPU batch size (global batch = batch_size * num_process)
train_epochs=300
learning_rate=1.5e-04
lradj=constant
warmup_epochs=5
patience=30
use_grad_clip=False

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

# Derive a short split tag from split_json_path for checkpoint/cache isolation
split_tag=$(basename "$split_json_path" .json)
checkpoints="./checkpoints/${model_name}_${dataset}_${split_tag}_dm${d_model}_dff${d_ff}_el${e_layers}_dl${d_layers}_bs${batch_size}_lr${learning_rate}_seed${seed}"

# ==========================================
# 6. Execution Command
# ==========================================
# Note: Remove --resume_existing if you do not need to resume training
# Note: --mixed_precision fp16 corresponds to --use_amp in the original script

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
  --d_llm $d_llm \
  --cache_root $cache_root \
  --gpu 0 \
  --seed $seed \
  --use_amp \
  --num_query $num_query \
  --accumulation_steps 1 \
  --use_multi_gpu \
  --resume_existing \
  --use_capacity_resample \
  --lambda_recovery $lambda_recovery \
  --num_slots $num_slots \
  --temperature $temperature \
  --top_k $top_k \
  --lambda_mem $lambda_mem \
  --num_segments $num_segments \
  --lambda_life_loss $lambda_life_loss \
  --k_dim $k_dim \
  --enc_in $enc_in \
  --kernel_size $kernel_size \
  --cnn_channels $cnn_channels \
  --stride $stride \
  --d_ffs $d_ffs"


if [ "$use_grad_clip" = "True" ] || [ "$use_grad_clip" = "true" ]; then
    echo "Enable gradient clipping..."
    FULL_CMD="$BASE_CMD --use_grad_clip"
else
    echo "Do not use gradient clipping..."
    FULL_CMD="$BASE_CMD"
fi

echo "Executing command:"
echo "$FULL_CMD"
echo ""

eval $FULL_CMD
