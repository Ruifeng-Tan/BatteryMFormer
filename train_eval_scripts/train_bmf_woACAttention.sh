#!/bin/bash

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
# Specify GPU IDs to use (e.g., "0" or "0,1,2,3")
gpu_ids=0,1
# Specify the number of GPUs (must match the count in gpu_ids)
num_process=2
# Main process port (avoid conflicts among multiple runs)
master_port=29477
seed=2024
# ==========================================
# 2. Model & Data Configuration
# ==========================================
model_name=BatteryMFormer_woACAttention
dataset=NA-ion
input_mode=current_voltage  # Options: current_voltage or soh_to_soh
# ---------- Evaluation Protocol ----------
# We use Leave-One-Aging-condition-Out (LOAO) evaluation by default.
# Each split holds out one aging condition for testing.
# Available LOAO splits:
#   CALB:   data_provider/split_json/loao/CALB_loao_cond{396..399}_valseed2024.json  (4 folds)
#   NA-ion: data_provider/split_json/loao/NA-ion_loao_cond{400..411}_valseed2024.json (12 folds)
# For Li-ion pure-OOD evaluation:
#   data_provider/split_json/liion_pure_ood/Li_ion_pure_ood_seed{2021,42,2024}.json
# To use the legacy random split, leave split_json_path empty.
split_json_path=./data_provider/split_json/loao/NA-ion_loao_cond400_valseed2024.json

# ========================================== 
# 3. Model Architecture Hyperparameters
# ==========================================
enc_in=3
num_slots=64
kernel_size=10
stride=10
cnn_channels=16
num_segments=50
top_k=2
num_query=10
temperature=1.0

k_dim=256
d_model=64
n_heads=8
e_layers=2
d_layers=8
d_ff=128
d_ffs=128
dropout=0.12
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
batch_size=64       # Per-GPU batch size (global batch = batch_size * num_process)
train_epochs=200
learning_rate=7.78e-05
lradj=constant
warmup_epochs=5
patience=20
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

# Paths (modify these to match your environment)
root_path=/path/to/your/dataset
processed_SOH_path=/path/to/your/processed_SOH


# Modify the checkpoint path by adding kernel_size, stride, and cnn_channels
split_tag=$(basename "$split_json_path" .json)
checkpoints="./checkpoints/${model_name}_${dataset}_${split_tag}_te${train_epochs}_ks${kernel_size}_st${stride}_ch${cnn_channels}_dff${d_ff}_dffs${d_ffs}_dm${d_model}_kdm${k_dim}_el${e_layers}_dl${d_layers}_bs${batch_size}_dr${dropout}_nh${n_heads}_lr${learning_rate}_nq${num_query}_${lambda_recovery}_${num_slots}_clip${use_grad_clip}_seed${seed}"

cache_root=/path/to/your/cache
# ==========================================
# 6. Execution Command
# ==========================================
# Note: Remove --resume_existing if you do not need to resume training
# Note: --mixed_precision fp16 corresponds to --use_amp in the original script

# Build the base command
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


# Add gradient clipping arguments based on use_grad_clip
if [ "$use_grad_clip" = "True" ] || [ "$use_grad_clip" = "true" ]; then
    echo "Enable gradient clipping..."
    # Add gradient clipping flag (add other related arguments here if needed)
    FULL_CMD="$BASE_CMD --use_grad_clip"
    
    # If you want to set a clipping threshold, add extra arguments (optional)
    # max_grad_norm=1.0
    # FULL_CMD="$BASE_CMD --use_grad_clip --max_grad_norm $max_grad_norm"
else
    echo "Do not use gradient clipping..."
    FULL_CMD="$BASE_CMD"
fi

# Print the full command
echo "Executing command:"
echo "$FULL_CMD"
echo ""

# Execute the command
eval $FULL_CMD
