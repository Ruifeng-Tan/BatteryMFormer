#!/bin/bash

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
# 指定使用的GPU编号 (例如: "0" 或 "0,1,2,3")
gpu_ids=0,1
# 指定使用的GPU数量 (必须与 gpu_ids 的数量一致)
num_process=2
# 主进程端口 (防止多任务冲突)
master_port=19511
seed=2024
# ==========================================
# 2. Model & Data Configuration
# ==========================================
model_name=iTransformer
dataset=Li_ion  #  Li_ion 
input_mode=current_voltage  # 选项: current_voltage 或 soh_to_soh

# ==========================================
# 3. Model Architecture Hyperparameters
# ==========================================
enc_in=3
num_slots=64
stride=10
cnn_channels=16
num_segments=10
num_routed_experts=8
num_shared_experts=2
top_k=2
num_query=1
temperature=1.0


k_dim=8
d_model=128
n_heads=8
e_layers=2
d_layers=2
e_layers2=2
d_ff=128
dropout=0.1   #  0.05 -> 0.1,
activation=gelu
factor=1
d_llm=1024
mem_dim=512


# ==========================================
# 4. Training Hyperparameters
# ==========================================
lambda_recovery=100.0
lambda_mem=1.0
lambda_life_loss=1.0
batch_size=16    # 单卡 Batch Size (总 Batch = batch_size * num_process)
train_epochs=100
learning_rate=0.00005
lradj=constant
warmup_epochs=5
weight_decay=0.0
patience=30
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

# Paths
root_path="/ai/dl_project/MemoryNet/dataset/cleaned_data"
processed_SOH_path="/ai/dl_project/MemoryNet/dataset/processed_SOH"

# 处理 kernel_sizes 中的空格，用下划线替换，以便在路径中使用
kernel_str=$(echo $kernel_sizes | tr ' ' '_')

# 修改 checkpoint 路径，添加 kernel_sizes, stride 和 cnn_channels
checkpoints="./checkpoints/${model_name}_${dataset}_${input_mode}_ks${kernel_str}_st${stride}_ch${cnn_channels}_dff${d_ff}_md${mem_dim}_dm${d_model}_kdm${k_dim}_el${e_layers}_el2${e_layers2}_dl${d_layers}_bs${batch_size}_dr${dropout}_nh${n_heads}_lr${learning_rate}_nq${num_query}_${lambda_recovery}_${num_slots}_seed${seed}"

cache_root="./.cache/"
# ==========================================
# 6. Execution Command
# ==========================================
# 注意: --resume_existing 如果不需要断点续训请删除该行
# 注意: --mixed_precision fp16 对应原脚本的 --use_amp

CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch \
  --mixed_precision no \
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
  --grad_clip $grad_clip \
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
  --num_query $num_query \
  --accumulation_steps 1 \
  --use_multi_gpu \
  --use_capacity_resample \
  --lambda_recovery $lambda_recovery \
  --num_slots $num_slots \
  --temperature $temperature \
  --top_k $top_k \
  --lambda_mem $lambda_mem \
  --num_segments $num_segments \
  --lambda_life_loss $lambda_life_loss \
  --num_routed_experts $num_routed_experts \
  --num_shared_experts $num_shared_experts \
  --e_layers2 $e_layers2 \
  --k_dim $k_dim \
  --enc_in $enc_in \
  --mem_dim $mem_dim
