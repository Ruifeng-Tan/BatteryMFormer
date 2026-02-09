#!/bin/bash

# Fine-tune a pretrained BatteryMFormer on a new target dataset (full fine-tuning).
# The pretrained checkpoint directory must contain: args.json, model.safetensors (or checkpoint.pth), label_scaler.
# Model architecture and data parameters are loaded from the pretrained args.json automatically.

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
# Specify GPU IDs to use (e.g., "0" or "0,1")
gpu_ids=0
# Specify the number of GPUs (must match the count in gpu_ids)
num_process=1
# Main process port (avoid conflicts among multiple runs)
master_port=19520
seed=2024

# ==========================================
# 2. Pretrained Model & Target Dataset
# ==========================================
# Path to the pretrained checkpoint directory
pretrained_path=/path/to/your/pretrained/checkpoint
# Target dataset to fine-tune on (Li_ion, CALB, NA-ion, ZN-coin)
finetune_dataset=CALB

# ==========================================
# 3. Fine-tuning Hyperparameters
# ==========================================
finetune_lr=0.00001
finetune_epochs=50
batch_size=128
patience=20
weight_decay=0.01
grad_clip=1.0
lradj=warmup_cosine
warmup_epochs=5

# ==========================================
# 4. Execution Command
# ==========================================
CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch \
  --mixed_precision fp16 \
  --num_processes $num_process \
  --main_process_port $master_port \
  finetune_model.py \
  --pretrained_path $pretrained_path \
  --finetune_dataset $finetune_dataset \
  --finetune_lr $finetune_lr \
  --finetune_epochs $finetune_epochs \
  --batch_size $batch_size \
  --patience $patience \
  --weight_decay $weight_decay \
  --grad_clip $grad_clip \
  --lradj $lradj \
  --warmup_epochs $warmup_epochs \
  --seed $seed \
  --use_amp
