#!/bin/bash

# Evaluate a trained BatteryMFormer checkpoint on a target dataset.
# The checkpoint directory must contain: args.json, model.safetensors (or checkpoint.pth), label_scaler.
# Results (seen/unseen analysis, per-cycle metrics) are saved into the checkpoint directory.

# ==========================================
# 1. Hardware & Environment Setup
# ==========================================
# Specify GPU IDs to use (e.g., "0" or "0,1,2,3")
gpu_ids=0,1
# Specify the number of GPUs (must match the count in gpu_ids)
num_process=2
# Main process port (avoid conflicts among multiple runs)
master_port=29100

# ==========================================
# 2. Evaluation Configuration
# ==========================================
# Path to the checkpoint directory (must contain args.json, model weights, label_scaler)
args_path=/path/to/your/checkpoint
# Target dataset to evaluate on (Li_ion, CALB, NA-ion, ZN-coin)
target_dataset=Li_ion

# ==========================================
# 3. Execution Command
# ==========================================
CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch \
  --mixed_precision fp16 \
  --num_processes $num_process \
  --main_process_port $master_port \
  evaluate_model.py \
  --args_path $args_path \
  --target_dataset $target_dataset
