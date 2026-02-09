#!/bin/bash

# DLinear Training Script for MemoryNet
# Linear decomposition model for SOH trajectory forecasting
# Paper: https://arxiv.org/pdf/2205.13504.pdf
# Usage:
#   Single GPU: bash train_eval_scripts/train_dlinear.sh [current_voltage|soh_to_soh|both]
#   Multi GPU:  bash train_eval_scripts/train_dlinear.sh [current_voltage|soh_to_soh|both] [num_gpus]

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Default: Single GPU
export CUDA_VISIBLE_DEVICES=2

# Model configuration
MODEL=DLinear
DATASET=Li_ion

# Get input mode and GPU count from command line arguments
MODE=${1:-both}
NUM_GPUS=${2:-1}

# Set multi-GPU environment if needed
if [ "$NUM_GPUS" -gt 2 ]; then
    export CUDA_VISIBLE_DEVICES=2,3,4
    USE_MULTI_GPU="--use_multi_gpu"
    echo "Multi-GPU training enabled: ${NUM_GPUS} GPUs"
else
    export CUDA_VISIBLE_DEVICES=2
    USE_MULTI_GPU=""
    echo "Single GPU training"
fi

# Shared training parameters
LR=0.0001
LRADJ=constant
WARMUP_EPOCHS=5
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0
PATIENCE=50
INPUT_CYCLES=100

# Model architecture parameters (DLinear is simpler)
D_MODEL=128
N_HEADS=4        # Not used in DLinear, kept for consistency
E_LAYERS=4       # Not used in DLinear
D_LAYERS=2       # Not used in DLinear
D_FF=256
DROPOUT=0.2
ACTIVATION=gelu
FACTOR=1
MOVING_AVG=25    # DLinear-specific: moving average window size

# Data parameters
SEQ_LEN=1
PRED_LEN=5000
EARLY_CYCLE_THRESHOLD=100
CHARGE_DISCHARGE_LENGTH=300
EOL_THRESHOLD=0.8
TRUNCATE_START_CYCLE=100

# Paths
ROOT_PATH="/ai/dl_project/MemoryNet/dataset/cleaned_data"
processed_SOH_path="/ai/dl_project/MemoryNet/dataset/processed_SOH"
CACHE_ROOT="./.cache/"
INPUT_MODE=current_voltage
BATCH_SIZE=512
EPOCHS=250

# Function to train a single mode
train_mode() {
    # Adjust checkpoint dir for multi-GPU
    if [ "$NUM_GPUS" -gt 1 ]; then
        CHECKPOINT_DIR="./checkpoints/${MODEL}_${DATASET}_${INPUT_MODE}_${NUM_GPUS}gpu"
    else
        CHECKPOINT_DIR="./checkpoints/${MODEL}_${DATASET}_${INPUT_MODE}"
    fi

    echo "========================================="
    echo "Training ${MODEL} with ${INPUT_MODE} mode"
    echo "GPUs: ${NUM_GPUS}"
    echo "Batch size per GPU: ${BATCH_SIZE}"
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "Total batch size: $((BATCH_SIZE * NUM_GPUS))"
    fi
    echo "Epochs: ${EPOCHS}"
    echo "========================================="

    # Create checkpoint directory
    mkdir -p $CHECKPOINT_DIR

    # Choose training command based on GPU count
    if [ "$NUM_GPUS" -gt 1 ]; then
        TRAIN_CMD="accelerate launch --num_processes ${NUM_GPUS} --mixed_precision fp16 run_main.py"
    else
        TRAIN_CMD="python run_main.py"
    fi

    # Run training
    $TRAIN_CMD \
        --model $MODEL \
        --dataset $DATASET \
        --root_path $ROOT_PATH \
        --processed_SOH_path $processed_SOH_path \
        --checkpoints $CHECKPOINT_DIR \
        --input_mode $INPUT_MODE \
        --batch_size $BATCH_SIZE \
        --train_epochs $EPOCHS \
        --learning_rate $LR \
        --lradj $LRADJ \
        --warmup_epochs $WARMUP_EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --grad_clip $GRAD_CLIP \
        --patience $PATIENCE \
        --d_model $D_MODEL \
        --n_heads $N_HEADS \
        --e_layers $E_LAYERS \
        --d_layers $D_LAYERS \
        --d_ff $D_FF \
        --dropout $DROPOUT \
        --activation $ACTIVATION \
        --factor $FACTOR \
        --moving_avg $MOVING_AVG \
        --seq_len $SEQ_LEN \
        --pred_len $PRED_LEN \
        --eol_threshold $EOL_THRESHOLD \
        --truncate_start_cycle $TRUNCATE_START_CYCLE \
        --early_cycle_threshold $EARLY_CYCLE_THRESHOLD \
        --charge_discharge_length $CHARGE_DISCHARGE_LENGTH \
        --task_name soh_forecast \
        --cache_root $CACHE_ROOT \
        --gpu 0 \
        --seed 2024 \
        --use_amp \
        --accumulation_steps 1 \
        $USE_MULTI_GPU \
        --use_capacity_resample \
        --resume_existing

    echo ""
    echo "Completed training ${INPUT_MODE} mode"
    echo ""
}

# Train based on mode selection
if [ "$MODE" = "current_voltage" ]; then
    train_mode "current_voltage" 128 500
elif [ "$MODE" = "soh_to_soh" ]; then
    train_mode "soh_to_soh" 48 600
elif [ "$MODE" = "both" ]; then
    echo "Training both input modes..."
    echo ""
    train_mode "soh_to_soh" 128 600
    train_mode "current_voltage" 256 800
    echo "========================================="
    echo "All training completed!"
    echo "========================================="
else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: bash train_eval_scripts/train_dlinear.sh [current_voltage|soh_to_soh|both] [num_gpus]"
    exit 1
fi
