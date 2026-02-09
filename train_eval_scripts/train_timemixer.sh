#!/bin/bash

# TimeMixer Training Script for MemoryNet
# Memory-efficient version with CycleEncoder
# Usage:
#   Single GPU: bash train_eval_scripts/train_timemixer.sh [current_voltage|soh_to_soh|both]
#   Multi GPU:  bash train_eval_scripts/train_timemixer.sh [current_voltage|soh_to_soh|both] [num_gpus]
#
# Examples:
#   bash train_eval_scripts/train_timemixer.sh current_voltage      # Single GPU
#   bash train_eval_scripts/train_timemixer.sh current_voltage 4    # 4 GPUs

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Default: Single GPU
export CUDA_VISIBLE_DEVICES=1

# Model configuration
MODEL=TimeMixer
DATASET=Li_ion  # Li_ion to use 13 Li-ion battery datasets

# Get input mode and GPU count from command line arguments
MODE=${1:-both}        # Default: both
NUM_GPUS=${2:-1}       # Default: 1 (single GPU)

# Set multi-GPU environment if needed
if [ "$NUM_GPUS" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs
    USE_MULTI_GPU="--use_multi_gpu"
    echo "Multi-GPU training enabled: ${NUM_GPUS} GPUs"
else
    export CUDA_VISIBLE_DEVICES=0
    USE_MULTI_GPU=""
    echo "Single GPU training"
fi

# Shared training parameters (aligned with CPTransformer)
LR=0.00005
LRADJ=constant
WARMUP_EPOCHS=5
# Regularization to prevent overfitting
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0
PATIENCE=50
INPUT_CYCLES=100

# Variable length training parameters
VARIABLE_LENGTH_TRAINING=true
MIN_INPUT_CYCLES=1

# Model architecture parameters (aligned with CPTransformer)
D_MODEL=128
N_HEADS=4
E_LAYERS=6
D_LAYERS=1
D_FF=256
DROPOUT=0.2
ACTIVATION=gelu
FACTOR=1

# TimeMixer-specific parameters
# With CycleEncoder, seq_len is now num_cycles (100), not flattened (30000)
# So we can use moderate downsampling
DOWN_SAMPLING_WINDOW=2
DOWN_SAMPLING_LAYERS=3
DOWN_SAMPLING_METHOD=avg
DECOMP_METHOD=moving_avg
MOVING_AVG=25
TOP_K=5
USE_NORM=1

# Data parameters (aligned with CPTransformer)
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
    local INPUT_MODE=$1
    local BATCH_SIZE=$2
    local EPOCHS=$3

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
    echo "CycleEncoder seq_len: $((EARLY_CYCLE_THRESHOLD - SEQ_LEN + 1)) cycles"
    echo "========================================="

    mkdir -p "$CHECKPOINT_DIR"

    # Choose training command based on GPU count
    if [ "$NUM_GPUS" -gt 1 ]; then
        TRAIN_CMD="accelerate launch --num_processes ${NUM_GPUS} --mixed_precision fp16 run_main.py"
    else
        TRAIN_CMD="python run_main.py"
    fi

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
        --seq_len $SEQ_LEN \
        --pred_len $PRED_LEN \
        --eol_threshold $EOL_THRESHOLD \
        --truncate_start_cycle $TRUNCATE_START_CYCLE \
        --early_cycle_threshold $EARLY_CYCLE_THRESHOLD \
        --charge_discharge_length $CHARGE_DISCHARGE_LENGTH \
        --task_name soh_forecast \
        --cache_root $CACHE_ROOT \
        --gpu 0 \
        --seed 2021 \
        --use_amp \
        --accumulation_steps 1 \
        --down_sampling_window $DOWN_SAMPLING_WINDOW \
        --down_sampling_layers $DOWN_SAMPLING_LAYERS \
        --down_sampling_method $DOWN_SAMPLING_METHOD \
        --decomp_method $DECOMP_METHOD \
        --moving_avg $MOVING_AVG \
        --top_k $TOP_K \
        --use_norm $USE_NORM \
        $USE_MULTI_GPU \
        --resume_existing

    echo ""
    echo "Completed training ${INPUT_MODE} mode"
    echo ""
}

# Train based on mode selection
if [ "$MODE" = "current_voltage" ]; then
    train_mode "current_voltage" 512 250
elif [ "$MODE" = "soh_to_soh" ]; then
    train_mode "soh_to_soh" 512 250
elif [ "$MODE" = "both" ]; then
    echo "Training both input modes..."
    echo ""

    # 1. soh_to_soh mode
    train_mode "soh_to_soh" 512 250

    # 2. current_voltage mode
    train_mode "current_voltage" 512 250

    echo "========================================="
    echo "All training completed!"
    echo "========================================="
else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: bash train_eval_scripts/train_timemixer.sh [current_voltage|soh_to_soh|both] [num_gpus]"
    echo "Examples:"
    echo "  bash train_eval_scripts/train_timemixer.sh current_voltage     # Single GPU"
    echo "  bash train_eval_scripts/train_timemixer.sh current_voltage 4   # 4 GPUs"
    exit 1
fi
