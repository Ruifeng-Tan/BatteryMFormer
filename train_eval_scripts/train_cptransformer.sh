#!/bin/bash

# CPTransformer Training Script for MemoryNet with Accelerate Support
# Supports two input modes: current_voltage and soh_to_soh
# Usage:
#   Single GPU: bash train_eval_scripts/train_cptransformer.sh [current_voltage|soh_to_soh|both]
#   Multi GPU:  bash train_eval_scripts/train_cptransformer.sh [current_voltage|soh_to_soh|both] [num_gpus]
#
# Examples:
#   bash train_eval_scripts/train_cptransformer.sh current_voltage      # Single GPU
#   bash train_eval_scripts/train_cptransformer.sh current_voltage 4    # 4 GPUs

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Default: Single GPU
export CUDA_VISIBLE_DEVICES=0

# Model configuration
MODEL=CPTransformer
DATASET=Li_ion # Li_ion to use 13 Li-ion battery datasets

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

# Shared training parameters
LR=0.00005
LRADJ=constant
WARMUP_EPOCHS=5
# IMPORTANT: Regularization to prevent overfitting with variable length training
WEIGHT_DECAY=0.01            # L2 regularization (was 0.0)
GRAD_CLIP=1.0
PATIENCE=50                  # Early stop faster to prevent overfitting (was 100)
INPUT_CYCLES=100

# Variable length training parameters
# When enabled, generates multiple samples per battery with different input lengths
# This significantly increases training data (e.g., 91 samples per battery with min=10, max=100)
# WARNING: Same-battery samples are highly correlated, need strong regularization!
VARIABLE_LENGTH_TRAINING=true
MIN_INPUT_CYCLES=1           # Use all cycles from 1 to 100 for variable length training

# Model architecture parameters
D_MODEL=128
N_HEADS=4
E_LAYERS=6                   # Slightly smaller model (was 8)
D_LAYERS=1
D_FF=256
DROPOUT=0.2                  # Stronger dropout to prevent overfitting (was 0.1)
ACTIVATION=gelu
FACTOR=1

# Data parameters
SEQ_LEN=1
PRED_LEN=5000
EARLY_CYCLE_THRESHOLD=100
CHARGE_DISCHARGE_LENGTH=300
EOL_THRESHOLD=0.8
TRUNCATE_START_CYCLE=100

# Paths
# Note: Using cleaned_data for raw cycle data, processed_SOH for SOH trajectories
ROOT_PATH="/data/trf/python_works/BatteryLife/dataset"
processed_SOH_path="/data/trf/python_works/BatteryLife/dataset/processed_SOH"
CACHE_ROOT="./.cache/"
INPUT_MODE=current_voltage
BATCH_SIZE=512
EPOCHS=250

# Alternative: Use processed SOH directory (same as baseline2)
# ROOT_PATH="/ai/dl_project/BatteryLife/dataset/SOH_processed_data"

# Function to train a single mode
train_mode() {
    # local INPUT_MODE=$1
    # local BATCH_SIZE=$2
    # local EPOCHS=$3

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
        # Multi-GPU training with Accelerate
        TRAIN_CMD="accelerate launch --num_processes ${NUM_GPUS} --mixed_precision fp16 run_main.py"
    else
        # Single GPU training (original method)
        TRAIN_CMD="python run_main.py"
    fi

    # Build variable length training flag
    # if [ "$VARIABLE_LENGTH_TRAINING" = "true" ]; then
    #     VAR_LEN_FLAG="--variable_length_training --min_input_cycles $MIN_INPUT_CYCLES"
    # else
    #     VAR_LEN_FLAG=""
    # fi

    # Run training (make sure you've activated batterylife conda environment)
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
        --seed 42 \
        --use_amp \
        --accumulation_steps 1 \
        $USE_MULTI_GPU \
        $VAR_LEN_FLAG \
        --use_capacity_resample \
        --resume_existing

    echo ""
    echo "Completed training ${INPUT_MODE} mode"
    echo ""
}

# Train based on mode selection
if [ "$MODE" = "current_voltage" ]; then
    # Train only current_voltage mode
    train_mode "current_voltage" 128 500
elif [ "$MODE" = "soh_to_soh" ]; then
    # Train only soh_to_soh mode
    train_mode "soh_to_soh" 48 600
elif [ "$MODE" = "both" ]; then
    # Train both modes sequentially
    echo "Training both input modes..."
    echo ""

    # 1. soh_to_soh mode (batch_size=48, epochs=600)
    train_mode "soh_to_soh" 128 600

    # 2. current_voltage mode (batch_size=128, epochs=500)
    train_mode "current_voltage" 256 800

    echo "========================================="
    echo "All training completed!"
    echo "========================================="
else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: bash train_eval_scripts/train_cptransformer.sh [current_voltage|soh_to_soh|both] [num_gpus]"
    echo "Examples:"
    echo "  bash train_eval_scripts/train_cptransformer.sh current_voltage     # Single GPU"
    echo "  bash train_eval_scripts/train_cptransformer.sh current_voltage 4   # 4 GPUs"
    exit 1
fi
