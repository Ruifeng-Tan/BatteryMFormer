#!/bin/bash

# TimeMixer++ Training Script for MemoryNet
# Paper: TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis
# Usage:
#   Single GPU: bash train_eval_scripts/train_timemixerpp.sh [current_voltage|soh_to_soh|both]
#   Multi GPU:  bash train_eval_scripts/train_timemixerpp.sh [current_voltage|soh_to_soh|both] [num_gpus]
#
# Examples:
#   bash train_eval_scripts/train_timemixerpp.sh current_voltage      # Single GPU
#   bash train_eval_scripts/train_timemixerpp.sh current_voltage 4    # 4 GPUs

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Default: Single GPU
export CUDA_VISIBLE_DEVICES=1

# Model configuration
MODEL=TimeMixerPP
DATASET=Li_ion

# Get input mode and GPU count from command line arguments
MODE=${1:-current_voltage}  # Default: current_voltage
NUM_GPUS=${2:-1}            # Default: 1 (single GPU)

# Set multi-GPU environment if needed
if [ "$NUM_GPUS" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    USE_MULTI_GPU="--use_multi_gpu"
    echo "Multi-GPU training enabled: ${NUM_GPUS} GPUs"
else
    export CUDA_VISIBLE_DEVICES=1
    USE_MULTI_GPU=""
    echo "Single GPU training"
fi

# Shared training parameters
LR=0.0001
LRADJ=warmup_cosine
WARMUP_EPOCHS=5
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0
PATIENCE=30

# Model architecture parameters
D_MODEL=128
N_HEADS=4
E_LAYERS=3
D_LAYERS=1
D_FF=256
DROPOUT=0.1
ACTIVATION=gelu

# TimeMixer++ specific parameters
DOWN_SAMPLING_WINDOW=2
DOWN_SAMPLING_LAYERS=2
DOWN_SAMPLING_METHOD=avg
NUM_KERNELS=6
TOP_K=3
CHANNEL_INDEPENDENCE=1

# Data parameters
SEQ_LEN=1
PRED_LEN=5000
EARLY_CYCLE_THRESHOLD=100
CHARGE_DISCHARGE_LENGTH=300
EOL_THRESHOLD=0.8
TRUNCATE_START_CYCLE=100

# Paths
ROOT_PATH="/ai/dl_project/MemoryNet/dataset/cleaned_data"
PROCESSED_SOH_PATH="/ai/dl_project/MemoryNet/dataset/processed_SOH"
CACHE_ROOT="./.cache/"

# Training parameters
BATCH_SIZE=64
EPOCHS=100

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
    echo "Dataset: ${DATASET}"
    echo "GPUs: ${NUM_GPUS}"
    echo "Batch size: ${BATCH_SIZE}"
    echo "Epochs: ${EPOCHS}"
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
        --processed_SOH_path $PROCESSED_SOH_PATH \
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
        --down_sampling_window $DOWN_SAMPLING_WINDOW \
        --down_sampling_layers $DOWN_SAMPLING_LAYERS \
        --down_sampling_method $DOWN_SAMPLING_METHOD \
        --num_kernels $NUM_KERNELS \
        --top_k $TOP_K \
        --channel_independence $CHANNEL_INDEPENDENCE \
        $USE_MULTI_GPU

    echo ""
    echo "Completed training ${INPUT_MODE} mode"
    echo ""
}

# Train based on mode selection
if [ "$MODE" = "current_voltage" ]; then
    train_mode "current_voltage" $BATCH_SIZE $EPOCHS
elif [ "$MODE" = "soh_to_soh" ]; then
    train_mode "soh_to_soh" $BATCH_SIZE $EPOCHS
elif [ "$MODE" = "both" ]; then
    echo "Training both input modes..."
    echo ""
    train_mode "current_voltage" $BATCH_SIZE $EPOCHS
    train_mode "soh_to_soh" $BATCH_SIZE $EPOCHS
    echo "========================================="
    echo "All training completed!"
    echo "========================================="
else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: bash train_eval_scripts/train_timemixerpp.sh [current_voltage|soh_to_soh|both] [num_gpus]"
    echo "Examples:"
    echo "  bash train_eval_scripts/train_timemixerpp.sh current_voltage     # Single GPU"
    echo "  bash train_eval_scripts/train_timemixerpp.sh current_voltage 4   # 4 GPUs"
    exit 1
fi
