#!/bin/bash

# MultiPatchFormer Training Script for MemoryNet with Accelerate Support
# Usage:
#   Single GPU: bash train_eval_scripts/train_multipatchformer.sh [current_voltage|soh_to_soh|both]
#   Multi GPU:  bash train_eval_scripts/train_multipatchformer.sh [current_voltage|soh_to_soh|both] [num_gpus]

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Model configuration
MODEL=MultiPatchFormer
DATASET=MIX_large

# Get input mode and GPU count from command line arguments
MODE=${1:-both}
NUM_GPUS=${2:-1}

# Set multi-GPU environment if needed
if [ "$NUM_GPUS" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    USE_MULTI_GPU="--use_multi_gpu"
    echo "Multi-GPU training enabled: ${NUM_GPUS} GPUs"
else
    export CUDA_VISIBLE_DEVICES=0
    USE_MULTI_GPU=""
    echo "Single GPU training"
fi

# Shared training parameters (参考CPTransformer)
LR=0.0001
LRADJ=cosine
WARMUP_EPOCHS=5
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0
PATIENCE=70

# Model architecture parameters (参考CPTransformer)
D_MODEL=128
N_HEADS=4
E_LAYERS=8
D_LAYERS=1
D_FF=256
DROPOUT=0.1
ACTIVATION=gelu
FACTOR=1

# Data parameters
SEQ_LEN=100
PRED_LEN=5000
EARLY_CYCLE_THRESHOLD=100
CHARGE_DISCHARGE_LENGTH=100
EOL_THRESHOLD=0.8
TRUNCATE_START_CYCLE=100

# Paths
ROOT_PATH="/ai/dl_project/MemoryNet/dataset/cleaned_data"
CACHE_ROOT="/ai/dl_project/MemoryNet/.cache"

# Function to train a single mode
train_mode() {
    local INPUT_MODE=$1
    local BATCH_SIZE=$2
    local EPOCHS=$3

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

    mkdir -p $CHECKPOINT_DIR

    if [ "$NUM_GPUS" -gt 1 ]; then
        TRAIN_CMD="accelerate launch --num_processes ${NUM_GPUS} --mixed_precision fp16 run_main.py"
    else
        TRAIN_CMD="python run_main.py"
    fi

    $TRAIN_CMD \
        --model $MODEL \
        --dataset $DATASET \
        --root_path $ROOT_PATH \
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
        $USE_MULTI_GPU \
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
    train_mode "soh_to_soh" 128 600
elif [ "$MODE" = "both" ]; then
    # Train both modes sequentially
    echo "Training both input modes..."
    echo ""

    # 1. soh_to_soh mode (batch_size=128, epochs=600)
    train_mode "soh_to_soh" 128 600

    # 2. current_voltage mode (batch_size=128, epochs=500)
    train_mode "current_voltage" 128 500

    echo "========================================="
    echo "All training completed!"
    echo "========================================="
else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: bash train_eval_scripts/train_multipatchformer.sh [current_voltage|soh_to_soh|both] [num_gpus]"
    exit 1
fi
