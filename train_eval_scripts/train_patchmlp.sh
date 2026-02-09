#!/bin/bash

# PatchMLP Training Script
# Usage: bash train_eval_scripts/train_patchmlp.sh [current_voltage|soh_to_soh|both] [num_gpus]
# Examples:
#   bash train_eval_scripts/train_patchmlp.sh current_voltage        # Single GPU
#   bash train_eval_scripts/train_patchmlp.sh current_voltage 2      # 2 GPUs

cd "$(dirname "$0")/.." || exit 1

MODEL=PatchMLP
DATASET=Li_ion
MODE=${1:-current_voltage}
NUM_GPUS=${2:-1}

# Set multi-GPU environment if needed
if [ "$NUM_GPUS" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    USE_MULTI_GPU="--use_multi_gpu"
    echo "Multi-GPU training enabled: ${NUM_GPUS} GPUs (0,1)"
else
    export CUDA_VISIBLE_DEVICES=0
    USE_MULTI_GPU=""
    echo "Single GPU training (GPU 0)"
fi

# Model architecture
D_MODEL=128
E_LAYERS=3
DROPOUT=0.1

# Training hyperparameters
LR=0.0001
WEIGHT_DECAY=0.01
PATIENCE=50
EPOCHS=250
SEED=2024

# Data params (MUST match for cache compatibility)
SEQ_LEN=1
PRED_LEN=5000
EARLY_CYCLE_THRESHOLD=100
CHARGE_DISCHARGE_LENGTH=300
EOL_THRESHOLD=0.8

# Paths
ROOT_PATH="/data/trf/python_works/BatteryLife/dataset"
PROCESSED_SOH_PATH="/data/trf/python_works/BatteryLife/dataset/processed_SOH"
CACHE_ROOT="/home/djt/Trajectory_forecasting/.cache/"

train_mode() {
    local INPUT_MODE=$1
    local BATCH_SIZE=$2

    # Adjust checkpoint dir for multi-GPU
    if [ "$NUM_GPUS" -gt 1 ]; then
        CHECKPOINT_DIR="./checkpoints/${MODEL}_${DATASET}_${INPUT_MODE}_seed${SEED}_${NUM_GPUS}gpu"
    else
        CHECKPOINT_DIR="./checkpoints/${MODEL}_${DATASET}_${INPUT_MODE}_seed${SEED}"
    fi

    echo "========================================="
    echo "Training ${MODEL} [${INPUT_MODE}]"
    echo "GPUs: ${NUM_GPUS}"
    echo "Batch: ${BATCH_SIZE}, Epochs: ${EPOCHS}"
    echo "========================================="

    mkdir -p $CHECKPOINT_DIR

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
        --weight_decay $WEIGHT_DECAY \
        --dropout $DROPOUT \
        --patience $PATIENCE \
        --d_model $D_MODEL \
        --e_layers $E_LAYERS \
        --activation gelu \
        --seq_len $SEQ_LEN \
        --pred_len $PRED_LEN \
        --eol_threshold $EOL_THRESHOLD \
        --early_cycle_threshold $EARLY_CYCLE_THRESHOLD \
        --charge_discharge_length $CHARGE_DISCHARGE_LENGTH \
        --task_name soh_forecast \
        --cache_root $CACHE_ROOT \
        --gpu 0 \
        --seed $SEED \
        --use_norm 1 \
        --use_capacity_resample \
        $USE_MULTI_GPU

    echo "Completed ${INPUT_MODE}"
}

case "$MODE" in
    current_voltage)
        train_mode "current_voltage" 32
        ;;
    soh_to_soh)
        train_mode "soh_to_soh" 256
        ;;
    both)
        train_mode "current_voltage" 32
        train_mode "soh_to_soh" 256
        echo "All training completed!"
        ;;
    *)
        echo "Usage: bash train_eval_scripts/train_patchmlp.sh [current_voltage|soh_to_soh|both]"
        exit 1
        ;;
esac
