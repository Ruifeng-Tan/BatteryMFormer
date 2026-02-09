#!/bin/bash
# TimesFM Baseline Evaluation Script
# TimesFM is a pretrained foundation model - no training needed, just evaluation

cd "$(dirname "$0")/.." || exit 1

# Use GPU 1 (or change to available GPU)
GPU=${3:-1}

# Dataset to evaluate (Li_ion, CALB, NA-ion, ZN-coin)
DATASET=${1:-Li_ion}
SEED=${2:-2024}

# Must match the cache file parameters
EARLY_CYCLE_THRESHOLD=100
CHARGE_DISCHARGE_LENGTH=300
SEQ_LEN=1
MAX_TRAJECTORY_LEN=5000

echo "========================================="
echo "TimesFM Baseline Evaluation"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"
echo "========================================="

python train_eval_scripts/eval_timesfm.py \
    --dataset $DATASET \
    --seed $SEED \
    --flag test \
    --early_cycle_threshold $EARLY_CYCLE_THRESHOLD \
    --charge_discharge_length $CHARGE_DISCHARGE_LENGTH \
    --seq_len $SEQ_LEN \
    --max_trajectory_len $MAX_TRAJECTORY_LEN \
    --use_capacity_resample \
    --batch_size 32 \
    --gpu $GPU \
    --output_dir ./results/timesfm

echo ""
echo "Evaluation completed!"
