#!/bin/bash
# TimesFM Baseline Evaluation Script
#
# This script evaluates TimesFM foundation model on SOH prediction task.
# Must be run in the TSorchestra (tso) conda environment.
#
# Usage:
#   bash train_eval_scripts/evaluate_timesfm.sh [seed] [input_cycles]
#
# Examples:
#   bash train_eval_scripts/evaluate_timesfm.sh 2021 100
#   bash train_eval_scripts/evaluate_timesfm.sh 2024 50

set -e

# Default parameters
SEED=${1:-2021}
INPUT_CYCLES=${2:-100}
EOL_THRESHOLD=0.8
MAX_PRED_LEN=500

# Output directory
OUTPUT_DIR="./results/timesfm_baseline"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "TimesFM Zero-shot SOH Prediction Evaluation"
echo "=============================================="
echo "Seed: $SEED"
echo "Input cycles: $INPUT_CYCLES"
echo "EOL threshold: $EOL_THRESHOLD"
echo ""

# Check if we're in tso environment
if [[ "$CONDA_DEFAULT_ENV" != "tso" ]]; then
    echo "WARNING: Not in 'tso' conda environment!"
    echo "Please run: conda activate tso"
    echo ""
    echo "Attempting to run with conda run..."
    /opt/conda/condabin/conda run -n tso python test/evaluate_timesfm_baseline.py \
        --seed $SEED \
        --input_cycles $INPUT_CYCLES \
        --eol_threshold $EOL_THRESHOLD \
        --max_pred_len $MAX_PRED_LEN \
        --output_dir $OUTPUT_DIR \
        --save_predictions
else
    python test/evaluate_timesfm_baseline.py \
        --seed $SEED \
        --input_cycles $INPUT_CYCLES \
        --eol_threshold $EOL_THRESHOLD \
        --max_pred_len $MAX_PRED_LEN \
        --output_dir $OUTPUT_DIR \
        --save_predictions
fi

echo ""
echo "Results saved to: $OUTPUT_DIR"
