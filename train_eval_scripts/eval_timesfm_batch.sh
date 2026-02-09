#!/bin/bash
# Batch evaluation script for TimesFM baseline
# Tests all combinations of seeds and datasets

# Seeds to test
SEEDS=(42 2021 2024)

# Datasets to test
DATASETS=("Li_ion" "NA-ion" "CALB")

# GPU to use (can be overridden by command line)
GPU=${1:-0}

echo "=============================================="
echo "TimesFM Batch Evaluation"
echo "Seeds: ${SEEDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "GPU: $GPU"
echo "=============================================="

# Run all combinations sequentially (model loading takes time, so one at a time)
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "----------------------------------------------"
        echo "Evaluating: $dataset with seed $seed"
        echo "----------------------------------------------"

        python /ai/dl_project/MemoryNet/train_eval_scripts/eval_timesfm.py \
            --dataset "$dataset" \
            --seed "$seed" \
            --gpu "$GPU" \
            --flag test

        echo "Completed: $dataset seed=$seed"
    done
done

echo ""
echo "=============================================="
echo "All evaluations completed!"
echo "Results saved to: ./results/timesfm/"
echo "=============================================="

# Print summary
echo ""
echo "Summary of results:"
echo "=============================================="
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Dataset: $dataset"
    for seed in "${SEEDS[@]}"; do
        result_file="./results/timesfm/timesfm_${dataset}_seed${seed}_test.json"
        if [ -f "$result_file" ]; then
            mape=$(python -c "import json; d=json.load(open('$result_file')); print(f'MAPE={d[\"avg_mape\"]:.4f}%')")
            echo "  seed=$seed: $mape"
        else
            echo "  seed=$seed: [not found]"
        fi
    done
done
