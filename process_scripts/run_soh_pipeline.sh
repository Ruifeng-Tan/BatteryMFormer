#!/bin/bash

# SOH Processing Pipeline
# This script runs the complete SOH processing pipeline:
#   Step 0: (Optional) Convert ISU_ILCC raw JSON to cleaned_data format
#   Step 1: Generate SOH from cleaned_data
#   Step 2: Run dataset-specific preprocess scripts
#   Step 3: Merge MICH and MICH_EXP into total_MICH
#   Step 4: (Optional) Time normalization on cleaned_data
#   Step 5: Rename Tongji files (-# to --)

set -e

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Path Configuration
# Path to cleaned_data directory (contains battery pkl files)
INPUT_DIR="/data/trf/python_works/BatteryLife/dataset"
# Intermediate SOH output directory
SOH_DIR="/data/trf/python_works/BatteryLife/dataset/SOH"
# Final processed SOH output directory
OUTPUT_DIR="/data/trf/python_works/BatteryLife/dataset/processed_SOH"
# Path to CALB cycling data (used in Step 1)
CALB_CAPACITY_FILE="/data/trf/python_works/Trajectory_forecasting/process_scripts/overall_CALB_cycling_data.xlsx"
# Path to raw ISU_ILCC dataset (all zip files should be unzipped)
# Set to "" to skip ISU_ILCC conversion in Step 0
ISU_ILCC_RAW_DIR="/data/trf/raw_ISU"

# 2. Pipeline Options
# Python command (e.g., "python" or "conda run -n battery python")
PYTHON_CMD="python"
# Whether to run time normalization in Step 4 (true/false)
TIME_NORMALIZE=true
# Number of parallel workers for ISU_ILCC preprocessing and time normalization
NUM_WORKERS=50

# 3. Print Configuration
echo "========================================"
echo "SOH Processing Pipeline"
echo "========================================"
echo "Input directory (cleaned_data): $INPUT_DIR"
echo "Intermediate SOH directory:     $SOH_DIR"
echo "Output directory (processed):   $OUTPUT_DIR"
echo "Python command:                 $PYTHON_CMD"
echo "ISU_ILCC raw data directory:    ${ISU_ILCC_RAW_DIR:-'(not specified)'}"
echo "Time normalization:             $TIME_NORMALIZE"
echo "========================================"

mkdir -p "$SOH_DIR"
mkdir -p "$OUTPUT_DIR"

# Step 0: Convert ISU_ILCC Raw JSON to cleaned_data
if [[ -n "$ISU_ILCC_RAW_DIR" ]]; then
    echo ""
    echo "[Step 0] Converting ISU_ILCC JSON to cleaned_data format..."
    if [[ -d "$ISU_ILCC_RAW_DIR" ]]; then
        $PYTHON_CMD "$SCRIPT_DIR/convert_isu_ilcc_to_pkl.py" \
            --input_dir "$ISU_ILCC_RAW_DIR" \
            --output_dir "$INPUT_DIR/ISU_ILCC"
    else
        echo "Warning: ISU_ILCC raw directory not found: $ISU_ILCC_RAW_DIR, skipping."
    fi
fi

# Step 1: Generate SOH from cleaned_data
echo ""
echo "[Step 1] Generating SOH from cleaned_data..."

$PYTHON_CMD "process_scripts/generate_soh.py" \
    --cleaned_data_root "$INPUT_DIR" \
    --output_root "$SOH_DIR"

$PYTHON_CMD "process_scripts/generate_CALB_soh.py" \
    --raw_data_file_path "$CALB_CAPACITY_FILE" \
    --output_path "$SOH_DIR"

# Step 2: Run Dataset-Specific Preprocess Scripts
echo ""
echo "[Step 2] Running preprocess scripts..."

# Dataset-specific scripts
$PYTHON_CMD "$SCRIPT_DIR/preprocess_HNEI.py"     --input "$SOH_DIR/HNEI"     --output "$OUTPUT_DIR/HNEI"
$PYTHON_CMD "$SCRIPT_DIR/preprocess_MICH.py"     --input "$SOH_DIR/MICH"     --output "$OUTPUT_DIR/MICH"
$PYTHON_CMD "$SCRIPT_DIR/preprocess_MICH_EXP.py" --input "$SOH_DIR/MICH_EXP" --output "$OUTPUT_DIR/MICH_EXP"
$PYTHON_CMD "$SCRIPT_DIR/preprocess_RWTH.py"     --input "$SOH_DIR/RWTH"     --output "$OUTPUT_DIR/RWTH"
$PYTHON_CMD "$SCRIPT_DIR/preprocess_SNL.py"      --input "$SOH_DIR/SNL"      --output "$OUTPUT_DIR/SNL"
$PYTHON_CMD "$SCRIPT_DIR/preprocess_Tongji.py"   --input "$SOH_DIR/Tongji"   --output "$OUTPUT_DIR/Tongji"

$PYTHON_CMD "$SCRIPT_DIR/preprocess_ISU_ILCC.py" \
    --input "$SOH_DIR/ISU_ILCC" \
    --output "$OUTPUT_DIR/ISU_ILCC" \
    --rpt_json "$SCRIPT_DIR/all_rpt_positions.json" \
    --workers "$NUM_WORKERS"

# General preprocess script (handles remaining datasets)
$PYTHON_CMD "$SCRIPT_DIR/preprocess.py" \
    --base_input "$SOH_DIR" \
    --base_output "$OUTPUT_DIR"

# Step 3: Merge MICH and MICH_EXP into total_MICH
echo ""
echo "[Step 3] Merging MICH and MICH_EXP into total_MICH..."

TARGET_MICH="$OUTPUT_DIR/total_MICH"
mkdir -p "$TARGET_MICH"

if [[ -d "$OUTPUT_DIR/MICH" ]]; then
    cp -r "$OUTPUT_DIR/MICH"/. "$TARGET_MICH/"
else
    echo "Warning: $OUTPUT_DIR/MICH does not exist, skipping."
fi

if [[ -d "$OUTPUT_DIR/MICH_EXP" ]]; then
    cp -r "$OUTPUT_DIR/MICH_EXP"/. "$TARGET_MICH/"
else
    echo "Warning: $OUTPUT_DIR/MICH_EXP does not exist, skipping."
fi

# Step 4: Time Normalization (Optional)
if [[ "$TIME_NORMALIZE" == "true" ]]; then
    echo ""
    echo "[Step 4] Running time normalization on cleaned_data..."
    $PYTHON_CMD "$SCRIPT_DIR/time_normalization.py" \
        --input "$INPUT_DIR" \
        --workers "$NUM_WORKERS"
fi

# Step 5: Rename Tongji Files (-# to --)
echo ""
echo "[Step 5] Renaming Tongji files..."

if [[ -f "$SCRIPT_DIR/rename_Tongji_cells.py" ]]; then
    $PYTHON_CMD "$SCRIPT_DIR/rename_Tongji_cells.py" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Warning: rename_Tongji_cells.py not found in $SCRIPT_DIR, skipping."
fi

# Done
echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Final output: $OUTPUT_DIR"
