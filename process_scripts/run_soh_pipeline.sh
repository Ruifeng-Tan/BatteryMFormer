#!/bin/bash
# SOH Processing Pipeline Script
#
# This script runs the complete SOH processing pipeline:
# 1. generate_soh.py - Generate SOH from cleaned_data
# 2. preprocess_*.py - Process SOH data (smooth, fix anomalies)
#
# Usage:
#   ./run_soh_pipeline.sh --input INPUT_DIR [--soh SOH_DIR] [--output OUTPUT_DIR] [--python PYTHON_CMD] [--isu_ilcc_raw ISU_ILCC_RAW_DIR]
#
# Example:
#   ./run_soh_pipeline.sh --input /path/to/cleaned_data --output /path/to/processed_SOH
#
# Required:
#   --input   Path to cleaned_data directory (contains battery pkl files)
#

set -e

# Default paths
CALB_capacity_file_path="/data/trf/python_works/Trajectory_forecasting/process_scripts/overall_CALB_cycling_data.xlsx"
INPUT_DIR="/data/trf/python_works/BatteryLife/dataset"
SOH_DIR="/data/trf/python_works/BatteryLife/dataset/SOH"
OUTPUT_DIR="/data/trf/python_works/BatteryLife/dataset/processed_SOH"
PYTHON_CMD="python"
ISU_ILCC_RAW_DIR="/data/trf/raw_ISU" # path to raw ISU dataset. All original zip files should be unziped
TIME_NORMALIZE=true  # true to normalize time for BatteryLife verion v9 and before

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --soh)
            SOH_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --isu_ilcc_raw)
            ISU_ILCC_RAW_DIR="$2"
            shift 2
            ;;
        --time-normalize)
            TIME_NORMALIZE=true
            shift
            ;;
        -h|--help)
            echo "SOH Processing Pipeline"
            echo ""
            echo "Usage: $0 --input INPUT_DIR [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --input   Path to cleaned_data directory containing battery pkl files"
            echo ""
            echo "Optional:"
            echo "  --soh          Intermediate SOH directory (default: <PROJECT_ROOT>/dataset/SOH)"
            echo "  --output       Final output directory for processed_SOH (default: <PROJECT_ROOT>/dataset/processed_SOH)"
            echo "  --python       Python command to use (default: python)"
            echo "  --isu_ilcc_raw Path to ISU_ILCC raw data directory (for JSON to pkl conversion)"
            echo "                 If specified, will run convert_isu_ilcc_to_pkl.py first"
            echo "  --time-normalize  Run time normalization on all datasets after processing"
            echo ""
            echo "Example:"
            echo "  $0 --input /path/to/cleaned_data --output /path/to/processed_SOH"
            echo "  $0 --input ./data/cleaned_data --python 'conda run -n battery python'"
            echo "  $0 --input ./data/cleaned_data --isu_ilcc_raw ./data/ISU_ILCC"
            echo ""
            echo "Input directory structure (cleaned_data):"
            echo "  cleaned_data/"
            echo "    ├── CALB/"
            echo "    │   ├── CALB_0_B182.pkl"
            echo "    │   └── ..."
            echo "    ├── CALCE/"
            echo "    └── ..."
            echo ""
            echo "Output directory structure (processed_SOH):"
            echo "  processed_SOH/"
            echo "    ├── CALB/"
            echo "    │   ├── CALB_0_B182.pkl  (contains SOH trajectory)"
            echo "    │   └── ..."
            echo "    ├── CALCE/"
            echo "    └── ..."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check Python availability
if ! command -v ${PYTHON_CMD%% *} &> /dev/null; then
    echo "Error: Python command not found: $PYTHON_CMD"
    echo "Please specify a valid Python command with --python"
    exit 1
fi

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

# Create output directories
mkdir -p "$SOH_DIR"
mkdir -p "$OUTPUT_DIR"

# Step 0: Convert ISU_ILCC raw JSON to cleaned_data format (if specified)
if [[ -n "$ISU_ILCC_RAW_DIR" ]]; then
    echo ""
    echo "========================================"
    echo "Step 0: Converting ISU_ILCC JSON to cleaned_data"
    echo "========================================"
    if [[ ! -d "$ISU_ILCC_RAW_DIR" ]]; then
        echo "Warning: ISU_ILCC raw directory does not exist: $ISU_ILCC_RAW_DIR"
        echo "Skipping ISU_ILCC conversion..."
    else
        $PYTHON_CMD "$SCRIPT_DIR/convert_isu_ilcc_to_pkl.py" \
            --input_dir "$ISU_ILCC_RAW_DIR" \
            --output_dir "$INPUT_DIR/ISU_ILCC"
    fi
fi

# Step 1: Generate SOH from cleaned_data
echo ""
echo "========================================"
echo "Step 1: Generating SOH from cleaned_data"
echo "========================================"
$PYTHON_CMD "process_scripts/generate_soh.py" \
    --cleaned_data_root "$INPUT_DIR" \
    --output_root "$SOH_DIR"

$PYTHON_CMD "process_scripts/generate_CALB_soh.py" \
    --raw_data_file_path "$CALB_capacity_file_path" \
    --output_path "$SOH_DIR"

# Step 2: Run preprocess scripts
# These scripts have hardcoded paths, so we need to modify them temporarily

echo ""
echo "========================================"
echo "Step 2: Running preprocess scripts"
echo "========================================"

# Function to run a preprocess script with modified paths
run_preprocess() {
    local script_name=$1
    local dataset_name=$2
    local script_path="process_scripts/$script_name"

    if [[ ! -f "$script_path" ]]; then
        echo "Warning: Script not found: $script_path"
        return
    fi

    echo ""
    echo "----------------------------------------"
    echo "Running: $script_name"
    echo "----------------------------------------"

    # Create a temporary modified script
    local temp_script=$(mktemp)

    # Replace paths in the script
    # Handle both single dataset scripts (INPUT_DIR/OUTPUT_DIR) and general script (BASE_INPUT_DIR/BASE_OUTPUT_DIR)
    sed -e "s|INPUT_DIR = Path('.*')|INPUT_DIR = Path('$SOH_DIR/$dataset_name')|g" \
        -e "s|OUTPUT_DIR = Path('.*')|OUTPUT_DIR = Path('$OUTPUT_DIR/$dataset_name')|g" \
        -e "s|BASE_INPUT_DIR = Path('.*')|BASE_INPUT_DIR = Path('$SOH_DIR')|g" \
        -e "s|BASE_OUTPUT_DIR = Path('.*')|BASE_OUTPUT_DIR = Path('$OUTPUT_DIR')|g" \
        "$script_path" > "$temp_script"

    # Run the modified script
    $PYTHON_CMD "$temp_script"

    # Clean up
    rm -f "$temp_script"
}

# Run dataset-specific preprocess scripts first
run_preprocess "preprocess_HNEI.py" "HNEI"

# ISU_ILCC uses command line arguments (no temp script needed)
echo ""
echo "----------------------------------------"
echo "Running: preprocess_ISU_ILCC.py"
echo "----------------------------------------"
$PYTHON_CMD "$SCRIPT_DIR/preprocess_ISU_ILCC.py" \
    --input "$SOH_DIR/ISU_ILCC" \
    --output "$OUTPUT_DIR/ISU_ILCC" \
    --rpt_json "$SCRIPT_DIR/all_rpt_positions.json" \
    --workers 50

run_preprocess "preprocess_MICH.py" "MICH"
run_preprocess "preprocess_MICH_EXP.py" "MICH_EXP"
run_preprocess "preprocess_RWTH.py" "RWTH"
run_preprocess "preprocess_SNL.py" "SNL"
run_preprocess "preprocess_Tongji.py" "Tongji"

# Run general preprocess script (handles remaining datasets)
echo ""
echo "----------------------------------------"
echo "Running: preprocess.py (general)"
echo "----------------------------------------"

# Create a temporary modified script for the general preprocess
temp_script=$(mktemp)
sed -e "s|BASE_INPUT_DIR = Path('.*')|BASE_INPUT_DIR = Path('$SOH_DIR')|g" \
    -e "s|BASE_OUTPUT_DIR = Path('.*')|BASE_OUTPUT_DIR = Path('$OUTPUT_DIR')|g" \
    "process_scripts/preprocess.py" > "$temp_script"

$PYTHON_CMD "$temp_script"

rm -f "$temp_script"



echo ""
echo "========================================"
echo "Step 3: Merging MICH and MICH_EXP"
echo "========================================"

TARGET_MICH="$OUTPUT_DIR/total_MICH"
SRC_MICH="$OUTPUT_DIR/MICH"
SRC_MICH_EXP="$OUTPUT_DIR/MICH_EXP"

# 1. 创建 total_MICH 文件夹
if [ ! -d "$TARGET_MICH" ]; then
    echo "Creating directory: $TARGET_MICH"
    mkdir -p "$TARGET_MICH"
else
    echo "Directory exists: $TARGET_MICH"
fi

# 2. 拷贝 MICH 下的所有文件
if [ -d "$SRC_MICH" ]; then
    echo "Copying from $SRC_MICH to $TARGET_MICH..."
    # 使用 /. 确保拷贝的是文件夹下的内容，而不是文件夹本身
    cp -r "$SRC_MICH"/. "$TARGET_MICH/"
else
    echo "Warning: $SRC_MICH does not exist, skipping."
fi

# 3. 拷贝 MICH_EXP 下的所有文件 (同名文件会覆盖)
if [ -d "$SRC_MICH_EXP" ]; then
    echo "Copying from $SRC_MICH_EXP to $TARGET_MICH..."
    cp -r "$SRC_MICH_EXP"/. "$TARGET_MICH/"
else
    echo "Warning: $SRC_MICH_EXP does not exist, skipping."
fi

# Step 4: Time normalization (optional)
# Normalizes time_in_s in cleaned_data to cumulative seconds format
if [[ "$TIME_NORMALIZE" == "true" ]]; then
    echo ""
    echo "========================================"
    echo "Step 4: Time Normalization (cleaned_data)"
    echo "========================================"
    $PYTHON_CMD "$SCRIPT_DIR/time_normalization.py" \
        --input "$INPUT_DIR" \
        --workers 50
fi

# ========================================================
# Step 5: Rename Tongji files (New Addition)
# ========================================================
echo ""
echo "========================================"
echo "Step 5: Renaming Tongji files (-# to --)"
echo "========================================"

# 检查脚本是否存在
if [[ -f "$SCRIPT_DIR/rename_Tongji_cells.py" ]]; then
    $PYTHON_CMD "$SCRIPT_DIR/rename_Tongji_cells.py" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Warning: rename_Tongji_cells.py not found in $SCRIPT_DIR"
fi
echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Final output: $OUTPUT_DIR"
echo ""
