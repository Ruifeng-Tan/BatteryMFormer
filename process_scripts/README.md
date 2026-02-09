# Data Processing Pipeline

## Quick Start

1. Open `run_soh_pipeline.sh` and modify the path configuration:

```bash
INPUT_DIR="/path/to/cleaned_data"
SOH_DIR="/path/to/SOH"
OUTPUT_DIR="/path/to/processed_SOH"
CALB_CAPACITY_FILE="/path/to/overall_CALB_cycling_data.xlsx"
ISU_ILCC_RAW_DIR="/path/to/raw_ISU"  # set to "" to skip
```

2. Run the pipeline:

```bash
bash process_scripts/run_soh_pipeline.sh
```

