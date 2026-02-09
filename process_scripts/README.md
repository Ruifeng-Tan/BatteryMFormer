# Data Processing Pipeline

## Quick Start

1. Open `run_soh_pipeline.sh` and modify the path configuration:

```bash
INPUT_DIR="/path/to/cleaned_data"
SOH_DIR="/path/to/SOH"
OUTPUT_DIR="/path/to/processed_SOH"
CALB_CAPACITY_FILE="/path/to/overall_CALB_cycling_data.xlsx"
ISU_ILCC_RAW_DIR="/path/to/raw_ISU"  # set to "" to skip ISU_ILCC raw-to-pkl conversion (Step 0)
```

Optional settings in `run_soh_pipeline.sh`:

| Variable | Default | Description |
|---|---|---|
| `TIME_NORMALIZE` | `true` | Set to `false` to skip time normalization (Step 4) |
| `ISU_ILCC_RAW_DIR` | `""` | Set to `""` to skip ISU_ILCC JSON-to-pkl conversion (Step 0) |
| `NUM_WORKERS` | `50` | Number of parallel workers |
| `PYTHON_CMD` | `python` | Python command (e.g. `conda run -n myenv python`) |

2. Run the pipeline:

```bash
bash process_scripts/run_soh_pipeline.sh
```

