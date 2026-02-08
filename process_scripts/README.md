# SOH Data Processing Scripts

This folder contains scripts for processing battery SOH (State of Health) data.

## Processing Pipeline Overview

```
cleaned_data/  -->  generate_soh.py  -->  SOH/  -->  preprocess_*.py  -->  processed_SOH/
```

1. **generate_soh.py**: Calculate SOH trajectories from raw battery cycle data
2. **preprocess_*.py**: Smooth SOH data, detect and fix anomalies

## Quick Start

Run the complete pipeline with default paths:

```bash
cd /ai/dl_project/MemoryNet/process_scripts
./run_soh_pipeline.sh
```

Run with custom paths:

```bash
./run_soh_pipeline.sh \
    --input /path/to/cleaned_data \
    --soh /path/to/intermediate_soh \
    --output /path/to/processed_soh \
    --threads 100
```

## Scripts Description

### Main Pipeline Script

| Script | Description |
|--------|-------------|
| `run_soh_pipeline.sh` | Master script that runs the complete processing pipeline |

### SOH Generation

| Script | Description |
|--------|-------------|
| `generate_soh.py` | Generate SOH trajectories from cleaned battery data |

**Usage:**
```bash
/opt/conda/condabin/conda run -n batterylife python generate_soh.py \
    --cleaned_data_root /path/to/cleaned_data \
    --output_root /path/to/SOH \
    --num_threads 100
```

**SOH Calculation Formula:**
- **CALB dataset (EOL=0.9)**: `SOH = (current_capacity / first_cycle_capacity) / discharge_depth`
- **Other datasets (EOL=0.8)**: `SOH = (current_capacity / nominal_capacity) / discharge_depth`

**Filtering Rules:**
- CALB: Discard if `min(SOH) > 0.925`, extrapolate if `0.9 < min(SOH) <= 0.925`
- Others: Discard if `min(SOH) > 0.825`, extrapolate if `0.8 < min(SOH) <= 0.825`

### Preprocess Scripts

| Script | Dataset | Processing Logic |
|--------|---------|------------------|
| `preprocess.py` | CALB, CALCE, HUST, ISU_ILCC, MATR, NA-ion, Stanford, Stanford_2, UL_PUR, XJTU, ZN-coin | Zero detection + PCHIP interpolation; Large change (>10%) replacement |
| `preprocess_HNEI.py` | HNEI | Dataset-specific anomaly handling |
| `preprocess_MICH.py` | MICH | Dataset-specific anomaly handling |
| `preprocess_MICH_EXP.py` | MICH_EXP | Dataset-specific anomaly handling |
| `preprocess_RWTH.py` | RWTH | Dataset-specific anomaly handling |
| `preprocess_SNL.py` | SNL | Dataset-specific anomaly handling |
| `preprocess_Tongji.py` | Tongji | Simple replacement (>1% change) |

**Note:** `ZN-coin` dataset is copied directly without any processing.

## Directory Structure

```
/ai/dl_project/MemoryNet/dataset/
├── cleaned_data/          # Input: Raw battery cycle data
│   ├── CALB/
│   ├── CALCE/
│   └── ...
├── SOH/                   # Intermediate: Generated SOH trajectories
│   ├── CALB/
│   ├── CALCE/
│   └── ...
└── processed_SOH/         # Output: Processed SOH (smoothed, anomaly-corrected)
    ├── CALB/
    ├── CALCE/
    └── ...
```

## Output Data Format

Each processed SOH file (`.pkl`) contains:

```python
{
    'cell_id': str,                      # Battery identifier
    'cycle_numbers': list[int],          # Cycle numbers [1, 2, 3, ...]
    'SOH': list[float],                  # Processed SOH values
    'SOH_original': list[float],         # Original unprocessed SOH
    'cycle_start_time_in_s': list[float],# Start time of each cycle
    'anomalies': list[dict],             # Detected and corrected anomalies
    'processing_params': dict            # Processing parameters used
}
```
