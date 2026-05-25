# BatteryMFormer (KDD 2026)

Official implementation of "BatteryMFormer: Multi-level Learning for Battery Degradation Trajectory Forecasting." If you find this repository useful, we would appreciate citations to our paper and stars to this repository.

## Requirements

```bash
pip install -r requirements.txt
```

The experiments were run with Python 3.9 and NVIDIA RTX 3090 GPUs.

## Data availability

BatteryMFormer uses the BatteryLife v11 processed data, available from Hugging Face:

- [Battery-Life/BatteryLife_Processed](https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed/tree/main)

Detailed download instructions are provided in [assets/Data_download.md](./assets/Data_download.md).

## Quick start

The default example below uses the Li-ion pure-OOD split `data_provider/split_json/pure_ood/Liion_split_seed2024.json`.

### 1. Generate SOH

If you use the BatteryLife v11 processed data, the generated SOH files are already included in `processed_SOH`.

To regenerate SOH from raw data, edit the paths in `process_scripts/run_soh_pipeline.sh` and run:

```bash
bash process_scripts/run_soh_pipeline.sh
```

### 2. Model training

Edit the path variables at the top of `train_eval_scripts/train_BattMFormer.sh`:

```bash
root_path=/path/to/BatteryLife_Processed_snapshot
processed_SOH_path=/path/to/BatteryLife_Processed_snapshot/processed_SOH
cache_root=/path/to/cache
```

Then train BatteryMFormer on Li-ion:

```bash
bash train_eval_scripts/train_BattMFormer.sh
```

Checkpoints are saved under `checkpoints/`.

### 3. Evaluation

Set `args_path` in `train_eval_scripts/evaluate_model.sh` to the checkpoint directory and set `target_dataset=Li_ion`, then run:

```bash
bash train_eval_scripts/evaluate_model.sh
```

Evaluation metrics are written to the checkpoint directory.

### 4. For other datasets and splits

To run another dataset or split, change `dataset` and `split_json_path` in `train_eval_scripts/train_BattMFormer.sh`, then use the matching `target_dataset` during evaluation.

| Dataset | `dataset` | Example split |
| --- | --- | --- |
| Li-ion | `Li_ion` | `data_provider/split_json/pure_ood/Liion_split_seed{2021,42,2024}.json` |
| Zn-ion | `ZN-coin` | `data_provider/split_json/pure_ood/ZNcoin_split_seed{2021,42,2024}.json` |
| CALB | `CALB` | `data_provider/split_json/loao/CALB_loao_cond{396..399}_seed2021.json` |
| NA-ion | `NA-ion` | `data_provider/split_json/loao/NA-ion_loao_cond{400..411}_seed2021.json` |

Per-fold hyperparameters used in our experiments are listed in [assets/per_fold_hyperparameters.md](./assets/per_fold_hyperparameters.md).


## Acknowledgements
This repo is constructed based on the following repos:
- https://github.com/Ruifeng-Tan/BatteryLife
- https://github.com/thuml/Time-Series-Library