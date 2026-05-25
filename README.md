# BatteryMFormer

Official implementation of "BatteryMFormer: Multi-level Learning for Battery Degradation Trajectory Forecasting." If you find this repository useful, please cite our paper and star the repo.

## Requirements

Tested with Python 3.9 on an RTX 3090. Install dependencies with:

```
pip install -r requirements.txt
```

## Data availability

We use the **BatteryLife v11 processed data**, hosted on Hugging Face:

- [Battery-Life/BatteryLife_Processed](https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed/tree/main)

Quick download steps (see the [BatteryLife data tutorial](https://github.com/Ruifeng-Tan/BatteryLife/blob/main/assets/Data_download.md) for the full walk-through):

1. Create a Hugging Face access token at https://huggingface.co/settings/tokens (read access to public datasets is enough).
2. Install / upgrade the CLI and download:

   ```
   pip install --upgrade huggingface_hub
   huggingface-cli download --repo-type dataset \
     --token <YOUR_HF_TOKEN> \
     --revision v11 \
     --resume-download Battery-Life/BatteryLife_Processed \
     --cache-dir /path/to/your/dataset
   ```

3. After download you should see the snapshot directory:

   ```
   /path/to/your/dataset/datasets--Battery-Life--BatteryLife_Processed/snapshots/<commit_hash>/
   ```

   Inside that snapshot directory there is a `processed_SOH` subfolder used during training. In the training scripts, set:

   - `root_path` = the snapshot directory above (contains per-dataset folders, e.g. `Li_ion`, `CALB`, `NA-ion`)
   - `processed_SOH_path` = `<snapshot>/processed_SOH`

If you would rather start from the raw datasets, see the optional [Data preprocessing](#data-preprocessing-optional) section at the bottom.

## Quick start: BMF on Li-ion

End-to-end run on the Li-ion pure-OOD split. Hyperparameters follow the Li_ion / seed2024 row in [`asset/per_fold_hyperparameters.md`](./asset/per_fold_hyperparameters.md).

### 1. Aging-condition embeddings (already shipped)

The Qwen3-Embedding aging-condition embeddings are already provided at `data_provider/prompt_embeddings/Qwen3_total.pkl`. **You can skip this step.**

To regenerate them yourself:

```
python download_llm.py                              # download Qwen3-Embedding-0.6B
jupyter nbconvert --execute generate_aging_condition_new_dataset.ipynb
```

### 2. Train

Edit the three path variables near the top of `train_eval_scripts/train_BattMFormer.sh`:

```bash
root_path=/path/to/your/dataset                 # processed data root (from the HF download)
processed_SOH_path=/path/to/your/processed_SOH  # the processed_SOH subfolder
cache_root=/path/to/your/cache                  # any writable directory for cached samples
```

Then launch:

```
bash train_eval_scripts/train_BattMFormer.sh
```

The script defaults to `dataset=Li_ion` with the pure-OOD split `pure_ood/Liion_split_seed2024.json`. Checkpoints land under `./checkpoints/BatteryMFormer_Li_ion_.../`.

### 3. Evaluate

Set `args_path` in `train_eval_scripts/evaluate_model.sh` to the checkpoint directory printed by the training run, then:

```
bash train_eval_scripts/evaluate_model.sh
```

Detailed per-cycle / seen-vs-unseen metrics are written back into the checkpoint directory.

## Running other datasets

The same scripts cover the other datasets — just change the variables at the top:

| Dataset | `dataset` | Example split json |
| --- | --- | --- |
| Li-ion | `Li_ion` | `pure_ood/Liion_split_seed{2021,42,2024}.json` |
| Zn-ion | `ZN-coin` | `pure_ood/ZNcoin_split_seed{2021,42,2024}.json` |
| CALB | `CALB` | `loao/CALB_loao_cond{396..399}_seed2021.json` |
| NA-ion | `NA-ion` | `loao/NA-ion_loao_cond{400..411}_seed2021.json` |

Per-fold best hyperparameters (learning rate, `d_model`, `d_layers`, …) are listed in [`asset/per_fold_hyperparameters.md`](./asset/per_fold_hyperparameters.md).

## Data preprocessing (optional)

If you would rather start from the raw datasets and reproduce the processed files yourself, follow this section.

### Download raw data

The Zn-ion, Na-ion, and CALB raw data are hosted on Hugging Face / Zenodo by the BatteryLife project. All other raw datasets are released by their original authors.

- **Zn-ion / Na-ion / CALB** — [Battery-Life/BatteryLife_Raw](https://huggingface.co/datasets/Battery-Life/BatteryLife_Raw/tree/main) (also mirrored on [Zenodo](https://zenodo.org/records/17960956))

  ```
  huggingface-cli download --repo-type dataset \
    --token <YOUR_HF_TOKEN> \
    --resume-download Battery-Life/BatteryLife_Raw \
    --cache-dir /path/to/your/raw_data
  ```

- **Li-ion raw datasets** — download from the original sources:

  | Dataset | Link |
  | --- | --- |
  | CALCE | https://calce.umd.edu/battery-data |
  | MATR | [batches 1-3](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) / [batch 9](https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94) |
  | HUST | https://data.mendeley.com/datasets/nsc7hnsg4s/2 |
  | RWTH | https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip |
  | ISU-ILCC | https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234 |
  | XJTU | https://zenodo.org/records/10963339 |
  | Tongji | https://zenodo.org/records/6405084 |
  | Stanford | https://data.matr.io/8/ |
  | HNEI / SNL / MICH / MICH_EXP / UL_PUR | https://www.batteryarchive.org/index.html |
  | SDU | https://zenodo.org/records/14859405 |

After downloading, organize them in the BatteryLife layout (one folder per dataset, each containing per-cell `.pkl` files). See the [BatteryLife data structure description](https://github.com/Ruifeng-Tan/BatteryLife/blob/main/assets/Data_structure_description.md) for details.

### Run the SOH pipeline

The ISU-ILCC conversion step uses [microsoft/BatteryML](https://github.com/microsoft/BatteryML). Install it once if you plan to process the Li-ion datasets:

```
git clone https://github.com/microsoft/BatteryML.git
pip install -e BatteryML
```

Edit the path variables at the top of `process_scripts/run_soh_pipeline.sh`:

```bash
INPUT_DIR=/path/to/your/cleaned_data      # raw per-dataset pkl folders (BatteryLife layout)
SOH_DIR=/path/to/your/dataset/SOH         # intermediate SOH output
OUTPUT_DIR=/path/to/your/processed_SOH    # final processed output (used as processed_SOH_path)
ISU_ILCC_RAW_DIR=/path/to/your/raw_ISU    # leave empty to skip ISU-ILCC conversion
```

Then run:

```
bash process_scripts/run_soh_pipeline.sh
```

The resulting `OUTPUT_DIR` should be passed as `processed_SOH_path` to the training scripts.
