# Data Download

BatteryMFormer uses the BatteryLife v11 processed data:

- [Battery-Life/BatteryLife_Processed](https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed/tree/main)

## Download Processed Data

Install the Hugging Face CLI:

```bash
pip install --upgrade huggingface_hub
```

Download the v11 processed dataset:

```bash
huggingface-cli download --repo-type dataset \
  --token <YOUR_HF_TOKEN> \
  --revision v11 \
  --resume-download Battery-Life/BatteryLife_Processed \
  --cache-dir /path/to/your/dataset
```

The downloaded files are stored under:

```text
/path/to/your/dataset/datasets--Battery-Life--BatteryLife_Processed/snapshots/<commit_hash>/
```

Use this snapshot directory when configuring the training script:

```bash
root_path=/path/to/your/dataset/datasets--Battery-Life--BatteryLife_Processed/snapshots/<commit_hash>
processed_SOH_path=/path/to/your/dataset/datasets--Battery-Life--BatteryLife_Processed/snapshots/<commit_hash>/processed_SOH
```

## Hugging Face Token

Create an access token at:

```text
https://huggingface.co/settings/tokens
```

A read token is sufficient for downloading the public dataset.

## Raw Data

Raw Zn-ion, NA-ion, and CALB data are available from:

- [Battery-Life/BatteryLife_Raw](https://huggingface.co/datasets/Battery-Life/BatteryLife_Raw/tree/main)
- [Zenodo mirror](https://zenodo.org/records/17960956)

Li-ion raw datasets should be downloaded from their original dataset providers:

| Dataset | Source |
| --- | --- |
| CALCE | https://calce.umd.edu/battery-data |
| MATR | https://data.matr.io/ |
| HUST | https://data.mendeley.com/datasets/nsc7hnsg4s/2 |
| RWTH | https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip |
| ISU-ILCC | https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234 |
| XJTU | https://zenodo.org/records/10963339 |
| Tongji | https://zenodo.org/records/6405084 |
| Stanford | https://data.matr.io/8/ |
| HNEI / SNL / MICH / MICH_EXP / UL_PUR | https://www.batteryarchive.org/index.html |
| SDU | https://zenodo.org/records/14859405 |

Use raw data only if you want to regenerate SOH with `process_scripts/run_soh_pipeline.sh`.
