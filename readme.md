# BatteryMFormer

This is the official repository for "BatteryMFormer: Multi-level Learning for Battery Degradation Trajectory Forecasting." If you find this repository useful, we would appreciate citations to our paper and stars to this repository.

# Requirements
Please install the required packages listed in the `requirements.txt` file:
```
# Python 3.9.25
# RTX 3090
pip install -r requirements.txt
```

# Data availability 

The raw ISU_ILCC dataset is downloadable at [ISU_ILCC dataset](https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234). We rely on the some files in the raw ISU_ILCC data to do the preprocessing of the degradation trajectory. For other datasets, please directly download the .pkl files from [version v9 BatteryLife data](https://zenodo.org/records/17958489) for later data processing.

# Quick start

## Data preprocessing

After downloading BatteryLife (v9) and ISU_ILCC data, run:

```
bash process_scripts/run_soh_pipeline.sh
```
Note that you should revise the args like `CALB_capacity_file_path` and `INPUT_DIR` in the script before running. The OUTPUT_DIR should be consistent with the processed_SOH_path in the training scripts

To skip this step, you can also download the processed SOH data from [link](https://huggingface.co/datasets/JintaoDong256/BatteryMFormer), and processed cycling data from [link](https://huggingface.co/datasets/JintaoDong256/BatteryMFormer_cycling_data).

## Training

You can run the following command to train the model:
```
sh train_eval_scripts/train_BattMFormer.sh
Tip: set random seeds and hyperparameters consistently if you plan to reproduce results in our paper.
```

## Evaluation

To obtain detailed test-set results (e.g., performance vs. increasing usable cycle number):

```
sh train_eval_scripts/evaluate_model.sh
```

