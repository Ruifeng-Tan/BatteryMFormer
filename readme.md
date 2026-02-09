# BatteryMFormer
This is the official repository for BatteryMFormer: Multi-level Learning for Battery Degradation Trajectory Forecasting. If you find this repository useful, we would appreciate citations to our paper and stars to this repository.

# Requirements


# Data availability 
Please directly download the .pkl files from [version v9 BatteryLife data](https://zenodo.org/records/17958489) for later data processing. The ISU_ILCC dataset is downloadable at [ISU_ILCC dataset](https://huggingface.co/datasets/JintaoDong256/BatteryMFormer_Stanford).


# Quick start
## Data preprocessing
After downloading the processed data from BatteryLife and the ISU_ILCC link. You can run the following command to preprocess the data:

```
bash process_scripts/run_soh_pipeline.sh
# Note that you should revise the args like `CALB_capacity_file_path` and `INPUT_DIR` in the script before running.
# The OUTPUT_DIR should be consistent with the processed_SOH_path in the training scripts
```

You can also download the processed data from [link](https://huggingface.co/datasets/JintaoDong256/BatteryMFormer) to skip this step.

## Training
You can run the following command to train the model:
```
sh train_eval_scripts/train_BattMFormer.sh
```

## Evaluation
You can run the following command to obtain the detailed results on the test set (such as model performance with increasing usable cycle number):
```
sh train_eval_scripts/evaluate_model.sh
```


