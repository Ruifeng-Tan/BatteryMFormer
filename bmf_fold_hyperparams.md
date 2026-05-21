# BatteryMFormer Per-Fold Hyperparameters

## Hyperparameter Search Range

Each row lists the distinct values that the corresponding hyperparameter took across all 22 BMF folds. These are the values that were actually selected by tuning, not the full grid that was searched.

CALB NAion seed 2021,贝叶斯优化的搜索范围，patch_len stride改成一致，trainepoch，patience

| Hyperparameter | Distinct values used | Values (sorted)                                               |
| -------------- | -------------------- | ------------------------------------------------------------- |
| learning_rate  | 8                    | 2.5e-05, 3e-05, 5e-05, 6e-05, 8e-05, 0.0001, 0.00012, 0.00015 |
| batch_size     | 2                    | 64, 128                                                       |
| train_epochs   | 4                    | 300, 500, 700, 950                                            |
| patience       | 5                    | 20, 30, 80, 120, 180                                          |
| dropout        | 8                    | 0.05, 0.092, 0.15, 0.2, 0.25, 0.28, 0.4, 0.5                  |
| d_model        | 3                    | 64, 128, 256                                                  |
| d_ff           | 3                    | 32, 64, 128                                                   |
| d_ffs          | 4                    | 32, 64, 128, 256                                              |
| k_dim          | 2                    | 512, 768                                                      |
| mem_dim        | 2                    | 128, 512                                                      |
| e_layers       | 2                    | 2, 4                                                          |
| d_layers       | 5                    | 2, 4, 6, 7, 8                                                 |
| num_query      | 6                    | 4, 8, 10, 12, 20, 50                                          |
| num_slots      | 2                    | 64, 96                                                        |
| patch_len      | 2                    | 16, 20                                                        |
| stride         | 3                    | 10, 20, 30                                                    |
| kernel_size    | 2                    | 10, 30                                                        |


## Per-Fold Hyperparameters

### Li_ion  (3 folds)


| Fold     | seed | Best epoch | Total epochs | learning_rate | batch_size | train_epochs | patience | dropout | d_model | d_ff | d_ffs | k_dim | mem_dim | e_layers | d_layers | num_query | num_slots | patch_len | stride | kernel_size | eol_threshold |
| -------- | ---- | ---------- | ------------ | ------------- | ---------- | ------------ | -------- | ------- | ------- | ---- | ----- | ----- | ------- | -------- | -------- | --------- | --------- | --------- | ------ | ----------- | ------------- |
| seed2021 | 2021 | 25         | 45           | 8e-05         | 128        | 300          | 20       | 0.28    | 256     | 128  | 128   | 512   | 128     | 4        | 7        | 10        | 96        | 16        | 10     | 10          | 0.8           |
| seed42   | 42   | 6          | 126          | 8e-05         | 128        | 700          | 120      | 0.05    | 128     | 64   | 128   | 512   | 128     | 4        | 4        | 12        | 96        | 20        | 10     | 10          | 0.8           |
| seed2024 | 2024 | 10         | 40           | 0.00015       | 128        | 300          | 30       | 0.2     | 128     | 128  | 128   | 512   | 128     | 2        | 4        | 8         | 64        | 16        | 10     | 10          | 0.8           |


### CALB  (4 folds)


| Fold    | seed | Best epoch | Total epochs | learning_rate | batch_size | train_epochs | patience | dropout | d_model | d_ff | d_ffs | k_dim | mem_dim | e_layers | d_layers | num_query | num_slots | patch_len | stride | kernel_size | eol_threshold |
| ------- | ---- | ---------- | ------------ | ------------- | ---------- | ------------ | -------- | ------- | ------- | ---- | ----- | ----- | ------- | -------- | -------- | --------- | --------- | --------- | ------ | ----------- | ------------- |
| cond396 | 99   | 57         | 237          | 2.5e-05       | 64         | 950          | 180      | 0.4     | 128     | 64   | 256   | 768   | 128     | 2        | 6        | 8         | 64        | 16        | 20     | 10          | 0.9           |
| cond397 | 2021 | 83         | 163          | 0.00012       | 64         | 500          | 80       | 0.15    | 128     | 128  | 256   | 768   | 128     | 2        | 6        | 8         | 64        | 16        | 10     | 10          | 0.9           |
| cond398 | 2021 | 10         | 40           | 0.00012       | 128        | 300          | 30       | 0.5     | 64      | 32   | 32    | 512   | 128     | 2        | 2        | 4         | 64        | 16        | 10     | 10          | 0.9           |
| cond399 | 42   | 95         | 125          | 6e-05         | 64         | 300          | 30       | 0.092   | 128     | 64   | 256   | 768   | 512     | 2        | 6        | 8         | 64        | 16        | 30     | 30          | 0.9           |


### NA-ion  (12 folds)


| Fold    | seed | Best epoch | Total epochs | learning_rate | batch_size | train_epochs | patience | dropout | d_model | d_ff | d_ffs | k_dim | mem_dim | e_layers | d_layers | num_query | num_slots | patch_len | stride | kernel_size | eol_threshold |
| ------- | ---- | ---------- | ------------ | ------------- | ---------- | ------------ | -------- | ------- | ------- | ---- | ----- | ----- | ------- | -------- | -------- | --------- | --------- | --------- | ------ | ----------- | ------------- |
| cond400 | 2021 | 144        | 174          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond401 | 2021 | 12         | 42           | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond402 | 2021 | 235        | 265          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond403 | 2021 | 138        | 168          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond404 | 2021 | 95         | 125          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond405 | 2021 | 65         | 95           | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond406 | 2021 | 37         | 67           | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond407 | 2021 | 297        | 300          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond408 | 2021 | 189        | 219          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond409 | 2021 | 154        | 184          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond410 | 2021 | 139        | 169          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| cond411 | 2021 | 154        | 184          | 5e-05         | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |


### Zn-ion  (3 folds)


| Fold     | seed | Best epoch | Total epochs | learning_rate | batch_size | train_epochs | patience | dropout | d_model | d_ff | d_ffs | k_dim | mem_dim | e_layers | d_layers | num_query | num_slots | patch_len | stride | kernel_size | eol_threshold |
| -------- | ---- | ---------- | ------------ | ------------- | ---------- | ------------ | -------- | ------- | ------- | ---- | ----- | ----- | ------- | -------- | -------- | --------- | --------- | --------- | ------ | ----------- | ------------- |
| seed2021 | 2021 | 90         | 120          | 0.0001        | 128        | 300          | 30       | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 16        | 10     | 10          | 0.8           |
| seed42   | 42   | 143        | 173          | 5e-05         | 128        | 300          | 30       | 0.5     | 64      | 128  | 64    | 512   | 128     | 4        | 6        | 50        | 64        | 16        | 10     | 10          | 0.8           |
| seed2024 | 2024 | 85         | 115          | 3e-05         | 128        | 300          | 30       | 0.05    | 64      | 32   | 64    | 512   | 128     | 4        | 8        | 20        | 64        | 16        | 10     | 10          | 0.8           |


## Constants (Identical Across All 22 Folds)


| Hyperparameter          | Value                                   |
| ----------------------- | --------------------------------------- |
| model                   | BatteryMFormer                          |
| lradj                   | constant                                |
| warmup_epochs           | 5                                       |
| weight_decay            | 0.0                                     |
| embedding_dropout       | 0.1                                     |
| head_dropout            | 0.0                                     |
| activation              | gelu                                    |
| d_llm                   | 1024                                    |
| n_heads                 | 8                                       |
| e_layers2               | 3                                       |
| top_k                   | 2                                       |
| temperature             | 1                                       |
| seq_len                 | 1                                       |
| label_len               | 48                                      |
| pred_len                | 5000                                    |
| cnn_channels            | 16                                      |
| lambda_recovery         | 100                                     |
| lambda_mem              | 10                                      |
| lambda_life_loss        | 0.0                                     |
| num_segments            | 50                                      |
| early_cycle_threshold   | 100                                     |
| charge_discharge_length | 300                                     |
| truncate_start_cycle    | 100                                     |
| input_mode              | current_voltage                         |
| cache_root              | /home/djt/Trajectory_forecasting/.cache |


