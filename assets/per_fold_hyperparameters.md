# BatteryMFormer Per-Fold Hyperparameters

## Optimization Protocol

Per-fold hyperparameters are selected by Bayesian optimization with the Optuna
TPE sampler. For each fold we run 30 trials, train each trial on the training
split, evaluate it on the held-out validation split, and pick the trial with
the lowest validation MAPE as the reported configuration.

The training-epoch budget and the early-stopping patience are fixed at 300 and 30.

## Bayesian Optimization Search Space

Column names below use the code-side parameter names; the corresponding
symbols used in the paper are `d_model` → $d$, `d_ff` → $d_{\mathrm{ff}}$,
`d_ffs` → $d_{\mathrm{ffs}}$, `e_layers` → $L_{\mathrm{intra}}$,
`d_layers` → $L_{de}$, and `num_slots` → $N_{\mathrm{mem}}$.

| Hyperparameter | Range / Choices                        |
| -------------- | -------------------------------------- |
| learning_rate  | [2e-5, 2e-4]                           |
| batch_size     | {64, 128}                              |
| dropout        | [0.05, 0.5]                            |
| d_model        | {64, 128, 256}                         |
| d_ff           | {32, 64, 128}                          |
| d_ffs          | {32, 64, 128, 256}                     |
| k_dim          | {512, 768}                             |
| mem_dim        | {128, 512}                             |
| e_layers       | {2, 4}                                 |
| d_layers       | {2, 4, 6, 8}                           |
| num_query      | {4, 8, 10, 12, 20, 50}                 |
| num_slots      | {64, 96}                               |
| kernel_size    | {10, 16, 20, 30}  (stride = kernel)    |

## Per-Fold Hyperparameters

### Li_ion  (3 folds)

| Fold     | seed | Best epoch | Total epochs | learning_rate | batch_size | dropout | d_model | d_ff | d_ffs | k_dim | mem_dim | e_layers | d_layers | num_query | num_slots | kernel_size | eol_threshold |
| -------- | ---- | ---------- | ------------ | ------------- | ---------- | ------- | ------- | ---- | ----- | ----- | ------- | -------- | -------- | --------- | --------- | ----------- | ------------- |
| seed2021 | 2021 | 25         | 45           | 8e-05         | 128        | 0.28    | 256     | 128  | 128   | 512   | 128     | 4        | 7        | 10        | 96        | 10          | 0.8           |
| seed42   | 42   | 6          | 126          | 8e-05         | 128        | 0.05    | 128     | 64   | 128   | 512   | 128     | 4        | 4        | 12        | 96        | 10          | 0.8           |
| seed2024 | 2024 | 10         | 40           | 1.5e-04       | 128        | 0.2     | 128     | 128  | 128   | 512   | 128     | 2        | 4        | 8         | 64        | 10          | 0.8           |

### CALB  (4 folds)

| Fold    | seed | Best epoch | Total epochs | learning_rate | batch_size | dropout | d_model | d_ff | d_ffs | k_dim | mem_dim | e_layers | d_layers | num_query | num_slots | kernel_size | eol_threshold |
| ------- | ---- | ---------- | ------------ | ------------- | ---------- | ------- | ------- | ---- | ----- | ----- | ------- | -------- | -------- | --------- | --------- | ----------- | ------------- |
| cond396 | 2021 | 15         | 45           | 7.75e-05      | 64         | 0.18    | 256     | 64   | 64    | 768   | 512     | 2        | 8        | 12        | 96        | 30          | 0.9           |
| cond397 | 2021 | 153        | 183          | 1.98e-04      | 64         | 0.30    | 64      | 64   | 64    | 768   | 512     | 4        | 6        | 4         | 96        | 16          | 0.9           |
| cond398 | 2021 | 29         | 59           | 1.99e-04      | 64         | 0.31    | 256     | 32   | 256   | 768   | 128     | 4        | 8        | 8         | 96        | 10          | 0.9           |
| cond399 | 2021 | 132        | 162          | 1.99e-04      | 64         | 0.33    | 256     | 128  | 256   | 768   | 128     | 4        | 4        | 4         | 64        | 16          | 0.9           |

### NA-ion  (12 folds)

| Fold    | seed | Best epoch | Total epochs | learning_rate | batch_size | dropout | d_model | d_ff | d_ffs | k_dim | mem_dim | e_layers | d_layers | num_query | num_slots | kernel_size | eol_threshold |
| ------- | ---- | ---------- | ------------ | ------------- | ---------- | ------- | ------- | ---- | ----- | ----- | ------- | -------- | -------- | --------- | --------- | ----------- | ------------- |
| cond400 | 2021 | 144        | 174          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond401 | 2021 | 12         | 42           | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond402 | 2021 | 235        | 265          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond403 | 2021 | 138        | 168          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond404 | 2021 | 95         | 125          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond405 | 2021 | 65         | 95           | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond406 | 2021 | 37         | 67           | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond407 | 2021 | 297        | 300          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond408 | 2021 | 189        | 219          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond409 | 2021 | 154        | 184          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond410 | 2021 | 139        | 169          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| cond411 | 2021 | 154        | 184          | 5e-05         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |

### Zn-ion  (3 folds)

| Fold     | seed | Best epoch | Total epochs | learning_rate | batch_size | dropout | d_model | d_ff | d_ffs | k_dim | mem_dim | e_layers | d_layers | num_query | num_slots | kernel_size | eol_threshold |
| -------- | ---- | ---------- | ------------ | ------------- | ---------- | ------- | ------- | ---- | ----- | ----- | ------- | -------- | -------- | --------- | --------- | ----------- | ------------- |
| seed2021 | 2021 | 90         | 120          | 1e-04         | 128        | 0.25    | 128     | 128  | 256   | 512   | 128     | 2        | 6        | 4         | 64        | 10          | 0.8           |
| seed42   | 42   | 143        | 173          | 5e-05         | 128        | 0.5     | 64      | 128  | 64    | 512   | 128     | 4        | 6        | 50        | 64        | 10          | 0.8           |
| seed2024 | 2024 | 85         | 115          | 3e-05         | 128        | 0.05    | 64      | 32   | 64    | 512   | 128     | 4        | 8        | 20        | 64        | 10          | 0.8           |

## Constants (Identical Across All 22 Folds)

| Hyperparameter          | Value           |
| ----------------------- | --------------- |
| model                   | BatteryMFormer  |
| train_epochs            | 300             |
| patience                | 30              |
| lradj                   | constant        |
| warmup_epochs           | 5               |
| weight_decay            | 0.0             |
| embedding_dropout       | 0.1             |
| head_dropout            | 0.0             |
| activation              | gelu            |
| d_llm                   | 1024            |
| n_heads                 | 8               |
| e_layers2               | 3               |
| top_k                   | 2               |
| temperature             | 1               |
| seq_len                 | 1               |
| label_len               | 48              |
| pred_len                | 5000            |
| cnn_channels            | 16              |
| lambda_recovery         | 100             |
| lambda_mem              | 10              |
| lambda_life_loss        | 0.0             |
| num_segments            | 50              |
| early_cycle_threshold   | 100             |
| charge_discharge_length | 300             |
| truncate_start_cycle    | 100             |
| input_mode              | current_voltage |
