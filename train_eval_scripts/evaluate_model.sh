args_path=/data/trf/python_works/Trajectory_forecasting/backup/model_result/BatteryMFormer_Li_ion_te300_ks10_st10_ch16_dff128_dffs128_dm128_kdm512_el2_dl4_bs128_dr0.5_nh8_lr0.00005_nq10_100.0_64_clipFalse_seed2024
# args_path=/data/trf/checkpoints/BatteryMFormer_final_checkpoints/BatteryMFormer_seed2021_dffs128_dl6/
target_dataset=CALB
# 指定使用的GPU编号 (例如: "0" 或 "0,1,2,3")
gpu_ids=0,1
# 指定使用的GPU数量 (必须与 gpu_ids 的数量一致)
num_process=2
# 主进程端口 (防止多任务冲突)
master_port=29100


CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch \
  --mixed_precision fp16 \
  --num_processes $num_process \
  --main_process_port $master_port \
  evaluate_model.py \
  --args_path $args_path \
  --target_dataset $target_dataset
