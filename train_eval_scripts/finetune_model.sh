#!/bin/bash

# 指定使用的GPU编号 (例如: "0" 或 "0,1")
gpu_ids=0
# 指定使用的GPU数量
num_process=1
# 主进程端口 (防止多任务冲突)
master_port=19520
seed=2021

pretrained_path="./checkpoints/CPTransformer_Li_ion_current_voltage"

# 微调目标数据集
finetune_dataset=CALB

finetune_lr=0.00001

# 微调轮数
finetune_epochs=50

# 批大小 (留空则使用预训练设置)
batch_size=128

# 早停耐心值
patience=20

# 权重衰减
weight_decay=0.01

# 梯度裁剪
grad_clip=1.0

freeze_mode=none

# 渐进解冻: N 轮后解冻所有参数 (0 = 不解冻)
unfreeze_after_epochs=0

# 学习率调度: constant / cosine / warmup_cosine
lradj=warmup_cosine

# Warmup 轮数 (lradj=warmup_cosine 时使用)
warmup_epochs=5

# ==========================================
# 6. Output Configuration
# ==========================================
# 输出目录 (留空则自动生成: pretrained_path + _FT_ + dataset)
# output_dir=""

# ==========================================
# 7. Execution Command
# ==========================================
echo "=========================================="
echo "Fine-tuning Configuration"
echo "=========================================="
echo "Pretrained model:  $pretrained_path"
echo "Target dataset:    $finetune_dataset"
echo "Learning rate:     $finetune_lr"
echo "Epochs:            $finetune_epochs"
echo "Batch size:        $batch_size"
echo "Freeze mode:       $freeze_mode"
echo "GPU:               $gpu_ids"
echo "=========================================="

CMD="CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch \
  --mixed_precision fp16 \
  --num_processes $num_process \
  --main_process_port $master_port \
  finetune_model.py \
  --pretrained_path $pretrained_path \
  --finetune_dataset $finetune_dataset \
  --finetune_lr $finetune_lr \
  --finetune_epochs $finetune_epochs \
  --batch_size $batch_size \
  --patience $patience \
  --weight_decay $weight_decay \
  --grad_clip $grad_clip \
  --freeze_mode $freeze_mode \
  --lradj $lradj \
  --warmup_epochs $warmup_epochs \
  --seed $seed \
  --use_amp"

if [ "$unfreeze_after_epochs" -gt 0 ]; then
  CMD="$CMD --unfreeze_after_epochs $unfreeze_after_epochs"
fi

# 执行
echo ""
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Fine-tuning completed!"
echo "Output: ${pretrained_path}_FT_${finetune_dataset}"
echo "=========================================="
