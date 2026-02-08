import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
import random
import pickle
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin, load_checkpoint_in_model
from accelerate.utils import set_seed as accelerate_set_seed
from data_provider.data_factory import data_provider_soh
from utils.tools import EarlyStopping
import joblib
from models import PatchTST, iTransformer, DLinear, Autoformer, CPTransformer, CPMLP, BatteryMFormer
from models import TimeMixerPP, IC2ML, PatchMLP
from models import ConvTimeNet
import warnings
import wandb  # <--- 1. 导入 wandb

warnings.filterwarnings('ignore')

def print_detailed_parameters(model, accelerator):
    total_params = 0
    trainable_params = 0
    

    accelerator.print(f"{'Parameter':<50} {'Shape':<25} {'Size':<10} {'Trainable':<10}")
    accelerator.print("-" * 100)
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        param_shape = list(param.shape)
        
        total_params += param_size
        if param.requires_grad:
            trainable_params += param_size
        
        accelerator.print(f"{name:<50} {str(param_shape):<25} {param_size:<10,} {param.requires_grad:<10}")
    
    accelerator.print("-" * 100)
    accelerator.print(f"\nTotal parameters: {total_params:,}")
    accelerator.print(f"Trainable parameters: {trainable_params:,}")
    accelerator.print(f"Percentage of trainable parameters: {100 * trainable_params/total_params:.2f}%")

def set_seed(seed):
    """Set random seed for reproducibility"""
    accelerate_set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(args):
    """Get model based on model name"""
    model_dict = {
        'PatchTST': PatchTST,
        'iTransformer': iTransformer,
        'DLinear': DLinear,
        'Autoformer': Autoformer,
        'CPTransformer': CPTransformer,
        'CPMLP': CPMLP,
        'TimeMixerPP': TimeMixerPP,
        'PatchMLP': PatchMLP,
        'BatteryMFormer': BatteryMFormer,
        'IC2ML': IC2ML,
        'ConvTimeNet': ConvTimeNet,
    }

    if args.model not in model_dict:
        raise ValueError(f"Model {args.model} not supported")

    model = model_dict[args.model].Model(args)
    return model


def masked_mse_loss(pred, target, mask):
    """
    Calculate MSE loss only on valid positions (from cycle 101 to EOL)
    """
    # Only calculate loss on valid positions
    valid_pred = pred * mask
    valid_target = target * mask

    # Calculate squared error
    squared_error = (valid_pred - valid_target) ** 2

    # Average over valid positions only
    loss = squared_error.sum() / mask.sum()

    return loss


def get_eol_threshold_for_dataset(dataset_name):
    """Get EOL threshold based on dataset name. CALB uses 0.9, others use 0.8."""
    return 0.9 if dataset_name == 'CALB' else 0.8


def calculate_masked_metrics(pred, target, mask, eol_threshold=0.8):
    """
    Calculate MAPE, RMSE, MAE on masked regions

    Args:
        pred: predicted SOH values (normalized)
        target: target SOH values (normalized)
        mask: mask array
        eol_threshold: EOL threshold used for normalization (0.8 for most, 0.9 for CALB)
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Reverse normalization to real SOH values
    # Normalization was: (soh - eol_threshold) / (1.0 - eol_threshold)
    # So reverse is: soh = pred * (1.0 - eol_threshold) + eol_threshold
    pred_real = pred * (1.0 - eol_threshold) + eol_threshold
    target_real = target * (1.0 - eol_threshold) + eol_threshold

    # Get valid positions
    valid_mask = mask > 0
    if valid_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    valid_pred = pred_real[valid_mask]
    valid_target = target_real[valid_mask]

    # Calculate metrics on real SOH values
    mae = np.abs(valid_pred - valid_target).mean()
    mse = ((valid_pred - valid_target) ** 2).mean()
    rmse = np.sqrt(mse)

    # MAPE (only for targets > 0.01 to avoid division by zero)
    valid_for_mape = valid_target > 0.01
    if valid_for_mape.sum() > 0:
        mape = np.abs((valid_target[valid_for_mape] - valid_pred[valid_for_mape]) / valid_target[valid_for_mape]).mean() * 100
    else:
        mape = 0.0

    return mape, rmse, mae


def vali_model(loader, model, criterion, accelerator, args, print_progress=False):
    """
    Encapsulated validation/test function.
    Returns: avg_loss, mape, mae, rmse
    """
    model.eval()
    count = 0
    preds = []
    targets = []
    masks = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            cycle_curve_data, curve_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names = batch

            if args.input_mode == 'current_voltage':
                outputs = model(cycle_curve_data=cycle_curve_data, curve_attn_mask=curve_attn_mask, aging_condition_embedding=aging_condition_embedding, soh_trajectory=soh_trajectory, trajectory_mask=trajectory_mask, soc_input=soc_curves, soh_input=soh_input, cycle_level_features=cycle_level_features)
            elif args.input_mode == 'capacity_increment':
                soh, outputs, RUL = model(capacity_increment=cycle_curve_data, capacity_mask=curve_attn_mask)
            else:
                outputs = model(soh_input=soh_input)

            count += 1

            # Gather for metrics
            gathered_outputs, gathered_targets, gathered_masks = accelerator.gather_for_metrics(
                (outputs, soh_trajectory, trajectory_mask)
            )
            preds.append(gathered_outputs.detach().cpu().numpy())
            targets.append(gathered_targets.detach().cpu().numpy())
            masks.append(gathered_masks.detach().cpu().numpy())

            if print_progress and (batch_idx + 1) % 10 == 0:
                accelerator.print(f'  Processed {batch_idx + 1}/{len(loader)} batches')

    preds_np = np.concatenate(preds, axis=0)
    targets_np = np.concatenate(targets, axis=0)
    masks_np = np.concatenate(masks, axis=0)
    eol_threshold = get_eol_threshold_for_dataset(args.dataset)
    mape, rmse, mae = calculate_masked_metrics(preds_np, targets_np, masks_np, eol_threshold)
    return mape, mape, mae, rmse


def train(model, train_loader, val_loader, test_loader, optimizer, criterion, accelerator, args, scheduler=None):
    """Training function with Test evaluation included"""
    best_val_loss = float('inf')
    
    # History containers
    train_losses = []
    val_losses = []
    test_losses = []
    test_metrics_history = {'mape': [], 'rmse': [], 'mae': []}
    
    # 新增：记录最佳验证MAPE对应的指标
    best_val_mape = float('inf')
    best_epoch_info = {
        'epoch': 0,
        'test_mape': 0.0,
        'test_mae': 0.0,
        'test_rmse': 0.0,
        'val_mape': float('inf'),
        'val_mae': 0.0,
        'val_rmse': 0.0
    }
    
    # 新增：记录最后一个epoch的训练指标
    final_train_metrics = {
        'avg_train_loss': 0.0,
        'train_mape': 0.0,
        'train_mae': 0.0,
        'train_rmse': 0.0
    }
    
    checkpoint_path = args.checkpoints
    early_stopping = EarlyStopping(args, accelerator=accelerator, patience=args.patience, least_epochs=1)
    
    for epoch in range(args.train_epochs):
        # ===========================
        # 1. Training Phase
        # ===========================
        model.train()
        train_loss = 0
        train_count = 0
        train_preds = []
        train_targets = []
        train_masks = []

        for batch_idx, batch in enumerate(train_loader):
            cycle_curve_data, curve_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names = batch

            optimizer.zero_grad()

            if args.input_mode == 'current_voltage':
                outputs, recovery_loss, life_pred_loss, embedding_alignment_loss, aug_loss = model(cycle_curve_data=cycle_curve_data, curve_attn_mask=curve_attn_mask, aging_condition_embedding=aging_condition_embedding, soh_trajectory=soh_trajectory, trajectory_mask=trajectory_mask, soc_input=soc_curves, soh_input=soh_input, cycle_level_features=cycle_level_features, life_labels=life_labels)
            elif args.input_mode == 'capacity_increment':
                soh, outputs, RUL = model(capacity_increment=cycle_curve_data, capacity_mask=curve_attn_mask)
            else:
                outputs, recovery_loss, life_pred_loss, embedding_alignment_loss, aug_loss = model(soh_input=soh_input)

            if args.model in ['BatteryMFormer']:
                loss = criterion(outputs, soh_trajectory, trajectory_mask) + args.lambda_recovery * recovery_loss + args.lambda_mem * embedding_alignment_loss 
            else:
                loss = criterion(outputs, soh_trajectory, trajectory_mask)
            accelerator.backward(loss)

            # Apply gradient clipping to prevent NaN
            # if args.use_grad_clip > 0:
            #     accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            train_loss += loss.item()
            train_count += 1

            # Gather for metrics
            gathered_outputs, gathered_targets, gathered_masks = accelerator.gather_for_metrics(
                (outputs, soh_trajectory, trajectory_mask)
            )
            train_preds.append(gathered_outputs.detach().cpu().numpy())
            train_targets.append(gathered_targets.detach().cpu().numpy())
            train_masks.append(gathered_masks.detach().cpu().numpy())

            # Print progress occasionally
            print_freq = 1 if len(train_loader) < 10 else 100
            if batch_idx % print_freq == 0:
                if args.model in ['BatteryMFormer']:
                    accelerator.print(f'Epoch {epoch+1}/{args.train_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}, Embedding Alignment Loss: {embedding_alignment_loss.item():.6f}, Recovery Loss: {recovery_loss.item():.6f}')
                else:
                    accelerator.print(f'Epoch {epoch+1}/{args.train_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}')

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = train_loss / train_count
        train_losses.append(avg_train_loss)

        # Calculate Train Metrics
        eol_threshold = get_eol_threshold_for_dataset(args.dataset)
        if accelerator.is_main_process:
            train_preds_np = np.concatenate(train_preds, axis=0)
            train_targets_np = np.concatenate(train_targets, axis=0)
            train_masks_np = np.concatenate(train_masks, axis=0)
            train_mape, train_rmse, train_mae = calculate_masked_metrics(train_preds_np, train_targets_np, train_masks_np, eol_threshold)
        else:
            train_mape, train_rmse, train_mae = 0.0, 0.0, 0.0

        # ===========================
        # 2. Validation Phase
        # ===========================
        avg_val_loss, val_mape, val_mae, val_rmse = vali_model(
            val_loader, model, criterion, accelerator, args
        )
        val_losses.append(avg_val_loss)

        # ===========================
        # 3. Testing Phase
        # ===========================
        avg_test_loss, test_mape, test_mae, test_rmse = vali_model(
            test_loader, model, criterion, accelerator, args
        )
        test_losses.append(avg_test_loss)

        if accelerator.is_main_process:
            # Save metrics to history
            test_metrics_history['mape'].append(test_mape)
            test_metrics_history['rmse'].append(test_rmse)
            test_metrics_history['mae'].append(test_mae)
            
            # 新增：检查是否为最佳验证MAPE
            # 使用 <= 与 EarlyStopping 保持一致，确保记录的metrics与保存的模型对应
            if val_mape <= best_val_mape:
                best_val_mape = val_mape
                best_epoch_info.update({
                    'epoch': epoch + 1,
                    'test_mape': test_mape,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'val_mape': val_mape,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse
                })
            
            # 新增：更新最后一个epoch的训练指标
            final_train_metrics.update({
                'avg_train_loss': avg_train_loss,
                'train_mape': train_mape,
                'train_mae': train_mae,
                'train_rmse': train_rmse
            })

        # ===========================
        # 4. Logging & Saving
        # ===========================
        current_lr = optimizer.param_groups[0]['lr']
        accelerator.print(f'Epoch {epoch+1}/{args.train_epochs}:')
        if accelerator.is_main_process:
            accelerator.print(f'  Train -> Loss: {avg_train_loss:.6f}, MAPE: {train_mape:.2f}%, RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}')
            accelerator.print(f'  Val   -> Loss: {avg_val_loss:.6f}, MAPE: {val_mape:.2f}%, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}')
            accelerator.print(f'  Test  -> Loss: {avg_test_loss:.6f}, MAPE: {test_mape:.2f}%, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}')
            
            # WandB Log: 记录所有指标
            wandb.log({
                "epoch": epoch + 1,
                # Training Metrics
                "Train/Loss": avg_train_loss,
                "Train/MAE": train_mae,
                "Train/RMSE": train_rmse,
                "Train/MAPE": train_mape,
                # Validation Metrics
                "Val/MAE": val_mae,
                "Val/RMSE": val_rmse,
                "Val/MAPE": val_mape,
                # Test Metrics
                "Test/MAE": test_mae,
                "Test/RMSE": test_rmse,
                "Test/MAPE": test_mape,
                # Learning Rate
                "LR": current_lr
            })

        accelerator.print(f'  LR: {current_lr:.6e}')

        early_stopping(epoch+1, val_mape, val_mae, test_mae, model, checkpoint_path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            accelerator.set_trigger()
            
        if accelerator.check_trigger():
            break

    # ===========================
    # 5. 训练结束后记录总结指标
    # ===========================
    if accelerator.is_main_process:
        accelerator.print("\n" + "="*80)
        accelerator.print("TRAINING SUMMARY")
        accelerator.print("="*80)
        
        # 打印最佳验证MAPE对应的指标
        accelerator.print(f"\nBest Validation MAPE at Epoch {best_epoch_info['epoch']}:")
        accelerator.print(f"  Validation -> MAPE: {best_epoch_info['val_mape']:.4f}%, MAE: {best_epoch_info['val_mae']:.6f}, RMSE: {best_epoch_info['val_rmse']:.6f}")
        accelerator.print(f"  Test       -> MAPE: {best_epoch_info['test_mape']:.4f}%, MAE: {best_epoch_info['test_mae']:.6f}, RMSE: {best_epoch_info['test_rmse']:.6f}")
        
        # 打印最后一个epoch的训练指标
        accelerator.print(f"\nFinal Epoch (Epoch {epoch+1}) Training Metrics:")
        accelerator.print(f"  Loss: {final_train_metrics['avg_train_loss']:.6f}")
        accelerator.print(f"  MAPE: {final_train_metrics['train_mape']:.2f}%")
        accelerator.print(f"  MAE:  {final_train_metrics['train_mae']:.6f}")
        accelerator.print(f"  RMSE: {final_train_metrics['train_rmse']:.6f}")
        
        # 保存重要指标到wandb的summary
        wandb.run.summary["best_val_mape_epoch"] = best_epoch_info['epoch']
        wandb.run.summary["best_val_mape"] = best_epoch_info['val_mape']
        wandb.run.summary["best_val_mae_at_best_val"] = best_epoch_info['val_mae']
        wandb.run.summary["best_val_rmse_at_best_val"] = best_epoch_info['val_rmse']
        wandb.run.summary["best_test_mape_at_best_val"] = best_epoch_info['test_mape']
        wandb.run.summary["best_test_mae_at_best_val"] = best_epoch_info['test_mae']
        wandb.run.summary["best_test_rmse_at_best_val"] = best_epoch_info['test_rmse']
        wandb.run.summary["final_train_loss"] = final_train_metrics['avg_train_loss']
        
        accelerator.print("="*80)
        accelerator.print("Summary metrics have been logged to wandb")
        accelerator.print("="*80)
        accelerator.print(f'Checkpoint saved to {checkpoint_path}')
    return train_losses, val_losses, test_losses, test_metrics_history, epoch, best_epoch_info, final_train_metrics

def main():
    parser = argparse.ArgumentParser(description='SOH Trajectory Prediction')

    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (PatchTST, iTransformer, DLinear, Autoformer, CPTransformer, CPMLP, TimeMixer, WPMixer, BatteryDformer_*, MultiPatchFormer)')
    parser.add_argument('--checkpoints', type=str, required=True,
                        help='Path to save model checkpoints')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='MIX_large',
                        help='Dataset name')
    parser.add_argument('--root_path', type=str, default='/ai/dl_project/MemoryNet/dataset/cleaned_data',
                        help='Root path of dataset')
    parser.add_argument('--processed_SOH_path', type=str, default='/ai/dl_project/MemoryNet/dataset/Trajectory_forecasting/processed_SOH',
                        help='Root path of processed SOH trajectories')
    parser.add_argument('--input_mode', type=str, default='current_voltage',
                        choices=['current_voltage', 'soh_to_soh', 'capacity_increment'],
                        help='Input mode for model')

    # SOH truncation parameters
    parser.add_argument('--eol_threshold', type=float, default=0.8,
                        help='End of Life threshold (default: 0.8)')
    parser.add_argument('--truncate_start_cycle', type=int, default=100,
                        help='Start truncation from this cycle (default: 100)')

    # Model architecture parameters
    parser.add_argument('--kernel_size', type=int, default=10,
                        help='The kernel size for modeling SOC interval')
    parser.add_argument('--cnn_channels', type=int, default=16,
                        help='The number of channels in the CNN layers')
    parser.add_argument('--num_routed_experts', type=int, default=8,
                        help='The number of expert networks in one MoE layer')
    parser.add_argument('--num_shared_experts', type=int, default=2,
                        help='The number of shared expert networks in one MoE layer')
    parser.add_argument('--num_query', type=int, default=6,
                        help='The number of aging-condition queries')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Input sequence length for SOH-to-SOH mode')
    parser.add_argument('--num_slots', type=int, default=128,
                        help='Number of memory slots')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Top-k selection for memory query')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softmax in memory query')
    parser.add_argument('--pred_len', type=int, default=5000,
                        help='Prediction sequence length (max trajectory length, default: 5000 based on data statistics)')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimension of model')
    parser.add_argument('--d_llm', type=int, default=1024,
                        help='Embedding dimension of LLM')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--e_layers2', type=int, default=3,
                        help='Number of inter-cycle encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Dimension of feed-forward network')
    parser.add_argument('--d_ffs', type=int, default=256,
                        help='Dimension of feed-forward network in SOCPatchEncoder')
    parser.add_argument('--k_dim', type=int, default=4,
                        help='The number of the kept features for linear projection')
    parser.add_argument('--mem_dim', type=int, default=128,
                        help='The embedding dim of memory')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='Activation function')

    # Training parameters
    parser.add_argument('--train_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--lradj', type=str, default='constant',
                        choices=['constant', 'cosine', 'warmup_cosine'],
                        help='Learning rate adjustment')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs for warmup_cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--use_grad_clip', action='store_true', default=False,
                        help='Enable gradient clipping')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Gradient clipping value (0 means disabled)')
    parser.add_argument('--early_cycle_threshold', type=int, default=100,
                        help='Number of early cycles for input')
    parser.add_argument('--charge_discharge_length', type=int, default=100,
                        help='Length of charge/discharge curves')
    parser.add_argument('--num_segments', type=int, default=3,
                        help='Number of segments per period for hierarchical embedding')
    parser.add_argument('--use_capacity_resample', action='store_true',
                        help='Use capacity-based resampling for physical alignment')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loader')

    # Device and optimization parameters
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='Use multiple GPUs')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Random seed')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--lambda_recovery', type=float, default=0.1,
                        help='Recovery loss weight')
    parser.add_argument('--lambda_mem', type=float, default=0.1,
                        help='Memory loss weight')
    parser.add_argument('--lambda_life_loss', type=float, default=0.1,
                        help='Loss weight for life prediction')
    # Cache and resume parameters
    parser.add_argument('--cache_root', type=str, default='/ai/dl_project/MemoryNet/.cache',
                        help='Root directory for cache')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reload data from disk (ignore cache)')
    parser.add_argument('--resume_existing', action='store_true',
                        help='Resume training from existing checkpoint')

    # Additional model-specific parameters
    parser.add_argument('--factor', type=int, default=1,
                        help='Attention factor for sparse attention')
    parser.add_argument('--task_name', type=str, default='soh_forecast',
                        help='Task name')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='Moving average window size (for Autoformer)')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Label length (for Autoformer)')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch length (for PatchTST)')
    parser.add_argument('--stride', type=int, default=8,
                        help='Stride (for PatchTST and WPMixer)')
    # WPMixer specific parameters
    parser.add_argument('--tfactor', type=int, default=5,
                        help='Token mixer expansion factor (for WPMixer)')
    parser.add_argument('--dfactor', type=int, default=5,
                        help='Embedding mixer expansion factor (for WPMixer)')
    parser.add_argument('--wavelet', type=str, default='db2',
                        help='Wavelet name for decomposition (for WPMixer)')
    parser.add_argument('--level', type=int, default=1,
                        help='Wavelet decomposition level (for WPMixer)')
    parser.add_argument('--no_decomposition', action='store_true',
                        help='Disable wavelet decomposition (for WPMixer ablation)')
    parser.add_argument('--embedding_dropout', type=float, default=0.1,
                        help='Embedding dropout rate (for WPMixer)')
    parser.add_argument('--enc_in', type=int, default=3,
                        help='Encoder input size')

    # TimeMixer specific parameters
    parser.add_argument('--down_sampling_window', type=int, default=2,
                        help='Down sampling window size (for TimeMixer)')
    parser.add_argument('--down_sampling_layers', type=int, default=2,
                        help='Number of down sampling layers (for TimeMixer)')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        choices=['avg', 'max', 'conv'],
                        help='Down sampling method (for TimeMixer)')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='Channel independence flag: 0 or 1 (for TimeMixer)')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        choices=['moving_avg', 'dft_decomp'],
                        help='Decomposition method (for TimeMixer)')
    parser.add_argument('--use_norm', type=int, default=1,
                        help='Whether to use normalization: 0 or 1 (for TimeMixer)')

    # TimeMixer++ specific parameters
    parser.add_argument('--num_kernels', type=int, default=6,
                        help='Number of kernels in Inception block (for TimeMixer++)')
    parser.add_argument('--channel_mixing', action='store_true',
                        help='Enable channel mixing attention (for TimeMixer++)')

    # MultiPatchFormer specific parameters
    parser.add_argument('--use_cycle_encode', action='store_true',
                        help='Use CycleEncoder for memory-efficient input (for MultiPatchFormer)')

    # IC2ML specific parameters
    parser.add_argument('--horizon', type=int, default=5000, help='')
    parser.add_argument('--context', type=int, default=100, help='')

    # ConvTimeNet specific parameters
    parser.add_argument('--patch_ks', type=int, default=300,
                        help='Patch kernel size (for ConvTimeNet)')
    parser.add_argument('--patch_sd', type=int, default=300,
                        help='Patch stride (for ConvTimeNet)')
    parser.add_argument('--dw_ks', type=str, default='11,15,21,29,39,51',
                        help='Depthwise conv kernel sizes, comma-separated (for ConvTimeNet)')
    parser.add_argument('--padding_patch', type=str, default='end',
                        help='Patch padding mode (for ConvTimeNet)')
    parser.add_argument('--re_param', type=int, default=1,
                        help='Enable reparameterization: 0 or 1 (for ConvTimeNet)')
    parser.add_argument('--re_param_kernel', type=int, default=3,
                        help='Small kernel size for reparameterization (for ConvTimeNet)')
    parser.add_argument('--enable_res_param', type=int, default=1,
                        help='Enable learnable residual parameter: 0 or 1 (for ConvTimeNet)')
    parser.add_argument('--head_dropout', type=float, default=0.0,
                        help='Head dropout rate (for ConvTimeNet)')
    parser.add_argument('--revin', type=int, default=0,
                        help='Enable RevIN: 0 or 1 (for ConvTimeNet)')
    parser.add_argument('--affine', type=int, default=0,
                        help='RevIN affine: 0 or 1 (for ConvTimeNet)')
    parser.add_argument('--subtract_last', type=int, default=0,
                        help='RevIN subtract last: 0 or 1 (for ConvTimeNet)')
    parser.add_argument('--deformable', type=int, default=1,
                        help='Enable deformable patching: 0 or 1 (for ConvTimeNet)')

    # Few-shot learning parameters
    parser.add_argument('--fewshot_ratio', type=float, default=None,
                        help='Ratio of training data to use for few-shot learning (e.g., 0.5 for 50%)')

    args = parser.parse_args()


    # Set random seed
    set_seed(args.seed)

    # Create checkpoint directory
    os.makedirs(args.checkpoints, exist_ok=True)

    args_json_path = os.path.join(args.checkpoints, "args.json")
    with open(args_json_path, "w") as f:
        json.dump(args.__dict__, f)

    # ===== Create Accelerator for distributed training =====
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Check if DeepSpeed config exists
    deepspeed_config_path = './ds_config_zero2.json'
    if os.path.exists(deepspeed_config_path) and args.use_multi_gpu:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config_path)
        accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            deepspeed_plugin=deepspeed_plugin,
            gradient_accumulation_steps=args.accumulation_steps,
            mixed_precision='fp16' if args.use_amp else 'no'
        )
    else:
        # Standard DDP without DeepSpeed
        accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=args.accumulation_steps,
            mixed_precision='fp16' if args.use_amp else 'no'
        )

    # Print configuration
    accelerator.print(f'Using {accelerator.num_processes} GPU(s)')
    accelerator.print(f'Model: {args.model}')
    accelerator.print(f'Dataset: {args.dataset}')
    accelerator.print(f'Mixed Precision: {accelerator.mixed_precision}')

    # <--- 3. WandB Init: 初始化项目并上传 args (只在主进程执行)
    if accelerator.is_main_process:
        wandb.init(
            project="BattMFormer_paper",
            config=args.__dict__,  # 将 args 转为 dict 上传
            name=f"{args.model}_{args.dataset}"  # 给 run 起个名字，方便区分
        )
    # <--- End WandB Init

    # Get model
    accelerator.print(args)
    model = get_model(args)

    # Print model info (only on main process)
    if accelerator.is_main_process:
        print_detailed_parameters(model, accelerator)

    # Create data loaders (Load ALL loaders here)
    accelerator.print('Loading training data...')
    train_set, train_loader = data_provider_soh(args, 'train', args.input_mode)
    label_scaler = train_set.label_scaler

    accelerator.print('Loading validation data...')
    _, val_loader = data_provider_soh(args, 'val', args.input_mode, label_scaler)
    
    accelerator.print('Loading test data...')
    _, test_loader = data_provider_soh(args, 'test', args.input_mode, label_scaler)

    accelerator.wait_for_everyone()
    joblib.dump(label_scaler, f'{args.checkpoints}/label_scaler') # save the label scaler
    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = masked_mse_loss

    # Create learning rate scheduler
    scheduler = None
    if args.lradj == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-8)
        accelerator.print(f'Using CosineAnnealingLR scheduler')
    elif args.lradj == 'warmup_cosine':
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return float(epoch + 1) / float(args.warmup_epochs)
            progress = (epoch - args.warmup_epochs) / max(1, args.train_epochs - args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        accelerator.print(f'Using WarmupCosine scheduler with {args.warmup_epochs} warmup epochs')

    # ===== Prepare model, optimizer, dataloaders with Accelerator =====
    # Note: We now include test_loader in prepare
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # Train model
    accelerator.print('\nStarting training...')
    start_time = time.time()

    # Pass test_loader to train function
    train_losses, val_losses, test_losses, test_metrics_history, epoch, best_epoch_info, final_train_metrics = train(
        model, train_loader, val_loader, test_loader,
        optimizer, criterion, accelerator, args, scheduler=scheduler
    )

    end_time = time.time()

    accelerator.print(f'\nTraining completed in {end_time - start_time:.2f} seconds')

    # Save training history for hyperparameter search
    if accelerator.is_local_main_process:
        history = {
            'final_val_mape': best_epoch_info['val_mape'],
            'final_val_mae': best_epoch_info['val_mae'],
            'final_val_rmse': best_epoch_info['val_rmse'],
            'final_test_mape': best_epoch_info['test_mape'],
            'final_test_mae': best_epoch_info['test_mae'],
            'final_test_rmse': best_epoch_info['test_rmse'],
            'best_epoch': best_epoch_info['epoch'],
            'total_epochs': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
        }
        history_path = os.path.join(args.checkpoints, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        accelerator.print(f'Training history saved to {history_path}')

    accelerator.set_trigger()
    if accelerator.check_trigger() and accelerator.is_local_main_process:
        wandb.finish()
        # <--- End WandB Finish


if __name__ == '__main__':
    main()
