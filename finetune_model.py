"""
Fine-tuning script for pretrained models.

Usage:
    python finetune_model.py \
        --pretrained_path ./checkpoints/BatteryMFormer_Li_ion \
        --finetune_dataset CALB \
        --finetune_lr 0.00001 \
        --finetune_epochs 50
"""

import argparse
import torch
import torch.optim as optim
import numpy as np
import os
import time
import json
import random
from accelerate import Accelerator, DistributedDataParallelKwargs, load_checkpoint_in_model
from accelerate.utils import set_seed as accelerate_set_seed
from data_provider.data_factory import data_provider_soh
from utils.tools import EarlyStopping
import joblib
from models import PatchTST, iTransformer, DLinear, CPTransformer, CPMLP
from models import TimeMixerPP, IC2ML, PatchMLP
from models import BatteryMFormer, ConvTimeNet
import warnings

warnings.filterwarnings('ignore')


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
    valid_pred = pred * mask
    valid_target = target * mask
    squared_error = (valid_pred - valid_target) ** 2
    loss = squared_error.sum() / mask.sum()
    return loss


def get_eol_threshold(dataset_name):
    """Get EOL threshold based on dataset name (CALB=0.9, others=0.8)"""
    return 0.9 if dataset_name == 'CALB' else 0.8


def calculate_masked_metrics(pred, target, mask, eol_threshold=0.8):
    """
    Calculate MAPE, RMSE, MAE on masked regions

    Args:
        pred: predictions (normalized)
        target: targets (normalized)
        mask: mask for valid positions
        eol_threshold: EOL threshold for denormalization (CALB=0.9, others=0.8)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Reverse normalization to real SOH values
    pred_real = pred * (1.0 - eol_threshold) + eol_threshold
    target_real = target * (1.0 - eol_threshold) + eol_threshold

    valid_mask = mask > 0
    if valid_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    valid_pred = pred_real[valid_mask]
    valid_target = target_real[valid_mask]

    mae = np.abs(valid_pred - valid_target).mean()
    mse = ((valid_pred - valid_target) ** 2).mean()
    rmse = np.sqrt(mse)

    valid_for_mape = valid_target > 0.01
    if valid_for_mape.sum() > 0:
        mape = np.abs((valid_target[valid_for_mape] - valid_pred[valid_for_mape]) / valid_target[valid_for_mape]).mean() * 100
    else:
        mape = 0.0

    return mape, rmse, mae


def print_parameter_stats(model, accelerator):
    """Print parameter statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Parameter Statistics:")
    accelerator.print(f"  Total parameters:     {total_params:,}")
    accelerator.print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    accelerator.print(f"  Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    accelerator.print(f"{'='*60}\n")


def vali_model(loader, model, criterion, accelerator, args):
    """
    Validation/test function.
    Returns: avg_loss, mape, mae, rmse
    """
    model.eval()
    preds = []
    targets = []
    masks = []

    with torch.no_grad():
        for batch in loader:
            cycle_curve_data, curve_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names = batch

            if args.input_mode == 'current_voltage':
                outputs = model(cycle_curve_data=cycle_curve_data, curve_attn_mask=curve_attn_mask,
                              aging_condition_embedding=aging_condition_embedding, soh_trajectory=soh_trajectory,
                              trajectory_mask=trajectory_mask, soc_input=soc_curves, soh_input=soh_input,
                              cycle_level_features=cycle_level_features)
            elif args.input_mode == 'capacity_increment':
                outputs = model(capacity_increment=cycle_curve_data, capacity_mask=curve_attn_mask, tgt_soh=soh_trajectory)
                # IC2ML returns (soh, trajectory, RUL), take trajectory at index 1
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    outputs = outputs[1]
            else:
                outputs = model(soh_input=soh_input)

            # Handle model outputs (some models return tuples)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            gathered_outputs, gathered_targets, gathered_masks = accelerator.gather_for_metrics(
                (outputs, soh_trajectory, trajectory_mask)
            )
            preds.append(gathered_outputs.detach().cpu().numpy())
            targets.append(gathered_targets.detach().cpu().numpy())
            masks.append(gathered_masks.detach().cpu().numpy())

    preds_np = np.concatenate(preds, axis=0)
    targets_np = np.concatenate(targets, axis=0)
    masks_np = np.concatenate(masks, axis=0)
    eol_threshold = get_eol_threshold(args.dataset)
    mape, rmse, mae = calculate_masked_metrics(preds_np, targets_np, masks_np, eol_threshold)
    return mape, mape, mae, rmse


def finetune_train(model, train_loader, val_loader, test_loader, optimizer, criterion,
                   accelerator, args, scheduler=None):
    """Fine-tuning training loop"""
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

    checkpoint_path = args.checkpoints
    early_stopping = EarlyStopping(args, accelerator=accelerator, patience=args.patience, least_epochs=1)

    for epoch in range(args.train_epochs):
        # Training phase
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
                outputs = model(cycle_curve_data=cycle_curve_data, curve_attn_mask=curve_attn_mask,
                              aging_condition_embedding=aging_condition_embedding, soh_trajectory=soh_trajectory,
                              trajectory_mask=trajectory_mask, soc_input=soc_curves, soh_input=soh_input,
                              cycle_level_features=cycle_level_features, life_labels=life_labels)
            elif args.input_mode == 'capacity_increment':
                # IC2ML model returns (soh, trajectory, RUL), we need trajectory (index 1)
                outputs = model(capacity_increment=cycle_curve_data, capacity_mask=curve_attn_mask, tgt_soh=soh_trajectory)
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    main_output = outputs[1]  # trajectory is at index 1
                    loss = criterion(main_output, soh_trajectory, trajectory_mask)
                    outputs = main_output
                else:
                    loss = criterion(outputs, soh_trajectory, trajectory_mask)
            else:
                outputs = model(soh_input=soh_input)

            # Handle model outputs (skip for capacity_increment which is already handled)
            if args.input_mode != 'capacity_increment':
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                    # Some models return additional losses
                    if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor) and outputs[1].dim() == 0:
                        aux_loss = sum(o for o in outputs[1:] if isinstance(o, torch.Tensor) and o.dim() == 0)
                        loss = criterion(main_output, soh_trajectory, trajectory_mask) + 0.1 * aux_loss
                    else:
                        loss = criterion(main_output, soh_trajectory, trajectory_mask)
                    outputs = main_output
                else:
                    loss = criterion(outputs, soh_trajectory, trajectory_mask)

            accelerator.backward(loss)

            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            train_loss += loss.item()
            train_count += 1

            gathered_outputs, gathered_targets, gathered_masks = accelerator.gather_for_metrics(
                (outputs, soh_trajectory, trajectory_mask)
            )
            train_preds.append(gathered_outputs.detach().cpu().numpy())
            train_targets.append(gathered_targets.detach().cpu().numpy())
            train_masks.append(gathered_masks.detach().cpu().numpy())

            # Print progress
            print_freq = max(1, len(train_loader) // 10)
            if batch_idx % print_freq == 0:
                accelerator.print(f'Epoch {epoch+1}/{args.train_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}')

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = train_loss / train_count

        # Calculate train metrics
        if accelerator.is_main_process:
            train_preds_np = np.concatenate(train_preds, axis=0)
            train_targets_np = np.concatenate(train_targets, axis=0)
            train_masks_np = np.concatenate(train_masks, axis=0)
            eol_threshold = get_eol_threshold(args.dataset)
            train_mape, train_rmse, train_mae = calculate_masked_metrics(train_preds_np, train_targets_np, train_masks_np, eol_threshold)
        else:
            train_mape, train_rmse, train_mae = 0.0, 0.0, 0.0

        # Validation
        _, val_mape, val_mae, val_rmse = vali_model(val_loader, model, criterion, accelerator, args)

        # Test
        _, test_mape, test_mae, test_rmse = vali_model(test_loader, model, criterion, accelerator, args)

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        accelerator.print(f'\nEpoch {epoch+1}/{args.train_epochs}:')
        if accelerator.is_main_process:
            accelerator.print(f'  Train -> Loss: {avg_train_loss:.6f}, MAPE: {train_mape:.2f}%, RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}')
        accelerator.print(f'  Val   -> MAPE: {val_mape:.2f}%, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}')
        accelerator.print(f'  Test  -> MAPE: {test_mape:.2f}%, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}')
        accelerator.print(f'  LR: {current_lr:.6e}')

        # Track best model
        if accelerator.is_main_process:
            if val_mape < best_val_mape:
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

        # Early stopping
        early_stopping(epoch+1, val_mape, val_mae, test_mae, model, checkpoint_path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping triggered!")
            accelerator.set_trigger()

        if accelerator.check_trigger():
            break

    # Print summary
    if accelerator.is_main_process:
        accelerator.print("\n" + "="*80)
        accelerator.print("FINE-TUNING SUMMARY")
        accelerator.print("="*80)
        accelerator.print(f"\nBest Validation MAPE at Epoch {best_epoch_info['epoch']}:")
        accelerator.print(f"  Validation -> MAPE: {best_epoch_info['val_mape']:.4f}%, MAE: {best_epoch_info['val_mae']:.6f}, RMSE: {best_epoch_info['val_rmse']:.6f}")
        accelerator.print(f"  Test       -> MAPE: {best_epoch_info['test_mape']:.4f}%, MAE: {best_epoch_info['test_mae']:.6f}, RMSE: {best_epoch_info['test_rmse']:.6f}")
        accelerator.print("="*80)

    return best_epoch_info


def load_pretrained_args(pretrained_path):
    """Load args.json from pretrained model directory"""
    args_path = os.path.join(pretrained_path, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {pretrained_path}")

    with open(args_path, 'r') as f:
        args_dict = json.load(f)

    return argparse.Namespace(**args_dict)


def merge_args(pretrained_args, finetune_args):
    """
    Merge pretrained args with finetune args.
    Finetune args take precedence when specified.
    """
    merged = argparse.Namespace(**vars(pretrained_args))

    # Override dataset
    merged.dataset = finetune_args.finetune_dataset

    # Learning rate handling
    if finetune_args.finetune_lr is not None:
        merged.learning_rate = finetune_args.finetune_lr
    else:
        # Default: use 1/10 of pretrained learning rate
        merged.learning_rate = pretrained_args.learning_rate * finetune_args.lr_decay_ratio

    # Override other parameters
    if finetune_args.batch_size is not None:
        merged.batch_size = finetune_args.batch_size

    merged.train_epochs = finetune_args.finetune_epochs
    merged.patience = finetune_args.patience
    merged.lradj = finetune_args.lradj
    merged.warmup_epochs = finetune_args.warmup_epochs
    merged.seed = finetune_args.seed
    merged.grad_clip = finetune_args.grad_clip
    merged.use_amp = finetune_args.use_amp
    merged.weight_decay = finetune_args.weight_decay

    # Data paths (optional override)
    if finetune_args.root_path is not None:
        merged.root_path = finetune_args.root_path
    if finetune_args.processed_SOH_path is not None:
        merged.processed_SOH_path = finetune_args.processed_SOH_path

    # Store pretrained path for reference
    merged.pretrained_path = finetune_args.pretrained_path
    merged.finetune_dataset = finetune_args.finetune_dataset

    return merged


def main():
    parser = argparse.ArgumentParser(description='Fine-tune MemoryNet Models')

    # Required arguments
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pretrained model directory (containing args.json, checkpoint, label_scaler)')
    parser.add_argument('--finetune_dataset', type=str, required=True,
                        help='Target dataset for fine-tuning (e.g., CALB, CALCE, HUST)')

    # Fine-tuning hyperparameters
    parser.add_argument('--finetune_lr', type=float, default=None,
                        help='Fine-tuning learning rate (default: 1/10 of pretrained lr)')
    parser.add_argument('--finetune_epochs', type=int, default=200,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (default: use pretrained setting)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')

    # Learning rate scheduling
    parser.add_argument('--lradj', type=str, default='warmup_cosine',
                        choices=['constant', 'cosine', 'warmup_cosine'],
                        help='Learning rate adjustment strategy')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs for warmup_cosine scheduler')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1,
                        help='Learning rate multiplier compared to pretraining (default: 0.1)')

    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: pretrained_path + "_FT_" + finetune_dataset)')

    # System parameters
    parser.add_argument('--seed', type=int, default=2021,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')

    # Data paths (optional override)
    parser.add_argument('--root_path', type=str, default=None,
                        help='Override root path for data')
    parser.add_argument('--processed_SOH_path', type=str, default=None,
                        help='Override processed SOH path')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load pretrained args
    print(f"Loading pretrained configuration from: {args.pretrained_path}")
    pretrained_args = load_pretrained_args(args.pretrained_path)

    # Normalize legacy model names to current names
    _MODEL_ALIASES = {
        'BatteryMFormer_Trial2': 'BatteryMFormer',
    }
    if pretrained_args.model in _MODEL_ALIASES:
        pretrained_args.model = _MODEL_ALIASES[pretrained_args.model]

    # Merge args
    merged_args = merge_args(pretrained_args, args)

    # Set output directory with _FT suffix
    if args.output_dir is None:
        base_name = os.path.basename(args.pretrained_path.rstrip('/'))
        merged_args.checkpoints = f"./checkpoints/{base_name}_FT_{args.finetune_dataset}"
    else:
        merged_args.checkpoints = args.output_dir

    # Ensure we don't overwrite pretrained weights
    if os.path.abspath(merged_args.checkpoints) == os.path.abspath(args.pretrained_path):
        raise ValueError("Output directory cannot be the same as pretrained path! This would overwrite the pretrained weights.")

    # Create output directory
    os.makedirs(merged_args.checkpoints, exist_ok=True)

    # Create Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision='fp16' if merged_args.use_amp else 'no'
    )

    # Print configuration
    accelerator.print("="*80)
    accelerator.print("FINE-TUNING CONFIGURATION")
    accelerator.print("="*80)
    accelerator.print(f"Pretrained model: {args.pretrained_path}")
    accelerator.print(f"Target dataset: {args.finetune_dataset}")
    accelerator.print(f"Output directory: {merged_args.checkpoints}")
    accelerator.print(f"Model: {merged_args.model}")
    accelerator.print(f"Input mode: {merged_args.input_mode}")
    accelerator.print(f"Learning rate: {merged_args.learning_rate:.2e}")
    accelerator.print(f"Epochs: {merged_args.train_epochs}")
    accelerator.print(f"Batch size: {merged_args.batch_size}")
    accelerator.print(f"Mixed Precision: {accelerator.mixed_precision}")
    accelerator.print("="*80)

    # Get model
    model = get_model(merged_args)

    # Load pretrained weights
    accelerator.print(f"\nLoading pretrained weights from: {args.pretrained_path}")
    load_checkpoint_in_model(model, args.pretrained_path)
    accelerator.print("Pretrained weights loaded successfully!")

    print_parameter_stats(model, accelerator)

    # Load label_scaler from pretrained
    label_scaler_path = os.path.join(args.pretrained_path, 'label_scaler')
    if os.path.exists(label_scaler_path):
        label_scaler = joblib.load(label_scaler_path)
        accelerator.print(f"Loaded label_scaler from pretrained model")
    else:
        label_scaler = None
        accelerator.print("Warning: label_scaler not found in pretrained directory, will create new one")

    # Load data for fine-tuning dataset
    accelerator.print(f'\nLoading data for fine-tuning dataset: {merged_args.dataset}')

    accelerator.print('Loading training data...')
    train_set, train_loader = data_provider_soh(merged_args, 'train', merged_args.input_mode, label_scaler)
    if label_scaler is None:
        label_scaler = train_set.label_scaler

    accelerator.print('Loading validation data...')
    _, val_loader = data_provider_soh(merged_args, 'val', merged_args.input_mode, label_scaler)

    accelerator.print('Loading test data...')
    _, test_loader = data_provider_soh(merged_args, 'test', merged_args.input_mode, label_scaler)

    # Save label_scaler to output directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        joblib.dump(label_scaler, os.path.join(merged_args.checkpoints, 'label_scaler'))

        # Save merged args
        with open(os.path.join(merged_args.checkpoints, 'args.json'), 'w') as f:
            json.dump(vars(merged_args), f, indent=2)

        # Save finetune info
        finetune_info = {
            'pretrained_path': args.pretrained_path,
            'finetune_dataset': args.finetune_dataset,
            'finetune_lr': merged_args.learning_rate,
            'finetune_epochs': merged_args.train_epochs
        }
        with open(os.path.join(merged_args.checkpoints, 'finetune_info.json'), 'w') as f:
            json.dump(finetune_info, f, indent=2)

    # Setup optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=merged_args.learning_rate, weight_decay=merged_args.weight_decay)
    criterion = masked_mse_loss

    # Create learning rate scheduler
    scheduler = None
    if merged_args.lradj == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=merged_args.train_epochs, eta_min=1e-8)
        accelerator.print(f'Using CosineAnnealingLR scheduler')
    elif merged_args.lradj == 'warmup_cosine':
        def lr_lambda(epoch):
            if epoch < merged_args.warmup_epochs:
                return float(epoch + 1) / float(merged_args.warmup_epochs)
            progress = (epoch - merged_args.warmup_epochs) / max(1, merged_args.train_epochs - merged_args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        accelerator.print(f'Using WarmupCosine scheduler with {merged_args.warmup_epochs} warmup epochs')

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # Start fine-tuning
    accelerator.print('\nStarting fine-tuning...')
    start_time = time.time()

    best_epoch_info = finetune_train(
        model, train_loader, val_loader, test_loader,
        optimizer, criterion, accelerator, merged_args, scheduler=scheduler
    )

    end_time = time.time()
    accelerator.print(f'\nFine-tuning completed in {end_time - start_time:.2f} seconds')
    accelerator.print(f'Model saved to: {merged_args.checkpoints}')

    # Save final results
    if accelerator.is_main_process:
        results = {
            'best_epoch': int(best_epoch_info['epoch']),
            'best_val_mape': float(best_epoch_info['val_mape']),
            'best_val_mae': float(best_epoch_info['val_mae']),
            'best_val_rmse': float(best_epoch_info['val_rmse']),
            'test_mape_at_best_val': float(best_epoch_info['test_mape']),
            'test_mae_at_best_val': float(best_epoch_info['test_mae']),
            'test_rmse_at_best_val': float(best_epoch_info['test_rmse']),
            'training_time_seconds': float(end_time - start_time)
        }
        with open(os.path.join(merged_args.checkpoints, 'finetune_results.json'), 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
