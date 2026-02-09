import argparse
import torch
import numpy as np
import os
import json
import random
import joblib
from collections import defaultdict
from accelerate import Accelerator, load_checkpoint_in_model
from accelerate.utils import set_seed as accelerate_set_seed
from data_provider.data_factory import data_provider_soh
from models import PatchTST, iTransformer, DLinear, CPTransformer, CPMLP, BatteryMFormer
from models import TimeMixerPP, IC2ML, PatchMLP
from models import ConvTimeNet
import warnings
warnings.filterwarnings('ignore')

import torch.distributed as dist


def _gather_object_compat(accelerator, obj):
    """Gather a python object from all processes (compat for older accelerate)."""
    if accelerator.num_processes == 1:
        return [obj]
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    gathered = [None for _ in range(accelerator.num_processes)]
    dist.all_gather_object(gathered, obj)
    return gathered


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
        'ConvTimeNet': ConvTimeNet
    }

    if args.model not in model_dict:
        raise ValueError(f"Model {args.model} not supported")

    model = model_dict[args.model].Model(args)
    return model


def get_eol_threshold(dataset_name):
    """Get EOL threshold based on dataset name (CALB=0.9, others=0.8)"""
    return 0.9 if dataset_name == 'CALB' else 0.8


def calculate_masked_metrics(pred, target, mask, eol_threshold=0.8):
    """Calculate MAPE, RMSE, MAE on masked regions"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Reverse normalization to real SOH values
    # Normalization was: (soh - eol_threshold) / (1.0 - eol_threshold)
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


def evaluate(loader, model, accelerator, args):
    """
    Evaluation loop that returns predictions, targets, masks, file_names, and useable_cycle_numbers
    """
    model.eval()
    preds = []
    targets = []
    masks = []
    all_file_names = []
    all_useable_cycles = []

    accelerator.print(f"Start evaluating on {len(loader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            cycle_curve_data, curve_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names = batch

            if args.input_mode == 'current_voltage':
                outputs = model(
                    cycle_curve_data=cycle_curve_data,
                    curve_attn_mask=curve_attn_mask,
                    aging_condition_embedding=aging_condition_embedding,
                    soh_trajectory=soh_trajectory,
                    trajectory_mask=trajectory_mask,
                    soc_input=soc_curves,
                    soh_input=soh_input,
                    cycle_level_features=cycle_level_features
                )
            elif args.input_mode == 'capacity_increment':
                soh, outputs, RUL = model(capacity_increment=cycle_curve_data, capacity_mask=curve_attn_mask)
            else:
                outputs = model(soh_input=soh_input)

            if curve_attn_mask is not None:
                useable_cycles = torch.sum(curve_attn_mask, dim=1).to(torch.int64)
            else:
                raise Exception('Not implemented')

            gathered_outputs, gathered_targets, gathered_masks = accelerator.gather_for_metrics(
                (outputs, soh_trajectory, trajectory_mask)
            )
            gathered_useable_cycles = accelerator.gather_for_metrics(useable_cycles)

            preds.append(gathered_outputs.detach().cpu().numpy())
            targets.append(gathered_targets.detach().cpu().numpy())
            masks.append(gathered_masks.detach().cpu().numpy())

            gathered_file_names = _gather_object_compat(accelerator, list(file_names))

            if accelerator.is_main_process:
                for sub in gathered_file_names:
                    all_file_names.extend(sub)
                all_useable_cycles.extend(gathered_useable_cycles.detach().cpu().tolist())

            if (batch_idx + 1) % 50 == 0:
                accelerator.print(f'  Processed {batch_idx + 1}/{len(loader)} batches')

    preds_np = np.concatenate(preds, axis=0)
    targets_np = np.concatenate(targets, axis=0)
    masks_np = np.concatenate(masks, axis=0)

    eol_threshold = get_eol_threshold(args.dataset)
    mape, rmse, mae = calculate_masked_metrics(preds_np, targets_np, masks_np, eol_threshold)

    # Strict alignment check + safe trimming for the known "remainder" behavior in gather_for_metrics
    if accelerator.is_main_process:
        n = int(preds_np.shape[0])
        fn_len = len(all_file_names)
        uc_len = len(all_useable_cycles)

        if fn_len != n or uc_len != n:
            accelerator.print(
                f"[Warning] Metadata length mismatch before alignment: "
                f"preds={n}, file_names={fn_len}, useable_cycles={uc_len}. "
                f"Trimming metadata to preds length (handles remainder truncation)."
            )
            all_file_names = all_file_names[:n]
            all_useable_cycles = all_useable_cycles[:n]

        assert len(all_file_names) == n, (len(all_file_names), n)
        assert len(all_useable_cycles) == n, (len(all_useable_cycles), n)
    else:
        all_file_names = []
        all_useable_cycles = []

    return mape, rmse, mae, preds_np, targets_np, masks_np, all_file_names, np.array(all_useable_cycles, dtype=int)


def get_seen_aging_conditions(args):
    """
    Get the set of aging condition IDs that were seen during training and validation
    """
    name2aging_path = os.path.join(os.path.dirname(__file__), 'name2agingConditionID.json')
    with open(name2aging_path, 'r') as f:
        name2aging = json.load(f)

    dataset_name = args.dataset
    seed = args.seed

    if dataset_name != 'Li_ion':
        split_json_path = f'./data_provider/split_json/{dataset_name}_split_{seed}.json'
    else:
        split_json_path = f'./data_provider/split_json/total_split_{seed}.json'

    with open(split_json_path, 'r') as f:
        split_data = json.load(f)

    train_files = split_data.get('train', [])
    val_files = split_data.get('val', [])
    seen_files = train_files + val_files

    seen_aging_ids = set()
    for file_name in seen_files:
        if file_name in name2aging:
            seen_aging_ids.add(name2aging[file_name])

    return seen_aging_ids, name2aging


def evaluate_seen_unseen(preds, targets, masks, file_names, args, accelerator):
    """
    Evaluate model performance on seen vs unseen aging conditions
    """
    seen_aging_ids, name2aging = get_seen_aging_conditions(args)
    eol_threshold = get_eol_threshold(args.dataset)

    seen_indices = []
    unseen_indices = []

    for i, file_name in enumerate(file_names):
        aging_id = name2aging.get(file_name, None)
        if aging_id is not None and aging_id in seen_aging_ids:
            seen_indices.append(i)
        else:
            unseen_indices.append(i)

    results = {}

    if len(seen_indices) > 0:
        seen_preds = preds[seen_indices]
        seen_targets = targets[seen_indices]
        seen_masks = masks[seen_indices]
        seen_mape, seen_rmse, seen_mae = calculate_masked_metrics(seen_preds, seen_targets, seen_masks, eol_threshold)
        results['seen'] = {
            'mape': float(seen_mape),
            'rmse': float(seen_rmse),
            'mae': float(seen_mae),
            'num_samples': len(seen_indices)
        }
        accelerator.print(f"\nSeen Aging Conditions ({len(seen_indices)} samples):")
        accelerator.print(f"  MAPE: {seen_mape:.4f}%  RMSE: {seen_rmse:.6f}  MAE: {seen_mae:.6f}")
    else:
        results['seen'] = {'mape': 0.0, 'rmse': 0.0, 'mae': 0.0, 'num_samples': 0}
        accelerator.print("\nNo samples with seen aging conditions")

    if len(unseen_indices) > 0:
        unseen_preds = preds[unseen_indices]
        unseen_targets = targets[unseen_indices]
        unseen_masks = masks[unseen_indices]
        unseen_mape, unseen_rmse, unseen_mae = calculate_masked_metrics(unseen_preds, unseen_targets, unseen_masks, eol_threshold)
        results['unseen'] = {
            'mape': float(unseen_mape),
            'rmse': float(unseen_rmse),
            'mae': float(unseen_mae),
            'num_samples': len(unseen_indices)
        }
        accelerator.print(f"\nUnseen Aging Conditions ({len(unseen_indices)} samples):")
        accelerator.print(f"  MAPE: {unseen_mape:.4f}%  RMSE: {unseen_rmse:.6f}  MAE: {unseen_mae:.6f}")
    else:
        results['unseen'] = {'mape': 0.0, 'rmse': 0.0, 'mae': 0.0, 'num_samples': 0}
        accelerator.print("\nNo samples with unseen aging conditions")

    return results


def evaluate_by_cycle_number(preds, targets, masks, useable_cycles, accelerator, eol_threshold=0.8):
    """
    Evaluate model performance grouped by useable cycle number
    Returns dictionaries for MAE, MAPE, RMSE keyed by cycle number
    """
    cycle_groups = defaultdict(list)
    for i, cycle_num in enumerate(useable_cycles):
        cycle_groups[int(cycle_num)].append(i)

    mae_by_cycle = {}
    mape_by_cycle = {}
    rmse_by_cycle = {}

    accelerator.print(f"\nEvaluating performance by useable cycle number...")
    accelerator.print(f"Found {len(cycle_groups)} unique cycle numbers")

    for cycle_num in sorted(cycle_groups.keys()):
        indices = cycle_groups[cycle_num]
        if len(indices) == 0:
            continue

        group_preds = preds[indices]
        group_targets = targets[indices]
        group_masks = masks[indices]

        mape, rmse, mae = calculate_masked_metrics(group_preds, group_targets, group_masks, eol_threshold)

        mae_by_cycle[cycle_num] = float(mae)
        mape_by_cycle[cycle_num] = float(mape)
        rmse_by_cycle[cycle_num] = float(rmse)

    cycle_nums = sorted(mae_by_cycle.keys())
    if len(cycle_nums) > 0:
        accelerator.print(f"\nCycle Number Range: {min(cycle_nums)} - {max(cycle_nums)}")
        accelerator.print(f"\nAll cycle performances:")
        accelerator.print(f"{'Cycle':<8} {'MAPE(%)':<12} {'RMSE':<12} {'MAE':<12}")
        accelerator.print("-" * 44)
        for c in cycle_nums:
            accelerator.print(f"{c:<8} {mape_by_cycle[c]:<12.4f} {rmse_by_cycle[c]:<12.6f} {mae_by_cycle[c]:<12.6f}")

    return mae_by_cycle, mape_by_cycle, rmse_by_cycle


def set_seed(seed):
    accelerate_set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Evaluate SOH Model')
    parser.add_argument('--args_path', type=str, required=True,
                        help='Path to the directory containing args.json and checkpoint.pth')
    parser.add_argument('--target_dataset', type=str, required=True,
                        help='Name of the dataset to evaluate on (e.g., MIX_large)')

    cmd_args = parser.parse_args()

    accelerator = Accelerator()

    args_file = os.path.join(cmd_args.args_path, 'args.json')
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Could not find args.json in {cmd_args.args_path}")

    accelerator.print(f"Loading arguments from {args_file}")
    with open(args_file, 'r') as f:
        args_dict = json.load(f)

    args = argparse.Namespace(**args_dict)

    # Normalize legacy model names to current names
    _MODEL_ALIASES = {
        'BatteryMFormer_Trial2': 'BatteryMFormer',
    }
    if args.model in _MODEL_ALIASES:
        args.model = _MODEL_ALIASES[args.model]

    set_seed(args.seed)

    original_train_dataset = args.dataset
    target_eval_dataset = cmd_args.target_dataset

    accelerator.print(f"Model trained on: {original_train_dataset}")
    accelerator.print(f"Evaluating on:    {target_eval_dataset}")
    accelerator.print(f"Checkpoint path:  {args.checkpoints}")

    accelerator.print("Loading original training set to retrieve scaler...")
    label_scaler_path = f'{cmd_args.args_path}/label_scaler'
    if not os.path.exists(label_scaler_path):
        raise FileNotFoundError(f"label_scaler missing: {label_scaler_path}")

    if os.path.exists(label_scaler_path):
        label_scaler = joblib.load(label_scaler_path)
    else:
        accelerator.print(f"Warning: label_scaler not found at {label_scaler_path}, using None")
        # label_scaler = None
        raise Exception("label_scaler is None, please check the checkpoint.")

    args.dataset = target_eval_dataset
    accelerator.print(f"Loading target dataset: {args.dataset}...")
    _, test_loader = data_provider_soh(args, 'test', args.input_mode, label_scaler)

    model = get_model(args)

    checkpoint_path = cmd_args.args_path
    checkpoint_pth = os.path.join(checkpoint_path, 'checkpoint.pth')
    model_safetensors = os.path.join(checkpoint_path, 'model.safetensors')

    if os.path.exists(model_safetensors):
        load_checkpoint_in_model(model, model_safetensors)
    elif os.path.exists(checkpoint_pth):
        accelerator.print(f"Loading checkpoint from {checkpoint_pth}")
        checkpoint = torch.load(checkpoint_pth, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif os.path.exists(checkpoint_path):
        load_checkpoint_in_model(model, checkpoint_path)
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model, test_loader = accelerator.prepare(model, test_loader)

    accelerator.print("\n" + "=" * 50)
    accelerator.print("Starting Inference")
    accelerator.print("=" * 50)

    mape, rmse, mae, preds, targets, masks, file_names, useable_cycles = evaluate(
        test_loader, model, accelerator, args
    )

    accelerator.print("\n" + "=" * 50)
    accelerator.print(f"Overall Evaluation Results on {args.dataset}")
    accelerator.print("=" * 50)
    accelerator.print(f"MAE:  {mae:.6f}")
    accelerator.print(f"RMSE: {rmse:.6f}")
    accelerator.print(f"MAPE: {mape:.2f}%")
    accelerator.print(f"Total samples: {len(preds)}")

    accelerator.print("\n" + "=" * 50)
    accelerator.print("Seen vs Unseen Aging Conditions Analysis")
    accelerator.print("=" * 50)

    seen_unseen_results = evaluate_seen_unseen(preds, targets, masks, file_names, args, accelerator)
    seen_unseen_results["overall"] = {
        "mape": float(mape),
        "rmse": float(rmse),
        "mae": float(mae),
        "num_samples": int(len(preds))
    }

    seen_unseen_path = os.path.join(cmd_args.args_path, 'seen_unseen_results.json')
    mae_path = os.path.join(cmd_args.args_path, 'mae_useable_cycle_numbers.json')
    mape_path = os.path.join(cmd_args.args_path, 'mape_useable_cycle_numbers.json')
    rmse_path = os.path.join(cmd_args.args_path, 'rmse_useable_cycle_numbers.json')

    accelerator.print("\n" + "=" * 50)
    accelerator.print("Performance by Useable Cycle Number")
    accelerator.print("=" * 50)

    mae_by_cycle, mape_by_cycle, rmse_by_cycle = evaluate_by_cycle_number(
        preds, targets, masks, useable_cycles, accelerator, get_eol_threshold(args.dataset)
    )

    # Avoid concurrent writes: only main process writes files
    if accelerator.is_main_process:
        with open(seen_unseen_path, 'w') as f:
            json.dump(seen_unseen_results, f, indent=2)
        accelerator.print(f"\nSeen/Unseen/Overall results saved to: {seen_unseen_path}")

        with open(mae_path, 'w') as f:
            json.dump(mae_by_cycle, f, indent=2)
        with open(mape_path, 'w') as f:
            json.dump(mape_by_cycle, f, indent=2)
        with open(rmse_path, 'w') as f:
            json.dump(rmse_by_cycle, f, indent=2)

        accelerator.print(f"\nCycle number results saved to:")
        accelerator.print(f"  MAE:  {mae_path}")
        accelerator.print(f"  MAPE: {mape_path}")
        accelerator.print(f"  RMSE: {rmse_path}")

    accelerator.wait_for_everyone()

    accelerator.print("\n" + "=" * 50)
    accelerator.print("Evaluation Complete")
    accelerator.print("=" * 50)


if __name__ == '__main__':
    main()