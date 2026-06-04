"""
TimesFM baseline evaluation for SOH trajectory prediction.

TimesFM only consumes a single-variable input (the SOH sequence) and requires the
input length to be a multiple of patch_size (32). Because masked positions output
NaN, we pad at the end with the last observed SOH and keep mask=False everywhere;
the padded region uses the padded SOH itself as the prediction and the model's
real forecast starts at cycle padded_len+1.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
import json
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='TimesFM Baseline Evaluation')

    # Dataset
    parser.add_argument('--dataset', type=str, default='Li_ion',
                        choices=['Li_ion', 'CALB', 'NA-ion', 'ZN-coin'])
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--flag', type=str, default='test', choices=['train', 'val', 'test'])

    # Data params (must match cache)
    parser.add_argument('--early_cycle_threshold', type=int, default=100)
    parser.add_argument('--charge_discharge_length', type=int, default=300)
    parser.add_argument('--seq_len', type=int, default=1)
    parser.add_argument('--max_trajectory_len', type=int, default=5000)
    parser.add_argument('--use_capacity_resample', action='store_true', default=True)

    # TimesFM params
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')

    # EOL threshold
    parser.add_argument('--eol_threshold', type=float, default=0.8)

    # Split (must match the split used at training time so the cache lines up)
    parser.add_argument('--split_json_path', type=str, default='',
                        help='Path to split json (LOAO / pure-OOD). Leave empty for legacy random split.')
    parser.add_argument('--split_tag', type=str, default='',
                        help='Optional split tag. If empty, derived from --split_json_path filename.')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results/timesfm')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cache_root', type=str, default='./.cache',
                        help='Root directory for cached preprocessed data')

    return parser.parse_args()


def load_cache_data(args):
    """Load preprocessed data from cache. Key must match data_loader_soh_optimized.py."""
    cache_key = (f"{args.dataset}_{args.seed}_{args.flag}_current_voltage_"
                 f"{args.early_cycle_threshold}_{args.charge_discharge_length}_"
                 f"{args.seq_len}_{args.max_trajectory_len}_"
                 f"capresample{args.use_capacity_resample}_v3_soc_alignchk1")

    split_tag = args.split_tag
    if not split_tag and args.split_json_path:
        split_tag = Path(args.split_json_path).stem
    if split_tag:
        cache_key += f"_split{split_tag}"

    cache_file = f"{args.cache_root}/{cache_key}.pkl"

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}\n"
                                f"Please run training first to generate cache.")

    print(f"Loading cache: {cache_file}")
    with open(cache_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")
    return samples


def get_visible_soh(soh_input):
    """
    Extract visible SOH values from input
    soh_input: [early_cycle_threshold, 1] or [early_cycle_threshold]
    Returns: 1D array of visible SOH values
    """
    if isinstance(soh_input, torch.Tensor):
        soh_input = soh_input.numpy()

    soh_input = soh_input.squeeze()  # [early_cycle_threshold]

    # Find last non-zero index (visible cycles)
    non_zero_mask = soh_input != 0
    if not non_zero_mask.any():
        return np.array([])

    # Find the last non-zero position
    last_valid_idx = np.where(non_zero_mask)[0][-1] + 1

    return soh_input[:last_valid_idx]


def denormalize_soh(soh_normalized, eol_threshold=0.8):
    """Convert normalized SOH back to real scale"""
    # Normalization was: (soh - eol_threshold) / (1.0 - eol_threshold)
    # Reverse: soh = soh_normalized * (1.0 - eol_threshold) + eol_threshold
    return soh_normalized * (1.0 - eol_threshold) + eol_threshold


def calculate_metrics_global(all_preds, all_targets, all_masks, eol_threshold=0.8):
    """
    Calculate MAPE, RMSE, MAE on ALL valid positions globally.
    This matches the calculation method in run_main.py.

    Args:
        all_preds: [N, pred_len] normalized predictions
        all_targets: [N, pred_len] normalized targets
        all_masks: [N, pred_len] masks
        eol_threshold: EOL threshold for denormalization
    """
    # Denormalize to real SOH values
    pred_real = all_preds * (1.0 - eol_threshold) + eol_threshold
    target_real = all_targets * (1.0 - eol_threshold) + eol_threshold

    # Get all valid positions
    valid_mask = all_masks > 0
    if valid_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    valid_pred = pred_real[valid_mask]
    valid_target = target_real[valid_mask]

    # MAE
    mae = np.abs(valid_pred - valid_target).mean()

    # RMSE
    mse = ((valid_pred - valid_target) ** 2).mean()
    rmse = np.sqrt(mse)

    # MAPE (only for targets > 0.01 to avoid division by zero)
    valid_for_mape = valid_target > 0.01
    if valid_for_mape.sum() > 0:
        mape = np.abs((valid_target[valid_for_mape] - valid_pred[valid_for_mape]) / valid_target[valid_for_mape]).mean() * 100
    else:
        mape = 0.0

    return mape, rmse, mae


def normalize_soh(soh_real, eol_threshold=0.8):
    """Convert real SOH to normalized scale"""
    return (soh_real - eol_threshold) / (1.0 - eol_threshold)


def timesfm_predict_batch(model, soh_inputs, horizon, patch_size=32):
    """Run TimesFM on a batch of SOH inputs that share the same padded length.

    Each input is padded at the end with its last value to the next multiple
    of patch_size; mask stays all False to avoid the NaN bug. Group by
    padded length before calling.
    """
    if len(soh_inputs) == 0:
        return []

    device = model.model.device

    # Pad each input at the end with its last value to next multiple of patch_size
    processed_inputs = []
    for soh in soh_inputs:
        padded_len = ((len(soh) + patch_size - 1) // patch_size) * patch_size
        if padded_len > len(soh):
            soh = np.pad(soh, (0, padded_len - len(soh)), 'edge')
        processed_inputs.append(soh)

    # Stack into batch
    batch_input = np.array(processed_inputs, dtype=np.float32)
    batch_mask = np.zeros_like(batch_input, dtype=bool)

    # Convert to tensors
    inputs_t = torch.from_numpy(batch_input).to(device)
    masks_t = torch.from_numpy(batch_mask).to(device)

    # Run decode on entire batch
    with torch.no_grad():
        pf_outputs, _, ar_outputs = model.model.decode(horizon, inputs_t, masks_t)

        # Combine prefill and autoregressive outputs
        to_cat = [pf_outputs[:, -1, ...]]
        if ar_outputs is not None:
            batch_size = inputs_t.shape[0]
            to_cat.append(ar_outputs.reshape(batch_size, -1, 10))
        full_forecast = torch.cat(to_cat, dim=1)

        # Get median prediction (index 5) for all samples
        predictions = full_forecast[:, :horizon, 5].cpu().numpy()

    return list(predictions)


def evaluate_timesfm(args):
    """Main evaluation function"""
    import timesfm

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.set_float32_matmul_precision("high")

    # Load TimesFM model
    print("Loading TimesFM model...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False  # Disable compile for more stable inference
    )
    print(f"Model loaded on device: {model.model.device}")

    # Load data
    samples = load_cache_data(args)

    # Determine EOL threshold based on dataset
    if args.dataset == 'CALB':
        eol_threshold = 0.9
    else:
        eol_threshold = 0.8

    # Group samples by file_name to avoid duplicate evaluations
    # (multiple samples from same battery with different input lengths)
    samples_by_file = defaultdict(list)
    for sample in samples:
        samples_by_file[sample['file_name']].append(sample)

    print(f"Total unique batteries: {len(samples_by_file)}")

    # Prepare all samples
    all_samples_info = []
    for file_name, file_samples in samples_by_file.items():
        for sample in file_samples:
            soh_input = sample['soh_input']
            soh_trajectory = sample['soh_trajectory']
            trajectory_mask = sample['trajectory_mask']

            # Get visible SOH (normalized)
            visible_soh_norm = get_visible_soh(soh_input)
            if len(visible_soh_norm) < 1:
                continue

            input_len = len(visible_soh_norm)

            # Denormalize for TimesFM input (it expects real values)
            visible_soh_real = denormalize_soh(visible_soh_norm, eol_threshold)

            all_samples_info.append({
                'file_name': file_name,
                'input_len': input_len,
                'visible_soh_real': visible_soh_real,
                'target_normalized': soh_trajectory,
                'mask': trajectory_mask,
                'eol_threshold': eol_threshold
            })

    print(f"Total samples to evaluate: {len(all_samples_info)}")

    # Use same horizon as other models (pred_len=5000)
    max_horizon = args.max_trajectory_len
    print(f"Using horizon: {max_horizon}")

    # Group samples by padded input length (next multiple of patch_size) for efficient batching
    patch_size = 32
    samples_by_padded_len = defaultdict(list)
    for idx, info in enumerate(all_samples_info):
        input_len = info['input_len']
        padded_len = ((input_len + patch_size - 1) // patch_size) * patch_size
        samples_by_padded_len[padded_len].append((idx, info))

    print(f"Grouped into {len(samples_by_padded_len)} padded length groups")

    # Evaluate with batched inference - collect all predictions in normalized form
    # to match run_main.py's global metric calculation
    all_preds = np.zeros((len(all_samples_info), args.max_trajectory_len), dtype=np.float32)
    all_targets = np.zeros((len(all_samples_info), args.max_trajectory_len), dtype=np.float32)
    all_masks = np.zeros((len(all_samples_info), args.max_trajectory_len), dtype=np.float32)
    all_input_lens = np.zeros(len(all_samples_info), dtype=np.int32)

    batch_size = args.batch_size

    for padded_len, group in tqdm(samples_by_padded_len.items(), desc="Processing groups"):
        # Process this group in batches
        for batch_start in range(0, len(group), batch_size):
            batch = group[batch_start:batch_start + batch_size]
            batch_indices = [item[0] for item in batch]
            batch_infos = [item[1] for item in batch]

            # Prepare batch inputs
            batch_soh_inputs = [info['visible_soh_real'] for info in batch_infos]

            # Run batched prediction
            batch_predictions = timesfm_predict_batch(
                model, batch_soh_inputs, max_horizon, patch_size
            )

            # Store predictions in normalized form
            for i, (idx, info, pred) in enumerate(zip(batch_indices, batch_infos, batch_predictions)):
                input_len = info['input_len']
                visible_soh_real = info['visible_soh_real']
                target_normalized = info['target_normalized']
                mask = info['mask']
                eol_thresh = info['eol_threshold']

                # Real cycles + padded region (last SOH) + model forecast from padded_len+1
                full_pred_real = np.zeros(args.max_trajectory_len)
                full_pred_real[:input_len] = visible_soh_real
                pad_end = min(padded_len, args.max_trajectory_len)
                if pad_end > input_len:
                    full_pred_real[input_len:pad_end] = visible_soh_real[-1]
                pred_start = padded_len
                if pred_start < args.max_trajectory_len:
                    pred_len = min(len(pred), args.max_trajectory_len - pred_start)
                    if pred_len > 0:
                        full_pred_real[pred_start:pred_start+pred_len] = pred[:pred_len]

                # Convert prediction to normalized scale (same as target)
                full_pred_normalized = normalize_soh(full_pred_real, eol_thresh)

                # Store in arrays
                all_preds[idx] = full_pred_normalized
                all_targets[idx] = target_normalized
                all_masks[idx] = mask
                all_input_lens[idx] = input_len

    # Calculate global metrics (matching run_main.py)
    avg_mape, avg_rmse, avg_mae = calculate_metrics_global(
        all_preds, all_targets, all_masks, eol_threshold
    )

    print("\n" + "="*60)
    print(f"TimesFM Evaluation Results on {args.dataset} (seed={args.seed})")
    print("="*60)
    print(f"Total samples evaluated: {len(all_samples_info)}")
    print(f"Average MAPE: {avg_mape:.4f}%")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Average MAE:  {avg_mae:.6f}")

    # Per input length results (also using global calculation for each group)
    print("\n" + "-"*60)
    print("Results by Input Length:")
    print("-"*60)
    key_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    per_input_length_results = {}

    for input_len in range(1, args.early_cycle_threshold + 1):
        mask_for_len = all_input_lens == input_len
        if mask_for_len.sum() == 0:
            continue

        preds_for_len = all_preds[mask_for_len]
        targets_for_len = all_targets[mask_for_len]
        masks_for_len = all_masks[mask_for_len]

        mape, rmse, mae = calculate_metrics_global(
            preds_for_len, targets_for_len, masks_for_len, eol_threshold
        )

        per_input_length_results[input_len] = {
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'count': int(mask_for_len.sum())
        }

        if input_len in key_lengths:
            print(f"Input {input_len:3d} cycles: MAPE={mape:7.4f}%, RMSE={rmse:.6f}, MAE={mae:.6f}, n={int(mask_for_len.sum())}")

    os.makedirs(args.output_dir, exist_ok=True)
    split_tag = args.split_tag
    if not split_tag and args.split_json_path:
        split_tag = Path(args.split_json_path).stem
    suffix = f"_{split_tag}" if split_tag else ""
    result_file = os.path.join(
        args.output_dir,
        f"timesfm_{args.dataset}_seed{args.seed}{suffix}_{args.flag}.json"
    )

    results = {
        'dataset': args.dataset,
        'seed': args.seed,
        'flag': args.flag,
        'split_tag': split_tag,
        'avg_mape': float(avg_mape),
        'avg_rmse': float(avg_rmse),
        'avg_mae': float(avg_mae),
        'num_samples': len(all_samples_info),
        'per_input_length': {str(k): v for k, v in per_input_length_results.items()}
    }

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    return results


if __name__ == '__main__':
    args = parse_args()
    evaluate_timesfm(args)
