#!/usr/bin/env python3
"""
ISU_ILCC SOH PCHIP Interpolation Smoothing Script

Smooth RPT-induced anomalies in SOH trajectories using PCHIP interpolation.
RPT positions are determined by rpt_start_time from all_rpt_positions.json.

Algorithm:
1. For each RPT, use rpt_start_time to find the last normal cycle before RPT
2. Find recovery point where SOH returns to baseline (within tolerance)
3. Use 5 cycles before RPT as front anchors, 5 cycles after recovery as back anchors
4. Apply PCHIP interpolation to smooth the anomaly region

Usage:
    python preprocess_ISU_ILCC.py --input INPUT_DIR --output OUTPUT_DIR --workers 120
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.interpolate import PchipInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Default parameters
ANCHOR_COUNT = 5
RECOVERY_TOLERANCE = 0.015  # SOH recovers to baseline ± 1.5%
MAX_INTERPOLATION_LENGTH = 200  # Maximum cycles to interpolate


def load_rpt_positions(json_path):
    """Load RPT positions from all_rpt_positions.json"""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object"""
    if not ts_str:
        return None
    try:
        ts_str = str(ts_str)[:19]
        if 'T' in ts_str:
            return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
        else:
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return None


def find_last_cycle_before_rpt(cycle_timestamps, rpt_start):
    """
    Find the last cycle whose timestamp is before RPT start time.

    Args:
        cycle_timestamps: List of cycle start timestamps (datetime or string)
        rpt_start: RPT start time (datetime)

    Returns:
        int: Index of last cycle before RPT, or None if not found
    """
    last_before_idx = None

    for i, ts in enumerate(cycle_timestamps):
        if isinstance(ts, str):
            ts = parse_timestamp(ts)
        if ts is None:
            continue

        if ts < rpt_start:
            last_before_idx = i
        else:
            # Timestamps are sorted, so we can break early
            break

    return last_before_idx


def smooth_single_rpt(soh, cycle_numbers, cycle_timestamps, rpt_start_str,
                      anchor_count=ANCHOR_COUNT, tolerance=RECOVERY_TOLERANCE,
                      max_interp_length=MAX_INTERPOLATION_LENGTH):
    """
    Smooth a single RPT region using PCHIP interpolation.

    Args:
        soh: SOH array (will be modified in place for sequential processing)
        cycle_numbers: Cycle numbers array
        cycle_timestamps: Cycle start timestamps
        rpt_start_str: RPT start time string
        anchor_count: Number of anchor points (default 5)
        tolerance: Recovery tolerance (default 0.015)
        max_interp_length: Maximum interpolation length (default 200)

    Returns:
        tuple: (smoothed_soh, start_idx, end_idx) or (None, None, None) if skipped
    """
    rpt_start = parse_timestamp(rpt_start_str)
    if rpt_start is None:
        return None, None, None

    # Find last cycle before RPT
    before_rpt_idx = find_last_cycle_before_rpt(cycle_timestamps, rpt_start)

    if before_rpt_idx is None:
        # RPT is before the first cycle in pkl
        return None, None, None

    # Check if we have enough front anchors
    if before_rpt_idx < anchor_count:
        # Not enough cycles before RPT for anchors
        return None, None, None

    # First cycle after RPT start
    after_rpt_idx = before_rpt_idx + 1

    if after_rpt_idx >= len(soh):
        # RPT is after the last cycle in pkl (SOH truncated)
        return None, None, None

    # Calculate baseline SOH (average of 5 cycles before RPT)
    front_anchor_start = before_rpt_idx - anchor_count + 1
    front_anchor_end = before_rpt_idx + 1  # exclusive
    baseline_soh = np.mean(soh[front_anchor_start:front_anchor_end])

    # Find recovery point: where SOH returns to baseline ± tolerance
    # Limit search to max_interp_length cycles after front anchors
    recovery_idx = None
    max_search_idx = min(len(soh), front_anchor_end + max_interp_length)

    for j in range(after_rpt_idx, max_search_idx):
        if abs(soh[j] - baseline_soh) <= tolerance:
            recovery_idx = j
            break

    # If no recovery found within tolerance and max length, use max length as recovery point
    if recovery_idx is None:
        recovery_idx = max_search_idx

    # Check if we have enough back anchors after recovery
    if recovery_idx + anchor_count > len(soh):
        # Not enough cycles after recovery for anchors
        return None, None, None

    # Define anchor regions (excluding the anomaly region)
    # Front anchors: anchor_count cycles ending at before_rpt_idx (inclusive)
    front_anchor_cycles = cycle_numbers[front_anchor_start:front_anchor_end].tolist()
    front_anchor_soh = soh[front_anchor_start:front_anchor_end].tolist()

    # Back anchors: anchor_count cycles starting from recovery_idx (inclusive)
    back_anchor_start = recovery_idx
    back_anchor_end = recovery_idx + anchor_count
    back_anchor_cycles = cycle_numbers[back_anchor_start:back_anchor_end].tolist()
    back_anchor_soh = soh[back_anchor_start:back_anchor_end].tolist()

    # Combine anchors for PCHIP interpolation
    anchor_cycles = front_anchor_cycles + back_anchor_cycles
    anchor_soh_values = front_anchor_soh + back_anchor_soh

    # Ensure unique cycles (shouldn't happen, but just in case)
    if len(set(anchor_cycles)) < len(anchor_cycles):
        unique_data = {}
        for c, s in zip(anchor_cycles, anchor_soh_values):
            if c not in unique_data:
                unique_data[c] = s
        anchor_cycles = sorted(unique_data.keys())
        anchor_soh_values = [unique_data[c] for c in anchor_cycles]

    if len(anchor_cycles) < 4:
        return None, None, None

    # Apply PCHIP interpolation to the anomaly region
    # Anomaly region: from (before_rpt_idx + 1) to (recovery_idx - 1) inclusive
    # But we interpolate the entire region between front and back anchors
    smooth_start = front_anchor_end  # First cycle after front anchors
    smooth_end = back_anchor_start   # First cycle of back anchors (exclusive for smoothing)

    if smooth_start >= smooth_end:
        # No cycles to smooth
        return None, None, None

    try:
        pchip = PchipInterpolator(anchor_cycles, anchor_soh_values)
        smoothed_cycles = cycle_numbers[smooth_start:smooth_end]
        smoothed_soh = pchip(smoothed_cycles)
        return smoothed_soh, smooth_start, smooth_end
    except Exception:
        return None, None, None


def process_battery(args):
    """Process a single battery file"""
    input_path, output_path, rpt_info, anchor_count, tolerance, max_interp_length = args

    try:
        # Load SOH data
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        cell_id = data['cell_id']
        cycle_numbers = np.array(data['cycle_numbers'])
        soh_original = np.array(data['SOH'])
        cycle_start_time_in_s = data.get('cycle_start_time_in_s', [])
        cycle_timestamps = data.get('cycle_start_datetime', [])

        # Skip if no RPT info available
        if rpt_info is None:
            output_data = {
                'cell_id': cell_id,
                'cycle_numbers': cycle_numbers.tolist(),
                'SOH': soh_original.tolist(),
                'SOH_original': soh_original.tolist(),
                'cycle_start_time_in_s': cycle_start_time_in_s,
                'cycle_start_datetime': cycle_timestamps,
                'rpt_smoothed_regions': [],
                'processing_params': {
                    'anchor_count': anchor_count,
                    'recovery_tolerance': tolerance,
                    'note': 'No RPT info available'
                }
            }
            with open(output_path, 'wb') as f:
                pickle.dump(output_data, f)
            return cell_id, 0, 'no_rpt_info'

        if not cycle_timestamps:
            output_data = {
                'cell_id': cell_id,
                'cycle_numbers': cycle_numbers.tolist(),
                'SOH': soh_original.tolist(),
                'SOH_original': soh_original.tolist(),
                'cycle_start_time_in_s': cycle_start_time_in_s,
                'cycle_start_datetime': [],
                'rpt_smoothed_regions': [],
                'processing_params': {
                    'anchor_count': anchor_count,
                    'recovery_tolerance': tolerance,
                    'note': 'No cycle timestamps available in SOH data'
                }
            }
            with open(output_path, 'wb') as f:
                pickle.dump(output_data, f)
            return cell_id, 0, 'no_timestamps'

        # Get RPT positions
        rpt_positions = rpt_info.get('rpt_positions', [])

        # Process RPTs sequentially (use smoothed SOH for next RPT)
        soh_smoothed = soh_original.copy()
        smoothed_regions = []

        for rpt in rpt_positions:
            rpt_start = rpt.get('rpt_start_time')

            if not rpt_start:
                continue

            smoothed, start_idx, end_idx = smooth_single_rpt(
                soh_smoothed, cycle_numbers, cycle_timestamps,
                rpt_start, anchor_count, tolerance, max_interp_length
            )

            if smoothed is not None and len(smoothed) > 0:
                soh_smoothed[start_idx:end_idx] = smoothed

                smoothed_regions.append({
                    'rpt_index': rpt.get('rpt_index', -1),
                    'rpt_start_time': rpt_start,
                    'cycles_before_rpt': rpt.get('cycles_before_rpt', -1),
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx),
                    'region_length': int(end_idx - start_idx)
                })

        # Save output
        output_data = {
            'cell_id': cell_id,
            'cycle_numbers': cycle_numbers.tolist(),
            'SOH': soh_smoothed.tolist(),
            'SOH_original': soh_original.tolist(),
            'cycle_start_time_in_s': cycle_start_time_in_s,
            'cycle_start_datetime': cycle_timestamps,  # Preserve datetime timestamps
            'rpt_smoothed_regions': smoothed_regions,
            'processing_params': {
                'anchor_count': anchor_count,
                'recovery_tolerance': tolerance,
                'max_interp_length': max_interp_length
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)

        return cell_id, len(smoothed_regions), 'success'

    except Exception as e:
        return str(input_path), 0, f'error: {str(e)}'


def main():
    parser = argparse.ArgumentParser(description='ISU_ILCC SOH PCHIP Smoothing')
    parser.add_argument('--input', type=str,
                        default='/path/to/your/dataset/SOH/ISU_ILCC',
                        help='Input directory with raw SOH pkl files')
    parser.add_argument('--output', type=str,
                        default='/path/to/your/dataset/processed_SOH/ISU_ILCC',
                        help='Output directory for smoothed SOH pkl files')
    parser.add_argument('--rpt_json', type=str,
                        default='all_rpt_positions.json',
                        help='Path to all_rpt_positions.json')
    parser.add_argument('--workers', type=int, default=120,
                        help='Number of parallel workers')
    parser.add_argument('--anchor_count', type=int, default=5,
                        help='Number of anchor points before and after RPT')
    parser.add_argument('--tolerance', type=float, default=0.015,
                        help='Recovery tolerance (SOH returns to baseline +/- this value)')
    parser.add_argument('--max_interp_length', type=int, default=200,
                        help='Maximum interpolation length in cycles')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ISU_ILCC SOH PCHIP Interpolation Smoothing")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"RPT JSON: {args.rpt_json}")
    print(f"Workers: {args.workers}")
    print(f"Anchor count: {args.anchor_count}")
    print(f"Recovery tolerance: {args.tolerance}")
    print(f"Max interp length: {args.max_interp_length}")
    print("=" * 70)

    # Load RPT positions
    print("\nLoading RPT positions...")
    rpt_data = load_rpt_positions(args.rpt_json)
    print(f"Loaded RPT info for {len(rpt_data)} batteries")

    # Get input files
    pkl_files = sorted(list(input_dir.glob('*.pkl')))
    print(f"Found {len(pkl_files)} SOH files to process")

    # Prepare tasks
    tasks = []
    for pkl_file in pkl_files:
        cell_id = pkl_file.stem
        rpt_info = rpt_data.get(cell_id, None)
        output_path = output_dir / pkl_file.name
        tasks.append((pkl_file, output_path, rpt_info, args.anchor_count, args.tolerance, args.max_interp_length))

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_battery, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    success_count = sum(1 for r in results if r[2] == 'success')
    no_rpt_count = sum(1 for r in results if r[2] == 'no_rpt_info')
    no_ts_count = sum(1 for r in results if r[2] == 'no_timestamps')
    error_count = sum(1 for r in results if r[2].startswith('error'))
    total_regions = sum(r[1] for r in results)

    print(f"Total batteries: {len(results)}")
    print(f"Successfully processed: {success_count}")
    print(f"No RPT info: {no_rpt_count}")
    print(f"No timestamps: {no_ts_count}")
    print(f"Errors: {error_count}")
    print(f"Total RPT regions smoothed: {total_regions}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)

    # Show batteries with smoothed regions
    smoothed_batteries = [(r[0], r[1]) for r in results if r[1] > 0]
    if smoothed_batteries:
        print(f"\nBatteries with smoothed RPT regions ({len(smoothed_batteries)}):")
        for cell_id, count in sorted(smoothed_batteries):
            print(f"  {cell_id}: {count} regions")

    errors = [r for r in results if r[2].startswith('error')]
    if errors:
        print("\nErrors:")
        for cell_id, _, err in errors[:10]:
            print(f"  {cell_id}: {err}")


if __name__ == '__main__':
    main()
