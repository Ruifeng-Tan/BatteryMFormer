"""
General Dataset SOH Processing Script

Process remaining datasets: CALB, CALCE, HUST, MATR, NA-ion,
Stanford, Stanford_2, UL_PUR, XJTU, ZN-coin

Processing logic:
1. Detect zero values -> Use PCHIP interpolation with surrounding normal values
2. Detect changes > 5% -> Replace with previous cycle's SOH
3. Record all anomalies

Note: Already processed datasets (HNEI, ISU_ILCC, MICH, MICH_EXP, RWTH, SNL, Tongji) are skipped
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from typing import Dict, List

# Parameter Configuration
THRESHOLD_CHANGE = 10.0      # 5% change threshold for replacement
ANCHOR_POINTS = 5           # Number of points before/after for PCHIP interpolation

# Datasets to process
DATASETS_TO_PROCESS = [
    'CALB', 'CALCE', 'HUST', 'MATR', 'NA-ion',
    'Stanford', 'Stanford_2', 'UL_PUR', 'XJTU', 'ZN-coin'
]

# Datasets to copy without processing (no anomaly detection/correction)
DATASETS_COPY_ONLY = ['ZN-coin']

# Base directories (set via command-line arguments)
BASE_INPUT_DIR: Path = None
BASE_OUTPUT_DIR: Path = None


def detect_zero_anomalies(soh: np.ndarray, cycle_numbers: np.ndarray) -> List[Dict]:
    """Detect zero-value regions"""
    anomalies = []

    i = 0
    while i < len(soh):
        if soh[i] == 0:
            start_idx = i
            start_cycle = cycle_numbers[i]

            # Find extent of zero region
            while i < len(soh) and soh[i] == 0:
                i += 1

            end_idx = i - 1
            end_cycle = cycle_numbers[end_idx] if end_idx < len(cycle_numbers) else cycle_numbers[-1]

            anomalies.append({
                'type': 'zero',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_cycle': int(start_cycle),
                'end_cycle': int(end_cycle),
                'method': 'pchip'
            })
        else:
            i += 1

    return anomalies


def apply_pchip_interpolation(soh: np.ndarray, cycle_numbers: np.ndarray,
                               anomaly: Dict) -> np.ndarray:
    """Fix zero-value anomaly using PCHIP interpolation"""
    corrected_soh = soh.copy()

    start_idx = anomaly['start_idx']
    end_idx = anomaly['end_idx']

    # Collect anchor points before anomaly
    anchor_start = max(0, start_idx - ANCHOR_POINTS)
    anchor_cycles = []
    anchor_soh = []

    for i in range(anchor_start, start_idx):
        if soh[i] > 0:
            anchor_cycles.append(cycle_numbers[i])
            anchor_soh.append(soh[i])

    # Collect anchor points after anomaly
    anchor_end = min(len(soh) - 1, end_idx + ANCHOR_POINTS)
    for i in range(end_idx + 1, anchor_end + 1):
        if soh[i] > 0:
            anchor_cycles.append(cycle_numbers[i])
            anchor_soh.append(soh[i])

    if len(anchor_soh) < 2:
        print(f"    Warning: Insufficient anchor points ({len(anchor_soh)}), using constant value")
        if len(anchor_soh) >= 1:
            corrected_soh[start_idx:end_idx+1] = anchor_soh[0]
        return corrected_soh

    # Perform PCHIP interpolation
    try:
        pchip = PchipInterpolator(anchor_cycles, anchor_soh)
        interpolated_cycles = cycle_numbers[start_idx:end_idx+1]
        interpolated_soh = pchip(interpolated_cycles)
        interpolated_soh = np.maximum(interpolated_soh, 0.0)  # Prevent negative values
        corrected_soh[start_idx:end_idx+1] = interpolated_soh
    except Exception as e:
        print(f"    Warning: PCHIP interpolation failed: {e}")
        # Fallback to linear interpolation
        if len(anchor_soh) >= 2:
            slope = (anchor_soh[-1] - anchor_soh[0]) / (anchor_cycles[-1] - anchor_cycles[0])
            for i in range(start_idx, end_idx + 1):
                corrected_soh[i] = anchor_soh[0] + slope * (cycle_numbers[i] - anchor_cycles[0])
                corrected_soh[i] = max(0.0, corrected_soh[i])

    return corrected_soh


def detect_anomaly_regions(soh: np.ndarray, cycle_numbers: np.ndarray) -> List[Dict]:
    """
    Detect anomaly regions (sudden jumps that return to normal trend).

    A jump is detected when:
    - Change exceeds threshold
    - After the jump, SOH returns to within RECOVERY_THRESHOLD of expected trend

    Returns:
        List of anomaly region dicts with start_idx, end_idx
    """
    RECOVERY_THRESHOLD = 0.03  # 3% tolerance for recovery detection
    MAX_ANOMALY_LENGTH = 10    # Maximum length of an anomaly region

    anomaly_regions = []

    i = 1
    while i < len(soh):
        if soh[i-1] == 0 or soh[i] == 0:
            i += 1
            continue

        rel_change = (soh[i] - soh[i-1]) / soh[i-1] * 100

        # Check if this is a sudden jump (potential anomaly start)
        if abs(rel_change) > THRESHOLD_CHANGE:
            start_idx = i
            pre_anomaly_soh = soh[i-1]

            # Look for the end of anomaly (when SOH returns to expected trend)
            # Expected trend: slight decline from pre_anomaly_soh
            end_idx = start_idx

            for j in range(start_idx + 1, min(len(soh), start_idx + MAX_ANOMALY_LENGTH + 1)):
                if soh[j] == 0:
                    break

                # Check if SOH has returned to near pre-anomaly level
                # Allow for some normal degradation
                expected_soh = pre_anomaly_soh * (1 - 0.001 * (j - start_idx + 1))  # ~0.1% per cycle decline
                if abs(soh[j] - expected_soh) <= RECOVERY_THRESHOLD:
                    end_idx = j - 1  # End before recovery point
                    break
                end_idx = j

            # Only mark as anomaly if it's a short region (likely test-induced)
            if end_idx - start_idx + 1 <= MAX_ANOMALY_LENGTH:
                anomaly_regions.append({
                    'type': 'jump_anomaly',
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_cycle': int(cycle_numbers[start_idx]),
                    'end_cycle': int(cycle_numbers[end_idx]),
                    'pre_anomaly_soh': float(pre_anomaly_soh),
                    'change_percent': float(rel_change)
                })
                i = end_idx + 1
            else:
                # This is likely normal battery degradation, not an anomaly
                i += 1
        else:
            i += 1

    return anomaly_regions


def replace_large_changes(soh: np.ndarray, cycle_numbers: np.ndarray) -> tuple:
    """
    Detect and fix anomaly regions (sudden jumps that may be test-induced).

    IMPORTANT: This function now detects anomaly REGIONS instead of single points,
    which prevents the "avalanche effect" where continuous replacement leads to
    flat lines in the output.

    Returns:
        corrected_soh: Corrected SOH array
        large_change_anomalies: List of large change anomaly records
    """
    corrected_soh = soh.copy()
    large_change_anomalies = []

    # First, detect all anomaly regions
    anomaly_regions = detect_anomaly_regions(soh, cycle_numbers)

    # Then fix each region using interpolation
    for region in anomaly_regions:
        start_idx = region['start_idx']
        end_idx = region['end_idx']

        # Get anchor points before and after the anomaly region
        anchor_before_idx = start_idx - 1
        anchor_after_idx = min(end_idx + 1, len(soh) - 1)

        if anchor_before_idx >= 0 and anchor_after_idx < len(soh):
            # Linear interpolation between anchor points
            if anchor_before_idx != anchor_after_idx:
                soh_before = soh[anchor_before_idx]
                soh_after = soh[anchor_after_idx]

                for idx in range(start_idx, end_idx + 1):
                    # Linear interpolation
                    t = (idx - anchor_before_idx) / (anchor_after_idx - anchor_before_idx)
                    corrected_soh[idx] = soh_before + t * (soh_after - soh_before)

        # Record anomaly
        large_change_anomalies.append({
            'type': 'large_change',
            'start_cycle': region['start_cycle'],
            'end_cycle': region['end_cycle'],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'original_soh_range': f"{soh[start_idx]:.4f}-{soh[end_idx]:.4f}",
            'change_percent': region['change_percent'],
            'method': 'linear_interpolation'
        })

    return corrected_soh, large_change_anomalies


def process_battery(pkl_file: Path, output_dir: Path) -> Dict:
    """Process a single battery file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    cell_id = data['cell_id']
    cycle_numbers = np.array(data['cycle_numbers'])
    soh_original = np.array(data['SOH'])

    print(f"\n  Processing: {cell_id}")
    print(f"    Total cycles: {len(cycle_numbers)}, SOH range: {soh_original.min():.4f} - {soh_original.max():.4f}")

    # Step 1: Detect and fix zero values using PCHIP
    zero_anomalies = detect_zero_anomalies(soh_original, cycle_numbers)
    soh_corrected = soh_original.copy()

    if len(zero_anomalies) > 0:
        print(f"    Step 1: Detected {len(zero_anomalies)} zero-value regions")
        for idx, anomaly in enumerate(zero_anomalies):
            print(f"      [{idx+1}] Cycles {anomaly['start_cycle']}-{anomaly['end_cycle']} "
                  f"({anomaly['end_idx'] - anomaly['start_idx'] + 1} cycles)")
            soh_corrected = apply_pchip_interpolation(soh_corrected, cycle_numbers, anomaly)
    else:
        print(f"    Step 1: No zero values detected")

    # Step 2: Detect and replace large changes
    soh_corrected, large_change_anomalies = replace_large_changes(soh_corrected, cycle_numbers)

    if len(large_change_anomalies) > 0:
        print(f"    Step 2: Detected {len(large_change_anomalies)} anomaly regions")
        for idx, anomaly in enumerate(large_change_anomalies[:10]):  # Show first 10
            print(f"      [{idx+1}] Cycles {anomaly['start_cycle']}-{anomaly['end_cycle']}: "
                  f"SOH {anomaly['original_soh_range']} "
                  f"({anomaly['change_percent']:+.2f}%)")
        if len(large_change_anomalies) > 10:
            print(f"      ... and {len(large_change_anomalies) - 10} more")
    else:
        print(f"    Step 2: No large changes detected")

    # Combine all anomalies
    all_anomalies = zero_anomalies + large_change_anomalies

    # Prepare output data
    output_data = {
        'cell_id': cell_id,
        'cycle_numbers': cycle_numbers.tolist(),
        'SOH': soh_corrected.tolist(),
        'SOH_original': soh_original.tolist(),
        'cycle_start_time_in_s': data.get('cycle_start_time_in_s', []),
        'anomalies': all_anomalies,
        'processing_params': {
            'threshold_change': THRESHOLD_CHANGE,
            'anchor_points': ANCHOR_POINTS,
            'zero_method': 'pchip_interpolation',
            'large_change_method': 'region_detection_with_interpolation'
        }
    }

    output_file = output_dir / pkl_file.name
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    return {
        'cell_id': cell_id,
        'num_zero_anomalies': len(zero_anomalies),
        'num_large_changes': len(large_change_anomalies),
        'total_anomalies': len(all_anomalies),
        'processed': True
    }


def copy_dataset_without_processing(dataset_name: str):
    """Copy dataset files directly without any processing"""
    import shutil

    input_dir = BASE_INPUT_DIR / dataset_name
    output_dir = BASE_OUTPUT_DIR / dataset_name

    if not input_dir.exists():
        print(f"  Warning: Input directory not found: {input_dir}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(list(input_dir.glob('*.pkl')))
    pkl_files = [f for f in pkl_files if not f.name.endswith('.backup')]

    if len(pkl_files) == 0:
        print(f"  Warning: No PKL files found in {input_dir}")
        return None

    print(f"\n{'='*80}")
    print(f"Copying Dataset (no processing): {dataset_name}")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(pkl_files)} battery files")

    for pkl_file in pkl_files:
        output_file = output_dir / pkl_file.name
        shutil.copy2(pkl_file, output_file)
        print(f"  Copied: {pkl_file.name}")

    print(f"\nCopied {len(pkl_files)} files without processing")

    return {
        'dataset': dataset_name,
        'num_batteries': len(pkl_files),
        'total_zero': 0,
        'total_large': 0,
        'total_anomalies': 0,
        'copied_only': True
    }


def process_dataset(dataset_name: str):
    """Process all batteries in a dataset"""
    input_dir = BASE_INPUT_DIR / dataset_name
    output_dir = BASE_OUTPUT_DIR / dataset_name

    if not input_dir.exists():
        print(f"  ⚠️  Input directory not found: {input_dir}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(list(input_dir.glob('*.pkl')))
    pkl_files = [f for f in pkl_files if not f.name.endswith('.backup')]

    if len(pkl_files) == 0:
        print(f"  ⚠️  No PKL files found in {input_dir}")
        return None

    print(f"\n{'='*80}")
    print(f"Processing Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(pkl_files)} battery files")

    results = []

    for pkl_file in pkl_files:
        try:
            result = process_battery(pkl_file, output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ ERROR processing {pkl_file.name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'-'*80}")
    print(f"Dataset {dataset_name} Summary:")
    print(f"{'-'*80}")

    total_zero = sum(r['num_zero_anomalies'] for r in results)
    total_large = sum(r['num_large_changes'] for r in results)
    total_anomalies = sum(r['total_anomalies'] for r in results)

    print(f"Processed batteries: {len(results)}")
    print(f"Total zero-value anomalies: {total_zero}")
    print(f"Total large-change replacements: {total_large}")
    print(f"Total anomalies: {total_anomalies}")

    return {
        'dataset': dataset_name,
        'num_batteries': len(results),
        'total_zero': total_zero,
        'total_large': total_large,
        'total_anomalies': total_anomalies
    }


def main():
    """Main function"""
    print("=" * 80)
    print("General Dataset SOH Processing")
    print("=" * 80)
    print(f"\nParameter Configuration:")
    print(f"  Large change threshold: {THRESHOLD_CHANGE}%")
    print(f"  PCHIP anchor points: {ANCHOR_POINTS}")
    print(f"\nDatasets to process:")
    for ds in DATASETS_TO_PROCESS:
        print(f"  - {ds}")
    print("=" * 80)

    dataset_results = []

    for dataset_name in DATASETS_TO_PROCESS:
        if dataset_name in DATASETS_COPY_ONLY:
            result = copy_dataset_without_processing(dataset_name)
        else:
            result = process_dataset(dataset_name)
        if result:
            dataset_results.append(result)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_datasets = len(dataset_results)
    total_batteries = sum(r['num_batteries'] for r in dataset_results)
    total_zero = sum(r['total_zero'] for r in dataset_results)
    total_large = sum(r['total_large'] for r in dataset_results)
    total_anomalies = sum(r['total_anomalies'] for r in dataset_results)

    print(f"Total datasets processed: {total_datasets}")
    print(f"Total batteries processed: {total_batteries}")
    print(f"Total zero-value anomalies fixed: {total_zero}")
    print(f"Total large changes replaced: {total_large}")
    print(f"Total anomalies handled: {total_anomalies}")

    print("\nPer-dataset summary:")
    print(f"{'Dataset':<15} {'Batteries':>10} {'Zeros':>8} {'Large':>8} {'Total':>8}")
    print("-" * 60)
    for r in dataset_results:
        print(f"{r['dataset']:<15} {r['num_batteries']:>10} {r['total_zero']:>8} "
              f"{r['total_large']:>8} {r['total_anomalies']:>8}")

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Output directory: {BASE_OUTPUT_DIR}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='General dataset SOH preprocessing')
    parser.add_argument('--base_input', type=str, required=True, help='Base input SOH directory')
    parser.add_argument('--base_output', type=str, required=True, help='Base output processed SOH directory')
    args = parser.parse_args()
    BASE_INPUT_DIR = Path(args.base_input)
    BASE_OUTPUT_DIR = Path(args.base_output)
    main()
