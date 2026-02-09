"""
Tongji Dataset SOH Simple Replacement Script

Simple processing logic:
- Detect anomaly REGIONS (consecutive cycles with abnormal increase)
- Use interpolation to smooth the region
- Record all problematic cycles

"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List

# Parameter Configuration
THRESHOLD_CHANGE = 1.0  # 1% threshold for detecting anomaly start
MAX_ANOMALY_LENGTH = 10  # Maximum length of an anomaly region

# Path Configuration
INPUT_DIR = Path('/ai/dl_project/MemoryNet/dataset/SOH/Tongji')
OUTPUT_DIR = Path('/ai/dl_project/MemoryNet/dataset/processed_SOH/Tongji')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_relative_changes(soh: np.ndarray) -> np.ndarray:
    """Calculate relative change rates (percentage)"""
    relative_changes = np.zeros(len(soh))
    for i in range(1, len(soh)):
        if soh[i-1] != 0:
            relative_changes[i] = (soh[i] - soh[i-1]) / soh[i-1] * 100
        else:
            relative_changes[i] = 0
    return relative_changes


def detect_and_replace_anomalies(soh: np.ndarray, cycle_numbers: np.ndarray) -> tuple:
    """
    Detect anomaly REGIONS (not single points) and fix using interpolation.

    This prevents the "avalanche effect" where continuous replacement leads to
    flat lines in the output.

    An anomaly region is defined as:
    - Starts with a cycle that has >1% increase
    - Ends when SOH returns to near expected trend

    Returns:
        corrected_soh: Corrected SOH array
        anomalies: List of anomaly records
    """
    corrected_soh = soh.copy()
    anomalies = []
    relative_changes = calculate_relative_changes(soh)

    i = 1
    while i < len(soh):
        if soh[i-1] == 0:
            i += 1
            continue

        rel_change = relative_changes[i]

        # Only detect positive changes (SOH increase) as anomaly start
        if rel_change > THRESHOLD_CHANGE:
            start_idx = i
            pre_anomaly_soh = soh[i-1]

            # Find end of anomaly region
            end_idx = start_idx
            for j in range(start_idx + 1, min(len(soh), start_idx + MAX_ANOMALY_LENGTH + 1)):
                if soh[j] == 0:
                    break
                # Check if SOH has returned to expected trend
                # Expected: should be close to or below pre_anomaly_soh
                if soh[j] <= pre_anomaly_soh * 1.005:  # Within 0.5% tolerance
                    end_idx = j - 1
                    break
                end_idx = j

            # Only process if this is a short anomaly (likely test-induced)
            if end_idx - start_idx + 1 <= MAX_ANOMALY_LENGTH:
                # Use linear interpolation to fix the region
                anchor_before_idx = start_idx - 1
                anchor_after_idx = min(end_idx + 1, len(soh) - 1)

                if anchor_before_idx >= 0:
                    soh_before = soh[anchor_before_idx]
                    soh_after = soh[anchor_after_idx] if anchor_after_idx < len(soh) else soh_before * 0.99

                    for idx in range(start_idx, end_idx + 1):
                        if anchor_after_idx != anchor_before_idx:
                            t = (idx - anchor_before_idx) / (anchor_after_idx - anchor_before_idx)
                            corrected_soh[idx] = soh_before + t * (soh_after - soh_before)
                        else:
                            corrected_soh[idx] = soh_before

                    # Record anomaly
                    anomalies.append({
                        'start_cycle': int(cycle_numbers[start_idx]),
                        'end_cycle': int(cycle_numbers[end_idx]),
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'original_soh_range': f"{soh[start_idx]:.4f}-{soh[end_idx]:.4f}",
                        'change_percent': float(rel_change),
                        'replaced': True
                    })

                i = end_idx + 1
            else:
                # This is likely normal behavior, not an anomaly
                i += 1
        else:
            i += 1

    return corrected_soh, anomalies


def process_battery(pkl_file: Path) -> Dict:
    """Process a single battery file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    cell_id = data['cell_id']
    cycle_numbers = np.array(data['cycle_numbers'])
    soh_original = np.array(data['SOH'])

    print(f"\nProcessing: {cell_id}")
    print(f"  Total cycles: {len(cycle_numbers)}, SOH range: {soh_original.min():.4f} - {soh_original.max():.4f}")

    # Detect and replace anomalies
    soh_corrected, anomalies = detect_and_replace_anomalies(soh_original, cycle_numbers)

    if len(anomalies) == 0:
        print(f"  No anomalies detected (all changes < {THRESHOLD_CHANGE}%)")
    else:
        print(f"  Detected {len(anomalies)} anomaly regions:")
        for idx, anomaly in enumerate(anomalies):
            print(f"    [{idx+1}] Cycles {anomaly['start_cycle']}-{anomaly['end_cycle']}: "
                  f"SOH {anomaly['original_soh_range']} "
                  f"(change: {anomaly['change_percent']:+.2f}%)")

    # Prepare output data
    output_data = {
        'cell_id': cell_id,
        'cycle_numbers': cycle_numbers.tolist(),
        'SOH': soh_corrected.tolist(),
        'SOH_original': soh_original.tolist(),
        'cycle_start_time_in_s': data.get('cycle_start_time_in_s', []),
        'anomalies': anomalies,
        'processing_params': {
            'threshold_change': THRESHOLD_CHANGE,
            'max_anomaly_length': MAX_ANOMALY_LENGTH,
            'method': 'region_detection_with_interpolation'
        }
    }

    output_file = OUTPUT_DIR / pkl_file.name
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"  Saved to: {output_file}")

    return {
        'cell_id': cell_id,
        'num_anomalies': len(anomalies),
        'processed': True,
        'had_anomalies': len(anomalies) > 0,
        'output_file': output_file
    }


def main():
    """Main function"""
    print("=" * 80)
    print("Tongji Dataset SOH Simple Replacement")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nParameter Configuration:")
    print(f"  Change threshold: {THRESHOLD_CHANGE}%")
    print(f"  Method: Replace with previous cycle's SOH")
    print("=" * 80)

    pkl_files = sorted(list(INPUT_DIR.glob('*.pkl')))
    # Filter out backup files
    pkl_files = [f for f in pkl_files if not f.name.endswith('.backup')]
    print(f"\nFound {len(pkl_files)} battery files")

    results = []

    for pkl_file in pkl_files:
        try:
            result = process_battery(pkl_file)
            results.append(result)
        except Exception as e:
            print(f"\nERROR processing {pkl_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)

    total_processed = len(results)
    total_with_anomalies = sum(1 for r in results if r.get('had_anomalies', False))
    total_anomalies = sum(r['num_anomalies'] for r in results)

    print(f"Total batteries: {len(results)}")
    print(f"All batteries processed: {total_processed}")
    print(f"Batteries with anomalies (>5%): {total_with_anomalies}")
    print(f"Batteries without anomalies: {total_processed - total_with_anomalies}")
    print(f"Total cycles replaced: {total_anomalies}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
