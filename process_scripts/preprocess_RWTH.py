"""
RWTH Dataset SOH PCHIP Interpolation Smoothing Script

Features:
1. Detect anomalies in SOH curves (test-induced increases/decreases/time gaps)
2. Fix anomalies using PCHIP interpolation
3. Record anomaly cycle information
4. Save processed data to processed_SOH directory

Based on RWTH dataset analysis:
- Very high data quality (98.96% normal)
- Main anomalies: Characterization tests (SOH increases/decreases)
- Time gaps: ~2h during tests

"""

import pickle
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from typing import Dict, List

# Parameter Configuration (Based on RWTH analysis)
THRESHOLD_INCREASE = 0.06      # 0.6% relative increase threshold (99th percentile)
THRESHOLD_DECREASE = -0.6     # -0.6% relative decrease threshold (1st percentile)
THRESHOLD_TIME_GAP = 1.5      # 1.5 hours time gap threshold
STABILITY_THRESHOLD = 0.5     # Stability detection: consecutive changes < 0.5%
STABILITY_WINDOW = 10         # Stability detection: 10 consecutive cycles
ANCHOR_POINTS = 5             # Number of anchor points for PCHIP interpolation
RECOVERY_TOLERANCE = 0.002     # SOH needs to recover within 2% of pre-anomaly level

# Path Configuration (set via command-line arguments)
INPUT_DIR: Path = None
OUTPUT_DIR: Path = None


def calculate_relative_changes(soh: np.ndarray) -> np.ndarray:
    """Calculate relative change rates (percentage)"""
    relative_changes = np.zeros(len(soh))
    for i in range(1, len(soh)):
        if soh[i-1] != 0:
            relative_changes[i] = (soh[i] - soh[i-1]) / soh[i-1] * 100
        else:
            relative_changes[i] = 0
    return relative_changes


def calculate_time_gaps(timestamps: np.ndarray) -> np.ndarray:
    """Calculate time gaps in hours"""
    time_gaps = np.zeros(len(timestamps))
    for i in range(1, len(timestamps)):
        time_gaps[i] = (timestamps[i] - timestamps[i-1]) / 3600.0
    return time_gaps


def find_stable_point(soh: np.ndarray, relative_changes: np.ndarray, start_idx: int,
                      pre_anomaly_soh: float) -> int:
    """
    Find the point where SOH returns to pre-anomaly level after sudden increase/decrease

    Key logic change:
    - Primary goal: Find where SOH recovers to original level (not just "stabilizes")
    - Secondary: If never recovers, find stable point
    - Stricter recovery criteria to avoid premature termination

    Args:
        soh: SOH array
        relative_changes: relative change rates array
        start_idx: index where anomaly started
        pre_anomaly_soh: SOH value before anomaly (for recovery level check)

    Returns:
        Index where SOH has recovered to pre-anomaly level or reached stable state
    """
    i = start_idx + 1
    recovery_found = False
    recovery_idx = -1

    # First, try to find recovery point (back to pre-anomaly level)
    while i < len(soh):
        current_soh = soh[i]

        # Stricter recovery check: within 0.5% of pre-anomaly level
        soh_diff = abs(current_soh - pre_anomaly_soh)
        if soh_diff <= pre_anomaly_soh * 0.005:  # 0.5% relative tolerance
            # Verify that following cycles are not continuing to rise/fall
            if i + 3 < len(soh):
                # Check next 3 cycles don't have large changes
                next_changes = relative_changes[i+1:min(i+4, len(soh))]
                max_abs_change = np.max(np.abs(next_changes)) if len(next_changes) > 0 else 0

                if max_abs_change < 0.4:  # No large changes after recovery
                    recovery_found = True
                    recovery_idx = i
                    break
            else:
                recovery_found = True
                recovery_idx = i
                break

        i += 1

        # Stop searching after 50 cycles if no recovery
        if i > start_idx + 50:
            break

    # If recovery found, return that point
    if recovery_found:
        return recovery_idx

    # If no recovery, fall back to stability detection
    i = start_idx + 1
    while i < len(soh):
        end_window = min(i + STABILITY_WINDOW, len(soh))
        window_changes = relative_changes[i:end_window]

        # Check if stable (consecutive cycles with small change rates)
        is_stable = (len(window_changes) >= min(STABILITY_WINDOW, len(soh) - i) and
                     np.all(np.abs(window_changes) <= STABILITY_THRESHOLD) and
                     np.all(soh[i:end_window] > 0))

        if is_stable:
            return i

        i += 1

    return len(soh) - 1


def detect_anomalies(soh: np.ndarray, cycle_numbers: np.ndarray, timestamps: np.ndarray) -> List[Dict]:
    """Detect anomalies in SOH curve including time gap anomalies"""
    anomalies = []
    relative_changes = calculate_relative_changes(soh)
    time_gaps = calculate_time_gaps(timestamps)

    i = 1
    while i < len(soh):
        # Detect zero values
        if soh[i] == 0:
            start_idx = i
            start_cycle = cycle_numbers[i]

            while i < len(soh) and soh[i] == 0:
                i += 1

            end_idx = i - 1
            end_cycle = cycle_numbers[end_idx] if end_idx < len(cycle_numbers) else cycle_numbers[-1]

            anomalies.append({
                'type': 'zero',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_cycle': start_cycle,
                'end_cycle': end_cycle,
                'max_change': -100.0
            })
            continue

        # Detect large time gaps (characterization tests)
        if time_gaps[i] > THRESHOLD_TIME_GAP:
            start_idx = i - 1
            start_cycle = cycle_numbers[start_idx]
            pre_anomaly_soh = soh[start_idx]
            time_gap_value = time_gaps[i]

            end_idx = find_stable_point(soh, relative_changes, i, pre_anomaly_soh)
            end_cycle = cycle_numbers[end_idx] if end_idx < len(cycle_numbers) else cycle_numbers[-1]

            anomalies.append({
                'type': 'time_gap',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_cycle': start_cycle,
                'end_cycle': end_cycle,
                'max_change': time_gap_value,  # Store time gap in hours
                'time_gap_hours': time_gap_value
            })

            i = end_idx + 1
            continue

        # Detect large increases
        if relative_changes[i] > THRESHOLD_INCREASE:
            start_idx = i - 1
            start_cycle = cycle_numbers[start_idx]
            max_change = relative_changes[i]
            pre_anomaly_soh = soh[start_idx]

            end_idx = find_stable_point(soh, relative_changes, i, pre_anomaly_soh)
            end_cycle = cycle_numbers[end_idx] if end_idx < len(cycle_numbers) else cycle_numbers[-1]

            anomalies.append({
                'type': 'increase',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_cycle': start_cycle,
                'end_cycle': end_cycle,
                'max_change': max_change
            })

            i = end_idx + 1
            continue

        # Detect large decreases
        if relative_changes[i] < THRESHOLD_DECREASE:
            start_idx = i - 1
            start_cycle = cycle_numbers[start_idx]
            max_change = relative_changes[i]
            pre_anomaly_soh = soh[start_idx]

            end_idx = find_stable_point(soh, relative_changes, i, pre_anomaly_soh)
            end_cycle = cycle_numbers[end_idx] if end_idx < len(cycle_numbers) else cycle_numbers[-1]

            anomalies.append({
                'type': 'decrease',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_cycle': start_cycle,
                'end_cycle': end_cycle,
                'max_change': max_change
            })

            i = end_idx + 1
            continue

        i += 1

    return anomalies


def apply_pchip_interpolation(soh: np.ndarray, soh_original: np.ndarray,
                               cycle_numbers: np.ndarray, anomaly: Dict) -> np.ndarray:
    """
    Fix anomaly region using PCHIP interpolation

    Args:
        soh: Current SOH array (may have been modified by previous fixes)
        soh_original: Original unmodified SOH array (for collecting reliable anchors)
        cycle_numbers: Cycle number array
        anomaly: Anomaly dictionary

    Returns:
        Corrected SOH array
    """
    corrected_soh = soh.copy()

    start_idx = anomaly['start_idx']
    end_idx = anomaly['end_idx']

    anchor_start = max(0, start_idx - ANCHOR_POINTS)
    anchor_end = min(len(soh) - 1, end_idx + ANCHOR_POINTS)

    anchor_indices = []
    anchor_cycles = []
    anchor_soh = []

    # Collect anchor points before anomaly using ORIGINAL SOH
    for i in range(anchor_start, start_idx):
        if soh_original[i] > 0:
            anchor_indices.append(i)
            anchor_cycles.append(cycle_numbers[i])
            anchor_soh.append(soh_original[i])  # Use original, no clip

    # Collect anchor points after anomaly using ORIGINAL SOH
    for i in range(end_idx + 1, anchor_end + 1):
        if soh_original[i] > 0:
            anchor_indices.append(i)
            anchor_cycles.append(cycle_numbers[i])
            anchor_soh.append(soh_original[i])  # Use original, no clip

    anchors_before = len([i for i in anchor_indices if i < start_idx])
    anchors_after = len([i for i in anchor_indices if i > end_idx])

    if len(anchor_soh) < 2:
        print(f"    Warning: Insufficient anchor points ({len(anchor_soh)}), using constant value")
        if len(anchor_soh) >= 1:
            corrected_soh[start_idx:end_idx+1] = anchor_soh[0]
        return corrected_soh

    # Handle case where anomaly extends to end of data (EOL)
    if anchors_after == 0 and anchors_before >= 2:
        print(f"    Warning: Anomaly extends to end of data, using degradation-aware extrapolation")

        # Use more anchor points for reliable trend estimation
        extended_anchor_start = max(0, start_idx - ANCHOR_POINTS * 2)
        reliable_soh = []
        reliable_cycles = []

        for i in range(extended_anchor_start, start_idx):
            if soh_original[i] > 0:
                reliable_soh.append(soh_original[i])  # Use original, no clip
                reliable_cycles.append(cycle_numbers[i])

        if len(reliable_soh) >= 3:
            # Calculate slopes between consecutive points
            slopes = []
            for j in range(len(reliable_soh) - 1):
                slope = (reliable_soh[j+1] - reliable_soh[j]) / \
                       (reliable_cycles[j+1] - reliable_cycles[j])
                slopes.append(slope)

            # Use median slope, ensure it's non-positive (battery degrades)
            median_slope = np.median(slopes)
            median_slope = min(median_slope, 0)  # Force degradation trend

            # Extrapolate from last reliable point
            for i in range(start_idx, end_idx + 1):
                extrapolated = reliable_soh[-1] + median_slope * \
                              (cycle_numbers[i] - reliable_cycles[-1])
                corrected_soh[i] = max(0.0, extrapolated)  # Only prevent negative values

            print(f"    Extrapolated using median slope: {median_slope:.6f}")
            return corrected_soh
        else:
            print(f"    Warning: Insufficient data for extrapolation, using last known value")
            if len(reliable_soh) >= 1:
                corrected_soh[start_idx:end_idx+1] = reliable_soh[-1]
            return corrected_soh

    # Perform PCHIP interpolation
    try:
        pchip = PchipInterpolator(anchor_cycles, anchor_soh)
        interpolated_cycles = cycle_numbers[start_idx:end_idx+1]
        interpolated_soh = pchip(interpolated_cycles)

        # No clip - preserve original range behavior
        # Only prevent negative values
        interpolated_soh = np.maximum(interpolated_soh, 0.0)

        corrected_soh[start_idx:end_idx+1] = interpolated_soh

        print(f"    PCHIP interpolation: {len(anchor_soh)} anchors ({anchors_before} before, {anchors_after} after)")

    except Exception as e:
        print(f"    Warning: PCHIP interpolation failed: {e}")
        # Fallback to linear interpolation
        if len(anchor_soh) >= 2:
            slope = (anchor_soh[-1] - anchor_soh[0]) / (anchor_cycles[-1] - anchor_cycles[0])
            for i in range(start_idx, end_idx + 1):
                corrected_soh[i] = anchor_soh[0] + slope * (cycle_numbers[i] - anchor_cycles[0])
                corrected_soh[i] = max(0.0, corrected_soh[i])  # Only prevent negative

    return corrected_soh


def process_battery(pkl_file: Path) -> Dict:
    """Process a single battery file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    cell_id = data['cell_id']
    cycle_numbers = np.array(data['cycle_numbers'])
    soh_original = np.array(data['SOH'])
    timestamps = np.array(data['cycle_start_time_in_s'])

    print(f"\nProcessing: {cell_id}")
    print(f"  Total cycles: {len(cycle_numbers)}, SOH range: {soh_original.min():.4f} - {soh_original.max():.4f}")

    # Detect anomalies
    anomalies = detect_anomalies(soh_original, cycle_numbers, timestamps)
    print(f"  Detected {len(anomalies)} anomaly regions:")

    if len(anomalies) == 0:
        print(f"    No anomalies, skipping")
        return {
            'cell_id': cell_id,
            'num_anomalies': 0,
            'processed': False
        }

    for idx, anomaly in enumerate(anomalies):
        if anomaly['type'] == 'time_gap':
            print(f"    [{idx+1}] {anomaly['type']}: cycle {anomaly['start_cycle']}-{anomaly['end_cycle']} "
                  f"({anomaly['end_idx'] - anomaly['start_idx'] + 1} cycles), "
                  f"time_gap={anomaly['time_gap_hours']:.2f}h")
        else:
            print(f"    [{idx+1}] {anomaly['type']}: cycle {anomaly['start_cycle']}-{anomaly['end_cycle']} "
                  f"({anomaly['end_idx'] - anomaly['start_idx'] + 1} cycles), "
                  f"max_change={anomaly['max_change']:.2f}%")

    # Apply PCHIP interpolation to fix anomalies
    soh_corrected = soh_original.copy()
    for anomaly in anomalies:
        soh_corrected = apply_pchip_interpolation(soh_corrected, soh_original, cycle_numbers, anomaly)

    # Prepare output data
    output_data = {
        'cell_id': cell_id,
        'cycle_numbers': cycle_numbers.tolist(),
        'SOH': soh_corrected.tolist(),
        'SOH_original': soh_original.tolist(),
        'cycle_start_time_in_s': timestamps.tolist(),
        'anomalies': anomalies,
        'processing_params': {
            'threshold_increase': THRESHOLD_INCREASE,
            'threshold_decrease': THRESHOLD_DECREASE,
            'threshold_time_gap': THRESHOLD_TIME_GAP,
            'stability_threshold': STABILITY_THRESHOLD,
            'stability_window': STABILITY_WINDOW,
            'anchor_points': ANCHOR_POINTS,
            'recovery_tolerance': RECOVERY_TOLERANCE
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
        'output_file': output_file
    }


def main():
    """Main function"""
    print("=" * 80)
    print("RWTH Dataset SOH PCHIP Interpolation Smoothing")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nParameter Configuration:")
    print(f"  Increase threshold: {THRESHOLD_INCREASE}%")
    print(f"  Decrease threshold: {THRESHOLD_DECREASE}%")
    print(f"  Time gap threshold: {THRESHOLD_TIME_GAP} hours")
    print(f"  Stability threshold: ±{STABILITY_THRESHOLD}%")
    print(f"  Stability window: {STABILITY_WINDOW} cycles")
    print(f"  Recovery tolerance: ±{RECOVERY_TOLERANCE} (absolute SOH)")
    print(f"  Anchor points: {ANCHOR_POINTS} cycles (before and after)")
    print(f"\nKey features:")
    print(f"  - Allow SOH > 1.0 (capacity recovery)")
    print(f"  - Use original SOH for anchor points (avoid bug)")
    print(f"  - No clip on interpolation (preserve original behavior)")
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

    total_processed = sum(1 for r in results if r['processed'])
    total_anomalies = sum(r['num_anomalies'] for r in results)

    print(f"Total batteries: {len(results)}")
    print(f"Processed: {total_processed}")
    print(f"Skipped (no anomalies): {len(results) - total_processed}")
    print(f"Total anomalies detected: {total_anomalies}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RWTH Dataset SOH preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input SOH directory for RWTH')
    parser.add_argument('--output', type=str, required=True, help='Output processed SOH directory for RWTH')
    args = parser.parse_args()
    INPUT_DIR = Path(args.input)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
