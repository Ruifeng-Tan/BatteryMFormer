"""
SNL Dataset SOH PCHIP Interpolation Smoothing Script

Features:
1. Detect anomalies in SOH curves (test-induced increases/decreases)
2. Fix anomalies using PCHIP interpolation
3. Record anomaly cycle information
4. Save processed data to processed_SOH directory

"""

import pickle
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from typing import Dict, List

# Parameter Configuration (Conservative Approach)
THRESHOLD_INCREASE = 0.05      # 1.0% increase threshold (was 0.01, too small!)
THRESHOLD_DECREASE = -0.5
STABILITY_THRESHOLD = 0.5
STABILITY_WINDOW = 10
ANCHOR_POINTS = 5
RECOVERY_TOLERANCE = 0.02     # SOH needs to recover within 2% of pre-anomaly level
MAX_ANOMALY_LENGTH = 100       # Maximum length of an anomaly region (prevent avalanche)

# Path Configuration (set via command-line arguments)
INPUT_DIR: Path = None
OUTPUT_DIR: Path = None


def calculate_relative_changes(soh: np.ndarray) -> np.ndarray:
    """Calculate relative change rates"""
    relative_changes = np.zeros(len(soh))
    for i in range(1, len(soh)):
        if soh[i-1] != 0:
            relative_changes[i] = (soh[i] - soh[i-1]) / soh[i-1] * 100
        else:
            relative_changes[i] = 0
    return relative_changes


def find_stable_point(soh: np.ndarray, relative_changes: np.ndarray, start_idx: int,
                      pre_anomaly_soh: float) -> int:
    """
    Find the point where SOH stabilizes after anomaly

    Args:
        soh: SOH array
        relative_changes: relative change rates array
        start_idx: index where anomaly started
        pre_anomaly_soh: SOH value before anomaly (for recovery level check)

    Returns:
        Index where SOH has stabilized (either recovered or reached new stable level)

    Note:
        - Has MAX_ANOMALY_LENGTH limit to prevent excessive interpolation
        - If anomaly region exceeds limit, returns limit to avoid "avalanche effect"
    """
    i = start_idx + 1

    while i < len(soh):
        # IMPORTANT: Limit anomaly region length to prevent avalanche effect
        if i - start_idx > MAX_ANOMALY_LENGTH:
            # Return a reasonable end point, not the whole tail
            return start_idx + min(MAX_ANOMALY_LENGTH, 5)

        end_window = min(i + STABILITY_WINDOW, len(soh))
        window_changes = relative_changes[i:end_window]

        # Check if stable (consecutive cycles with small change rates)
        is_stable = (len(window_changes) >= min(STABILITY_WINDOW, len(soh) - i) and
                     np.all(np.abs(window_changes) <= STABILITY_THRESHOLD) and
                     np.all(soh[i:end_window] > 0))

        if is_stable:
            # Check if SOH has recovered to pre-anomaly level (within tolerance)
            current_soh = soh[i]
            if abs(current_soh - pre_anomaly_soh) <= RECOVERY_TOLERANCE:
                # Case 1: Recovered to original level (test-induced anomaly)
                return i - 1
            else:
                # Case 2: Stable at new level (permanent change, not test-induced)
                # Return current stable point instead of searching to end
                return i - 1

        i += 1

    # If we reach here, limit the anomaly region
    return min(start_idx + 5, len(soh) - 1)


def detect_anomalies(soh: np.ndarray, cycle_numbers: np.ndarray) -> List[Dict]:
    """Detect anomalies in SOH curve"""
    anomalies = []
    relative_changes = calculate_relative_changes(soh)

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


def apply_pchip_interpolation(soh: np.ndarray, cycle_numbers: np.ndarray,
                               anomaly: Dict) -> np.ndarray:
    """Fix anomaly region using PCHIP interpolation

    Special handling for tail anomalies:
    - If anomaly is at the tail (no valid anchor points after), use linear
      extrapolation based on pre-anomaly trend instead of PCHIP
    - This prevents PCHIP from creating unrealistic upward trends at the end
    """
    corrected_soh = soh.copy()

    start_idx = anomaly['start_idx']
    end_idx = anomaly['end_idx']

    anchor_start = max(0, start_idx - ANCHOR_POINTS)
    anchor_end = min(len(soh) - 1, end_idx + ANCHOR_POINTS)

    # Collect anchor points BEFORE anomaly
    anchor_indices_before = []
    anchor_cycles_before = []
    anchor_soh_before = []
    for i in range(anchor_start, start_idx):
        if soh[i] > 0:
            anchor_indices_before.append(i)
            anchor_cycles_before.append(cycle_numbers[i])
            anchor_soh_before.append(soh[i])

    # Collect anchor points AFTER anomaly
    anchor_indices_after = []
    anchor_cycles_after = []
    anchor_soh_after = []
    for i in range(end_idx + 1, anchor_end + 1):
        if soh[i] > 0:
            anchor_indices_after.append(i)
            anchor_cycles_after.append(cycle_numbers[i])
            anchor_soh_after.append(soh[i])

    # Check if this is a TAIL ANOMALY (no valid anchors after, or anomaly extends to end)
    is_tail_anomaly = (len(anchor_soh_after) == 0) or (end_idx >= len(soh) - 3)

    if is_tail_anomaly:
        # TAIL ANOMALY: Use linear extrapolation from pre-anomaly trend
        # This maintains the degradation trend instead of creating upward spikes
        print(f"    -> Tail anomaly detected, using linear extrapolation")

        if len(anchor_soh_before) >= 2:
            # Calculate trend from pre-anomaly points
            x_before = np.array(anchor_cycles_before)
            y_before = np.array(anchor_soh_before)

            # Use last few points to estimate slope (more recent trend)
            n_trend = min(len(x_before), 10)
            x_trend = x_before[-n_trend:]
            y_trend = y_before[-n_trend:]

            # Linear regression for slope
            slope = np.polyfit(x_trend, y_trend, 1)[0]

            # Ensure slope is non-positive (SOH should not increase)
            slope = min(slope, 0)

            # Extrapolate from last good point
            last_good_soh = anchor_soh_before[-1]
            last_good_cycle = anchor_cycles_before[-1]

            for i in range(start_idx, end_idx + 1):
                cycle_diff = cycle_numbers[i] - last_good_cycle
                new_soh = last_good_soh + slope * cycle_diff
                # Clip to reasonable range
                corrected_soh[i] = np.clip(new_soh, 0.5, last_good_soh)
        elif len(anchor_soh_before) == 1:
            # Only one anchor point, use constant value
            corrected_soh[start_idx:end_idx+1] = anchor_soh_before[0]
        else:
            # No anchor points before, keep original (shouldn't happen normally)
            print(f"    Warning: No anchor points available for tail anomaly")

        return corrected_soh

    # NORMAL CASE: Have anchor points both before and after
    anchor_indices = anchor_indices_before + anchor_indices_after
    anchor_cycles = anchor_cycles_before + anchor_cycles_after
    anchor_soh = anchor_soh_before + anchor_soh_after

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
        # FIXED: Don't clip to 0.5, use min of anchor_soh as lower bound
        # This prevents creating flat steps when SOH is below 0.5
        min_anchor = min(anchor_soh) - 0.05  # Allow some margin below anchors
        interpolated_soh = np.clip(interpolated_soh, max(0.0, min_anchor), 1.5)

        # Verify trend consistency: interpolated values should not create upward trend
        # if the overall trend is downward
        if len(anchor_soh_before) >= 2 and len(anchor_soh_after) >= 1:
            pre_trend = anchor_soh_before[-1] - anchor_soh_before[0]
            if pre_trend < 0:  # Downward trend before anomaly
                # Ensure interpolated values don't exceed the starting anchor
                max_allowed = anchor_soh_before[-1] + 0.01
                interpolated_soh = np.clip(interpolated_soh, 0.0, max_allowed)

        corrected_soh[start_idx:end_idx+1] = interpolated_soh

    except Exception as e:
        print(f"    Warning: PCHIP interpolation failed: {e}")
        if len(anchor_soh) >= 2:
            slope = (anchor_soh[-1] - anchor_soh[0]) / (anchor_cycles[-1] - anchor_cycles[0])
            min_anchor = min(anchor_soh) - 0.05
            for i in range(start_idx, end_idx + 1):
                corrected_soh[i] = anchor_soh[0] + slope * (cycle_numbers[i] - anchor_cycles[0])
                corrected_soh[i] = np.clip(corrected_soh[i], max(0.0, min_anchor), 1.5)

    return corrected_soh


def truncate_tail_rpt(soh: np.ndarray, cycle_numbers: np.ndarray,
                      cycle_start_time_in_s: list) -> tuple:
    """
    Detect and truncate tail RPT (Reference Performance Test) regions.

    RPT tests cause SOH to suddenly jump UP near the end of battery life.
    These should be truncated, not interpolated.

    Returns:
        Truncated (soh, cycle_numbers, cycle_start_time_in_s)
    """
    if len(soh) < 20:
        return soh, cycle_numbers, cycle_start_time_in_s

    # Look at the last 30 cycles for RPT patterns
    check_region = min(30, len(soh) // 3)

    # Find large upward jumps in the tail region
    for i in range(len(soh) - check_region, len(soh) - 1):
        if i < 1:
            continue
        diff = soh[i] - soh[i-1]
        # Large upward jump (>5%) indicates RPT start
        if diff > 0.05:
            # Check if this is followed by high SOH values (RPT region)
            remaining = soh[i:]
            # If remaining values are significantly higher than pre-jump value
            if np.mean(remaining) > soh[i-1] + 0.03:
                # Truncate at the point before the jump
                truncate_idx = i
                print(f"    -> Truncating tail RPT region at cycle {cycle_numbers[truncate_idx]} "
                      f"(SOH jumped from {soh[i-1]:.4f} to {soh[i]:.4f})")

                soh = soh[:truncate_idx]
                cycle_numbers = cycle_numbers[:truncate_idx]
                if len(cycle_start_time_in_s) >= truncate_idx:
                    cycle_start_time_in_s = cycle_start_time_in_s[:truncate_idx]
                break

    return soh, cycle_numbers, cycle_start_time_in_s


def process_battery(pkl_file: Path) -> Dict:
    """Process a single battery file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    cell_id = data['cell_id']
    cycle_numbers = np.array(data['cycle_numbers'])
    soh_original = np.array(data['SOH'])
    cycle_start_time_in_s = data.get('cycle_start_time_in_s', [])

    print(f"\nProcessing: {cell_id}")
    print(f"  Total cycles: {len(cycle_numbers)}, SOH range: {soh_original.min():.4f} - {soh_original.max():.4f}")

    # FIRST: Truncate tail RPT regions before anomaly detection
    soh_truncated, cycle_numbers_truncated, time_truncated = truncate_tail_rpt(
        soh_original.copy(), cycle_numbers.copy(),
        list(cycle_start_time_in_s) if cycle_start_time_in_s else []
    )

    # Update arrays if truncation happened
    if len(soh_truncated) < len(soh_original):
        print(f"  After RPT truncation: {len(soh_truncated)} cycles, "
              f"SOH range: {soh_truncated.min():.4f} - {soh_truncated.max():.4f}")
        soh_original = soh_truncated
        cycle_numbers = cycle_numbers_truncated
        cycle_start_time_in_s = time_truncated

    anomalies = detect_anomalies(soh_original, cycle_numbers)
    print(f"  Detected {len(anomalies)} anomaly regions:")

    if len(anomalies) == 0:
        print(f"    No anomalies, skipping")
        return {
            'cell_id': cell_id,
            'num_anomalies': 0,
            'processed': False
        }

    for idx, anomaly in enumerate(anomalies):
        print(f"    [{idx+1}] {anomaly['type']}: cycle {anomaly['start_cycle']}-{anomaly['end_cycle']} "
              f"({anomaly['end_idx'] - anomaly['start_idx'] + 1} cycles), "
              f"max_change={anomaly['max_change']:.2f}%")

    soh_corrected = soh_original.copy()
    for anomaly in anomalies:
        soh_corrected = apply_pchip_interpolation(soh_corrected, cycle_numbers, anomaly)

    output_data = {
        'cell_id': cell_id,
        'cycle_numbers': cycle_numbers.tolist(),
        'SOH': soh_corrected.tolist(),
        'SOH_original': soh_original.tolist(),
        'cycle_start_time_in_s': cycle_start_time_in_s,
        'anomalies': anomalies,
        'processing_params': {
            'threshold_increase': THRESHOLD_INCREASE,
            'threshold_decrease': THRESHOLD_DECREASE,
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
    print("SNL Dataset SOH PCHIP Interpolation Smoothing")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nParameter Configuration:")
    print(f"  Increase threshold: {THRESHOLD_INCREASE}%")
    print(f"  Decrease threshold: {THRESHOLD_DECREASE}%")
    print(f"  Stability threshold: ±{STABILITY_THRESHOLD}%")
    print(f"  Stability window: {STABILITY_WINDOW} cycles")
    print(f"  Recovery tolerance: ±{RECOVERY_TOLERANCE} (absolute SOH)")
    print(f"  Anchor points: {ANCHOR_POINTS} cycles (before and after)")
    print("=" * 80)

    pkl_files = sorted(list(INPUT_DIR.glob('*.pkl')))
    print(f"\nFound {len(pkl_files)} battery files")

    results = []

    for pkl_file in pkl_files:
        result = process_battery(pkl_file)
        results.append(result)

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
    parser = argparse.ArgumentParser(description='SNL Dataset SOH preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input SOH directory for SNL')
    parser.add_argument('--output', type=str, required=True, help='Output processed SOH directory for SNL')
    args = parser.parse_args()
    INPUT_DIR = Path(args.input)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
