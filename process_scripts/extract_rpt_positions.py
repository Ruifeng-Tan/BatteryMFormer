#!/usr/bin/env python3
"""
Extract RPT (Reference Performance Test) positions from ISU_ILCC dataset.

This script reads RPT and Cycling JSON files and determines the cycle indices
where each RPT test occurred by matching timestamps.

Output: all_rpt_positions.json with RPT position information for all cells.

Usage:
    python extract_rpt_positions.py [--output OUTPUT_FILE]

"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings


# Configuration
BASE_DIR = Path('/ai/dl_project/MemoryNet/dataset/ISU_ILCC')
CYCLING_DIR = BASE_DIR / 'Cycling_json'
RPT_DIR = BASE_DIR / 'RPT_json'
OUTPUT_FILE = Path('/ai/dl_project/MemoryNet/process_scripts/all_rpt_positions.json')

# Release 2.0 cells (updated data)
RELEASE_2_CELLS = [
    'G57C1', 'G57C2', 'G57C3', 'G57C4', 'G58C1',
    'G26C3', 'G49C1', 'G49C2', 'G49C3', 'G49C4',
    'G50C1', 'G50C3', 'G50C4'
]


def load_json_data(json_path):
    """Load JSON data from file."""
    with open(json_path, 'r') as f:
        data = json.loads(json.load(f))
    return data


def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object."""
    if ts_str == '[]' or not ts_str or ts_str is None:
        return None
    try:
        # Handle different timestamp formats
        if isinstance(ts_str, list):
            return None
        if 'T' in ts_str:
            # Remove nanoseconds if present
            ts_clean = ts_str.split('.')[0]
            return datetime.fromisoformat(ts_clean)
        else:
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError, AttributeError):
        return None


def extract_rpt_positions_for_cell(args):
    """Extract RPT positions for a single cell."""
    cell_name, subfolder = args

    try:
        # Load RPT and Cycling data
        rpt_path = RPT_DIR / subfolder / f'{cell_name}.json'
        cycling_path = CYCLING_DIR / subfolder / f'{cell_name}.json'

        if not rpt_path.exists() or not cycling_path.exists():
            return cell_name, None, f"File not found: {rpt_path} or {cycling_path}"

        rpt_data = load_json_data(rpt_path)
        cycling_data = load_json_data(cycling_path)

        # Get RPT start times
        rpt_start_times = rpt_data['start_stop_time']['start']
        rpt_stop_times = rpt_data['start_stop_time']['stop']
        rpt_capacities = rpt_data.get('capacity_discharge_C_5', [])

        # Get cycling discharge start times for each cycle
        cycle_start_times = cycling_data['time_series_discharge']['start']
        total_cycles = len(cycling_data['capacity_discharge'])

        # Parse all timestamps
        rpt_times_parsed = []
        for i, ts in enumerate(rpt_start_times):
            dt = parse_timestamp(ts)
            cap = rpt_capacities[i] if i < len(rpt_capacities) else None
            # Handle capacity being a list (empty data)
            if isinstance(cap, list):
                cap = None
            rpt_times_parsed.append({
                'rpt_index': i,
                'start_time': dt,
                'stop_time': parse_timestamp(rpt_stop_times[i]) if i < len(rpt_stop_times) else None,
                'capacity_c5': cap
            })

        cycle_times_parsed = []
        for i, ts in enumerate(cycle_start_times):
            dt = parse_timestamp(ts)
            cycle_times_parsed.append({
                'cycle_index': i,
                'start_time': dt
            })

        # Match RPT positions to cycles
        rpt_after_cycle_indices = []
        rpt_positions = []

        for rpt_info in rpt_times_parsed:
            rpt_start = rpt_info['start_time']

            if rpt_start is None:
                # RPT data missing, skip
                continue

            # Find the last cycle that started before this RPT
            cycles_before_rpt = 0
            for cycle_info in cycle_times_parsed:
                if cycle_info['start_time'] is None:
                    continue
                if cycle_info['start_time'] < rpt_start:
                    cycles_before_rpt = cycle_info['cycle_index'] + 1
                else:
                    break

            rpt_after_cycle_indices.append(cycles_before_rpt)
            rpt_positions.append({
                'rpt_index': rpt_info['rpt_index'],
                'rpt_start_time': rpt_info['start_time'].isoformat() if rpt_info['start_time'] else None,
                'rpt_stop_time': rpt_info['stop_time'].isoformat() if rpt_info['stop_time'] else None,
                'cycles_before_rpt': cycles_before_rpt,
                'capacity_c5': rpt_info['capacity_c5'],
                'cycles_in_interval': rpt_after_cycle_indices[-1] - rpt_after_cycle_indices[-2] if len(rpt_after_cycle_indices) > 1 else cycles_before_rpt
            })

        result = {
            'cell_id': f'ISU-ILCC_{cell_name}',
            'total_rpt_count': len(rpt_positions),
            'total_cycling_count': total_cycles,
            'rpt_after_cycle_indices': rpt_after_cycle_indices,
            'rpt_positions': rpt_positions
        }

        return cell_name, result, None

    except Exception as e:
        import traceback
        return cell_name, None, f"{str(e)}\n{traceback.format_exc()}"


def get_valid_cells():
    """Get list of valid cells from Valid_cells.csv."""
    csv_path = BASE_DIR / 'Valid_cells.csv'
    df = pd.read_csv(csv_path)
    return df.values.flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description='Extract RPT positions from ISU_ILCC dataset')
    parser.add_argument('--output', type=str, default=str(OUTPUT_FILE),
                       help='Output JSON file path')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of parallel workers')
    args = parser.parse_args()

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ISU_ILCC RPT Position Extractor")
    print("=" * 60)
    print(f"RPT Dir:     {RPT_DIR}")
    print(f"Cycling Dir: {CYCLING_DIR}")
    print(f"Output:      {output_file}")
    print("=" * 60)

    # Get all valid cells
    valid_cells = get_valid_cells()
    print(f"Found {len(valid_cells)} valid cells")

    # Prepare extraction tasks
    tasks = []
    for cell_name in valid_cells:
        if cell_name in RELEASE_2_CELLS:
            subfolder = 'Release 2.0'
        else:
            subfolder = 'Release 1.0'
        tasks.append((cell_name, subfolder))

    # Extract RPT positions in parallel
    all_rpt_data = {}
    success_count = 0
    error_count = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_rpt_positions_for_cell, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            cell_name, result, error = future.result()

            if error:
                error_count += 1
                errors.append((cell_name, error))
            elif result:
                cell_id = result['cell_id']
                all_rpt_data[cell_id] = result
                success_count += 1

    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_rpt_data, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Successfully extracted: {success_count}/{len(tasks)}")
    print(f"  Errors: {error_count}")
    print(f"  Output saved to: {output_file}")

    # Print some statistics
    if all_rpt_data:
        total_rpts = sum(d['total_rpt_count'] for d in all_rpt_data.values())
        avg_rpts = total_rpts / len(all_rpt_data)
        print(f"\n  Total RPTs across all cells: {total_rpts}")
        print(f"  Average RPTs per cell: {avg_rpts:.1f}")

        # Sample output for verification
        sample_cell = list(all_rpt_data.keys())[0]
        sample_data = all_rpt_data[sample_cell]
        print(f"\n  Sample: {sample_cell}")
        print(f"    Total RPTs: {sample_data['total_rpt_count']}")
        print(f"    Total cycles: {sample_data['total_cycling_count']}")
        print(f"    RPT positions: {sample_data['rpt_after_cycle_indices'][:5]}...")

    if errors:
        print("\nErrors:")
        for cell_name, err in errors[:5]:
            print(f"  {cell_name}: {err[:100]}...")

    print("=" * 60)


if __name__ == '__main__':
    main()
