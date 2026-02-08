#!/usr/bin/env python3
"""
Convert ISU_ILCC raw JSON data to cleaned_data format.

This script reads raw cycling JSON files from ISU_ILCC dataset and converts them
to the standard cleaned_data pkl format used by the MemoryNet pipeline.

Key features:
- Correctly parses timestamps with nanosecond precision
- time_in_s stores absolute time relative to battery test start
- Preserves start_stop_time for RPT position identification

Usage:
    python convert_isu_ilcc_to_cleaned.py [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from batteryml import BatteryData, CycleData, CyclingProtocol
from tqdm import tqdm
import warnings
CYCLING_RATES = {
    'ISU-ILCC_G1': [0.5, 0.5],
    'ISU-ILCC_G2': [0.5, 0.5],
    'ISU-ILCC_G3': [0.5, 0.5],
    'ISU-ILCC_G4': [1, 0.5],
    'ISU-ILCC_G5': [1, 0.5],
    'ISU-ILCC_G6': [2, 0.5],
    'ISU-ILCC_G7': [2, 0.5],
    'ISU-ILCC_G8': [2, 0.5],
    'ISU-ILCC_G9': [2, 0.5],
    'ISU-ILCC_G10': [2.5, 0.5],
    'ISU-ILCC_G12': [3, 0.5],
    'ISU-ILCC_G13': [3, 0.5],
    'ISU-ILCC_G14': [3, 0.5],
    'ISU-ILCC_G15': [3, 0.5],
    'ISU-ILCC_G16': [0.5, 0.5],
    'ISU-ILCC_G17': [1, 0.5],
    'ISU-ILCC_G18': [2.5, 0.5],
    'ISU-ILCC_G19': [2.5, 0.5],
    'ISU-ILCC_G20': [0.8, 0.5],
    'ISU-ILCC_G21': [1.2, 0.5],
    'ISU-ILCC_G22': [1.4, 0.5],
    'ISU-ILCC_G23': [1.6, 0.5],
    'ISU-ILCC_G24': [1.8, 0.5],
    'ISU-ILCC_G25': [1.8, 0.6],
    'ISU-ILCC_G26': [1.4, 2.2],
    'ISU-ILCC_G27': [0.6, 2.4],
    'ISU-ILCC_G28': [2.4, 1.6],
    'ISU-ILCC_G29': [1.6, 1.8],
    'ISU-ILCC_G30': [0.8, 0.8],
    'ISU-ILCC_G31': [1.2, 1],
    'ISU-ILCC_G32': [1, 1.4],
    'ISU-ILCC_G33': [2, 1.2],
    'ISU-ILCC_G34': [2.2, 2],
    'ISU-ILCC_G35': [1.825, 0.5],
    'ISU-ILCC_G36': [2.075, 0.5],
    'ISU-ILCC_G37': [0.725, 0.5],
    'ISU-ILCC_G38': [1.875, 0.5],
    'ISU-ILCC_G39': [1.475, 0.5],
    'ISU-ILCC_G40': [1.825, 1.025],
    'ISU-ILCC_G41': [2.075, 1.775],
    'ISU-ILCC_G42': [0.725, 2.375],
    'ISU-ILCC_G43': [1.875, 2.325],
    'ISU-ILCC_G44': [0.775, 1.275],
    'ISU-ILCC_G45': [1.125, 1.725],
    'ISU-ILCC_G46': [1.225, 2.025],
    'ISU-ILCC_G47': [2.325, 1.925],
    'ISU-ILCC_G48': [2.375, 2.225],
    'ISU-ILCC_G49': [0.975, 0.675],
    'ISU-ILCC_G50': [2.425, 1.625],
    'ISU-ILCC_G51': [2.275, 1.875],
    'ISU-ILCC_G52': [1.425, 0.875],
    'ISU-ILCC_G53': [2.025, 0.825],
    'ISU-ILCC_G54': [0.925, 1.125],
    'ISU-ILCC_G55': [1.025, 2.475],
    'ISU-ILCC_G56': [2.175, 0.975],
    'ISU-ILCC_G57': [1.775, 1.175],
    'ISU-ILCC_G58': [2.475, 0.575],
    'ISU-ILCC_G59': [1.325, 1.825],
    'ISU-ILCC_G60': [0.675, 1.325],
    'ISU-ILCC_G61': [2.125, 1.975],
    'ISU-ILCC_G62': [1.575, 2.425],
    'ISU-ILCC_G63': [1.975, 1.675],
    'ISU-ILCC_G64': [1.175, 1.425],
}

# Default paths (can be overridden by command line arguments)
DEFAULT_BASE_DIR = Path('/data/trf/raw_ISU')
DEFAULT_OUTPUT_DIR = Path('/data/trf/python_works/BatteryLife/dataset')

# Battery specifications
NOMINAL_CAPACITY = 0.25  # 250 mAh = 0.25 Ah
MAX_VOLTAGE = 4.2
MIN_VOLTAGE = 3.0

# Release 2.0 cells (updated data)
RELEASE_2_CELLS = [
    'G57C1', 'G57C2', 'G57C3', 'G57C4', 'G58C1',
    'G26C3', 'G49C1', 'G49C2', 'G49C3', 'G49C4',
    'G50C1', 'G50C3', 'G50C4'
]


def load_cycling_json(cell_name, subfolder, cycling_dir):
    """Load cycling JSON data for a specific cell."""
    json_path = cycling_dir / subfolder / f'{cell_name}.json'

    with open(json_path, 'r') as f:
        data = json.loads(json.load(f))

    return data


def load_rpt_json(cell_name, subfolder, rpt_dir):
    """Load RPT JSON data for a specific cell."""
    json_path = rpt_dir / subfolder / f'{cell_name}.json'

    with open(json_path, 'r') as f:
        data = json.loads(json.load(f))

    return data


def parse_timestamp_fast(ts_str):
    """Fast timestamp parsing - handles '2021-06-12T16:26:42.000000000' format."""
    if not ts_str or ts_str == '[]' or ts_str == 'NaT':
        return None

    # Fast path: extract just the datetime part (first 19 chars)
    # Format: '2021-06-12T16:26:42.000000000'
    ts_str = str(ts_str)[:19]
    return datetime(
        int(ts_str[0:4]),   # year
        int(ts_str[5:7]),   # month
        int(ts_str[8:10]),  # day
        int(ts_str[11:13]), # hour
        int(ts_str[14:16]), # minute
        int(ts_str[17:19])  # second
    )



def batch_convert_timestamps(timestamps, ref_time):
    """Batch convert timestamps to seconds - vectorized for speed."""
    if not timestamps:
        return []

    result = []
    last_valid = 0.0

    for ts in timestamps:
        dt = parse_timestamp_fast(ts)
        if dt is not None and ref_time is not None:
            last_valid = (dt - ref_time).total_seconds()
        result.append(last_valid)

    return result

def calculate_soc_start_and_end(df, name, nominal_capacity=0.25):
    charge_start_soc, discharge_end_soc = {}, {}

    charge_capacity = df['capacity_charge'][1] # The first cycle is RPT. We should skip RPT and use cycling data to calculate SOC interval.
    soc_charge_interval = charge_capacity / nominal_capacity
    if soc_charge_interval > 1:
        soc_charge_interval = 1
    charge_start_soc[name] = 1 - soc_charge_interval

    discharge_capacity = df['capacity_discharge'][1]
    soc_discharge_interval = discharge_capacity / nominal_capacity
    if soc_discharge_interval > 1:
        soc_discharge_interval = 0
    discharge_end_soc[name] = 1 - soc_discharge_interval

    if charge_start_soc[name] == 1 and discharge_end_soc[name] == 1:
        charge_start_soc[name] = 0
        discharge_end_soc[name] = 1
    return charge_start_soc, discharge_end_soc

def get_cycling_cycles_impacted_by_RPT(data_dict_cycle, data_dict_rpt):
    '''
    The cycles following RPT cycles have different cycling data. It will be charged from 0% SOC to 100% SOC.
    Thus, we detect these cycles and removed them from the cycling data.
    '''
    i = 0
    should_exclude_cycle_numbers = []
    # data_dict_cycle = convert_cycling_to_dict(zip_path, cell, subfolder)
    # data_dict_rpt = convert_RPT_to_dict(zip_path, cell, subfolder)
    cycle_start = data_dict_cycle['QV_discharge']['t']

    for cycle_number in range(len(cycle_start)):
        current_cycle_start = data_dict_cycle['QV_discharge']['t'][cycle_number][0]
        if i>= len(data_dict_rpt['start_stop_time']['start']) or not data_dict_rpt['start_stop_time']['start'][i]:
            continue
        elif data_dict_rpt['start_stop_time']['start'][i] < current_cycle_start:
            should_exclude_cycle_numbers.append(cycle_number+1)
            i = i + 1
    return should_exclude_cycle_numbers

def convert_cell(args):
    """Convert a single cell's data to cleaned_data format."""
    cell_name, subfolder, cycling_dir, rpt_dir = args
    name = 'ISU-ILCC_' + cell_name # e.g. ISU-ILCC_G1C1
    if cell_name == 'G42C4' or cell_name == 'G9C4' or cell_name == 'G25C4' or 'G26' in cell_name or 'G11' in cell_name:
        return None, None, None

    # Load raw data
    cycling_data = load_cycling_json(cell_name, subfolder, cycling_dir)
    rpt_data = load_rpt_json(cell_name, subfolder, rpt_dir)
    should_exclude_cycle_numbers = get_cycling_cycles_impacted_by_RPT(cycling_data, rpt_data)
    
    charge_start_soc, discharge_end_soc = calculate_soc_start_and_end(cycling_data, name)
    # Calculate SOC interval from first cycle's discharge capacity
    # DOD = DOC = SOC_interval[1] - SOC_interval[0]
    soc_interval = [charge_start_soc[name], 1.0]
    dod = soc_interval[1] - soc_interval[0] # Depth of discharge
    rates = CYCLING_RATES[name[:-2]]
    charge_protocol = [CyclingProtocol(
        rate_in_C=float(rates[0]), start_soc=charge_start_soc[name], end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C=float(rates[1]), start_soc=1.0, end_soc=discharge_end_soc[name]
    )]
    # Get reference time (first valid timestamp in the entire dataset)
    ref_time = None
    for ts in cycling_data['time_series_charge']['start']:
        ref_time = parse_timestamp_fast(ts)
        if ref_time is not None:
            break

    if ref_time is None:
        # Try QV_charge timestamps
        for cycle_ts in cycling_data['QV_charge']['t']:
            if cycle_ts:
                ref_time = parse_timestamp_fast(cycle_ts[0])
                if ref_time is not None:
                    break

    # Initialize cleaned data structure
    cleaned = {
        'cell_id': f'ISU-ILCC_{cell_name}',
        'form_factor': '502030-size Li-polymer',
        'anode_material': 'graphite',
        'cathode_material': 'NMC',
        'electrolyte_material': None,
        'nominal_capacity_in_Ah': NOMINAL_CAPACITY,
        'depth_of_charge': dod,  
        'depth_of_discharge': dod,  
        'already_spent_cycles': 0,
        'max_voltage_limit_in_V': MAX_VOLTAGE,
        'min_voltage_limit_in_V': MIN_VOLTAGE,
        'max_current_limit_in_A': None,
        'min_current_limit_in_A': None,
        'reference': None,
        'description': None,
        'charge_protocol': charge_protocol,
        'discharge_protocol': discharge_protocol,
        'SOC_interval': soc_interval,  # [charge_start_soc, 1.0] calculated from first cycle
        'cycle_data': [],
        # Store weekly start/stop times for RPT identification
        'weekly_start_stop_time': cycling_data.get('start_stop_time', {}),
        # Store per-cycle start times for reference
        'cycle_start_times': {
            'charge_start': cycling_data['time_series_charge']['start'],
            'charge_stop': cycling_data['time_series_charge']['stop'],
            'discharge_start': cycling_data['time_series_discharge']['start'],
            'discharge_stop': cycling_data['time_series_discharge']['stop'],
        }
    }

    # Get number of cycles
    n_cycles = len(cycling_data['capacity_discharge'])
    cycle_number = 0
    # Convert each cycle
    for cycle_idx in range(n_cycles):
        tmp_cycle_number = cycle_idx + 1
        if tmp_cycle_number in should_exclude_cycle_numbers:
            # Skip cycles impacted by RPT
            continue

        cycle_number = cycle_number + 1

        # Get charge data
        charge_Q = cycling_data['QV_charge']['Q'][cycle_idx] if cycle_idx < len(cycling_data['QV_charge']['Q']) else []
        charge_V = cycling_data['QV_charge']['V'][cycle_idx] if cycle_idx < len(cycling_data['QV_charge']['V']) else []
        charge_I = cycling_data['QV_charge']['I'][cycle_idx] if cycle_idx < len(cycling_data['QV_charge']['I']) else []
        charge_t = cycling_data['QV_charge']['t'][cycle_idx] if cycle_idx < len(cycling_data['QV_charge']['t']) else []

        # Get discharge data
        discharge_Q = cycling_data['QV_discharge']['Q'][cycle_idx] if cycle_idx < len(cycling_data['QV_discharge']['Q']) else []
        discharge_V = cycling_data['QV_discharge']['V'][cycle_idx] if cycle_idx < len(cycling_data['QV_discharge']['V']) else []
        discharge_I = cycling_data['QV_discharge']['I'][cycle_idx] if cycle_idx < len(cycling_data['QV_discharge']['I']) else []
        discharge_t = cycling_data['QV_discharge']['t'][cycle_idx] if cycle_idx < len(cycling_data['QV_discharge']['t']) else []

        # Skip cycles with no data
        if not charge_V and not discharge_V:
            continue

        # Convert timestamps to absolute seconds from reference time (fast batch)
        charge_time_s = batch_convert_timestamps(charge_t, ref_time)
        discharge_time_s = batch_convert_timestamps(discharge_t, ref_time)

        # Combine charge and discharge data
        combined_V = list(charge_V) + list(discharge_V)
        combined_I = list(charge_I) + list(discharge_I)
        combined_time = list(charge_time_s) + list(discharge_time_s)

        # For charge capacity, use charge Q values, then hold at max
        charge_cap = list(charge_Q)
        if charge_cap:
            max_charge = max(charge_cap) if charge_cap else 0
            charge_cap.extend([max_charge] * len(discharge_Q))
        else:
            charge_cap = [0.0] * (len(charge_Q) + len(discharge_Q))

        # For discharge capacity, start at 0 during charge, then use discharge Q
        discharge_cap = [0.0] * len(charge_Q)
        discharge_cap.extend(list(discharge_Q))

        # Ensure all arrays have same length
        min_len = min(len(combined_V), len(combined_I), len(combined_time),
                        len(charge_cap), len(discharge_cap))

        if min_len == 0:
            continue

        cycle = {
            'cycle_number': cycle_number,
            'voltage_in_V': combined_V[:min_len],
            'current_in_A': combined_I[:min_len],
            'charge_capacity_in_Ah': charge_cap[:min_len],
            'discharge_capacity_in_Ah': discharge_cap[:min_len],
            'time_in_s': combined_time[:min_len],
            'temperature_in_C': None,
            'internal_resistance_in_ohm': None
        }

        cleaned['cycle_data'].append(cycle)

    return cell_name, cleaned, None



def get_valid_cells(base_dir):
    """Get list of valid cells from Valid_cells.csv."""
    csv_path = base_dir / 'Valid_cells.csv'
    df = pd.read_csv(csv_path)
    return df.values.flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description='Convert ISU_ILCC JSON to cleaned_data format')
    parser.add_argument('--input_dir', type=str, default=str(DEFAULT_BASE_DIR),
                       help='Input directory containing ISU_ILCC raw data (Cycling_json, RPT_json, Valid_cells.csv)')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help='Output directory for cleaned_data pkl files')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of parallel workers')
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    cycling_dir = base_dir / 'Cycling_json'
    rpt_dir = base_dir / 'RPT_json'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ISU_ILCC JSON to cleaned_data Converter")
    print("=" * 60)
    print(f"Input:  {base_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Get all valid cells
    valid_cells = get_valid_cells(base_dir)
    print(f"Found {len(valid_cells)} valid cells")

    # Prepare conversion tasks
    tasks = []
    for cell_name in valid_cells:
        if cell_name in RELEASE_2_CELLS:
            subfolder = 'Release 2.0'
        else:
            subfolder = 'Release 1.0'
        tasks.append((cell_name, subfolder, cycling_dir, rpt_dir))

    # Convert cells in parallel
    success_count = 0
    error_count = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_cell, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            cell_name, cleaned_data, error = future.result()
            if cleaned_data is None:
                # The battery should be skiped
                error_count += 1
            elif cleaned_data:
                # Save cleaned data
                output_path = output_dir / f'ISU-ILCC_{cell_name}.pkl'
                with open(output_path, 'wb') as f:
                    pickle.dump(cleaned_data, f)
                success_count += 1

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Successfully converted: {success_count}/{len(tasks)}")
    print(f"  Errors: {error_count}")

    if errors:
        print("\nErrors:")
        for cell_name, err in errors[:5]:
            print(f"  {cell_name}: {err[:100]}...")

    print("=" * 60)


if __name__ == '__main__':
    main()
