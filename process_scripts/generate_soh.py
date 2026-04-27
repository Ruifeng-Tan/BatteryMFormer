import os
import sys
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import traceback
import copy
import matplotlib.pyplot as plt



def fix_spike_drops(raw_sohs, max_drop_per_cycle=0.04):
    """
    Fix abnormal spike drops in SOH trajectory.
    """
    if len(raw_sohs) <= 1:
        return raw_sohs, []

    fixed_sohs = list(raw_sohs)
    spike_indices = []

    for i in range(1, len(raw_sohs)):
        drop = raw_sohs[i-1] - raw_sohs[i]

        if drop > max_drop_per_cycle:
            fixed_sohs[i] = fixed_sohs[i-1]
            spike_indices.append(i)

    return fixed_sohs, spike_indices


# Excluded cells list
EXCLUDED_CELLS = {
    'ISU-ILCC_G26C1.pkl', 'ISU-ILCC_G26C2.pkl', 'ISU-ILCC_G26C3.pkl', 'ISU-ILCC_G26C4.pkl',
    'ISU-ILCC_G11C1.pkl', 'ISU-ILCC_G11C2.pkl', 'ISU-ILCC_G11C3.pkl', 'ISU-ILCC_G11C4.pkl',
    'ISU-ILCC_G42C4.pkl', 'ISU-ILCC_G9C4.pkl', 'ISU-ILCC_G25C4.pkl', 'ISU-ILCC_G40C3.pkl'
}


def process_one_cell(args):
    """
    Process a single cell.
    Returns: (cell_name, skip_reason)
             skip_reason is None if successful, otherwise a string string.
    """
    try:
        cleaned_data_path, output_path, cell_name = args

        # 1. Check Exclusion List
        if cell_name in EXCLUDED_CELLS:
            return cell_name, "In EXCLUDED_CELLS list"

        dataset_name = cell_name.split('_')[0]
        if dataset_name == 'MICH':
            dataset_name = 'total_MICH'
        filter_threshold = 0.925 if dataset_name == 'CALB' else 0.825
        eol_threshold = 0.9 if dataset_name == 'CALB' else 0.8

        is_isu_ilcc = 'ISU' in cell_name or 'ILCC' in cell_name
        cell_path = os.path.join(cleaned_data_path, cell_name)

        with open(cell_path, 'rb') as f:
            data = pickle.load(f)

        nominal_capacity = data['nominal_capacity_in_Ah']
        nominal_capacity = 1.85 if 'RWTH' in cell_name else nominal_capacity
        discharge_depth = data['SOC_interval'][1] - data['SOC_interval'][0]

        cycle_start_datetime_list = []
        if is_isu_ilcc:
            cycle_start_times = data.get('cycle_start_times', {})
            cycle_start_datetime_list = cycle_start_times.get('discharge_start', [])

        raw_sohs = []
        cycle_numbers = []
        cycle_start_time_in_s_list = []
        cycle_data = data['cycle_data']
        
        prefix = cell_name.split('_')[0]
        if prefix == 'CALB':
            prefix = cell_name.split('_')[:2]
            prefix = '_'.join(prefix)
            
        for index, one_cycle_data in enumerate(cycle_data):
            # current = np.array(one_cycle_data['current_in_A'])
            # current_c_rate = current / nominal_capacity
            
            discharge_capacity_in_Ah = np.array(one_cycle_data['discharge_capacity_in_Ah'])

            cycle_start_time_in_s = one_cycle_data['time_in_s'][0]
            soh = max(discharge_capacity_in_Ah) / nominal_capacity
            soh = soh / discharge_depth
            raw_sohs.append(soh)
            cycle_numbers.append(one_cycle_data['cycle_number'])
            cycle_start_time_in_s_list.append(cycle_start_time_in_s)

        # 2. Check if data exists
        if not raw_sohs:
            return cell_name, "No SOH data generated (empty cycle data)"

        if 'ZN-coin_441-1_20231227204855_08_4.' in cell_name:
            # The fix might cause inappropriate fixes on this cell and special handling is needed
            new_raw_sohs, spike_indices = fix_spike_drops(raw_sohs[:-5], max_drop_per_cycle=0.03)
            new_raw_sohs.extend(raw_sohs[-5:])
            raw_sohs = new_raw_sohs
        else:
            raw_sohs, spike_indices = fix_spike_drops(raw_sohs, max_drop_per_cycle=0.03)


        # 3. Check Filter Threshold (Battery too healthy)
        if raw_sohs[-1] > filter_threshold:
            return cell_name, f"Final SOH {raw_sohs[-1]:.4f} > threshold {filter_threshold}"

        if raw_sohs[-1] <= eol_threshold:
            eol = 0
            for cycle_number, soh in zip(cycle_numbers, raw_sohs):
                if soh <= eol_threshold:
                    eol = cycle_number
                    break
            eol = int(eol)
            raw_sohs = raw_sohs[:eol]
            cycle_numbers = cycle_numbers[:eol]
            cycle_start_time_in_s_list = cycle_start_time_in_s_list[:eol]
            if is_isu_ilcc and cycle_start_datetime_list:
                cycle_start_datetime_list = cycle_start_datetime_list[:eol]
        else:
            # Extrapolation parameters
            N = 20  # Number of cycles for linear regression
            MIN_SLOPE_THRESHOLD = 1e-4  # Minimum degradation rate (per cycle)
            MAX_EXTRAPOLATE_CYCLES = 50  # Maximum cycles to extrapolate

            n_cycles = len(cycle_numbers)

            if n_cycles < 2:
                pass
            else:
                N = min(N, n_cycles)
                X = np.array(cycle_numbers[-N:]).reshape(-1, 1)
                y_soh = np.array(raw_sohs[-N:])
                y_soh = np.minimum.accumulate(y_soh)
                y_time = np.array(cycle_start_time_in_s_list[-N:])

                reg_soh = LinearRegression().fit(X, y_soh)
                a = reg_soh.coef_[0]
                b = reg_soh.intercept_

                # Check 1: Slope must be negative and significant
                if a >= 0:
                    return cell_name, f"Slope is non-negative ({a:.2e}), no degradation trend"

                if abs(a) < MIN_SLOPE_THRESHOLD:
                    return cell_name, f"Slope too small ({a:.2e} < {MIN_SLOPE_THRESHOLD}), degradation too slow to extrapolate"

                eol_cycle_cont = (eol_threshold - b) / a
                last_cycle = int(cycle_numbers[-1])

                if eol_cycle_cont <= last_cycle:
                    pass
                else:
                    eol_cycle = int(np.ceil(eol_cycle_cont))
                    extra_cycles = list(range(last_cycle + 1, eol_cycle + 1))

                    # Check 2: Limit maximum extrapolation cycles
                    if len(extra_cycles) > MAX_EXTRAPOLATE_CYCLES:
                        extra_cycles = extra_cycles[:MAX_EXTRAPOLATE_CYCLES]

                    if len(extra_cycles) > 0:
                        X_extra = np.array(extra_cycles).reshape(-1, 1)
                        soh_extra = reg_soh.predict(X_extra)
                        reg_time = LinearRegression().fit(X, y_time)
                        t_extra = reg_time.predict(X_extra)

                        soh_extra = soh_extra.tolist()
                        t_extra = t_extra.tolist()

                        raw_sohs.extend(soh_extra)
                        cycle_numbers.extend(extra_cycles)
                        cycle_start_time_in_s_list.extend(t_extra)
                        # Set last SOH to EOL only if we reached it
                        if len(extra_cycles) == eol_cycle - last_cycle:
                            raw_sohs[-1] = eol_threshold

        cell_pkl = {
            'cell_id': cell_name.split('.pkl')[0],
            'cycle_numbers': cycle_numbers,
            'SOH': raw_sohs,
            'cycle_start_time_in_s': cycle_start_time_in_s_list,
        }

        if is_isu_ilcc and cycle_start_datetime_list:
            n_original = min(len(cycle_start_datetime_list), len(cycle_numbers))
            cell_pkl['cycle_start_datetime'] = cycle_start_datetime_list[:n_original]

        output_file = os.path.join(output_path, cell_name)
        with open(output_file, 'wb') as f:
            pickle.dump(cell_pkl, f)

        # Success return
        return cell_name, None

    except Exception as e:
        print(f"\n\n[CRITICAL ERROR] Failed processing file: {args[2]}")
        traceback.print_exc()
        raise e


def process_dataset(cleaned_data_root, output_root, dataset_name, num_workers=1):
    """
    Process a single dataset over cells.
    """
    cleaned_data_path = os.path.join(cleaned_data_root, dataset_name)
    output_path = os.path.join(output_root, dataset_name)
    os.makedirs(output_path, exist_ok=True)

    cell_names = [
        name for name in os.listdir(cleaned_data_path)
        if name.endswith('.pkl')
    ]

    tasks = [
        (cleaned_data_path, output_path, cell_name)
        for cell_name in cell_names
    ]

    # List to store skipped files: [(filename, reason), ...]
    skipped_summary = []

    if num_workers is None or num_workers <= 1:
        # -------- single-thread / serial mode ----------
        for t in tqdm(tasks, desc=f"Processing {dataset_name} (serial)", unit="cell"):
            result = process_one_cell(t)
            # result is (cell_name, reason)
            if result and result[1] is not None:
                skipped_summary.append(result)

    else:
        # -------- multi-process parallel mode ----------
        future_to_cell = {}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for t in tasks:
                future = executor.submit(process_one_cell, t)
                future_to_cell[future] = t[2]

            pbar = tqdm(as_completed(future_to_cell), total=len(future_to_cell),
                        desc=f"Processing {dataset_name} ({num_workers} workers)",
                        unit="cell")
            
            for future in pbar:
                try:
                    result = future.result()
                    # result is (cell_name, reason)
                    if result and result[1] is not None:
                        skipped_summary.append(result)
                        
                except Exception as e:
                    failed_cell = future_to_cell[future]
                    pbar.close()
                    print(f"\n[MAIN PROCESS] Exception detected in file: {failed_cell}")
                    print("Shutting down executor...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise e

    # ---- Print Skipped Summary ----
    if skipped_summary:
        print(f"\n{'='*30} SKIPPED CELLS REPORT: {dataset_name} {'='*30}")
        print(f"{'Cell Name':<40} | {'Reason'}")
        print("-" * 80)
        for name, reason in skipped_summary:
            print(f"{name:<40} | {reason}")
        print(f"{'='*80}\n")
    else:
        print(f"\n[INFO] All {len(tasks)} cells in {dataset_name} were processed successfully.\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate SOH trajectories from cleaned battery data'
    )
    parser.add_argument(
        '--cleaned_data_root',
        type=str,
        default='/path/to/your/dataset',
        help='Root directory of cleaned_data'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='/path/to/your/dataset/SOH',
        help='Root directory for output (SOH)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=[],
        help='List of datasets to process (default: all)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=64,
        help='Number of parallel workers (<=1 means serial)'
    )
    args = parser.parse_args()

    cleaned_data_root = args.cleaned_data_root
    output_root = args.output_root
    num_workers = args.num_workers

    datasets = args.datasets
    if len(datasets) == 0:
        datasets = [
            'CALCE', 'HNEI', 'MATR', 'UL_PUR', 'SNL', 'MICH_EXP', 'MICH',
            'RWTH', 'HUST', 'Tongji', 'Stanford', 'XJTU', 'ISU_ILCC',
            'NA-ion', 'CALB','ZN-coin','Stanford_2'
        ]

    for dataset_name in datasets:
        process_dataset(cleaned_data_root, output_root, dataset_name,
                        num_workers=num_workers)


if __name__ == '__main__':
    main()
