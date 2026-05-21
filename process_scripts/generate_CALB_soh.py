import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from datetime import datetime
import argparse


def fix_spike_drops(raw_sohs, max_drop_per_cycle=0.04):
    """Replace abnormal single-cycle SOH drops with the previous cycle's value."""
    if len(raw_sohs) <= 1:
        return list(raw_sohs)
    fixed = list(raw_sohs)
    for i in range(1, len(raw_sohs)):
        if fixed[i - 1] - raw_sohs[i] > max_drop_per_cycle:
            fixed[i] = fixed[i - 1]
    return fixed


def main(raw_data_file_path, output_path):
    sheets = ['0℃循环', '25℃ 循环', '35℃ 循环', '45℃循环']

    for sheet in sheets:
        cells_data = pd.read_excel(raw_data_file_path, sheet_name=sheet)
        columns_name = cells_data.columns.tolist()
        if sheet == '0℃循环':
            start_column_idx = [columns_name.index(i)  for i in columns_name if i.startswith('A1')]
            cells_name = ['CALB_0_' + i.replace('A', 'B') for i in columns_name if i.startswith('A1')]
        elif sheet == '25℃ 循环':
            start_column_idx = [columns_name.index(i)  for i in columns_name if i.startswith('T25')]
            cells_name = ['CALB_25_' + i for i in columns_name if i.startswith('T25')]
        else:
            start_column_idx = [columns_name.index(i) for i in columns_name if i.startswith('B')]
            if sheet == '35℃ 循环':
                cells_name = ['CALB_35_' + i for i in columns_name if i.startswith('B')]
            elif sheet == '45℃循环':
                cells_name = ['CALB_45_' + i for i in columns_name if i.startswith('B')]

        cycles_column_idx = [i + 1 for i in start_column_idx]
        times_column_idx = [i + 2 for i in start_column_idx]
        discharge_column_idx = [i + 4 for i in start_column_idx]

        for cell_name, cycle_idx, discharge_idx, time_idx in zip(tqdm(cells_name, desc='Extracting capacity data'), cycles_column_idx, discharge_column_idx, times_column_idx):
            cycles = cells_data.iloc[:, cycle_idx].tolist()
            discharge = cells_data.iloc[:, discharge_idx].tolist()
            times = cells_data.iloc[:, time_idx].values.tolist()

            cycle_df = pd.DataFrame()
            cycle_df['cycle_number'] = cycles
            cycle_df['time_in_s'] = times
            cycle_df['discharge_capacity_in_Ah'] = discharge

            # cycle_df.to_csv(f'./CALB/{cell_name}.csv', index=False)

            # Compute the SOH trajectory
            soc_interval = [0, 1]
            discharge_depth = soc_interval[1] - soc_interval[0]
            filter_threshold = 0.925
            eol_threshold = 0.9

            # refine the cycle_df
            cycle_df = cycle_df.dropna()
            cycle_number_list = cycle_df['cycle_number'].values.tolist()
            new_cycle_list = [i for i in range(1, len(cycle_number_list) + 1)]
            cycle_df['cycle_number'] = new_cycle_list

            nominal_capacity = cycle_df['discharge_capacity_in_Ah'].values.tolist()[0]

            discharge_capacity_in_Ah = cycle_df['discharge_capacity_in_Ah']
            cycle_start_time_in_s_list = cycle_df['time_in_s']
            soh = np.array(discharge_capacity_in_Ah) / nominal_capacity
            soh = soh / discharge_depth
            raw_sohs = soh
            cycle_numbers = list(cycle_df['cycle_number'])

            nan_mask = np.isnan(raw_sohs)
            raw_sohs = raw_sohs[~nan_mask].tolist()
            cycle_numbers = np.array(cycle_numbers)
            cycle_numbers = cycle_numbers[~nan_mask].tolist()
            cycle_start_time_in_s_list = np.array(cycle_start_time_in_s_list)
            cycle_start_time_in_s_list = cycle_start_time_in_s_list[~nan_mask].tolist()

            # convert datetime to seconds
            cycle_start_time_in_s_list = [str(i) for i in cycle_start_time_in_s_list]
            datetime_list = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in cycle_start_time_in_s_list]
            cycle_start_time_in_s_list = [dt.timestamp() for dt in datetime_list]
            time_start = cycle_start_time_in_s_list[0]
            cycle_start_time_in_s_list = [i - time_start for i in cycle_start_time_in_s_list]

            # Skip batteries whose final SOH is far from EOL
            if raw_sohs[-1] > filter_threshold:
                continue

            # CALB_35_B229 has an abnormal capacity dip around cycle 697 that
            # briefly pushes SOH below the EOL threshold before recovering;
            # without correction the EOL search would end its life ~474 cycles
            # early. BatteryLife v11 corrects this cell's life label likewise.
            if cell_name == 'CALB_35_B229':
                raw_sohs = fix_spike_drops(raw_sohs, max_drop_per_cycle=0.03)

            if raw_sohs[-1] <= eol_threshold:
                extrapolated = False
                # Already reaches EOL: find first cycle where SOH <= threshold
                eol = 0
                for cycle_number, soh in zip(cycle_numbers, raw_sohs):
                    if soh <= eol_threshold:
                        eol = cycle_number
                        break

                eol = int(eol)
                # Here you assume cycle_number is contiguous from 1, so slicing by index
                raw_sohs = raw_sohs[:eol]
                cycle_numbers = cycle_numbers[:eol]
                cycle_start_time_in_s_list = cycle_start_time_in_s_list[:eol]
            else:
                extrapolated = True
                # Use linear regression to extrapolate the trajectory to EOL
                N = 20  # number of last cycles used for regression
                n_cycles = len(cycle_numbers)

                if n_cycles < 2:
                    pass
                else:
                    N = min(N, n_cycles)
                    X = np.array(cycle_numbers[-N:]).reshape(-1, 1)
                    y_soh = np.array(raw_sohs[-N:])
                    # Force monotonic non-increasing SOH for the regression window
                    y_soh = np.minimum.accumulate(y_soh)
                    y_time = np.array(cycle_start_time_in_s_list[-N:])

                    # SOH regression: soh = a * cycle + b
                    reg_soh = LinearRegression().fit(X, y_soh)
                    a = reg_soh.coef_[0]
                    b = reg_soh.intercept_

                    # Compute the cycle number (continuous) when SOH hits EOL threshold:
                    # eol_cycle_cont satisfies: eol_threshold = a * eol_cycle_cont + b
                    eol_cycle_cont = (eol_threshold - b) / a  # ideally a < 0 ⇒ eol_cycle_cont > last_cycle
                    last_cycle = int(cycle_numbers[-1])

                    # If eol_cycle is before the last observed cycle, the fit is inconsistent
                    if eol_cycle_cont <= last_cycle:
                        eol_cycle_cont = last_cycle + 1
                    else:
                        # Round up to the nearest integer cycle
                        eol_cycle = int(np.ceil(eol_cycle_cont))

                        # Construct the sequence of cycles to extrapolate
                        extra_cycles = list(range(last_cycle + 1, eol_cycle + 1))
                        if len(extra_cycles) > 0:
                            X_extra = np.array(extra_cycles).reshape(-1, 1)

                            # Extrapolate SOH using the regression model
                            soh_extra = reg_soh.predict(X_extra)

                            # Time regression: t = alpha * cycle + beta
                            reg_time = LinearRegression().fit(X, y_time)
                            t_extra = reg_time.predict(X_extra)

                            soh_extra = soh_extra.tolist()
                            t_extra = t_extra.tolist()

                            # Append extrapolated part to the original trajectory
                            raw_sohs.extend(soh_extra)
                            cycle_numbers.extend(extra_cycles)
                            cycle_start_time_in_s_list.extend(t_extra)

                            # Optionally, force the last SOH value to exactly equal EOL threshold
                            raw_sohs[-1] = eol_threshold

            # Build the output dictionary
            cell_pkl = {
                'cell_id': cell_name.split('.pkl')[0],
                'cycle_numbers': cycle_numbers,
                'SOH': raw_sohs,
                'cycle_start_time_in_s': cycle_start_time_in_s_list,
            }

            # Save to output path (inside CALB subdirectory)
            calb_output_dir = os.path.join(output_path, 'CALB')
            os.makedirs(calb_output_dir, exist_ok=True)
            output_file = os.path.join(calb_output_dir, cell_name)
            with open(output_file + '.pkl', 'wb') as f:
                pickle.dump(cell_pkl, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate SOH trajectories from cleaned battery data'
    )
    parser.add_argument(
        '--raw_data_file_path',
        type=str,
        default='/path/to/your/dataset/',
        help='Root directory of cleaned_data'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/path/to/your/dataset/SOH',
        help='Root directory for output (processed_SOH)'
    )
    args = parser.parse_args()

    raw_data_file_path = args.raw_data_file_path
    output_path = args.output_path

    main(raw_data_file_path, output_path)
