import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import pickle
import json
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings('ignore')


def collate_fn_soh(samples):
    """
    Custom collate function for SOH trajectory prediction
    Handles both current/voltage input and SOH-to-SOH input modes
    """
    if 'cycle_curve_data' in samples[0]:
        # Mode 1: Current/Voltage/Capacity/SOC input
        cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples])
        curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])
    else:
        # Mode 2: SOH-to-SOH input
        cycle_curve_data = None
        curve_attn_mask = None

    soh_input = torch.vstack([i['soh_input'].unsqueeze(0) for i in samples])
    CEs = torch.vstack([i['CEs'].unsqueeze(0) for i in samples])
    EEs = torch.vstack([i['EEs'].unsqueeze(0) for i in samples])
    cycle_level_features = torch.cat([CEs, EEs], dim=-1)
    soh_trajectory = torch.vstack([i['soh_trajectory'].unsqueeze(0) for i in samples])
    trajectory_mask = torch.vstack([i['trajectory_mask'].unsqueeze(0) for i in samples])
    life_labels = torch.tensor([i['eol_index'] for i in samples])
    aging_condition_embedding = torch.vstack([i['aging_condition_embedding'].unsqueeze(0) for i in samples])

    # Collect file_names for evaluation (seen/unseen aging condition analysis)
    file_names = [i['file_name'] for i in samples]

    # Expand mask for the 4-channel data (Voltage, Current, Capacity, SOC)
    if cycle_curve_data is not None:
        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)  # [B, cycle_number, 4, fixed_len]
        cycle_curve_data[tmp_curve_attn_mask == 0] = 0  # set the unseen data as zeros

    soc_curves = cycle_curve_data[:, :, -1]  # [B, cycle_number, fixed_len]
    cycle_curve_data = cycle_curve_data[:, :, :3]  # [B, cycle_number, 3, fixed_len]

    return cycle_curve_data, curve_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names


def collate_fn_ic2ml(samples):
    """
    Custom collate function for SOH trajectory prediction (IC2ML)
    """
    cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples])
    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])

    aging_condition_embedding = None
    cycle_level_features = None
    soc_curves = None

    soh_input = torch.vstack([i['soh_input'].unsqueeze(0) for i in samples])
    soh_trajectory = torch.vstack([i['soh_trajectory'].unsqueeze(0) for i in samples])
    trajectory_mask = torch.vstack([i['trajectory_mask'].unsqueeze(0) for i in samples])
    life_labels = torch.tensor([i['rul_index'] for i in samples])

    # Collect file_names for evaluation (seen/unseen aging condition analysis)
    file_names = [i['file_name'] for i in samples]

    # Expand mask for the 4-channel data (Voltage, Current, Capacity, SOC)
    tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
    cycle_curve_data[tmp_curve_attn_mask == 0] = 0

    capacity_curves = cycle_curve_data[:, :, 2]  # [B, cycle_number, fixed_len]
    capacity_attn_mask = curve_attn_mask.unsqueeze(-1)

    return capacity_curves, capacity_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names


class Dataset_SOH_Forecasting(Dataset):
    def __init__(
        self,
        args,
        flag='train',
        input_mode='current_voltage',
        max_trajectory_len=5200,
        label_scaler=None,
        alignment_check: bool = True,
    ):
        """
        Dataset for SOH trajectory prediction with caching.

        New:
            alignment_check (bool): if True, run SOC-alignment validity self-checks (cheap, sampled).
        """
        self.args = args
        self.flag = flag
        self.input_mode = input_mode
        self.max_trajectory_len = max_trajectory_len
        self.label_scaler = label_scaler

        # --- NEW: alignment check switch ---
        self.alignment_check = bool(alignment_check)
        # Keep defaults internal (no need to change args)
        self._soc_check_max_cycles_per_battery = getattr(args, 'soc_check_max_cycles_per_battery', 3)
        self._soc_check_tol = getattr(args, 'soc_check_tol', 2e-3)

        # Basic parameters
        self.ZN_coin_charge_first_file_names = [
            'ZN-coin_402-1_20231209225636_01_1.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl',
            'ZN-coin_402-3_20231209225844_01_3.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl',
            'ZN-coin_428-1_20231212185048_01_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl',
            'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl',
            'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_430-2_20231212185305_02_7.pkl',
            'ZN-coin_430-3_20231212185323_03_2.pkl'
        ]
        self.random_seed = args.seed
        self.root_path = args.root_path if hasattr(args, 'root_path') else '/ai/dl_project/MemoryNet/dataset/cleaned_data'
        self.processed_SOH_path = args.processed_SOH_path
        self.dataset_name = args.dataset
        self.early_cycle_threshold = args.early_cycle_threshold if hasattr(args, 'early_cycle_threshold') else 100
        self.charge_discharge_len = args.charge_discharge_length if hasattr(args, 'charge_discharge_length') else 100
        self.seq_len = args.seq_len if hasattr(args, 'seq_len') else 100

        # Resampling
        self.use_capacity_resample = getattr(args, 'use_capacity_resample', False)
        self.num_segments = getattr(args, 'num_segments', 5)

        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']

        # Load splits
        self._load_data_files()

        # Cache
        cache_key = self._get_cache_key()
        cache_root = './.cache'
        os.makedirs(cache_root, exist_ok=True)
        cache_file = f"{cache_root}/{cache_key}.pkl"

        if os.path.exists(cache_file) and not getattr(args, 'force_reload', False):
            print(f"Loading preprocessed data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print(f"Preprocessing data for {self.dataset_name} {flag} {input_mode}...")
            self.samples = []
            self._prepare_samples()

            print(f"Saving preprocessed data to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)

        print(f"Loaded {len(self.samples)} samples for {flag} set")

    def read_cell_data_according_to_prefix(self, file_name):
        prefix = file_name.split('_')[0]

        if prefix.startswith('MATR'):
            data = pickle.load(open(f'{self.root_path}/MATR/{file_name}', 'rb'))
        elif prefix.startswith('HUST'):
            data = pickle.load(open(f'{self.root_path}/HUST/{file_name}', 'rb'))
        elif prefix.startswith('SNL'):
            data = pickle.load(open(f'{self.root_path}/SNL/{file_name}', 'rb'))
        elif prefix.startswith('CALCE'):
            data = pickle.load(open(f'{self.root_path}/CALCE/{file_name}', 'rb'))
        elif prefix.startswith('HNEI'):
            data = pickle.load(open(f'{self.root_path}/HNEI/{file_name}', 'rb'))
        elif prefix.startswith('MICH'):
            data = pickle.load(open(f'{self.root_path}/total_MICH/{file_name}', 'rb'))
        elif prefix.startswith('OX'):
            data = pickle.load(open(f'{self.root_path}/OX/{file_name}', 'rb'))
        elif prefix.startswith('RWTH'):
            data = pickle.load(open(f'{self.root_path}/RWTH/{file_name}', 'rb'))
        elif prefix.startswith('UL-PUR'):
            data = pickle.load(open(f'{self.root_path}/UL_PUR/{file_name}', 'rb'))
        elif prefix.startswith('SMICH'):
            data = pickle.load(open(f'{self.root_path}/MICH_EXP/{file_name[1:]}', 'rb'))
        elif prefix.startswith('BIT2'):
            data = pickle.load(open(f'{self.root_path}/BIT2/{file_name}', 'rb'))
        elif prefix.startswith('Tongji'):
            data = pickle.load(open(f'{self.root_path}/Tongji/{file_name}', 'rb'))
        elif prefix.startswith('Stanford'):
            file_name = file_name.replace('_Ref', '')
            data = pickle.load(open(f'{self.root_path}/Stanford_2/{file_name}', 'rb'))
        elif prefix.startswith('ISU-ILCC'):
            data = pickle.load(open(f'{self.root_path}/ISU_ILCC/{file_name}', 'rb'))
        elif prefix.startswith('XJTU'):
            data = pickle.load(open(f'{self.root_path}/XJTU/{file_name}', 'rb'))
        elif prefix.startswith('ZN-coin'):
            data = pickle.load(open(f'{self.root_path}/ZN-coin/{file_name}', 'rb'))
        elif prefix.startswith('CALB'):
            data = pickle.load(open(f'{self.root_path}/CALB/{file_name}', 'rb'))
        elif prefix.startswith('NA-ion'):
            data = pickle.load(open(f'{self.root_path}/NA-ion/{file_name}', 'rb'))
        else:
            raise ValueError(f"Unknown prefix: {prefix} (file={file_name})")

        return data

    def _get_cache_key(self):
        # Include alignment_check into cache key to avoid mixing checked/unchecked preprocessing
        key_str = (
            f"{self.dataset_name}_{self.args.seed}_{self.flag}_{self.input_mode}_"
            f"{self.early_cycle_threshold}_{self.charge_discharge_len}_{self.seq_len}_{self.max_trajectory_len}_"
            f"capresample{self.use_capacity_resample}_v3_soc_alignchk{int(self.alignment_check)}"
        )
        fewshot_ratio = getattr(self.args, 'fewshot_ratio', None)
        if fewshot_ratio and self.flag == 'train' and 0 < fewshot_ratio < 1:
            key_str += f"_fewshot{int(fewshot_ratio * 100)}"
        return key_str

    def _load_data_files(self):
        fewshot_ratio = getattr(self.args, 'fewshot_ratio', None)
        if fewshot_ratio and self.flag == 'train' and 0 < fewshot_ratio < 1:
            fewshot_pct = int(fewshot_ratio * 100)
            if self.dataset_name != 'Li_ion':
                split_json = f'./data_provider/split_json/fewshot/{self.dataset_name}_split_{self.random_seed}_fewshot{fewshot_pct}.json'
            else:
                split_json = f'./data_provider/split_json/fewshot/total_split_{self.random_seed}_fewshot{fewshot_pct}.json'
            print(f"Using data-efficient split: {split_json}")
        else:
            if self.dataset_name != 'Li_ion':
                split_json = f'./data_provider/split_json/{self.dataset_name}_split_{self.random_seed}.json'
            else:
                split_json = f'./data_provider/split_json/total_split_{self.random_seed}.json'

        if self.flag == 'train':
            files = json.load(open(split_json))['train']
        elif self.flag == 'val':
            files = json.load(open(split_json))['val']
        elif self.flag == 'test':
            files = json.load(open(split_json))['test']
        else:
            raise ValueError(f"flag must be in ['train','val','test'], got {self.flag}")

        if self.dataset_name == 'Li_ion':
            selected_files = []
            for one_file in files:
                if 'ZN-coin' in one_file or 'CALB' in one_file or 'NA-ion' in one_file:
                    continue
                selected_files.append(one_file)
            files = selected_files

        self.files = files

    @staticmethod
    def _sanitize_vector(values, fill_value=0.0):
        arr = np.asarray(values, dtype=np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
        return arr

    def _soc_alignment_sanity_check(
        self,
        file_name: str,
        cycle_idx: int,
        start_soc: float,
        end_soc: float,
        charge_s: np.ndarray,
        discharge_s: np.ndarray,
        tol: float,
    ):
        """
        SOC-alignment validity check (sampled, cheap).
        Raise ValueError with file/cycle info if check fails.
        """
        charge_s = np.asarray(charge_s, dtype=np.float32)
        discharge_s = np.asarray(discharge_s, dtype=np.float32)

        if charge_s.size == 0 or discharge_s.size == 0:
            raise ValueError(f"[SOCCheck] empty SOC curve: file={file_name}, cycle={cycle_idx}")

        # Monotonicity
        if np.any(np.diff(charge_s) < -1e-5):
            raise ValueError(
                f"[SOCCheck] charge SOC not non-decreasing: file={file_name}, cycle={cycle_idx}, "
                f"min_diff={float(np.diff(charge_s).min())}"
            )
        if np.any(np.diff(discharge_s) > 1e-5):
            raise ValueError(
                f"[SOCCheck] discharge SOC not non-increasing: file={file_name}, cycle={cycle_idx}, "
                f"max_diff={float(np.diff(discharge_s).max())}"
            )

        low, high = float(min(start_soc, end_soc)), float(max(start_soc, end_soc))

        # Range within battery SOC interval
        ch_min, ch_max = float(charge_s.min()), float(charge_s.max())
        dis_min, dis_max = float(discharge_s.min()), float(discharge_s.max())

        if ch_min < low - tol or ch_max > high + tol:
            raise ValueError(
                f"[SOCCheck] charge SOC out of interval: file={file_name}, cycle={cycle_idx}, "
                f"charge_range=[{ch_min:.4f},{ch_max:.4f}], interval=[{low:.4f},{high:.4f}]"
            )
        if dis_min < low - tol or dis_max > high + tol:
            raise ValueError(
                f"[SOCCheck] discharge SOC out of interval: file={file_name}, cycle={cycle_idx}, "
                f"discharge_range=[{dis_min:.4f},{dis_max:.4f}], interval=[{low:.4f},{high:.4f}]"
            )

        # Soft endpoint checks (use wider tolerance)
        if abs(float(charge_s[0]) - low) > 5 * tol:
            raise ValueError(
                f"[SOCCheck] charge start SOC deviates: file={file_name}, cycle={cycle_idx}, "
                f"charge_s0={float(charge_s[0]):.4f}, expected~{low:.4f}"
            )
        if abs(float(discharge_s[-1]) - low) > 5 * tol:
            raise ValueError(
                f"[SOCCheck] discharge end SOC deviates: file={file_name}, cycle={cycle_idx}, "
                f"discharge_send={float(discharge_s[-1]):.4f}, expected~{low:.4f}"
            )

    @staticmethod
    def _first_contiguous_block_last_index(idxs: np.ndarray) -> int:
        """
        Given sorted indices array (1D), return the last index of the first
        contiguous block. Contiguous means successive indices differ by 1.

        Example:
            idxs = [5,6,7, 10,11] -> returns 7
            idxs = [3,4,5] -> returns 5
        """
        if idxs.size == 0:
            return -1
        if idxs.size == 1:
            return int(idxs[0])

        diffs = np.diff(idxs)
        breaks = np.where(diffs > 1)[0]   # positions where continuity breaks
        if breaks.size == 0:
            return int(idxs[-1])          # fully contiguous
        # first break at position breaks[0], so first block ends at idxs[breaks[0]]
        return int(idxs[breaks[0]])

    def _prepare_charge_discharge_curves(self, file_name, cycle_data, nominal_capacity, start_soc, end_soc):
        """
        Prepare charge/discharge curves including SOC calculation.
        Returns:
            curves: [early_cycles, 4, charge_discharge_len]
            channels: [Voltage, Current, Capacity, SOC]
        """
        curves = []
        coulumbic_efficiencys = []
        energy_efficiencys = []

        soc_range = float(end_soc - start_soc)
        prefix = file_name.split('_')[0]
        if prefix == 'CALB':
            prefix = file_name.split('_')[:2]
            prefix = '_'.join(prefix)

        for cycle_idx in range(self.seq_len, self.early_cycle_threshold + 1):
            if cycle_idx < len(cycle_data):
                cycle = cycle_data[cycle_idx - 1]

                # Extract data
                cycle_df = pd.DataFrame()
                for key in self.need_keys:
                    cycle_df[key] = cycle[key]

                # Data cleaning
                cycle_df.loc[cycle_df['charge_capacity_in_Ah'] < 0] = np.nan
                cycle_df.loc[cycle_df['discharge_capacity_in_Ah'] < 0] = np.nan
                cycle_df.bfill(inplace=True)
                cycle_df.ffill(inplace=True)

                time_in_s_records = self._sanitize_vector(cycle_df['time_in_s'])
                time_in_h_records = time_in_s_records / 3600.0
                voltage = self._sanitize_vector(cycle_df['voltage_in_V'])
                current = self._sanitize_vector(cycle_df['current_in_A'])
                charge_cap = self._sanitize_vector(cycle_df['charge_capacity_in_Ah'])
                discharge_cap = self._sanitize_vector(cycle_df['discharge_capacity_in_Ah'])

                current_c_rate = current / float(nominal_capacity)

                # Determine charge/discharge end indices
                cutoff_charge = np.nonzero(current_c_rate >= 0.01)
                cutoff_discharge = np.nonzero(current_c_rate <= -0.01)
                if cutoff_charge[0].size == 0 or cutoff_discharge[0].size == 0:
                    # Bad cycle: pad zeros
                    if self.input_mode == 'capacity_increment':
                        curve = np.zeros((4, self.charge_discharge_len // 2), dtype=np.float32)
                    else:
                        curve = np.zeros((4, self.charge_discharge_len), dtype=np.float32)
                    CE, EE = 0.0, 0.0
                    coulumbic_efficiencys.append(CE)
                    energy_efficiencys.append(EE)
                    if self.input_mode == 'capacity_increment':
                        curves.append(curve.reshape(1, 4, self.charge_discharge_len // 2))
                    else:
                        curves.append(curve.reshape(1, 4, self.charge_discharge_len))
                    continue

                if 'MATR_b1c18' in file_name:
                    charge_end_index = cutoff_discharge[0][0] - 1
                else:
                    charge_end_index = cutoff_charge[0][-1]
                discharge_end_index = cutoff_discharge[0][-1]

                if prefix in ['RWTH', 'CALB_0', 'CALB_25', 'CALB_45'] or (
                    file_name not in self.ZN_coin_charge_first_file_names and prefix == 'ZN-coin'
                ):
                    # First discharge then charge
                    v_dis = voltage[:discharge_end_index]
                    c_dis = discharge_cap[:discharge_end_index]
                    i_dis = current[:discharge_end_index]
                    time_in_h_dis = time_in_h_records[:discharge_end_index]

                    v_chg = voltage[discharge_end_index:]
                    c_chg = charge_cap[discharge_end_index:]
                    i_chg = current[discharge_end_index:]
                    time_in_h_chg = time_in_h_records[discharge_end_index:]

                    i_chg_in_c_rates = i_chg / float(nominal_capacity)
                    valid_chg = np.abs(i_chg_in_c_rates) > 0.01
                    v_chg, c_chg, i_chg, time_in_h_chg = v_chg[valid_chg], c_chg[valid_chg], i_chg[valid_chg], time_in_h_chg[valid_chg]

                    power_dis = v_dis * np.abs(i_dis)
                    energy_discharge = np.trapz(power_dis, time_in_h_dis) if time_in_h_dis.size > 1 else 0.0

                    power_chg = v_chg * np.abs(i_chg)
                    energy_charge = np.trapz(power_chg, time_in_h_chg) if time_in_h_chg.size > 1 else 1e-6
                else:
                    # First charge then discharge
                    v_dis = voltage[charge_end_index:]
                    c_dis = discharge_cap[charge_end_index:]
                    i_dis = current[charge_end_index:]
                    time_in_h_dis = time_in_h_records[charge_end_index:]

                    i_dis_in_c_rates = i_dis / float(nominal_capacity)
                    valid_dis = np.abs(i_dis_in_c_rates) > 0.01
                    v_dis, c_dis, i_dis, time_in_h_dis = v_dis[valid_dis], c_dis[valid_dis], i_dis[valid_dis], time_in_h_dis[valid_dis]

                    v_chg = voltage[:charge_end_index]
                    c_chg = charge_cap[:charge_end_index]
                    i_chg = current[:charge_end_index]
                    time_in_h_chg = time_in_h_records[:charge_end_index]

                    power_dis = v_dis * np.abs(i_dis)
                    energy_discharge = np.trapz(power_dis, time_in_h_dis) if time_in_h_dis.size > 1 else 0.0

                    power_chg = v_chg * np.abs(i_chg)
                    energy_charge = np.trapz(power_chg, time_in_h_chg) if time_in_h_chg.size > 1 else 1e-6

                EE = float(energy_discharge) / float(max(energy_charge, 1e-6))
                CE = float(max(c_dis) / max(max(c_chg), 1e-6)) if len(c_dis) > 0 and len(c_chg) > 0 else 0.0

                # --- SOC construction with zero-division guards ---
                if len(c_chg) > 0:
                    c_chg_delta = float(max(c_chg.max() - c_chg.min(), 1e-6))
                    soc_chg = float(start_soc) + ((c_chg - c_chg[0]) / c_chg_delta) * soc_range
                else:
                    soc_chg = np.array([], dtype=np.float32)

                if len(c_dis) > 0:
                    c_dis_delta = float(max(c_dis.max() - c_dis.min(), 1e-6))
                    soc_dis = float(end_soc) - ((c_dis - c_dis[0]) / c_dis_delta) * soc_range
                else:
                    soc_dis = np.array([], dtype=np.float32)

                # --- Resample ---
                charge_len = self.charge_discharge_len // 2
                discharge_len = self.charge_discharge_len // 2

                if self.use_capacity_resample:
                    if 'BatteryMFormer' in self.args.model:
                        # STRICT SOC-aligned interpolation
                        charge_v, charge_i, charge_c, charge_s = self._soc_based_resample(
                            v_chg, i_chg, c_chg, soc_chg, charge_len, is_charge=True
                        )
                        discharge_v, discharge_i, discharge_c, discharge_s = self._soc_based_resample(
                            v_dis, i_dis, c_dis, soc_dis, discharge_len, is_charge=False
                        )

                        # --- NEW: validity self-check (sampled) ---
                        if (
                            self.alignment_check
                            and (cycle_idx - self.seq_len) < int(self._soc_check_max_cycles_per_battery)
                        ):
                            self._soc_alignment_sanity_check(
                                file_name=file_name,
                                cycle_idx=cycle_idx,
                                start_soc=float(start_soc),
                                end_soc=float(end_soc),
                                charge_s=np.asarray(charge_s),
                                discharge_s=np.asarray(discharge_s),
                                tol=float(self._soc_check_tol),
                            )
                    else:
                        # Keep original capacity-based for other models
                        charge_v, charge_i, charge_c, charge_s = self._capacity_based_resample(
                            v_chg, i_chg, c_chg, soc_chg, charge_len, is_charge=True
                        )
                        discharge_v, discharge_i, discharge_c, discharge_s = self._capacity_based_resample(
                            v_dis, i_dis, c_dis, soc_dis, discharge_len, is_charge=False
                        )
                else:
                    # Index-based interpolation
                    charge_v = self._fast_resample(v_chg, charge_len)
                    charge_i = self._fast_resample(i_chg, charge_len)
                    charge_c = self._fast_resample(c_chg, charge_len)
                    charge_s = self._fast_resample(soc_chg, charge_len)

                    discharge_v = self._fast_resample(v_dis, discharge_len)
                    discharge_i = self._fast_resample(i_dis, discharge_len)
                    discharge_c = self._fast_resample(c_dis, discharge_len)
                    discharge_s = self._fast_resample(soc_dis, discharge_len)

                # --- Combine and Stack ---
                if self.input_mode == 'capacity_increment':
                    voltage_curve = self._sanitize_vector(charge_v)
                    current_curve = self._sanitize_vector(charge_i)
                    capacity_curve = self._sanitize_vector(charge_c)
                    soc_curve = self._sanitize_vector(charge_s)

                    capacity_increment_curve = (capacity_curve - capacity_curve[0]) / float(nominal_capacity)

                    if nominal_capacity == 0 or not nominal_capacity:
                        raise Exception('Nominal capacity should not be none or 0 Ah!')
                    current_curve = current_curve / float(nominal_capacity)

                    curve = np.stack([voltage_curve, current_curve, capacity_increment_curve, soc_curve])
                    curve = self._sanitize_vector(curve)
                else:
                    voltage_curve = np.concatenate([charge_v, discharge_v])
                    current_curve = np.concatenate([charge_i, discharge_i])
                    capacity_curve = np.concatenate([charge_c, discharge_c])
                    soc_curve = np.concatenate([charge_s, discharge_s])

                    voltage_curve = self._sanitize_vector(voltage_curve)
                    current_curve = self._sanitize_vector(current_curve)
                    capacity_curve = self._sanitize_vector(capacity_curve)
                    soc_curve = self._sanitize_vector(soc_curve)

                    if nominal_capacity == 0 or not nominal_capacity:
                        raise Exception('Nominal capacity should not be none or 0 Ah!')
                    current_curve = current_curve / float(nominal_capacity)

                    curve = np.stack([voltage_curve, current_curve, capacity_curve, soc_curve])
                    curve = self._sanitize_vector(curve)
            else:
                # Pad with zeros
                if self.input_mode == 'capacity_increment':
                    curve = np.zeros((4, self.charge_discharge_len // 2), dtype=np.float32)
                else:
                    curve = np.zeros((4, self.charge_discharge_len), dtype=np.float32)
                CE, EE = 0.0, 0.0

            coulumbic_efficiencys.append(CE)
            energy_efficiencys.append(EE)

            if self.input_mode == 'capacity_increment':
                curves.append(curve.reshape(1, 4, self.charge_discharge_len // 2))
            else:
                curves.append(curve.reshape(1, 4, self.charge_discharge_len))

        curves = np.concatenate(curves, axis=0)
        return np.array(curves), np.array(coulumbic_efficiencys), np.array(energy_efficiencys)

    def _soc_based_resample(self, voltage, current, capacity, soc, target_length, is_charge=True):
        """
        Strict SOC-aligned resampling:
        interpolate v/i/c as functions of SOC onto an equally-spaced SOC grid.
        """
        if len(soc) == 0 or len(voltage) == 0:
            z = np.zeros(target_length, dtype=np.float32)
            return z, z, z, z

        voltage = self._sanitize_vector(voltage)
        current = self._sanitize_vector(current)
        capacity = self._sanitize_vector(capacity)
        soc = self._sanitize_vector(soc)

        # Enforce monotonic SOC (robust to small noise)
        if is_charge:
            soc_mono = np.maximum.accumulate(soc)      # non-decreasing
        else:
            soc_mono = np.minimum.accumulate(soc)      # non-increasing

        # Remove near-duplicate SOC points to keep x strictly monotonic for interp
        ds = np.diff(soc_mono)
        if is_charge:
            valid = np.concatenate([[True], ds > 1e-8])
        else:
            valid = np.concatenate([[True], ds < -1e-8])

        if valid.sum() < 2:
            return (
                self._fast_resample(voltage, target_length),
                self._fast_resample(current, target_length),
                self._fast_resample(capacity, target_length),
                self._fast_resample(soc, target_length),
            )

        soc_valid = soc_mono[valid]
        v_valid = voltage[valid]
        i_valid = current[valid]
        c_valid = capacity[valid]

        soc_min, soc_max = float(soc_valid.min()), float(soc_valid.max())
        if is_charge:
            target_soc = np.linspace(soc_min, soc_max, target_length, dtype=np.float32)
        else:
            target_soc = np.linspace(soc_max, soc_min, target_length, dtype=np.float32)

        # np.interp requires increasing x
        if is_charge:
            x = soc_valid
            v = np.interp(target_soc, x, v_valid)
            i = np.interp(target_soc, x, i_valid)
            c = np.interp(target_soc, x, c_valid)
            s = target_soc
        else:
            x = soc_valid[::-1]  # increasing
            v = np.interp(target_soc[::-1], x, v_valid[::-1])[::-1]
            i = np.interp(target_soc[::-1], x, i_valid[::-1])[::-1]
            c = np.interp(target_soc[::-1], x, c_valid[::-1])[::-1]
            s = target_soc

        return (
            self._sanitize_vector(v),
            self._sanitize_vector(i),
            self._sanitize_vector(c),
            self._sanitize_vector(s),
        )

    def _capacity_based_resample(self, voltage, current, capacity, soc, target_length, is_charge=True):
        """
        Capacity-aligned resampling (legacy / non-BatteryMFormer branch).
        """
        if len(capacity) == 0 or len(voltage) == 0:
            z = np.zeros(target_length, dtype=np.float32)
            return z, z, z, z

        voltage = self._sanitize_vector(voltage)
        current = self._sanitize_vector(current)
        capacity = self._sanitize_vector(capacity)
        soc = self._sanitize_vector(soc)

        capacity_monotonic = np.maximum.accumulate(capacity)
        max_cap = float(capacity_monotonic[-1]) if len(capacity_monotonic) > 0 else 1.0
        max_cap = max(max_cap, 1e-6)

        capacity_norm = capacity_monotonic / max_cap
        target_norm = np.linspace(0, 1, target_length)

        valid_mask = np.concatenate([[True], np.diff(capacity_norm) > 1e-8])
        if np.sum(valid_mask) < 2:
            return (
                self._fast_resample(voltage, target_length),
                self._fast_resample(current, target_length),
                self._fast_resample(capacity, target_length),
                self._fast_resample(soc, target_length),
            )

        x = capacity_norm[valid_mask]
        v_valid = voltage[valid_mask]
        i_valid = current[valid_mask]
        s_valid = soc[valid_mask]

        resampled_voltage = np.interp(target_norm, x, v_valid)
        resampled_current = np.interp(target_norm, x, i_valid)
        resampled_soc = np.interp(target_norm, x, s_valid)
        resampled_capacity = target_norm * max_cap

        return (
            self._sanitize_vector(resampled_voltage),
            self._sanitize_vector(resampled_current),
            self._sanitize_vector(resampled_capacity),
            self._sanitize_vector(resampled_soc),
        )

    def _fast_resample(self, curve, target_length):
        if len(curve) == 0:
            return np.zeros(target_length, dtype=np.float32)
        curve = self._sanitize_vector(curve)
        old_idx = np.linspace(0, len(curve) - 1, len(curve))
        new_idx = np.linspace(0, len(curve) - 1, target_length)
        return self._sanitize_vector(np.interp(new_idx, old_idx, curve))

    def _prepare_samples(self):
        Qwen3_aging_condition_embeddings = pickle.load(
            open('data_provider/prompt_embeddings/Qwen3_total.pkl', 'rb')
        )

        total_life_labels = []
        print('Files:', len(self.files))

        for file_name in self.files:
            data = self.read_cell_data_according_to_prefix(file_name)

            # dataset dir
            if self.dataset_name in ['NA-ion', 'ZN-coin', 'CALB']:
                dataset_dir = file_name.split('_')[0]
            else:
                dataset_dir = file_name.split('_')[0].replace('-', '_')
                if dataset_dir == 'MICH':
                    dataset_dir = 'total_MICH'
                elif 'Tongji' in file_name:
                    dataset_dir = 'Tongji'
                elif 'ISU-ILCC' in file_name:
                    dataset_dir = 'ISU_ILCC'
                elif 'Stanford' in file_name:
                    dataset_dir = 'Stanford_2'

            file_path = os.path.join(self.root_path, dataset_dir, file_name)
            if not os.path.exists(file_path):
                print('No file found for:', file_path)
                continue

            soh_file_path = os.path.join(self.processed_SOH_path, dataset_dir, file_name)
            if os.path.exists(soh_file_path):
                with open(soh_file_path, 'rb') as f:
                    soh_data = pickle.load(f)
                soh_trajectory = np.array(soh_data['SOH'], dtype=np.float32)
            else:
                print('No SOH file found for:', soh_file_path)
                continue

            eol_threshold = 0.8 if dataset_dir != 'CALB' else 0.9
            eol_cycle = len(soh_trajectory)

            total_life_labels.append(eol_cycle)
            if eol_cycle <= self.early_cycle_threshold:
                continue

            nominal_capacity = data['nominal_capacity_in_Ah']
            soc_interval = data['SOC_interval']
            start_soc, end_soc = soc_interval[0], soc_interval[1]
            if nominal_capacity is None or not np.isfinite(nominal_capacity) or nominal_capacity <= 0:
                raise Exception(f'{file_name} should have nominal capacity in Ah!')

            aging_condition_embedding_for_this_cell = Qwen3_aging_condition_embeddings[file_name]
            valid_cycle_number = len(data['cycle_data'])

            if self.input_mode == 'current_voltage':
                curves, coulumbic_efficiencys, energy_efficiencys = self._prepare_charge_discharge_curves(
                    file_name, data['cycle_data'], nominal_capacity, start_soc, end_soc
                )

                tmp_soh_trajectory = (soh_trajectory - eol_threshold) / (1.0 - eol_threshold)

                for useable_cycle_number in range(self.seq_len, self.early_cycle_threshold + 1):
                    if useable_cycle_number > valid_cycle_number:
                        break

                    sample = {'file_name': file_name, 'nominal_capacity': nominal_capacity}
                    sample['cycle_curve_data'] = curves

                    mask = np.ones(self.early_cycle_threshold - self.seq_len + 1)
                    mask[useable_cycle_number:] = 0
                    sample['curve_attn_mask'] = mask

                    padded_trajectory = np.zeros(self.max_trajectory_len)
                    trajectory_mask = np.zeros(self.max_trajectory_len)
                    padded_trajectory[:eol_cycle] = tmp_soh_trajectory
                    trajectory_mask[useable_cycle_number:] = 1
                    trajectory_mask[eol_cycle:] = 0

                    soh_input = tmp_soh_trajectory[:self.early_cycle_threshold].copy()
                    ces = np.zeros(self.early_cycle_threshold)
                    ees = np.zeros(self.early_cycle_threshold)
                    ces[:useable_cycle_number] = coulumbic_efficiencys[:useable_cycle_number]
                    ees[:useable_cycle_number] = energy_efficiencys[:useable_cycle_number]
                    soh_input[useable_cycle_number:] = 0

                    sample['soh_input'] = soh_input.reshape(-1, 1)
                    sample['CEs'] = ces.reshape(-1, 1)
                    sample['EEs'] = ees.reshape(-1, 1)
                    sample['soh_trajectory'] = padded_trajectory
                    sample['trajectory_mask'] = trajectory_mask
                    sample['eol_index'] = eol_cycle
                    sample['aging_condition_embedding'] = aging_condition_embedding_for_this_cell.reshape(1, -1)

                    self.samples.append(sample)

            elif self.input_mode == 'soh_to_soh':
                tmp_soh_trajectory = (soh_trajectory - eol_threshold) / (1.0 - eol_threshold)

                # NOTE: In your original code, soh_to_soh branch still uses CE/EE, but they are not computed here.
                # Keep it consistent with your original behavior: if you need CE/EE in this branch, compute curves first.
                # For now, we set them to zeros.
                for useable_cycle_number in range(self.seq_len, self.early_cycle_threshold + 1):
                    if useable_cycle_number > valid_cycle_number:
                        break

                    sample = {'file_name': file_name, 'nominal_capacity': nominal_capacity}

                    soh_input = tmp_soh_trajectory[:self.early_cycle_threshold].copy()
                    soh_input[useable_cycle_number:] = 0

                    ces = np.zeros(self.early_cycle_threshold)
                    ees = np.zeros(self.early_cycle_threshold)

                    sample['CEs'] = ces.reshape(-1, 1)
                    sample['EEs'] = ees.reshape(-1, 1)
                    sample['soh_input'] = soh_input.reshape(-1, 1)

                    padded_trajectory = np.zeros(self.max_trajectory_len)
                    trajectory_mask = np.zeros(self.max_trajectory_len)
                    padded_trajectory[:eol_cycle] = tmp_soh_trajectory
                    trajectory_mask[useable_cycle_number:] = 1
                    trajectory_mask[eol_cycle:] = 0

                    sample['soh_trajectory'] = padded_trajectory
                    sample['trajectory_mask'] = trajectory_mask
                    sample['eol_index'] = eol_cycle
                    sample['aging_condition_embedding'] = aging_condition_embedding_for_this_cell.reshape(1, -1)

                    self.samples.append(sample)

            elif self.input_mode == 'capacity_increment':
                curves, coulumbic_efficiencys, energy_efficiencys = self._prepare_charge_discharge_curves(
                    file_name, data['cycle_data'], nominal_capacity, start_soc, end_soc
                )

                tmp_soh_trajectory = (soh_trajectory - 0.8) / (1.0 - 0.8)
                for useable_cycle_number in range(self.seq_len, self.early_cycle_threshold + 1):
                    if useable_cycle_number > valid_cycle_number:
                        break

                    sample = {'file_name': file_name, 'nominal_capacity': nominal_capacity}
                    sample['cycle_curve_data'] = curves

                    mask = np.ones(self.early_cycle_threshold - self.seq_len + 1)
                    mask[useable_cycle_number:] = 0
                    sample['curve_attn_mask'] = mask

                    padded_trajectory = np.zeros(self.max_trajectory_len)
                    trajectory_mask = np.zeros(self.max_trajectory_len)
                    padded_trajectory[:eol_cycle] = tmp_soh_trajectory
                    trajectory_mask[useable_cycle_number:] = 1
                    trajectory_mask[eol_cycle:] = 0

                    soh_input = tmp_soh_trajectory[:self.early_cycle_threshold].copy()
                    soh_input[useable_cycle_number:] = 0

                    sample['soh_input'] = soh_input.reshape(-1, 1)
                    sample['soh_trajectory'] = padded_trajectory
                    sample['trajectory_mask'] = trajectory_mask
                    sample['rul_index'] = eol_cycle - useable_cycle_number

                    self.samples.append(sample)
            else:
                raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        # Label scaling
        if self.label_scaler is None:
            if self.input_mode != 'capacity_increment':
                scaler = StandardScaler()
                self.label_scaler = scaler.fit(np.array(total_life_labels).reshape(-1, 1))
                mean = self.label_scaler.mean_
                std = np.sqrt(self.label_scaler.var_)
                for sample in self.samples:
                    sample['eol_index'] = (sample['eol_index'] - mean[0]) / std[0]
        else:
            if self.input_mode != 'capacity_increment':
                for sample in self.samples:
                    sample['eol_index'] = (sample['eol_index'] - self.label_scaler.mean_) / np.sqrt(self.label_scaler.var_)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.input_mode == 'current_voltage':
            output = {
                'soh_trajectory': torch.FloatTensor(sample['soh_trajectory']),
                'trajectory_mask': torch.FloatTensor(sample['trajectory_mask']),
                'cycle_curve_data': torch.FloatTensor(sample['cycle_curve_data']),
                'curve_attn_mask': torch.FloatTensor(sample['curve_attn_mask']),
                'soh_input': torch.FloatTensor(sample['soh_input']),
                'eol_index': torch.FloatTensor([sample['eol_index']]),
                'aging_condition_embedding': torch.FloatTensor(sample['aging_condition_embedding']),
                'CEs': torch.FloatTensor(sample['CEs']),
                'EEs': torch.FloatTensor(sample['EEs']),
                'file_name': sample['file_name'],
            }
        elif self.input_mode == 'soh_to_soh':
            output = {
                'soh_trajectory': torch.FloatTensor(sample['soh_trajectory']),
                'trajectory_mask': torch.FloatTensor(sample['trajectory_mask']),
                'soh_input': torch.FloatTensor(sample['soh_input']),
                'eol_index': torch.FloatTensor([sample['eol_index']]),
                'aging_condition_embedding': torch.FloatTensor(sample['aging_condition_embedding']),
                'CEs': torch.FloatTensor(sample['CEs']),
                'EEs': torch.FloatTensor(sample['EEs']),
                'file_name': sample['file_name'],
            }
        elif self.input_mode == 'capacity_increment':
            output = {
                'soh_trajectory': torch.FloatTensor(sample['soh_trajectory']),
                'trajectory_mask': torch.FloatTensor(sample['trajectory_mask']),
                'cycle_curve_data': torch.FloatTensor(sample['cycle_curve_data']),
                'curve_attn_mask': torch.FloatTensor(sample['curve_attn_mask']),
                'soh_input': torch.FloatTensor(sample['soh_input']),
                'rul_index': torch.FloatTensor([sample['rul_index']]),
                'file_name': sample['file_name'],
            }
        else:
            raise Exception(f'{self.input_mode} is not supported! Please check the spell!')

        return output


class Dataset_SOH_to_SOH(Dataset_SOH_Forecasting):
    """Specialized dataset for SOH-to-SOH prediction"""
    def __init__(self, args, flag='train', max_trajectory_len=5200, alignment_check: bool = False):
        super().__init__(
            args,
            flag=flag,
            input_mode='soh_to_soh',
            max_trajectory_len=max_trajectory_len,
            alignment_check=alignment_check,
        )


class Dataset_IC2ML(Dataset_SOH_Forecasting):
    """Specialized dataset for IC2ML"""
    def __init__(self, args, flag='train', max_trajectory_len=5200, label_scaler=None, alignment_check: bool = False):
        super().__init__(
            args,
            flag=flag,
            input_mode='capacity_increment',
            max_trajectory_len=max_trajectory_len,
            label_scaler=label_scaler,
            alignment_check=alignment_check,
        )