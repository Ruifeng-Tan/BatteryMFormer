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
import hashlib
import tempfile
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
    DESs = torch.vstack([i['DESs'].unsqueeze(0) for i in samples])
    cycle_level_features = torch.cat([CEs, DESs], dim=-1)
    soh_trajectory = torch.vstack([i['soh_trajectory'].unsqueeze(0) for i in samples])
    trajectory_mask = torch.vstack([i['trajectory_mask'].unsqueeze(0) for i in samples])
    life_labels = torch.tensor([i['eol_index'] for i in samples])
    aging_condition_embedding = torch.vstack([i['aging_condition_embedding'].unsqueeze(0) for i in samples])

    # Collect file_names for evaluation (seen/unseen aging condition analysis)
    file_names = [i['file_name'] for i in samples]

    # Expand mask for the 4-channel data (Voltage, Current, Capacity, SOC)
    if cycle_curve_data is not None:
        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data) # [B, cycle_number, 4, fixed_len]
        cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

    soc_curves = cycle_curve_data[:,:,-1] # [B, cycle_number, fixed_len]
    cycle_curve_data = cycle_curve_data[:,:,:3] # [B, cycle_number, 3, fixed_len]

    return cycle_curve_data, curve_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names

# collate function for IC2ML model
def collate_fn_ic2ml(samples):
    """
    Custom collate function for SOH trajectory prediction
    Handles both current/voltage input and SOH-to-SOH input modes
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
    if cycle_curve_data is not None:
        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

    capacity_curves = cycle_curve_data[:,:,2] # [B, cycle_number, fixed_len]
    capacity_attn_mask = curve_attn_mask.unsqueeze(-1)

    return capacity_curves, capacity_attn_mask, soh_input, soh_trajectory, trajectory_mask, aging_condition_embedding, soc_curves, cycle_level_features, life_labels, file_names


class Dataset_SOH_Forecasting(Dataset):
    def __init__(self, args, flag='train', input_mode='current_voltage', max_trajectory_len=5200, label_scaler=None):
        """
        Optimized Dataset for SOH trajectory prediction with caching
        
        Args:
            args: Arguments containing dataset configuration
            flag: 'train', 'val', or 'test'
            input_mode: 'current_voltage' or 'soh_to_soh'
            max_trajectory_len: Maximum length of SOH trajectory (default 5200)
        """
        self.args = args
        self.flag = flag
        self.input_mode = input_mode
        self.max_trajectory_len = max_trajectory_len
        self.label_scaler = label_scaler
        
        # Basic parameters
        self.ZN_coin_charge_first_file_names = ['ZN-coin_402-1_20231209225636_01_1.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl', 'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_430-3_20231212185323_03_2.pkl']
        self.random_seed = args.seed
        self.root_path = args.root_path if hasattr(args, 'root_path') else '/ai/dl_project/MemoryNet/dataset/cleaned_data'
        self.processed_SOH_path = args.processed_SOH_path
        self.dataset_name = args.dataset
        self.early_cycle_threshold = args.early_cycle_threshold if hasattr(args, 'early_cycle_threshold') else 100
        self.charge_discharge_len = args.charge_discharge_length if hasattr(args, 'charge_discharge_length') else 100
        self.seq_len = args.seq_len if hasattr(args, 'seq_len') else 100  # For SOH-to-SOH input

        # New feature: capacity-based resampling (default: False for backward compatibility)
        self.use_capacity_resample = getattr(args, 'use_capacity_resample', False)
        # Number of segments for hierarchical embedding (default: 5)
        self.num_segments = getattr(args, 'num_segments', 5)

        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']


        # Cache directory for preprocessed data
        
        # Load data split
        self._load_data_files()
        
        # Try to load from cache first
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
            # Load and prepare data
            self.samples = []
            self._prepare_samples()


            print(f"Saving preprocessed data to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)

        print(f"Loaded {len(self.samples)} samples for {flag} set")

    def read_cell_data_according_to_prefix(self, file_name):
        '''
        Read the battery data and eol according to the file_name
        The dataset is indicated by the prefix of the file_name
        '''
        prefix = file_name.split('_')[0]

        if prefix.startswith('MATR'):
            data =  pickle.load(open(f'{self.root_path}/MATR/{file_name}', 'rb'))
        elif prefix.startswith('HUST'):
            data =  pickle.load(open(f'{self.root_path}/HUST/{file_name}', 'rb'))
        elif prefix.startswith('SNL'):
            data =  pickle.load(open(f'{self.root_path}/SNL/{file_name}', 'rb'))
        elif prefix.startswith('CALCE'):
            data =  pickle.load(open(f'{self.root_path}/CALCE/{file_name}', 'rb'))
        elif prefix.startswith('HNEI'):
            data =  pickle.load(open(f'{self.root_path}/HNEI/{file_name}', 'rb'))
        elif prefix.startswith('MICH'):
            data =  pickle.load(open(f'{self.root_path}/total_MICH/{file_name}', 'rb'))
        elif prefix.startswith('OX'):
            data =  pickle.load(open(f'{self.root_path}/OX/{file_name}', 'rb'))
        elif prefix.startswith('RWTH'):
            data =  pickle.load(open(f'{self.root_path}/RWTH/{file_name}', 'rb'))  
        elif prefix.startswith('UL-PUR'):
            data =  pickle.load(open(f'{self.root_path}/UL_PUR/{file_name}', 'rb'))  
        elif prefix.startswith('SMICH'):
            data =  pickle.load(open(f'{self.root_path}/MICH_EXP/{file_name[1:]}', 'rb')) 
        elif prefix.startswith('BIT2'):
            data =  pickle.load(open(f'{self.root_path}/BIT2/{file_name}', 'rb')) 
        elif prefix.startswith('Tongji'):
            data =  pickle.load(open(f'{self.root_path}/Tongji/{file_name}', 'rb'))
        elif prefix.startswith('Stanford'):
            file_name = file_name.replace('_Ref', '')
            data =  pickle.load(open(f'{self.root_path}/Stanford_2/{file_name}', 'rb'))
        elif prefix.startswith('ISU-ILCC'):
            data =  pickle.load(open(f'{self.root_path}/ISU_ILCC/{file_name}', 'rb'))
        elif prefix.startswith('XJTU'):
            data =  pickle.load(open(f'{self.root_path}/XJTU/{file_name}', 'rb'))
        elif prefix.startswith('ZN-coin'):
            data =  pickle.load(open(f'{self.root_path}/ZN-coin/{file_name}', 'rb'))
        elif prefix.startswith('CALB'):
            data =  pickle.load(open(f'{self.root_path}/CALB/{file_name}', 'rb'))
        elif prefix.startswith('NA-ion'):
            data =  pickle.load(open(f'{self.root_path}/NA-ion/{file_name}', 'rb'))

        return data

    def _get_cache_key(self):
        """Generate unique cache key based on configuration"""
        # Updated suffix to _v3_soc to invalidate old caches without SOC data
        key_str = f"{self.dataset_name}_{self.args.seed}_{self.flag}_{self.input_mode}_{self.early_cycle_threshold}_{self.charge_discharge_len}_{self.seq_len}_{self.max_trajectory_len}_capresample{self.use_capacity_resample}_v3_soc"
        # Append fewshot suffix for train set when fewshot_ratio is active
        fewshot_ratio = getattr(self.args, 'fewshot_ratio', None)
        if fewshot_ratio and self.flag == 'train' and 0 < fewshot_ratio < 1:
            key_str += f"_fewshot{int(fewshot_ratio * 100)}"
        return key_str
    
    def _load_data_files(self):
        """Load train/val/test file splits"""
        # Check if fewshot mode is active for training
        fewshot_ratio = getattr(self.args, 'fewshot_ratio', None)
        if fewshot_ratio and self.flag == 'train' and 0 < fewshot_ratio < 1:
            fewshot_pct = int(fewshot_ratio * 100)
            if self.dataset_name != 'Li_ion':
                split_json = f'./data_provider/split_json/fewshot/{self.dataset_name}_split_{self.random_seed}_fewshot{fewshot_pct}.json'
            else:
                split_json = f'./data_provider/split_json/fewshot/total_split_{self.random_seed}_fewshot{fewshot_pct}.json'
            print(f"Using fewshot split: {split_json}")
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
        """Replace NaN/Inf values with a safe fill value while preserving shape."""
        arr = np.asarray(values, dtype=np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            # print(arr)
            # raise Exception('NaN or Inf values found in vector')
            return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
        return arr

    def _calculate_soh_trajectory(self, cycle_data, dataset_name, nominal_capacity, soc_interval=None):
        """
        Calculate SOH trajectory from cycle data with discharge depth correction
        """
        if len(cycle_data) == 0:
            return np.array([])

        if soc_interval is None:
            soc_interval = [0, 1]

        discharge_depth = soc_interval[1] - soc_interval[0]
        discharge_depth = max(discharge_depth, 1e-6)

        if dataset_name == 'CALB':
            first_cycle_caps = self._sanitize_vector(cycle_data[0].get('discharge_capacity_in_Ah', []), fill_value=0.0)
            denominator = float(first_cycle_caps.max()) if first_cycle_caps.size > 0 else 1.0
            denominator = max(denominator, 1e-6)
        else:
            denominator = max(float(nominal_capacity), 1e-6)

        soh_trajectory = np.zeros(len(cycle_data))
        for i, cycle in enumerate(cycle_data):
            discharge_curve = self._sanitize_vector(cycle.get('discharge_capacity_in_Ah', []), fill_value=0.0)
            if discharge_curve.size > 0:
                current_capacity = float(discharge_curve.max(initial=0.0))
                capacity_ratio = current_capacity / denominator
                soh_trajectory[i] = capacity_ratio / discharge_depth
            else:
                raise Exception('The discharge data should not be empty!')

        return soh_trajectory
    
    def _prepare_charge_discharge_curves(self, file_name,  cycle_data, nominal_capacity, start_soc, end_soc):
        """
        Prepare charge/discharge curves including SOC calculation.
        Returns array of shape [early_cycles, 4, charge_discharge_len]
        Channels: [Voltage, Current, Capacity, SOC]
        """
        curves = []
        coulumbic_efficiencys = [] 
        energy_efficiencys = []
        
        # Calculate the SOC swing range
        soc_range = end_soc - start_soc
        prefix = file_name.split('_')[0]
        if prefix == 'CALB':
            prefix = file_name.split('_')[:2]
            prefix = '_'.join(prefix)

        for cycle_idx in range(self.seq_len, self.early_cycle_threshold+1):
            if cycle_idx < len(cycle_data):
                cycle = cycle_data[cycle_idx-1]

                # Extract data
                cycle_df = pd.DataFrame()
                for key in self.need_keys:
                    cycle_df[key] = cycle[key]

                # Data cleaning
                cycle_df.loc[cycle_df['charge_capacity_in_Ah']<0] = np.nan
                cycle_df.loc[cycle_df['discharge_capacity_in_Ah']<0] = np.nan
                cycle_df.bfill(inplace=True)
                cycle_df.ffill(inplace=True)

                time_in_s_records = self._sanitize_vector(cycle_df['time_in_s'])
                time_in_h_records = time_in_s_records / 3600
                voltage = self._sanitize_vector(cycle_df['voltage_in_V'])
                current = self._sanitize_vector(cycle_df['current_in_A'])
                charge_cap = self._sanitize_vector(cycle_df['charge_capacity_in_Ah'])
                discharge_cap = self._sanitize_vector(cycle_df['discharge_capacity_in_Ah'])
                

                # Find charge and discharge segments
                current_c_rate = current / nominal_capacity
                charge_indices = current_c_rate >= 0.01
                discharge_indices = current_c_rate <= -0.01

                cutoff_voltage_indices = np.nonzero(current_c_rate>=0.01) # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
                charge_end_index = cutoff_voltage_indices[0][-1] # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

                cutoff_voltage_indices = np.nonzero(current_c_rate<=-0.01) 
                discharge_end_index = cutoff_voltage_indices[0][-1]
                
                if prefix in ['RWTH', 'CALB_0', 'CALB_25', 'CALB_45'] or (file_name not in self.ZN_coin_charge_first_file_names and prefix=='ZN-coin'):
                    # Every cycle first discharge and then charge
                    #capacity_in_battery = np.where(charge_capacity_records==0, discharge_capacity_records, charge_capacity_records)
                    
                    v_dis = voltage[:discharge_end_index]
                    c_dis = discharge_cap[:discharge_end_index]
                    i_dis = current[:discharge_end_index]
                    time_in_h_dis = time_in_h_records[:discharge_end_index]
                    
                    v_chg = voltage[discharge_end_index:]
                    c_chg = charge_cap[discharge_end_index:]
                    i_chg = current[discharge_end_index:]
                    i_chg_in_c_rates = i_chg / nominal_capacity
                    time_in_h_chg = time_in_h_records[discharge_end_index:]
                    
                    v_chg = v_chg[np.abs(i_chg_in_c_rates)>0.01]
                    c_chg = c_chg[np.abs(i_chg_in_c_rates)>0.01]
                    time_in_h_chg = time_in_h_chg[np.abs(i_chg_in_c_rates)>0.01]
                    i_chg = i_chg[np.abs(i_chg_in_c_rates)>0.01]

                    power_dis = v_dis * np.abs(i_dis)
                    energy_discharge = np.trapz(power_dis, time_in_h_dis)

                    power_chg = v_chg * np.abs(i_chg)
                    energy_charge = np.trapz(power_chg, time_in_h_chg)
                else:
                    # Every cycle first charge and then discharge
                    #capacity_in_battery = np.where(np.logical_and(current_records>=-(nominal_capacity*0.01), discharge_capacity_records<=nominal_capacity*0.01), charge_capacity_records, discharge_capacity_records)
                    v_dis = voltage[charge_end_index:]
                    c_dis = discharge_cap[charge_end_index:]
                    i_dis = current[charge_end_index:]
                    i_dis_in_c_rates = i_dis / nominal_capacity
                    time_in_h_dis = time_in_h_records[charge_end_index:]
                    
                    v_dis = v_dis[np.abs(i_dis_in_c_rates)>0.01]
                    c_dis = c_dis[np.abs(i_dis_in_c_rates)>0.01]
                    time_in_h_dis = time_in_h_dis[np.abs(i_dis_in_c_rates)>0.01]
                    i_dis = i_dis[np.abs(i_dis_in_c_rates)>0.01]
                    
                    
                    v_chg = voltage[:charge_end_index]
                    c_chg = charge_cap[:charge_end_index]
                    i_chg = current[:charge_end_index]
                    time_in_h_chg = time_in_h_records[:charge_end_index]

                    power_dis = v_dis * np.abs(i_dis)
                    energy_discharge = np.trapz(power_dis, time_in_h_dis)

                    power_chg = v_chg * np.abs(i_chg)
                    energy_charge = np.trapz(power_chg, time_in_h_chg)
                
                EE = energy_discharge / energy_charge
                CE = max(c_dis) / max(c_chg) # Coulombic efficiency

                # if np.isinf(EE):
                #     print(cycle_idx)
                #     print(file_name, energy_discharge, energy_charge, time_in_h_chg, discharge_end_index, len(voltage))
                    
                # --- 2. Calculate Raw SOC Sequences ---
                # Charge: SOC increases from start_soc to end_soc based on accumulated capacity
                if len(c_chg) > 0:
                    c_chg_delta = c_chg.max() - c_chg.min()
                    # Normalize capacity 0~1 and map to SOC range
                    soc_chg = start_soc + ((c_chg - c_chg[0]) / c_chg_delta) * soc_range
                else:
                    soc_chg = np.array([])

                # Discharge: SOC decreases from end_soc to start_soc
                if len(c_dis) > 0:
                    c_dis_delta = c_dis.max() - c_dis.min()
                    # Discharge capacity usually starts at 0 and increases as battery empties
                    # So: SOC = End - (Capacity_Used / Total_Capacity_Used) * Range
                    soc_dis = end_soc - ((c_dis - c_dis[0]) / c_dis_delta) * soc_range
                else:
                    soc_dis = np.array([])

                # --- 3. Resample ---
                charge_len = self.charge_discharge_len // 2
                discharge_len = self.charge_discharge_len // 2

                if self.use_capacity_resample:
                    if 'BattMFormer' in self.args.model:
                        # SOC-based interpolation (Physical Alignment)
                        charge_v, charge_i, charge_c, charge_s = self._capacity_based_resample(
                            v_chg, i_chg, soc_chg, c_chg, charge_len, is_charge=True
                        )
                        discharge_v, discharge_i, discharge_c, discharge_s = self._capacity_based_resample(
                            v_dis, i_dis, soc_dis, c_dis, discharge_len, is_charge=False
                        )
                    else:
                        # Capacity-based interpolation (Physical Alignment)
                        charge_v, charge_i, charge_c, charge_s = self._capacity_based_resample(
                            v_chg, i_chg, c_chg, soc_chg, charge_len, is_charge=True
                        )
                        discharge_v, discharge_i, discharge_c, discharge_s = self._capacity_based_resample(
                            v_dis, i_dis, c_dis, soc_dis, discharge_len, is_charge=False
                        )
                else:
                    # Index-based interpolation (Fast)
                    charge_v = self._fast_resample(v_chg, charge_len)
                    charge_i = self._fast_resample(i_chg, charge_len)
                    charge_c = self._fast_resample(c_chg, charge_len)
                    charge_s = self._fast_resample(soc_chg, charge_len)

                    discharge_v = self._fast_resample(v_dis, discharge_len)
                    discharge_i = self._fast_resample(i_dis, discharge_len)
                    discharge_c = self._fast_resample(c_dis, discharge_len)
                    discharge_s = self._fast_resample(soc_dis, discharge_len)

                # --- 4. Combine and Stack ---
                if self.input_mode == 'capacity_increment':
                    # For capacity increment mode, we only need charge capacity data
                    voltage_curve = self._sanitize_vector(charge_v)
                    current_curve = self._sanitize_vector(charge_i)
                    capacity_curve = self._sanitize_vector(charge_c)
                    soc_curve = self._sanitize_vector(charge_s)

                    # Normalize capacity increment
                    capacity_increment_curve = (capacity_curve - capacity_curve[0]) / nominal_capacity

                    # Normalize current to C-rate
                    if nominal_capacity == 0 or not nominal_capacity:
                        raise Exception('Nominal capacity should not be none or 0 Ah!')
                    current_curve = current_curve / nominal_capacity

                    # Stack: [4, Length] -> [Voltage, Current, Capacity, SOC]
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

                    # Normalize current to C-rate
                    if nominal_capacity == 0 or not nominal_capacity:
                        raise Exception('Nominal capacity should not be none or 0 Ah!')
                    current_curve = current_curve / nominal_capacity

                    # Stack: [4, Length] -> [Voltage, Current, Capacity, SOC]
                    curve = np.stack([voltage_curve, current_curve, capacity_curve, soc_curve])
                    curve = self._sanitize_vector(curve)
            else:
                # Pad with zeros if not enough cycles (4 channels now)
                # raise Exception(f'{file_name} cell does not have enough cycles to form a complete input!')
                if self.input_mode == 'capacity_increment':
                    curve = np.zeros((4, self.charge_discharge_len//2))
                else:
                    curve = np.zeros((4, self.charge_discharge_len))
                
                CE = 0
                EE = 0

            coulumbic_efficiencys.append(CE)
            energy_efficiencys.append(EE)

            if self.input_mode == 'capacity_increment':
                # only use charge data
                curves.append(curve.reshape(1, 4, self.charge_discharge_len//2))
            else:
                curves.append(curve.reshape(1, 4, self.charge_discharge_len))

        curves = np.concatenate(curves, axis=0)
        return np.array(curves), np.array(coulumbic_efficiencys), np.array(energy_efficiencys)

    def _capacity_based_resample(self, voltage, current, capacity, soc, target_length, is_charge=True):
        """
        Resample curves including SOC based on capacity alignment.
        """
        if len(capacity) == 0 or len(voltage) == 0:
            return np.zeros(target_length), np.zeros(target_length), np.zeros(target_length), np.zeros(target_length)

        voltage = self._sanitize_vector(voltage)
        current = self._sanitize_vector(current)
        capacity = self._sanitize_vector(capacity)
        soc = self._sanitize_vector(soc)

        # Handle dirty data: ensure capacity is monotonically increasing
        capacity_monotonic = np.maximum.accumulate(capacity)

        # Get max capacity for this cycle (for normalization)
        max_cap = capacity_monotonic[-1] if len(capacity_monotonic) > 0 else 1.0
        max_cap = max(max_cap, 1e-6)

        # Normalize capacity to [0, 1] relative to this specific cycle
        capacity_norm = capacity_monotonic / max_cap

        # Create target points (equally spaced from 0 to 1)
        target_norm_indices = np.linspace(0, 1, target_length)

        # Find valid indices (strictly increasing capacity)
        valid_mask = np.concatenate([[True], np.diff(capacity_norm) > 1e-8])

        if np.sum(valid_mask) < 2:
            # Fallback to fast resample if capacity data is bad
            return (self._fast_resample(voltage, target_length),
                    self._fast_resample(current, target_length),
                    self._fast_resample(capacity, target_length),
                    self._fast_resample(soc, target_length))

        capacity_valid = capacity_norm[valid_mask]
        voltage_valid = voltage[valid_mask]
        current_valid = current[valid_mask]
        soc_valid = soc[valid_mask]

        # Interpolate all variables based on the normalized capacity axis
        resampled_voltage = np.interp(target_norm_indices, capacity_valid, voltage_valid)
        resampled_current = np.interp(target_norm_indices, capacity_valid, current_valid)
        resampled_soc = np.interp(target_norm_indices, capacity_valid, soc_valid)
        
        # Capacity is reconstructed from the target indices
        resampled_capacity = target_norm_indices * max_cap

        return (self._sanitize_vector(resampled_voltage),
                self._sanitize_vector(resampled_current),
                self._sanitize_vector(resampled_capacity),
                self._sanitize_vector(resampled_soc))
    
    def _fast_resample(self, curve, target_length):
        """Fast resampling using numpy interp"""
        if len(curve) == 0:
            return np.zeros(target_length)
        curve = self._sanitize_vector(curve)

        # Use linear interpolation for speed
        old_indices = np.linspace(0, len(curve)-1, len(curve))
        new_indices = np.linspace(0, len(curve)-1, target_length)
        resampled = np.interp(new_indices, old_indices, curve)
        return self._sanitize_vector(resampled)
    
    def _prepare_samples(self):
        """Prepare all samples for the dataset with optimizations"""
        # Pre-load embeddings once (instead of loading for each file)
        Qwen3_aging_condition_embeddings = pickle.load(
            open('data_provider/prompt_embeddings/Qwen3_total.pkl', 'rb')
        )

        total_life_labels = []
        print('Files:', len(self.files))
        for file_name in self.files:
            data = self.read_cell_data_according_to_prefix(file_name)

            # Load full cycle data from cleaned_data
            if self.dataset_name == 'NA-ion' or self.dataset_name == 'ZN-coin' or self.dataset_name == 'CALB':
                dataset_dir = file_name.split('_')[0]
            else:
                dataset_dir = file_name.split('_')[0]
                dataset_dir = dataset_dir.replace('-', '_')
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

            # Try to load SOH from separate processed_SOH directory first
            soh_dir_root = self.processed_SOH_path
            soh_file_path = os.path.join(soh_dir_root, dataset_dir, file_name)

            if os.path.exists(soh_file_path):
                with open(soh_file_path, 'rb') as f:
                    soh_data = pickle.load(f)
                soh_trajectory = np.array(soh_data['SOH'], dtype=np.float32)
            else:
                print('No SOH file found for:', soh_file_path)
                continue

            # Find EOL index
            eol_threshold = 0.8 if dataset_dir != 'CALB' else 0.9
            eol_cycle = len(soh_trajectory)
            # # Truncate trajectory if exceeds max_trajectory_len
            # if eol_cycle > self.max_trajectory_len:
            #     soh_trajectory = soh_trajectory[:self.max_trajectory_len]
            #     eol_cycle = self.max_trajectory_len

            total_life_labels.append(eol_cycle)
            if eol_cycle <= self.early_cycle_threshold:
                continue

            nominal_capacity = data['nominal_capacity_in_Ah']
            soc_interval = data['SOC_interval']
            start_soc, end_soc = soc_interval[0], soc_interval[1]
            if nominal_capacity is None or not np.isfinite(nominal_capacity) or nominal_capacity <= 0:
                raise Exception(f'{file_name} should have nominal capacity in Ah!')

            # Get embedding for this cell (pre-loaded above)
            aging_condition_embedding_for_this_cell = Qwen3_aging_condition_embeddings[file_name]
            valid_cycle_number = len(data['cycle_data']) 
            # Prepare sample based on input mode
            if self.input_mode == 'current_voltage':
                # Mode 1: Use current/voltage/capacity/SOC as input
                # Returns shape: [early_cycles, 4, fixed_len]
                curves, coulumbic_efficiencys, discharge_efficiencys = self._prepare_charge_discharge_curves(
                    file_name,
                    data['cycle_data'], 
                    nominal_capacity,
                    start_soc,
                    end_soc
                )
                
                tmp_soh_trajectory = (soh_trajectory - eol_threshold) / (1.0 - eol_threshold)

                for useable_cycle_number in range(self.seq_len, self.early_cycle_threshold+1):
                    if useable_cycle_number > valid_cycle_number:
                        # only effective for some CALB batteries that have only cycling data of 99 cycles available for modeling.
                        break
                    sample = {
                    'file_name': file_name,
                    'nominal_capacity': nominal_capacity
                    }
                    sample['cycle_curve_data'] = curves
                
                    # Create attention mask for early cycles
                    mask = np.ones(self.early_cycle_threshold - self.seq_len + 1)
                    mask[useable_cycle_number:] = 0 # zero indicates unseen data
                    sample['curve_attn_mask'] = mask

                    # Prepare SOH trajectory
                    padded_trajectory = np.zeros(self.max_trajectory_len)
                    trajectory_mask = np.zeros(self.max_trajectory_len)
                    padded_trajectory[:eol_cycle] = tmp_soh_trajectory
                    trajectory_mask[useable_cycle_number:] = 1 
                    trajectory_mask[eol_cycle:] = 0

                    soh_input = tmp_soh_trajectory[:self.early_cycle_threshold].copy()

                    ces = np.zeros(self.early_cycle_threshold)
                    des = np.zeros(self.early_cycle_threshold)
                    ces[:useable_cycle_number] = coulumbic_efficiencys[:useable_cycle_number]
                    des[:useable_cycle_number] = discharge_efficiencys[:useable_cycle_number]
                    soh_input[useable_cycle_number:] = 0
                    sample['soh_input'] = soh_input.reshape(-1, 1)
                    sample['CEs'] = ces.reshape(-1, 1)
                    sample['DESs'] = des.reshape(-1, 1)
                    sample['soh_trajectory'] = padded_trajectory
                    sample['trajectory_mask'] = trajectory_mask
                    sample['eol_index'] = eol_cycle
                    sample['aging_condition_embedding'] = aging_condition_embedding_for_this_cell.reshape(1, -1)
                    
                    self.samples.append(sample)

            elif self.input_mode == 'soh_to_soh':
                # Mode 2: Use historical SOH as input
                tmp_soh_trajectory = (soh_trajectory - eol_threshold) / (1.0 - eol_threshold)
                for useable_cycle_number in range(self.seq_len, self.early_cycle_threshold+1):
                    if useable_cycle_number > valid_cycle_number:
                        # only effective for some CALB batteries that have only cycling data of 99 cycles available for modeling.
                        break

                    sample = {
                    'file_name': file_name,
                    'nominal_capacity': nominal_capacity
                    }
                    soh_input = tmp_soh_trajectory[:self.early_cycle_threshold].copy()
                    soh_input[useable_cycle_number:] = 0
                    ces = np.zeros(self.early_cycle_threshold)
                    des = np.zeros(self.early_cycle_threshold)
                    ces[:useable_cycle_number] = coulumbic_efficiencys[:useable_cycle_number]
                    des[:useable_cycle_number] = discharge_efficiencys[:useable_cycle_number]
                    sample['CEs'] = ces.reshape(-1, 1)
                    sample['DESs'] = des.reshape(-1, 1)
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
                # Mode 3: Use charge capacity increment curves as input
                curves, coulumbic_efficiencys, discharge_efficiencys = self._prepare_charge_discharge_curves(
                    file_name,
                    data['cycle_data'], 
                    nominal_capacity,
                    start_soc,
                    end_soc
                )

                tmp_soh_trajectory = (soh_trajectory - 0.8) / (1.0 - 0.8)
                for useable_cycle_number in range(self.seq_len, self.early_cycle_threshold+1):
                    if useable_cycle_number > valid_cycle_number:
                        # only effective for some CALB batteries that have only cycling data of 99 cycles available for modeling.
                        break

                    sample = {
                    'file_name': file_name,
                    'nominal_capacity': nominal_capacity
                    }
                    sample['cycle_curve_data'] = curves
                
                    # Create attention mask for early cycles
                    mask = np.ones(self.early_cycle_threshold - self.seq_len + 1)
                    mask[useable_cycle_number:] = 0 # zero indicates unseen data
                    sample['curve_attn_mask'] = mask

                    # Prepare SOH trajectory
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

        
        if self.label_scaler is None:
            if self.input_mode != 'capacity_increment':
                scaler = StandardScaler()
                self.label_scaler = scaler.fit(np.array(total_life_labels).reshape(-1, 1))
                mean = self.label_scaler.mean_
                std = np.sqrt(self.label_scaler.var_)
                for i, sample in enumerate(self.samples):
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
                'DESs': torch.FloatTensor(sample['DESs']),
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
                'DESs': torch.FloatTensor(sample['DESs']),
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
    def __init__(self, args, flag='train', max_trajectory_len=5200):
        super().__init__(args, flag, input_mode='soh_to_soh', max_trajectory_len=max_trajectory_len)

# the dataloader for IC2ML model
class Dataset_IC2ML(Dataset_SOH_Forecasting):
    """Specialized dataset for SOH-to-SOH prediction"""
    def __init__(self, args, flag='train', max_trajectory_len=5200, label_scaler=None):
        super().__init__(args, flag, input_mode='capacity_increment', max_trajectory_len=max_trajectory_len, label_scaler=label_scaler)