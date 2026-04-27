import os
import pandas as pd
import json
import random
import math

# Set a seed for reproducibility
random_seed = 2024
random.seed(random_seed)

processed_SOH_dir = '/path/to/your/processed_SOH'
total_datasets = ['CALB', 'NA-ion', 'ZN-coin', 'CALCE', 'HNEI', 'HUST', 'ISU_ILCC', 'MATR', 'RWTH', 'SNL', 'Stanford_2', 'Tongji', 'total_MICH', 'UL_PUR', 'XJTU']

train_cells = []
val_cells = []
test_cells = []

name2agingConditionID = json.load(open('./name2agingConditionID.json'))

for dataset in total_datasets:
    dataset_path = os.path.join(processed_SOH_dir, dataset)
    if not os.path.isdir(dataset_path):
        continue
    
    dataset_cells = os.listdir(dataset_path)
    # if dataset == 'Tongji':
    #     dataset_cells = [file_name.replace('-#', '--') for file_name in dataset_cells]
    
    # 1. Group cells by Aging Condition ID
    # Structure: { condition_id: [full_path_to_cell1, full_path_to_cell2, ...] }
    condition_groups = {}
    
    for cell in dataset_cells:
        # Get ID from map. Assuming filename matches key. 
        # If keys in json don't have extensions but files do, you might need os.path.splitext(cell)[0]
        if cell in name2agingConditionID:
            cond_id = name2agingConditionID[cell]
            if cond_id not in condition_groups:
                condition_groups[cond_id] = []
            # Store full path for the final list
            condition_groups[cond_id].append(cell)
    
    unique_conditions = list(condition_groups.keys())
    random.shuffle(unique_conditions) # Shuffle to ensure random selection of conditions
    
    total_cells_count = sum(len(v) for v in condition_groups.values())
    
    # --- SCENARIO 1: Multiple Aging Conditions ---
    if len(unique_conditions) > 1:
        # Calculate minimum conditions needed for test (10%, at least 1)
        min_test_conds = max(1, math.ceil(len(unique_conditions) * 0.1))
        
        test_condition_ids = []
        current_test_count = 0
        
        # Select conditions for Test Set
        # Logic: Add conditions until we meet the 10% count rule AND the 20% volume rule
        # Constraint: Must leave at least 1 condition for Train/Val
        for cond_id in unique_conditions:
            # Stop if we have enough conditions AND enough volume, 
            # OR if adding this would take the last available condition (leaving none for train)
            if (len(test_condition_ids) >= min_test_conds and current_test_count >= 0.2 * total_cells_count):
                break
            
            # Safety check: Don't take the last condition if we already have at least one for test
            if len(test_condition_ids) > 0 and len(test_condition_ids) == len(unique_conditions) - 1:
                break
                
            test_condition_ids.append(cond_id)
            current_test_count += len(condition_groups[cond_id])
            
        # Assign cells based on selected conditions
        dataset_test = []
        dataset_train_val_pool = []
        
        for cond_id in unique_conditions:
            if cond_id in test_condition_ids:
                dataset_test.extend(condition_groups[cond_id])
            else:
                dataset_train_val_pool.extend(condition_groups[cond_id])
        
        # Split the remaining (Seen conditions) into Train (3) : Val (1)
        random.shuffle(dataset_train_val_pool)
        split_idx = int(len(dataset_train_val_pool) * 0.75) # 75% for train is 3:1 ratio
        
        dataset_train = dataset_train_val_pool[:split_idx]
        dataset_val = dataset_train_val_pool[split_idx:]

    # --- SCENARIO 2: Single Aging Condition ---
    else:
        # Flatten the list of cells
        all_cells = []
        for cells in condition_groups.values():
            all_cells.extend(cells)
            
        random.shuffle(all_cells)
        n = len(all_cells)
        
        # 6:2:2 Split
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        # Remainder goes to test to ensure no data loss due to rounding
        
        dataset_train = all_cells[:n_train]
        dataset_val = all_cells[n_train : n_train + n_val]
        dataset_test = all_cells[n_train + n_val:]

    # Add to global lists
    train_cells.extend(dataset_train)
    val_cells.extend(dataset_val)
    test_cells.extend(dataset_test)

    print(f"Dataset: {dataset} | Conditions: {len(unique_conditions)}")
    print(f"  Train: {len(dataset_train)} | {dataset_train}")
    print(f"  Val: {len(dataset_val)} | {dataset_val}")
    print(f"  Test: {len(dataset_test)} | {dataset_test}")
    print('\n')
    # save the dataset split for each dataset
    dataset_save_pkl = {
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test
    }

    json.dump(dataset_save_pkl, open(os.path.join('./data_provider/split_json', f'{dataset}_split_{random_seed}.json'), 'w'))


total_save_pkl = {
    'train': train_cells,
    'val': val_cells,
    'test': test_cells
}
json.dump(total_save_pkl, open(os.path.join('./data_provider/split_json', f'total_split_{random_seed}.json'), 'w'))
print("-" * 30)
print(f"Total Train: {len(train_cells)}")
print(f"Total Val:   {len(val_cells)}")
print(f"Total Test:  {len(test_cells)}")
