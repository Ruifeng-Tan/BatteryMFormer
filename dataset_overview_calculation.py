import json
import os
import pickle
# get all data split json files
data_split_path = './data_provider/split_json/'
data_split_json_files_path = os.listdir(data_split_path)
data_split_json_files = [i for i in data_split_json_files_path if i.endswith('.json')]
data_split_json_files = [i for i in data_split_json_files if not i.startswith('Stanford_split_') and not i.startswith('total')]

# get all labeled sample names
# tongji: -#
procssed_soh_path = '/data/trf/python_works/BatteryLife/dataset/processed_SOH'
dataset_names = ['CALB', 'NA-ion', 'ZN-coin', 'CALCE', 'HNEI', 'HUST', 'ISU_ILCC', 'MATR', 'MICH', 'MICH_EXP', 'RWTH', 'SNL', 'Stanford_2',
'Tongji', 'UL_PUR', 'XJTU']

label_names = []
# for file in label_json_files:
#     if file.startswith('Stanford_labels'):
#         continue
#     with open(os.path.join(label_path, file), 'r') as f:
#         label_data = json.load(f)
#     # print(file, len(label_data))
#     for key, value in label_data.items():
#         filename = key.split('.pkl')[0]
#         label_names.append(filename)
for dataset_name in dataset_names:
    dataset_dir = os.path.join(procssed_soh_path, dataset_name)
    cell_files = os.listdir(dataset_dir)
    for cell_file in cell_files:
        cell_data = pickle.load(open(os.path.join(dataset_dir, cell_file), 'rb'))
        life_label = len(cell_data['SOH'])
        if life_label > 100:
            label_names.append(cell_file)
        
    
print(len(label_names))
# get aging condition mapping
with open('./name2agingConditionID.json') as file:
    aging_conditions = json.load(file)


dataset_list = ['Li-ion', 'Na-ion', 'Zn-ion', 'CALB']
seed_list = [2021]
for dataset in dataset_list:
    for seed in seed_list:
        if dataset == 'Li-ion':
            datasets = [i for i in data_split_json_files if not i.startswith('NA') and not i.startswith('ZN') and not i.startswith('CALB')]
            if seed == 42:
                datasets = [i for i in datasets if '42' in i]
            elif seed == 2021:
                datasets = [i for i in datasets if '2021' in i]
            elif seed == 2024:
                datasets = [i for i in datasets if '2024' in i]
                
        elif dataset == 'Na-ion':
            datasets = [i for i in data_split_json_files if i.startswith('NA-ion')]
            if seed == 42:
                datasets = [i for i in datasets if '42' in i]
            elif seed == 2021:
                datasets = [i for i in datasets if '2021' in i]
            elif seed == 2024:
                datasets = [i for i in datasets if '2024' in i]
                
        elif dataset == 'Zn-ion':
            datasets = [i for i in data_split_json_files if i.startswith('ZN-coin')]
            if seed == 42:
                datasets = [i for i in datasets if '42' in i]
            elif seed == 2021:
                datasets = [i for i in datasets if '2021' in i]
            elif seed == 2024:
                datasets = [i for i in datasets if '2024' in i]

        elif dataset == 'CALB':
            datasets = [i for i in data_split_json_files if i.startswith('CALB')]
            if seed == 42:
                datasets = [i for i in datasets if '42' in i]
            elif seed == 2021:
                datasets = [i for i in datasets if '2021' in i]
            elif seed == 2024:
                datasets = [i for i in datasets if '2024' in i]
        

        print('Dataset:', dataset, 'Seed:', seed)
        total_files = []
        for dataset_name in datasets:
            dataset_path = os.path.join(data_split_path, dataset_name)
            split_record = json.load(open(dataset_path, 'r'))
            dataset_files = split_record['train'] + split_record['val'] + split_record['test']
            used_dataset_files = []
            for dataset_file in dataset_files:
                if dataset_file in label_names:
                    # The cell should be in the split and have soh degradation trajectory
                    used_dataset_files.append(dataset_file)
            total_files = total_files + used_dataset_files
        # # count the available battery number
        # # tongji: --
        # total_samples_name = []
        # for file in datasets:
        #     with open(os.path.join(data_split_path, file), 'r') as f:
        #         data_split = json.load(f)
        #     total_samples_name.extend([i.split('.pkl')[0] for i in data_split['train']])
        #     total_samples_name.extend([i.split('.pkl')[0] for i in data_split['val']])
        #     total_samples_name.extend([i.split('.pkl')[0] for i in data_split['test']])

        available_cells = [i for i in total_files if i in label_names]
        print('Available batteries:', len(available_cells))

        new_total_samples_name = []
        for i in available_cells:
            if 'Tongji' in i:
                i = i.replace('--', '-#')
            new_total_samples_name.append(i)
        # count aging conditions
        aging_conditions_id = [aging_conditions[i] for i in available_cells if i in aging_conditions]
        print('Aging conditions:', len(set(aging_conditions_id)))

    print('-------------------------------------')



