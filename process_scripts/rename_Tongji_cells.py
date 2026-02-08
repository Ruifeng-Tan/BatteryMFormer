import os
import argparse

# --- 修改 1: 添加参数解析，接收外部传入的 output_dir ---
parser = argparse.ArgumentParser(description="Rename Tongji dataset files")
parser.add_argument('--output_dir', type=str, required=True, help='Path to processed_SOH directory')
args = parser.parse_args()

# 使用传入的路径，而不是硬编码
processed_SOH_dir = args.output_dir
total_datasets = ['Tongji']

print(f"Starting rename process in: {processed_SOH_dir}")

for dataset in total_datasets:
    dataset_path = os.path.join(processed_SOH_dir, dataset)
    
    # Safety check: ensure directory exists
    if not os.path.isdir(dataset_path):
        print(f"Directory not found (skipping): {dataset_path}")
        continue
    
    # Get original file names
    dataset_cells = os.listdir(dataset_path)
    count = 0

    print(f"Scanning {dataset_path}...")

    for old_filename in dataset_cells:
        # Only process files that contain '-#'
        if '-#' in old_filename:
            new_filename = old_filename.replace('-#', '--')
            
            old_path = os.path.join(dataset_path, old_filename)
            new_path = os.path.join(dataset_path, new_filename)
            
            try:
                os.rename(old_path, new_path)
                # print(f"Renamed: {old_filename} -> {new_filename}") # Optional: uncomment for verbose
                count += 1
            except Exception as e:
                print(f"Error renaming {old_filename}: {e}")

    print(f"Done. Total files renamed in {dataset}: {count}")
