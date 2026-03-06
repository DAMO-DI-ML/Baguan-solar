import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

def merge_cldas_himawari(cldas_dir, himawari_dir, output_dir):
    """
    Merge CLDAS (Beijing Time) and Himawari (UTC) data
    CLDAS filename format: 20230913_1600.npy -> convert to UTC 20230913_0800.npy to match Himawari
    Merged files use UTC naming convention
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cldas_files = sorted([f for f in os.listdir(cldas_dir) if f.endswith('.npy')])

    success_count = 0
    failed_files = []

    print(f"Found {len(cldas_files)} CLDAS files")

    for cldas_file in tqdm(cldas_files, desc="Merging data"):
        try:
            # CLDAS timestamp (Beijing Time)
            bj_str = cldas_file.replace('.npy', '')
            bj_dt = datetime.strptime(bj_str, "%Y%m%d_%H%M")

            # Convert to UTC
            utc_dt = bj_dt - timedelta(hours=8)
            utc_str = utc_dt.strftime("%Y%m%d_%H%M")

            # File paths
            cldas_path = os.path.join(cldas_dir, cldas_file)
            himawari_path = os.path.join(himawari_dir, f"{utc_str}.npy")  # UTC
            output_path = os.path.join(output_dir, f"{utc_str}.npy")      # Merged file uses UTC naming

            if not os.path.exists(himawari_path):
                print(f"\nWarning: Himawari file not found: {himawari_path}")
                failed_files.append(bj_str)
                continue

            # Load data
            cldas_data = np.load(cldas_path)
            himawari_data = np.load(himawari_path)

            # Convert to numpy if needed
            if torch.is_tensor(cldas_data):
                cldas_data = cldas_data.numpy()
            if torch.is_tensor(himawari_data):
                himawari_data = himawari_data.numpy()

            # Validate dimensions
            if cldas_data.shape != (2, 512, 512):
                print(f"\nWarning: Invalid CLDAS dimensions {cldas_data.shape} in {cldas_file}")
                failed_files.append(bj_str)
                continue

            if himawari_data.shape != (4, 512, 512):
                print(f"\nWarning: Invalid Himawari dimensions {himawari_data.shape} in {utc_str}.npy")
                failed_files.append(bj_str)
                continue

            # Merge data
            merged_data = np.concatenate([cldas_data, himawari_data], axis=0)
            assert merged_data.shape == (6, 512, 512), f"Invalid merged dimensions: {merged_data.shape}"

            # Save
            np.save(output_path, merged_data)
            success_count += 1

        except Exception as e:
            print(f"\nError processing file {cldas_file}: {str(e)}")
            failed_files.append(bj_str)

    print(f"\nMerge completed!")
    print(f"Successfully merged: {success_count} files")
    print(f"Failed: {len(failed_files)} files")

    if failed_files:
        print(f"Failed timestamps (Beijing Time): {failed_files[:10]}...")

    return success_count, failed_files

def main():
    # Set paths
    cldas_dir = "./data/cldas_processed_2025"
    himawari_dir = "./data/himawari_processed_2025"
    output_dir = "./data/test"  # You can modify output path
    
    print("="*50)
    print("CLDAS and Himawari Data Merger")
    print("="*50)
    print(f"CLDAS data directory: {cldas_dir}")
    print(f"Himawari data directory: {himawari_dir}")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    # Execute merge
    success_count, failed_files = merge_cldas_himawari(cldas_dir, himawari_dir, output_dir)


if __name__ == "__main__":
    main()
