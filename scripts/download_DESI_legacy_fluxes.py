import os
import numpy as np
import pandas as pd
from astro_datasets import load_dataset
from astropy.table import Table
from dl import queryClient as qc  # Initialize the query client

def main(astroclip_dataset_path, save_path):
    # Load the official Hugging Face version of the AstroCLIP dataset
    print(f"Loading dataset from: {astroclip_dataset_path}")
    df_train = load_dataset(astroclip_dataset_path, split="train")
    df_test = load_dataset(astroclip_dataset_path, split="test")

    # Combine target IDs from train and test splits
    combined_targetids = df_train['targetid'] + df_test['targetid']

    # Split the combined_targetids array into smaller chunks
    chunk_size = 100
    chunks = [combined_targetids[i:i + chunk_size] for i in range(0, len(combined_targetids), chunk_size)]

    all_results = []

    # Iterate over chunks
    print(f"Processing {len(chunks)} chunks...")
    for idx, chunk in enumerate(chunks):
        if idx % 100 == 0:
            print(f"Processing chunk {idx + 1}/{len(chunks)}...")

        # Use the chunk in the SQL query
        query = f"""
        WITH target_ids (targetid) AS (
            VALUES {', '.join(f'({i})' for i in chunk)}
        )
        SELECT 
            ph.targetid, 
            tr.dered_flux_g, tr.dered_flux_z, tr.dered_flux_r, 
            tr.flux_g, tr.flux_z, tr.flux_r, 
            tr.dered_mag_g, tr.dered_mag_z, tr.dered_mag_r, 
            tr.mag_g, tr.mag_z, tr.mag_r, 
            tr.flux_ivar_r, tr.flux_ivar_g, tr.flux_ivar_z
        FROM desi_edr.photometry AS ph
        INNER JOIN ls_dr9.tractor AS tr ON ph.ls_id = tr.ls_id
        JOIN target_ids AS t ON ph.targetid = t.targetid
        """
        
        # Execute the query for this chunk
        zpix = qc.query(sql=query, fmt='table')
        
        # Convert the result to a Pandas DataFrame
        df = zpix.to_pandas()

        # Store the DataFrame result from each chunk
        all_results.append(df)

    # Concatenate all DataFrames from chunks
    final_results = pd.concat(all_results, ignore_index=True)

    # Save the final results to the specified path
    print(f"Saving results to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_results.to_hdf(save_path, key='df', mode='w')
    print("Processing completed.")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process AstroCLIP dataset and save query results.")
    parser.add_argument("astroclip_dataset_path", type=str, help="Path to the AstroCLIP dataset.")
    parser.add_argument("save_path", type=str, help="Path to save the final results (HDF5 file).")
    args = parser.parse_args()

    # Call the main function
    main(args.astroclip_dataset_path, args.save_path)