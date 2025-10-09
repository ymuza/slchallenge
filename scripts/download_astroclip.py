import argparse
import os
from astro_datasets import load_dataset

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download Hugging Face dataset to a specified path.")
    parser.add_argument("save_path", type=str, help="Path to save the downloaded dataset.")
    args = parser.parse_args()

    dset = load_dataset('astroclip/data/dataset.py')
    dset.save_to_disk(args.save_path)