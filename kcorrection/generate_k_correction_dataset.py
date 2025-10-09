import os
import sys
from astropy.table import Table, join
from astropy.io import fits
import h5py
from tqdm import tqdm

def load_fits_data(fits_path, selected_col_names):
    """Load and return selected columns from a FITS file as an Astropy Table."""
    with fits.open(fits_path) as hdul:
        fastspec_data = hdul['FASTSPEC'].data
        fastspec_table = Table(fastspec_data)
        return fastspec_table[selected_col_names]

def load_hdf5_data_in_chunks(hdf5_path, chunk_size=200):
    """Load astro_datasets from an HDF5 file in chunks and yield them."""
    with h5py.File(hdf5_path, "r") as f:
        num_objects = f['object_id'].shape[0]
        for start in tqdm(range(0, num_objects, chunk_size), desc="Reading HDF5 in chunks"):
            end = min(start + chunk_size, num_objects)
            yield {
                'image_embeddings': f['image_embeddings'][start:end],
                'spectrum_embeddings': f['spectrum_embeddings'][start:end],
                'TARGETID': f['object_id'][start:end],
                'images': f['image'][start:end],
                'spectra': f['spectrum'][start:end],
            }

def process_data_in_chunks(fits_table, hdf5_path, save_path, output_name, chunk_size=200):
    """Merge FITS and HDF5 data chunk by chunk and save the results incrementally."""
    output_file = os.path.join(save_path, output_name)
    
    with h5py.File(output_file, "w") as h5f:
        group = h5f.create_group("chunks")   # Group to store all chunks
        chunk_index = 0                      # To track chunk indices
        
        for hdf5_data in load_hdf5_data_in_chunks(hdf5_path, chunk_size):
            hdf5_table = Table(hdf5_data)
            merged_table = join(fits_table, hdf5_table, keys="TARGETID", join_type="inner")
            
            # Save each chunk with a unique path
            chunk_path = f"chunks/chunk_{chunk_index}"
            merged_table.write(output_file, path=chunk_path, append=True, format="hdf5")
            
            chunk_index += 1
            del hdf5_data, hdf5_table, merged_table  # Free memory

    print(f"Data saved in chunks under {output_file}")

    convert_hdf5_to_pandas(output_file)

def main(train_path, test_path, fits_path, save_path):
    # Define selected columns
    selected_col_names = [
        "TARGETID", "Z", "KCORR01_SDSS_R", "KCORR01_SDSS_G", "KCORR01_SDSS_Z",
        "ABSMAG01_SDSS_R", "ABSMAG01_SDSS_G", "ABSMAG01_SDSS_Z"
    ]
    
    # Load FITS data
    print("Loading FITS data...")
    fits_table = load_fits_data(fits_path, selected_col_names)
    
    # Process and save train data
    print("Processing train data...")
    process_data_in_chunks(fits_table, train_path, save_path, "kcorrs_train.hdf5")

    # Process and save test data
    print("Processing test data...")
    process_data_in_chunks(fits_table, test_path, save_path, "kcorrs_test.hdf5")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python generate_k_correction_dataset.py <train_path> <test_path> <fits_path> <save_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    fits_path = sys.argv[3]
    save_path = sys.argv[4]

    main(train_path, test_path, fits_path, save_path)
