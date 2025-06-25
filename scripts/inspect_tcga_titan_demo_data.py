import h5py
import torch
import numpy as np
from huggingface_hub import hf_hub_download

# This is a placeholder filename. Replace with the actual file you are using
# if it's different from the demo, or if you have it locally.
# For this script, we will still use the downloadable demo file and simulate
# the inspection of the other keys if they existed.
DEMO_FILENAME = "TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5"

def download_data(filename):
    """
    Downloads the TCGA demo feature file from the Hugging Face Hub.
    """
    print(f"Downloading {filename} from Hugging Face Hub...")
    file_path = hf_hub_download("MahmoodLab/TITAN", filename=filename)
    print(f"Download complete. File saved at: {file_path}")
    return file_path

def inspect_full_h5_file(file_path):
    """
    Loads and prints detailed information about all known datasets and attributes
    within the H5 file.

    Args:
        file_path (str): The path to the H5 file.
    """
    print("\n" + "="*50)
    print("INSPECTING H5 FILE:", file_path)
    print("="*50 + "\n")

    with h5py.File(file_path, 'r') as file:
        all_keys = list(file.keys())
        print(f"Keys found in H5 file: {all_keys}\n")

        # --- Inspect each key found in the file ---
        for key in all_keys:
            print(f"--- Inspecting Key: '{key}' ---")
            dataset = file[key]

            # Print basic dataset info
            print(f"Shape: {dataset.shape}")
            print(f"Data Type (dtype): {dataset.dtype}")

            # Print a sample of the data based on its likely type
            try:
                if key in ['features', 'coords', 'coords_patching']:
                    # For large coordinate/feature arrays, show the first 5 rows
                    print("Sample Data (first 5 rows):")
                    print(dataset[:5])

                elif key in ['annots', 'annots_patching']:
                    # For annotations, show first 20 values and unique labels
                    annots_data = dataset[:]
                    print(f"Sample Data (first 20 values): {annots_data[:20]}")
                    unique_labels, counts = np.unique(annots_data, return_counts=True)
                    print(f"Unique annotation labels and their counts:")
                    for label, count in zip(unique_labels, counts):
                        print(f"  Label {label}: {count} patches")

                elif key in ['mask', 'stitch']:
                    # For image-like data, shape and dtype are most important.
                    # Printing the array itself is not very useful.
                    print("Data is image-like (mask or stitch). Shape and dtype are most informative.")
                    # Optionally, print min/max values
                    data_min, data_max = np.min(dataset), np.max(dataset)
                    print(f"Min value: {data_min}, Max value: {data_max}")

                else:
                    print("Unknown key type, showing first few elements.")
                    print(dataset[:5])

            except Exception as e:
                print(f"Could not print sample data for key '{key}': {e}")

            print("\n" + "-"*30 + "\n")

if __name__ == "__main__":
    # Note: The standard demo file only has 'coords' and 'features'.
    # This script is designed to work with your more complete H5 file.
    # When run on the demo, it will only show results for the two available keys.
    # To inspect your file, replace DEMO_FILENAME with your local file path
    # and comment out the download_data call.
    #
    # Example for a local file:
    # my_local_h5_path = "path/to/your/file.h5"
    # inspect_full_h5_file(my_local_h5_path)

    h5_file_path = download_data(DEMO_FILENAME)
    inspect_full_h5_file(h5_file_path)
