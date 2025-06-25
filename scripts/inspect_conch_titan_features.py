"""
Inspects HDF5 (.h5) feature files.

This script can inspect a single .h5 file to determine the shape of its
feature matrix or process an entire directory of .h5 files to compute
overall statistics for the entire dataset.

Each .h5 file is assumed to contain a dataset (default key 'features')
which is a 2D NumPy array where each row represents a feature vector for a
patch from a Whole Slide Image (WSI).

Usage:
  # Inspect a single file
  python inspect_features.py /path/to/your/file.h5

  # Inspect an entire directory and compute statistics
  python inspect_features.py /path/to/your/directory/
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple

# --- Constants ---
# The default key inside the HDF5 file where the feature matrix is stored.
# Change this if your foundation model uses a different key.
DEFAULT_FEATURE_KEY = 'features'

def inspect_single_file(file_path: Path, feature_key: str) -> None:
    """
    Inspects a single HDF5 file and prints the shape of the feature matrix.
    Handles both slide-level (1D) and patch-level (2D) features.

    Args:
        file_path (Path): The path to the .h5 file.
        feature_key (str): The key for the dataset within the HDF5 file.
    """
    print(f"--- Inspecting Single File: {file_path.name} ---")
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            if feature_key in f:
                features = f[feature_key]
                print(f"  - Successfully loaded dataset with key: '{feature_key}'")
                print(f"  - Shape of feature data: {features.shape}")

                # Check if the features are a 2D matrix (patches) or a 1D vector (slide)
                if features.ndim == 2:  # Case: (n_patches, 768)
                    num_patches, feature_dim = features.shape
                    print("  - Type: Patch-level features")
                    print(f"  - Number of patches (vectors): {num_patches}")
                    print(f"  - Dimensionality of each vector: {feature_dim}")
                elif features.ndim == 1:  # Case: (768,)
                    feature_dim, = features.shape
                    print("  - Type: Slide-level feature")
                    print(f"  - Number of vectors: 1")
                    print(f"  - Dimensionality of the vector: {feature_dim}")
                else:
                    print(f"  - WARNING: Unexpected feature dimension. Expected 1D or 2D, but shape is {features.shape}.")

            else:
                print(f"Error: Could not find the dataset key '{feature_key}' in the file.")
                available_keys = list(f.keys())
                print(f"Available keys in the file are: {available_keys}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

def calculate_and_print_stats(
    all_features: np.ndarray,
    num_files: int,
    feature_dim: int
) -> None:
    """
    Calculates and prints statistics over a combined feature matrix.

    Args:
        all_features (np.ndarray): The combined 2D array of all features.
        num_files (int): The number of files that were processed.
        feature_dim (int): The dimensionality of the feature vectors.
    """
    total_patches = all_features.shape[0]

    print("\n--- Overall Dataset Statistics ---")
    print(f"  - Files processed: {num_files}")
    print(f"  - Total patches (vectors) found: {total_patches}")
    print(f"  - Dimensionality of features: {feature_dim}")
    print("-" * 20)
    print("  - Statistics across all vectors:")

    # Using np.float64 for more precise calculations
    stats_features = all_features.astype(np.float64)

    # Calculate statistics
    mean_val = np.mean(stats_features)
    std_val = np.std(stats_features)
    median_val = np.median(stats_features)
    min_val = np.min(stats_features)
    max_val = np.max(stats_features)

    print(f"    - Overall Mean:   {mean_val:.6f}")
    print(f"    - Overall Std Dev:  {std_val:.6f}")
    print(f"    - Overall Median: {median_val:.6f}")
    print(f"    - Overall Min:    {min_val:.6f}")
    print(f"    - Overall Max:    {max_val:.6f}")
    print("\nNote: 'Overall' stats are computed on the flattened array of all features.")


def process_directory(dir_path: Path, feature_key: str) -> None:
    """
    Processes all .h5 files in a directory, aggregates features, and computes stats.
    Handles both slide-level (1D) and patch-level (2D) features.

    Args:
        dir_path (Path): The path to the directory containing .h5 files.
        feature_key (str): The key for the dataset within the HDF5 files.
    """
    print(f"--- Processing Directory: {dir_path} ---")
    h5_files = sorted(list(dir_path.glob('*.h5')))

    if not h5_files:
        print(f"Error: No .h5 files found in the directory: {dir_path}")
        return

    feature_list: List[np.ndarray] = []
    feature_dim: int = -1
    files_processed = 0

    for file_path in h5_files:
        print(f"  - Reading {file_path.name}...")
        try:
            with h5py.File(file_path, 'r') as f:
                if feature_key not in f:
                    print(f"    - WARNING: Key '{feature_key}' not found. Skipping file.")
                    continue

                features = f[feature_key][:]  # [:] loads into memory

                # Reshape 1D slide-level features to 2D for consistent processing
                if features.ndim == 1:
                    features = features.reshape(1, -1) # Shape (768,) becomes (1, 768)

                # We now expect all features to be 2D
                if features.ndim != 2:
                    print(f"    - WARNING: Features are not 1D or 2D. Original shape was {f[feature_key].shape}. Skipping file.")
                    continue

                # On the first valid file, set the expected feature dimension
                if feature_dim == -1:
                    feature_dim = features.shape[1]

                # Check for consistent feature dimensions
                if features.shape[1] != feature_dim:
                    print(f"    - WARNING: Inconsistent feature dimension ({features.shape[1]}). Expected {feature_dim}. Skipping file.")
                    continue

                feature_list.append(features)
                files_processed += 1

        except Exception as e:
            print(f"    - WARNING: Could not read file {file_path.name}. Error: {e}")

    if not feature_list:
        print("\nCould not load any valid features from the directory.")
        return

    # Vertically stack all feature arrays into one large numpy array
    print("\nConcatenating all features for analysis. This may take a moment...")
    combined_features = np.vstack(feature_list)

    calculate_and_print_stats(combined_features, files_processed, feature_dim)


def main() -> None:
    """Main function to parse arguments and run the inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 (.h5) feature files from foundation models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to a single .h5 file or a directory containing .h5 files."
    )
    parser.add_argument(
        "--feature_key",
        type=str,
        default=DEFAULT_FEATURE_KEY,
        help=f"The key for the dataset inside the HDF5 file.\n(default: '{DEFAULT_FEATURE_KEY}')"
    )

    args = parser.parse_args()
    input_path: Path = args.path

    if not input_path.exists():
        print(f"Error: The specified path does not exist: {input_path}")
        return

    if input_path.is_dir():
        process_directory(input_path, args.feature_key)
    elif input_path.is_file():
        if input_path.suffix != '.h5':
            print(f"Warning: The specified file '{input_path.name}' is not a .h5 file.")
        inspect_single_file(input_path, args.feature_key)
    else:
        print(f"Error: The path {input_path} is not a valid file or directory.")

if __name__ == "__main__":
    main()
