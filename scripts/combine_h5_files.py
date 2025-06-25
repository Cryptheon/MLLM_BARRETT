import os
import h5py
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_h5_files(source_dir: str, output_file: str):
    """
    Combines multiple HDF5 files from a source directory into a single HDF5 file.

    Each file from the source directory is saved as a group in the output file,
    with the group name being the filename stem (e.g., 'RL-0006-I-HE').

    Args:
        source_dir (str): The path to the directory containing the .h5 files.
        output_file (str): The path for the output combined .h5 file.
    """
    if not os.path.isdir(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        return

    source_files = list(Path(source_dir).glob("*.h5"))
    if not source_files:
        logger.warning(f"No .h5 files found in {source_dir}")
        return

    logger.info(f"Found {len(source_files)} HDF5 files to combine.")

    with h5py.File(output_file, 'w') as f_out:
        for file_path in source_files:
            try:
                with h5py.File(file_path, 'r') as f_in:
                    file_stem = file_path.stem
                    # Create a group for each original file
                    group = f_out.create_group(file_stem)
                    # Copy all datasets from the source file to the new group
                    for key in f_in.keys():
                        f_in.copy(key, group)
                    logger.info(f"Copied data from {file_path.name} to group '{file_stem}'")
            except Exception as e:
                logger.error(f"Failed to process file {file_path.name}: {e}")

    logger.info(f"Successfully combined all files into {output_file}")


if __name__ == '__main__':
    # This block allows the script to be run from the command line.
    parser = argparse.ArgumentParser(
        description="Combine multiple HDF5 files into a single file."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        help="Directory containing the HDF5 files to be combined."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path for the combined output HDF5 file."
    )

    args = parser.parse_args()

    # Example usage from command line:
    # python combine_h5.py /path/to/slide_features_titan /path/to/combined_embeddings.h5
    combine_h5_files(args.source_dir, args.output_file)

