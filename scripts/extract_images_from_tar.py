import tarfile
import os
import argparse
import sys

def extract_specific_pngs(input_folder, output_folder, stain_type):
    """
    Scans a folder for .tar archives and extracts specific PNG files based on
    the chosen stain type into a structured output directory.

    Args:
        input_folder (str): The path to the folder containing .tar files.
        output_folder (str): The path to the folder where files will be extracted.
        stain_type (str): The type of stain to extract ('HE' or 'P53').
    """
    # --- 1. Validate Input and Output Paths ---
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found at '{input_folder}'")
        sys.exit(1)

    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{output_folder}'. Reason: {e}")
        sys.exit(1)

    # --- 2. Set Up Extraction Parameters ---
    # Dynamically create the search pattern based on the chosen stain type.
    # e.g., if stain_type is 'HE', search_pattern becomes '-HE.png'.
    search_pattern = f'{stain_type}'

    print(f"--- Starting Extraction ---")
    print(f"Source Folder: {input_folder}")
    print(f"Destination Folder: {output_folder}")
    print(f"Stain Type to Extract: {stain_type} (searching for *{search_pattern})")
    print("-" * 80)

    # --- 3. Iterate Through Files in the Input Folder ---
    found_tar_files = False
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith('.tar'):
            found_tar_files = True
            tar_path = os.path.join(input_folder, filename)
            
            # Get the base name of the tar file for the subfolder name.
            # e.g., 'RL-0562.tar' -> 'RL-0562'
            archive_basename = os.path.splitext(filename)[0]
            
            # Define the target directory for this specific archive's contents.
            target_extraction_path = os.path.join(output_folder, archive_basename)

            print(f"Processing Archive: {filename}")

            try:
                # --- 4. Open and Inspect the .tar Archive ---
                with tarfile.open(tar_path, 'r') as tar:
                    
                    # Find members that match the dynamic search pattern.
                    members_to_extract = [
                        member for member in tar.getmembers()
                        if member.isfile() and search_pattern in member.name
                    ]

                    if not members_to_extract:
                        print(f"  └─ No '{search_pattern}' files found to extract.")
                        continue # Move to the next .tar file

                    # Create the specific subdirectory for these files.
                    os.makedirs(target_extraction_path, exist_ok=True)
                    
                    # --- 5. Extract the Filtered Files ---
                    for member in members_to_extract:
                        print(f"  └─ Extracting: {member.name}")
                        tar.extract(member, path=target_extraction_path)

            except tarfile.ReadError:
                print(f"  └─ ERROR: Failed to read '{filename}'. It may be corrupted.")
            except Exception as e:
                print(f"  └─ ERROR: An unexpected error occurred. Reason: {e}")
    
    if not found_tar_files:
        print(f"Warning: No .tar files were found in '{input_folder}'.")

    print("-" * 80)
    print("--- Extraction Complete ---")


if __name__ == "__main__":
    # --- USAGE EXAMPLES ---
    # To extract HE images:
    # python your_script_name.py --input_folder data/raw/tar --output_folder data/raw/HE --stain_type HE
    #
    # To extract P53 images:
    # python your_script_name.py --input_folder data/raw/tar --output_folder data/raw/P53 --stain_type P53

    parser = argparse.ArgumentParser(
        description="Extracts specific PNG files (HE or P53) from a folder of .tar archives into a structured output directory."
    )
    
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True,
        help="The path to the folder containing the .tar files."
    )
    
    parser.add_argument(
        "--output_folder", 
        type=str, 
        required=True,
        help="The path to the folder where extracted files will be saved."
    )

    parser.add_argument(
        "--stain_type",
        type=str.upper, # Automatically convert input to uppercase
        required=True,
        choices=['HE', 'P53'], # Restrict input to these two options
        help="The type of stained image to extract. Choose between 'HE' or 'P53'."
    )
    
    args = parser.parse_args()
    
    extract_specific_pngs(args.input_folder, args.output_folder, args.stain_type)
