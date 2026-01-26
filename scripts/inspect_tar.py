import tarfile
import os
import argparse
import humanize  # A utility to make file sizes human-readable
from PIL import Image  # Import from the Pillow library to inspect images

def inspect_tar_file(tar_path):
    """
    Opens and inspects a .tar archive, lists its contents, and inspects
    any PNG files found within the archive.

    Args:
        tar_path (str): The full path to the .tar file.
    """
    Image.MAX_IMAGE_PIXELS = None

    try:
        # Check if the file exists before attempting to open it
        if not os.path.exists(tar_path):
            print(f"Error: File not found at '{tar_path}'")
            return

        print(f"--- Inspecting Archive: {tar_path} ---")

        # Open the .tar file for reading
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()

            if not members:
                print("The archive is empty.")
                return

            # Print a header for the output
            print(f"{'Name':<50} | {'Type':<10} | {'Size'}")
            print("-" * 80)

            # Iterate over each member in the archive
            for member in members:
                member_type = "File" if member.isfile() else "Directory" if member.isdir() else "Other"
                
                try:
                    size_str = humanize.naturalsize(member.size)
                except NameError:
                    size_str = f"{member.size} B"

                # Print the primary details of the member
                print(f"{member.name:<50} | {member_type:<10} | {size_str}")

                # If the member is a PNG file, extract and inspect it
                if member.isfile() and member.name.lower().endswith('.png'):
                    print("  └─ Inspecting PNG details...")
                    try:
                        # Extract the file into a memory buffer, don't write to disk
                        png_file_obj = tar.extractfile(member)
                        if png_file_obj:
                            # Open the image from the memory buffer
                            with Image.open(png_file_obj) as img:
                                print(f"    ├─ Format: {img.format}")
                                print(f"    ├─ Dimensions: {img.width}x{img.height} pixels")
                                print(f"    ├─ Color Mode: {img.mode}")
                                
                                # Display metadata if it exists
                                if img.info:
                                    print("    └─ Metadata:")
                                    for key, value in img.info.items():
                                        # Truncate long metadata values for readability
                                        value_str = str(value)
                                        if len(value_str) > 100:
                                            value_str = value_str[:100] + '...'
                                        print(f"        - {key}: {value_str}")
                                else:
                                    print("    └─ No metadata found.")
                    except Exception as e:
                        print(f"    └─ ERROR: Could not inspect PNG file. Reason: {e}")

        print(f"--- End of Archive ---")

    except tarfile.ReadError:
        print(f"Error: '{tar_path}' is not a valid tar archive or is corrupted.")
    except ImportError:
        print("Error: The 'Pillow' library is required to inspect images. Please install it using 'pip install Pillow'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- USAGE EXAMPLE ---
    # Run this script from your terminal like this:
    # python inspect_tar.py --tar_path LMM/RL-0006.tar
    
    # To use this script, you will need to install the 'humanize' and 'Pillow' libraries:
    # pip install humanize Pillow

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Inspect the contents of a .tar archive and its PNG files."
    )
    
    # Add an argument for the tar file path
    parser.add_argument(
        "--tar_path", 
        type=str, 
        required=True,
        help="The path to the .tar file to inspect."
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the inspection function with the provided file path
    inspect_tar_file(args.tar_path)

