import pandas as pd
import glob
import argparse
import re
import os

def extract_gpu_index(filename):
    match = re.search(r'_gpu(\d+)\.csv$', filename)
    return int(match.group(1)) if match else -1

def main():
    parser = argparse.ArgumentParser(description="Concatenate processed report CSVs from multiple GPUs.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the final combined CSV.")
    parser.add_argument("--partial_glob", type=str, default="*_gpu*.csv", help="Glob pattern to find partial CSV files.")
    args = parser.parse_args()

    expected_columns = ["patient_filename", "text", "processed_reports"]
    partial_files = sorted(glob.glob(args.partial_glob), key=extract_gpu_index)

    if not partial_files:
        raise ValueError("No matching GPU-partial CSV files found!")

    print(f"Found {len(partial_files)} files. Concatenating in order: {partial_files}")
    
    dfs = []
    for f in partial_files:
        df = pd.read_csv(f)

        # Add any missing expected columns with empty strings
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""

        # Remove unexpected columns and reorder
        df = df[expected_columns]
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    # Optional: check and clean up trailing whitespace
    final_df = final_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    final_df.to_csv(args.output_csv, index=False)
    print(f"Saved combined report to {args.output_csv}")

if __name__ == "__main__":
    main()
