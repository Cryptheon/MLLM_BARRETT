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

    partial_files = sorted(glob.glob(args.partial_glob), key=extract_gpu_index)

    if not partial_files:
        raise ValueError("No matching GPU-partial CSV files found!")

    print(f"Found {len(partial_files)} files. Concatenating in order: {partial_files}")
    dfs = [pd.read_csv(f) for f in partial_files]
    final_df = pd.concat(dfs, ignore_index=True)

    final_df.to_csv(args.output_csv, index=False)
    print(f"Saved combined report to {args.output_csv}")

if __name__ == "__main__":
    main()

