import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

# Define a type alias for the structured data we'll collect.
# It's a dictionary mapping metric names (like 'auc_macro') to a list of their values.
PlottingData = Dict[str, List[float]]

def find_latest_evaluation(checkpoint_path: Path) -> Optional[Path]:
    """
    Finds the most recent evaluation directory within a checkpoint folder.

    The most recent is determined by sorting the directory names lexicographically,
    as they are timestamped in a 'YYYY-MM-DD_HH-MM-SS' format.

    Args:
        checkpoint_path: The Path object for a single checkpoint directory
                         (e.g., './checkpoint-4000/').

    Returns:
        A Path object to the latest evaluation directory, or None if no
        evaluation directories are found.
    """
    evaluation_dirs = [p for p in checkpoint_path.iterdir() if p.is_dir() and p.name.startswith('evaluation_')]
    
    if not evaluation_dirs:
        print(f"Warning: No evaluation directories found in {checkpoint_path}")
        return None
    
    # Timestamps are in a sortable format, so the last one is the latest.
    latest_eval_dir = sorted(evaluation_dirs)[-1]
    return latest_eval_dir

def parse_evaluation_results(results_file: Path) -> Optional[Dict[str, Any]]:
    """
    Parses an evaluation_results.json file to extract key metrics.

    Args:
        results_file: The Path object for the JSON results file.

    Returns:
        A dictionary containing the extracted AUC metrics, or None if the file
        cannot be read, parsed, or if key metrics are missing.
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", {})
        auc_macro = metrics.get("auc_macro")
        auc_per_class = metrics.get("auc_per_class", {})

        if auc_macro is None:
            print(f"Warning: 'auc_macro' not found in {results_file}")
            return None

        return {
            "auc_macro": auc_macro,
            "HGD": auc_per_class.get("HGD"),
            "LGD": auc_per_class.get("LGD"),
            "ND": auc_per_class.get("ND"),
        }
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {results_file}")
        return None

def process_all_checkpoints(root_path: Path) -> Tuple[List[int], PlottingData]:
    """
    Iterates through all checkpoint directories to gather evaluation data.

    Args:
        root_path: The root directory containing all 'checkpoint-*' folders.

    Returns:
        A tuple containing:
        - A list of checkpoint numbers (for the x-axis).
        - A dictionary of metrics, where each key maps to a list of values
          (for the y-axis series).
    """
    checkpoint_dirs = sorted(
        [p for p in root_path.iterdir() if p.is_dir() and p.name.startswith('checkpoint-')],
        key=lambda p: int(p.name.split('-')[-1])
    )

    if not checkpoint_dirs:
        print(f"Error: No 'checkpoint-*' directories found in '{root_path}'.")
        return [], {}

    # Initialize data structures
    checkpoints = []
    plotting_data: PlottingData = {"auc_macro": [], "HGD": [], "LGD": [], "ND": []}

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_num = int(checkpoint_dir.name.split('-')[-1])
        print(f"Processing {checkpoint_dir.name}...")

        latest_eval_dir = find_latest_evaluation(checkpoint_dir)
        if not latest_eval_dir:
            continue

        results_json_path = latest_eval_dir / "evaluation_results.json"
        
        metrics = parse_evaluation_results(results_json_path)
        if not metrics:
            continue
        
        # Append data if all metrics are valid
        if all(metrics.get(key) is not None for key in plotting_data.keys()):
            checkpoints.append(checkpoint_num)
            for key in plotting_data.keys():
                plotting_data[key].append(metrics[key])
        else:
            print(f"Warning: Skipping checkpoint {checkpoint_num} due to missing AUC values.")


    return checkpoints, plotting_data

def create_and_save_plot(checkpoints: List[int], data: PlottingData, output_file: str) -> None:
    """
    Generates and saves a plot of AUC scores vs. checkpoints.

    Args:
        checkpoints: A list of checkpoint numbers (x-axis).
        data: A dictionary of metric lists (y-axis series).
        output_file: The filename to save the plot as.
    """
    if not checkpoints or not data:
        print("No data available to plot.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Plot each metric
    plt.plot(checkpoints, data["auc_macro"], marker='o', linestyle='-', label="Macro Average AUC")
    plt.plot(checkpoints, data["HGD"], marker='s', linestyle='--', label="HGD AUC")
    plt.plot(checkpoints, data["LGD"], marker='^', linestyle='--', label="LGD AUC")
    plt.plot(checkpoints, data["ND"], marker='d', linestyle='--', label="ND AUC")

    plt.title("AUC Score vs. Model Checkpoint", fontsize=16)
    plt.xlabel("Checkpoint Step", fontsize=12)
    plt.ylabel("Area Under Curve (AUC)", fontsize=12)
    plt.legend(title="Metrics")
    plt.xticks(checkpoints, rotation=45)
    plt.tight_layout() # Adjust layout to make room for labels

    try:
        plt.savefig(output_file)
        print(f"Plot successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")

def main() -> None:
    """
    Main function to orchestrate the script execution.
    """
    parser = argparse.ArgumentParser(
        description="Analyze model evaluation results across checkpoints and plot AUC scores."
    )
    parser.add_argument(
        "--path",
        type=str,
        help="The root path to the directory containing the 'checkpoint-*' folders."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="auc_vs_checkpoints.png",
        help="The name of the output plot file. (default: auc_vs_checkpoints.png)"
    )
    args = parser.parse_args()

    root_path = Path(args.path)
    if not root_path.is_dir():
        print(f"Error: The provided path '{root_path}' is not a valid directory.")
        return

    checkpoints, plotting_data = process_all_checkpoints(root_path)
    create_and_save_plot(checkpoints, plotting_data, args.output)

if __name__ == "__main__":
    main()

