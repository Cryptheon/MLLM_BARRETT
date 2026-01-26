import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import os
import json
from datetime import datetime
import numpy as np

# --- Constants ---
VALID_CLASSES = ["C", "HGD", "LGD", "IND", "ND"]

# --- Utility Functions ---

def find_valid_label(text, valid_classes):
    """Searches for a valid class label within a string."""
    if not isinstance(text, str) or not text.strip() or text.strip().upper() == 'NA':
        return None
    for label in valid_classes:
        if label in text:
            return label
    return None

def extract_label_from_end(text, valid_classes):
    """Extracts a valid label from the last non-empty line of a text block."""
    if not isinstance(text, str) or not text.strip():
        return None
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    return find_valid_label(lines[-1], valid_classes) if lines else None

# --- Plotting Functions ---

def plot_confusion_matrix(cm, class_names, save_path):
    """Plots and saves a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_multiclass_roc(y_true_bin, y_pred_bin, class_names, save_path):
    """Plots and saves a multiclass ROC curve."""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('viridis', len(class_names))

    for i, (name, color) in enumerate(zip(class_names, colors.colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC curve for {name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    plt.title('Multiclass Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve plot saved to {save_path}")

# --- Core Logic Functions ---

def setup_evaluation():
    """Parses command-line arguments and creates the output directory."""
    parser = argparse.ArgumentParser(description="Evaluate pathology report labels.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--label_source", type=str, default="keys", choices=["keys", "text"], help="Source of the labels for evaluation.")
    parser.add_argument("--output_dir", type=str, default=".", help="Base directory for results.")
    parser.add_argument("--cm_filename", type=str, default="confusion_matrix.png", help="Filename for the confusion matrix plot.")
    parser.add_argument("--roc_filename", type=str, default="roc_curve.png", help="Filename for the ROC curve plot.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving all results to: {output_path}")
    
    return args, output_path

def load_and_prepare_data(config, valid_classes):
    """Loads, cleans, and filters the evaluation data from the JSON file."""
    try:
        df = pd.read_json(config.input_json)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None, None

    if config.label_source == 'text':
        print("Parsing labels from 'original_report' and 'generated_report' text fields.")
        df['true_label'] = df['original_report'].apply(lambda txt: extract_label_from_end(txt, valid_classes))
        df['pred_label'] = df['generated_report'].apply(lambda txt: extract_label_from_end(txt, valid_classes))
    else:  # 'keys'
        print("Using labels from 'report_extracted_label' and 'gen_extracted_label' keys.")
        df['true_label'] = df['report_extracted_label'].apply(lambda txt: find_valid_label(txt, valid_classes))
        df['pred_label'] = df['gen_extracted_label'].apply(lambda txt: find_valid_label(txt, valid_classes))

    initial_count = len(df)
    df_filtered = df.dropna(subset=['true_label', 'pred_label']).copy()
    filtered_count = len(df_filtered)

    data_summary = {
        "initial_records": initial_count,
        "evaluated_records": filtered_count,
        "invalid_or_missing_records_excluded": initial_count - filtered_count,
    }
    print(f"Excluded {data_summary['invalid_or_missing_records_excluded']} records. Evaluating on {filtered_count} records.")
    return df_filtered, data_summary

def calculate_metrics(y_true, y_pred, all_possible_classes):
    """Computes all performance metrics based on true and predicted labels."""
    metrics = {}
    present_labels = [lbl for lbl in all_possible_classes if lbl in set(y_true.unique()) | set(y_pred.unique())]
    
    metrics['evaluated_classes'] = present_labels
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['classification_report'] = classification_report(y_true, y_pred, labels=present_labels, zero_division=0, output_dict=True)
    metrics['confusion_matrix'] = {
        "labels": present_labels,
        "matrix": confusion_matrix(y_true, y_pred, labels=present_labels).tolist()
    }
    
    # ROC/AUC calculations
    if len(present_labels) > 1:
        y_true_bin = label_binarize(y_true, classes=present_labels)
        y_pred_bin = label_binarize(y_pred, classes=present_labels)
        try:
            metrics['auc_macro'] = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
            auc_scores = roc_auc_score(y_true_bin, y_pred_bin, average=None, multi_class='ovr')
            metrics['auc_per_class'] = dict(zip(present_labels, auc_scores))
        except ValueError as e:
            print(f"Could not compute AUC score: {e}")
            metrics['auc_macro'] = None
            metrics['auc_per_class'] = None
            metrics['errors'] = [str(e)]
            
    return metrics

def generate_visualizations(y_true, y_pred, metrics, output_path, config):
    """Generates and saves the confusion matrix and ROC curve plots."""
    present_labels = metrics['evaluated_classes']
    
    # Plot Confusion Matrix
    cm_data = np.array(metrics['confusion_matrix']['matrix'])
    cm_path = os.path.join(output_path, config.cm_filename)
    plot_confusion_matrix(cm_data, present_labels, cm_path)

    # Plot ROC Curve
    if len(present_labels) > 1 and metrics.get('auc_macro') is not None:
        y_true_bin = label_binarize(y_true, classes=present_labels)
        y_pred_bin = label_binarize(y_pred, classes=present_labels)
        roc_path = os.path.join(output_path, config.roc_filename)
        plot_multiclass_roc(y_true_bin, y_pred_bin, present_labels, roc_path)

# --- Main Orchestration ---

def main():
    """Main function to run the complete evaluation pipeline."""
    config, output_path = setup_evaluation()
    
    df_eval, data_summary = load_and_prepare_data(config, VALID_CLASSES)
    
    if df_eval is None or df_eval.empty:
        print("No valid data to evaluate. Exiting.")
        return

    y_true, y_pred = df_eval['true_label'], df_eval['pred_label']
    
    print("\n--- Calculating Performance Metrics ---")
    metrics = calculate_metrics(y_true, y_pred, VALID_CLASSES)
    
    # Print key results to console
    print(f"\nOverall Accuracy: {metrics.get('accuracy', 0):.4f}")
    if 'auc_macro' in metrics and metrics['auc_macro'] is not None:
        print(f"AUC (Macro-Averaged): {metrics['auc_macro']:.4f}\n")

    print("\n--- Generating Visualizations ---")
    generate_visualizations(y_true, y_pred, metrics, output_path, config)
    
    # Assemble final results dictionary for JSON output
    final_results = {
        "metadata": {
            "run_timestamp": os.path.basename(output_path).replace("evaluation_", ""),
            "input_file": config.input_json,
            "label_source": config.label_source,
            "all_possible_classes": VALID_CLASSES,
        },
        "data_summary": data_summary,
        "metrics": metrics
    }
    
    # Save results to JSON
    results_json_path = os.path.join(output_path, "evaluation_results.json")
    with open(results_json_path, 'w') as f:
        # Handle numpy float types for JSON serialization
        json.dump(final_results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nStructured results saved to {results_json_path}")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    main()