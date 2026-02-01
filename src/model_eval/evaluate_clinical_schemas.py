import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,  # <-- Added for QWK
    precision_recall_fscore_support, # <-- Added for per-concept F1
)
from sklearn.preprocessing import MultiLabelBinarizer
import os
import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Set, Any, Tuple
import warnings

# --- Constants ---

# Defines the severity of each finding. Higher is more severe.
DYSPLASIA_SEVERITY_MAP = {
    "negative_for_dysplasia": 0,
    "none": 0,
    "reactive": 1,
    "reactive_changes": 1,
    "indefinite_for_dysplasia": 2,
    "low_grade_dysplasia": 3,
    "moderate_grade_dysplasia": 4, # Squamous-specific
    "high_grade_dysplasia": 5,
    "adenocarcinoma": 6,
    "squamous_cell_carcinoma": 6,
}

# All possible classes for the *critical finding*
# We derive this from the severity map
CRITICAL_CLASSES = sorted(list(DYSPLASIA_SEVERITY_MAP.keys()))

# The schema categories we are evaluating
SCHEMA_CATEGORIES = [
    "tissue_types",
    "columnar_findings",
    "squamous_findings",
    "metaplasia",
    "inflammation",
    "other_findings"
]

# --- Utility Functions ---

def get_highest_critical_finding(concepts: Dict[str, List[str]]) -> str:
    """Finds the most severe diagnostic finding from the extracted concepts."""
    highest_score = -1
    # Default if no scored findings are present
    highest_finding = "negative_for_dysplasia" 

    findings_to_check = concepts.get("columnar_findings", []) + \
                        concepts.get("squamous_findings", [])
    
    if not findings_to_check:
        return highest_finding

    for finding in findings_to_check:
        score = DYSPLASIA_SEVERITY_MAP.get(finding, -1)
        if score > highest_score:
            highest_score = score
            highest_finding = finding
            
    return highest_finding

def get_concepts_by_category(schema: Dict[str, List[str]], category: str) -> Set[str]:
    """Gets all unique, non-empty concepts for a single category as a set."""
    if not isinstance(schema, dict):
        return set()
    findings_list = schema.get(category, [])
    if not isinstance(findings_list, list):
        return set()
    return set(f for f in findings_list if f) # Ensure no empty strings

def calculate_set_metrics(original_set: Set[str], generated_set: Set[str]) -> Dict[str, float]:
    """Calculates Precision, Recall, and F1 for two sets."""
    metrics = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    true_positives = len(original_set.intersection(generated_set))
    
    if len(generated_set) > 0:
        metrics["precision"] = true_positives / len(generated_set)
    
    if len(original_set) > 0:
        metrics["recall"] = true_positives / len(original_set)
        
    pr_sum = metrics["precision"] + metrics["recall"]
    if pr_sum > 0:
        metrics["f1_score"] = (2 * metrics["precision"] * metrics["recall"]) / pr_sum
        
    return metrics


# --- Plotting Functions ---

def plot_confusion_matrix(cm, class_names, save_path):
    """Plots and saves a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Critical Finding Confusion Matrix", fontsize=14)
    plt.ylabel("True Label (from Original Report)", fontsize=12)
    plt.xlabel("Predicted Label (from Generated Report)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_per_category_f1(f1_scores: Dict[str, float], save_path: str):
    """Plots and saves a bar chart of F1 scores for each schema category."""
    df_f1 = pd.DataFrame(list(f1_scores.items()), columns=['Category', 'F1-Score'])
    df_f1 = df_f1.sort_values(by='F1-Score', ascending=False)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='F1-Score', y='Category', data=df_f1, palette='viridis')
    plt.title('Average F1-Score per Concept Category', fontsize=14)
    plt.xlabel('Average F1-Score', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Per-category F1 bar plot saved to {save_path}")

def plot_severity_error_histogram(errors: List[int], save_path: str):
    """Plots a histogram of severity grading errors."""
    plt.figure(figsize=(10, 6))
    
    if not errors:
        plt.text(0.5, 0.5, 'No error data to plot.', 
                 horizontalalignment='center', verticalalignment='center')
    else:
        min_err = min(errors)
        max_err = max(errors)
        # Create discrete bins for each integer error value
        bins = np.arange(min_err - 0.5, max_err + 1.5)
        
        sns.histplot(errors, bins=bins, kde=False, discrete=True, stat="count")
        
        plt.xlabel("Severity Error (Predicted - True)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title("Critical Finding Severity Error Distribution", fontsize=14)
        
        # Set integer ticks on x-axis
        plt.xticks(range(min_err, max_err + 1))
        
        # Add text labels for context
        plt.axvline(0, color='red', linestyle='--', lw=2)
        plt.text(0.1, plt.ylim()[1]*0.9, 'Over-grading (+)', 
                 color='darkred', ha='left')
        plt.text(-0.1, plt.ylim()[1]*0.9, 'Under-grading (-)', 
                 color='darkblue', ha='right')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Severity error histogram saved to {save_path}")

def plot_concept_level_f1(f1_scores: Dict[str, float], save_path: str, k: int = 15):
    """Plots a horizontal bar chart of the Top-K and Bottom-K concept F1-scores."""
    if not f1_scores:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No concept F1-scores to plot.', 
                 horizontalalignment='center', verticalalignment='center')
        plt.savefig(save_path)
        plt.close()
        print("No concept-level F1 scores to plot.")
        return

    # Sort scores
    sorted_scores = sorted(f1_scores.items(), key=lambda item: item[1])
    
    # Get Top-K and Bottom-K
    bottom_k = sorted_scores[:k]
    top_k = sorted_scores[-k:]
    
    # Combine, remove duplicates if k is large
    plot_data_dict = dict(bottom_k + top_k)
    plot_df = pd.DataFrame(list(plot_data_dict.items()), columns=['Concept', 'F1-Score'])
    plot_df = plot_df.sort_values(by='F1-Score', ascending=True)

    # Create color palette (e.g., green for high, red for low)
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(plot_df)))
    
    plt.figure(figsize=(14, max(8, len(plot_df) * 0.4)))
    plt.barh(plot_df['Concept'], plot_df['F1-Score'], color=colors)
    
    plt.xlabel('F1-Score', fontsize=12)
    plt.ylabel('Concept', fontsize=12)
    plt.title(f'Top/Bottom {k} Concept-Level F1-Scores', fontsize=14)
    plt.xlim(0, 1)
    
    # Add value labels
    for index, value in enumerate(plot_df['F1-Score']):
        plt.text(value + 0.01, index, f"{value:.2f}", va='center')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Concept-level F1 plot saved to {save_path}")


# --- Core Logic Functions ---

def setup_evaluation():
    """Parses command-line arguments and creates the output directory."""
    parser = argparse.ArgumentParser(description="Evaluate pathology report schema concordance.")
    parser.add_argument("--input_json", type=str, required=True, 
                        help="Path to the input JSON file containing extracted schemas.")
    parser.add_argument("--output_dir", type=str, default=".", 
                        help="Base directory for results.")
    parser.add_argument("--cm_filename", type=str, default="critical_finding_cm.png", 
                        help="Filename for the critical finding confusion matrix.")
    parser.add_argument("--f1_plot_filename", type=str, default="per_category_f1.png", 
                        help="Filename for the per-category F1 bar plot.")
    # --- New Arguments ---
    parser.add_argument("--severity_hist_filename", type=str, default="severity_error_hist.png",
                        help="Filename for the severity error histogram.")
    parser.add_argument("--concept_f1_filename", type=str, default="concept_level_f1.png",
                        help="Filename for the top/bottom concept F1 plot.")
    parser.add_argument("--label_source", type=str, default="keys", choices=["keys", "text"], 
                        help="Source of the labels for evaluation. (Note: Not used in this script but kept for compatibility).")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(args.output_dir, f"schema_evaluation_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving all results to: {output_path}")
    
    return args, output_path

def load_and_prepare_data(config: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads, cleans, and filters the evaluation data from the JSON file."""
    try:
        df = pd.read_json(config.input_json)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None, None

    # *** Corrected column names based on your original code ***
    if 'report_clinical_schema' not in df.columns or 'gen_clinical_schema' not in df.columns:
        print("Error: Input JSON must contain 'report_clinical_schema' and 'gen_clinical_schema' columns.")
        return None, None

    initial_count = len(df)
    # Drop rows where either schema is missing (NaN) or not a dictionary
    df_filtered = df.dropna(subset=['report_clinical_schema', 'gen_clinical_schema'])
    df_filtered = df_filtered[
        df_filtered['report_clinical_schema'].apply(isinstance, args=(dict,)) &
        df_filtered['gen_clinical_schema'].apply(isinstance, args=(dict,))
    ].copy()
    filtered_count = len(df_filtered)

    data_summary = {
        "initial_records": initial_count,
        "evaluated_records": filtered_count,
        "invalid_or_missing_schema_records_excluded": initial_count - filtered_count,
    }
    print(f"Excluded {data_summary['invalid_or_missing_schema_records_excluded']} records. "
          f"Evaluating on {filtered_count} records.")
    return df_filtered, data_summary

def calculate_metrics(df_eval: pd.DataFrame) -> Dict[str, Any]:
    """Computes all performance metrics based on the schema data."""
    
    y_true_critical = []
    y_pred_critical = []
    
    category_metrics = {cat: {"f1": [], "precision": [], "recall": []} for cat in SCHEMA_CATEGORIES}
    overall_concept_metrics = {"f1": [], "precision": [], "recall": []}
    
    # For concept-level F1 (using MultiLabelBinarizer)
    y_true_all_concepts = []
    y_pred_all_concepts = []
    
    # For severity error
    severity_errors = []

    for _, row in df_eval.iterrows():
        orig_schema = row['report_clinical_schema']
        gen_schema = row['gen_clinical_schema']

        # 1. Critical Finding
        true_crit = get_highest_critical_finding(orig_schema)
        pred_crit = get_highest_critical_finding(gen_schema)
        y_true_critical.append(true_crit)
        y_pred_critical.append(pred_crit)
        
        # --- New: Calculate Severity Error ---
        true_score = DYSPLASIA_SEVERITY_MAP.get(true_crit, 0)
        pred_score = DYSPLASIA_SEVERITY_MAP.get(pred_crit, 0)
        severity_errors.append(pred_score - true_score)

        # 2. Per-Category F1 & Concept-Level F1 prep
        all_orig_concepts = set()
        all_gen_concepts = set()

        for category in SCHEMA_CATEGORIES:
            orig_set = get_concepts_by_category(orig_schema, category)
            gen_set = get_concepts_by_category(gen_schema, category)
            
            # Add to this category's list
            cat_mets = calculate_set_metrics(orig_set, gen_set)
            category_metrics[category]["f1"].append(cat_mets["f1_score"])
            category_metrics[category]["precision"].append(cat_mets["precision"])
            category_metrics[category]["recall"].append(cat_mets["recall"])
            
            # Add to the overall set for this report pair
            # We prefix with category to avoid collisions (e.g., "reactive" in two places)
            prefixed_orig = {f"{category}:{c}" for c in orig_set}
            prefixed_gen = {f"{category}:{c}" for c in gen_set}
            all_orig_concepts.update(prefixed_orig)
            all_gen_concepts.update(prefixed_gen)
            
        # 3. Overall Concept F1 (for this one report pair)
        overall_mets = calculate_set_metrics(all_orig_concepts, all_gen_concepts)
        overall_concept_metrics["f1"].append(overall_mets["f1_score"])
        overall_concept_metrics["precision"].append(overall_mets["precision"])
        overall_concept_metrics["recall"].append(overall_mets["recall"])
        
        # 4. Add to lists for MultiLabelBinarizer
        y_true_all_concepts.append(all_orig_concepts)
        y_pred_all_concepts.append(all_gen_concepts)

    # --- Aggregate Metrics ---
    metrics = {}
    
    # 1. Critical Finding Metrics
    metrics['critical_finding_accuracy'] = accuracy_score(y_true_critical, y_pred_critical)
    
    # --- New: Calculate QWK ---
    metrics['critical_finding_qwk'] = cohen_kappa_score(
        y_true_critical, y_pred_critical, weights='quadratic'
    )
    
    present_labels = sorted(list(set(y_true_critical) | set(y_pred_critical)))
    
    metrics['critical_finding_report'] = classification_report(
        y_true_critical, y_pred_critical, labels=present_labels, zero_division=0, output_dict=True
    )
    metrics['critical_finding_confusion_matrix'] = {
        "labels": present_labels,
        "matrix": confusion_matrix(
            y_true_critical, y_pred_critical, labels=present_labels
        ).tolist()
    }
    # Store for plotting
    metrics['internal_y_true_critical'] = y_true_critical
    metrics['internal_y_pred_critical'] = y_pred_critical
    metrics['internal_severity_errors'] = severity_errors # <-- New

    # 2. Overall Concept Metrics (Averaged over all reports)
    metrics['overall_concept_metrics'] = {
        "avg_f1_score": np.mean(overall_concept_metrics["f1"]),
        "avg_precision": np.mean(overall_concept_metrics["precision"]),
        "avg_recall": np.mean(overall_concept_metrics["recall"]),
    }
    
    # 3. Per-Category F1 (Averaged over all reports)
    metrics['per_category_avg_metrics'] = {
        category: {
            "avg_f1_score": np.mean(val["f1"]),
            "avg_precision": np.mean(val["precision"]),
            "avg_recall": np.mean(val["recall"]),
        } for category, val in category_metrics.items()
    }
    
    # 4. --- New: Concept-Level F1 (Calculated over all reports) ---
    if y_true_all_concepts:
        mlb = MultiLabelBinarizer()
        y_true_bin = mlb.fit_transform(y_true_all_concepts)
        y_pred_bin = mlb.transform(y_pred_all_concepts)
        class_names = mlb.classes_
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Suppress UndefinedMetricWarning
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_bin, y_pred_bin, average=None, zero_division=0
            )
        
        metrics['concept_level_metrics'] = {
            class_names[i]: {
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "support": int(support[i])
            } for i in range(len(class_names))
        }
    else:
        metrics['concept_level_metrics'] = {}

    return metrics

def generate_visualizations(metrics: Dict[str, Any], output_path: str, config: argparse.Namespace):
    """Generates and saves all visualizations."""
    
    # 1. Plot Critical Finding Confusion Matrix
    cm_labels = metrics['critical_finding_confusion_matrix']['labels']
    cm_data = np.array(metrics['critical_finding_confusion_matrix']['matrix'])
    cm_path = os.path.join(output_path, config.cm_filename)
    plot_confusion_matrix(cm_data, cm_labels, cm_path)

    # 2. Plot Per-Category F1 Scores
    f1_scores_cat = {
        cat: data['avg_f1_score'] 
        for cat, data in metrics['per_category_avg_metrics'].items()
    }
    f1_plot_path = os.path.join(output_path, config.f1_plot_filename)
    plot_per_category_f1(f1_scores_cat, f1_plot_path)
    
    # 3. --- New: Plot Severity Error Histogram ---
    errors = metrics.get('internal_severity_errors', [])
    hist_path = os.path.join(output_path, config.severity_hist_filename)
    plot_severity_error_histogram(errors, hist_path)
    
    # 4. --- New: Plot Concept-Level F1 ---
    f1_scores_concept = {
        concept: data['f1_score']
        for concept, data in metrics.get('concept_level_metrics', {}).items()
        if data['support'] > 0 # Only plot concepts present in the true set
    }
    concept_f1_path = os.path.join(output_path, config.concept_f1_filename)
    plot_concept_level_f1(f1_scores_concept, concept_f1_path, k=15) # Show top/bottom 15

# --- Main Orchestration ---

def main():
    """Main function to run the complete evaluation pipeline."""
    config, output_path = setup_evaluation()
    
    df_eval, data_summary = load_and_prepare_data(config)
    
    if df_eval is None or df_eval.empty:
        print("No valid data to evaluate. Exiting.")
        return

    print("\n--- Calculating Performance Metrics ---")
    metrics = calculate_metrics(df_eval)
    
    # Print key results to console
    print(f"\nOverall Critical Finding Accuracy: "
          f"{metrics.get('critical_finding_accuracy', 0):.4f}")
    
    # --- New: Print QWK ---
    print(f"Critical Finding QWK: "
          f"{metrics.get('critical_finding_qwk', 0):.4f}")
    
    overall_f1 = metrics.get('overall_concept_metrics', {}).get('avg_f1_score')
    if overall_f1 is not None:
        print(f"Overall Concept F1-Score (Macro Avg): {overall_f1:.4f}\n")

    print("\n--- Generating Visualizations ---")
    generate_visualizations(metrics, output_path, config)
    
    # Clean up internal data before saving to JSON
    metrics.pop('internal_y_true_critical', None)
    metrics.pop('internal_y_pred_critical', None)
    metrics.pop('internal_severity_errors', None) # <-- New

    # Assemble final results dictionary for JSON output
    final_results = {
        "metadata": {
            "run_timestamp": os.path.basename(output_path).replace("schema_evaluation_", ""),
            "input_file": config.input_json,
            "all_schema_categories": SCHEMA_CATEGORIES,
        },
        "data_summary": data_summary,
        "metrics": metrics
    }
    
    # Save results to JSON
    results_json_path = os.path.join(output_path, "schema_evaluation_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(final_results, f, indent=4, 
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    print(f"\nStructured results saved to {results_json_path}")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    main()