import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import numpy as np
from typing import Dict, Any, Tuple
from collections import Counter
import warnings

# --- Constants ---

# The score columns we expect from the LLM Judge JSON
SCORE_COLUMNS = [
    "critical_finding_concordance",
    "factual_accuracy",
    "factual_completeness",
    "overall_equivalence"
]

# Keywords to search for in the reasoning field
REASONING_KEYWORDS = [
    'omit', 'omitted', 'omission', 'miss', 'missing',
    'hallucinate', 'hallucinated', 'hallucination', 'add', 'added', 'invented',
    'contradict', 'contradictory', 'mismatch', 'wrong', 'incorrect', 'error',
    'equivalent', 'match', 'concordant', 'good', 'excellent', 'perfect', 'similar'
]

# --- Plotting Functions ---

def plot_score_distributions(df_metrics: pd.DataFrame, save_path: str):
    """
    Plots a 100% stacked bar chart showing the distribution of scores
    for each evaluation criterion.
    """
    # Melt the dataframe to long format for easy plotting with seaborn
    df_melted = df_metrics.melt(var_name='Criterion', value_name='Score')
    
    # Get all possible integer scores for a consistent hue order
    all_scores = sorted(df_melted['Score'].dropna().unique().astype(int))
    palette = sns.color_palette("RdYlGn", n_colors=len(all_scores))
    
    plt.figure(figsize=(12, 8))
    # Use histplot with multiple="fill" to create a 100% stacked bar chart
    ax = sns.histplot(
        data=df_melted,
        x='Criterion',
        hue='Score',
        hue_order=all_scores, # Ensure 1 is at the bottom (red) and 4 is at the top (green)
        multiple='fill',
        palette=palette,
        discrete=True,
        shrink=0.8  # Adds a small gap between bars
    )
    
    plt.title('Distribution of LLM-as-a-Judge Scores', fontsize=16)
    plt.xlabel('Evaluation Criterion', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    
    # Format Y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=15, ha='right')
    
    # Move legend
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Score")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Score distribution plot saved to {save_path}")

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_path: str):
    """
    Plots a heatmap of the correlation between scoring criteria.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        fmt=".2f"
    )
    plt.title('Correlation Matrix of Judge Scores (Spearman Rank)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation heatmap saved to {save_path}")

# --- Core Logic Functions ---

def setup_evaluation():
    """Parses command-line arguments and creates the output directory."""
    parser = argparse.ArgumentParser(description="Evaluate LLM-as-a-Judge results.")
    parser.add_argument("--input_json", type=str, required=True, 
                        help="Path to the input JSON file containing judge scores.")
    parser.add_argument("--output_dir", type=str, default=".", 
                        help="Base directory for results.")
    parser.add_argument("--dist_plot_filename", type=str, default="judge_score_distribution.png", 
                        help="Filename for the score distribution plot.")
    parser.add_argument("--corr_plot_filename", type=str, default="judge_score_correlation.png", 
                        help="Filename for the score correlation heatmap.")
    parser.add_argument("--label_source", type=str, default="keys", choices=["keys", "text"], 
                        help="Source of the labels for evaluation. (Note: Not used in this script but kept for compatibility).")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(args.output_dir, f"llm_judge_evaluation_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving all results to: {output_path}")
    
    return args, output_path

def load_and_prepare_data(config: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads, cleans, and filters the evaluation data from the JSON file."""
    try:
        # Assumes the JSON is a list of objects, one for each report
        df = pd.read_json(config.input_json)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None, None
        
    initial_count = len(df)
    
    # Check if 'llm_judgement' column exists
    if 'llm_judgement' not in df.columns:
        print("Error: Input JSON must contain an 'llm_judgement' column.")
        return None, None
        
    # Normalize the 'llm_judgement' dictionary column into separate columns
    try:
        judge_df = pd.json_normalize(df['llm_judgement'])
    except Exception as e:
        print(f"Error normalizing 'llm_judgement' column. Is it a valid dictionary? Error: {e}")
        return None, None
    
    # Drop the original dict column and join the new normalized columns
    df = df.drop(columns=['llm_judgement'])
    df_eval = pd.concat([df, judge_df], axis=1)

    # Validate that all required score columns are present after normalization
    missing_cols = [col for col in SCORE_COLUMNS if col not in df_eval.columns]
    if missing_cols:
        print(f"Error: 'llm_judgement' is missing required keys: {missing_cols}")
        return None, None
        
    # Drop rows with any missing scores
    df_eval = df_eval.dropna(subset=SCORE_COLUMNS)
    
    # Ensure scores are numeric (int)
    for col in SCORE_COLUMNS:
        df_eval[col] = pd.to_numeric(df_eval[col], errors='coerce').astype('Int64')
        
    # Drop rows where scores couldn't be coerced
    df_final = df_eval.dropna(subset=SCORE_COLUMNS).copy()
    final_count = len(df_final)

    data_summary = {
        "initial_records": initial_count,
        "evaluated_records": final_count,
        "invalid_or_missing_score_records_excluded": initial_count - final_count,
    }
    
    excluded_count = data_summary['invalid_or_missing_score_records_excluded']
    print(f"Excluded {excluded_count} records. Evaluating on {final_count} records.")
    return df_final, data_summary

def calculate_metrics(df_eval: pd.DataFrame) -> Dict[str, Any]:
    """Computes all performance metrics based on the judge's scores."""
    
    metrics = {}
    
    # 1. Descriptive Statistics (Mean, Median, Std, etc.)
    metrics['descriptive_statistics'] = df_eval[SCORE_COLUMNS].describe().to_dict()
    
    # 2. Score Distributions (as percentages)
    distributions = {}
    for col in SCORE_COLUMNS:
        # Get counts for all possible scores (1-4) even if 0
        all_possible_scores = range(1, 5) # Assume 1-4 for all, plot will adjust
        dist = df_eval[col].value_counts(normalize=True).reindex(all_possible_scores, fill_value=0.0).sort_index()
        distributions[col] = {str(k): v for k, v in dist.to_dict().items() if v > 0} # Only report scores that exist
    metrics['score_distributions_percent'] = distributions
    
    # 3. Clinical Acceptability ("Pass Rates")
    # Based on your data:
    #   Concordance/Equivalence (1-4): Pass >= 3 (Good/Excellent)
    #   Accuracy/Completeness (1-3): Pass >= 2 (No major issues)
    metrics['clinical_acceptability'] = {
        "overall_equivalence_pass_rate (>=3)": (df_eval['overall_equivalence'] >= 3).mean(),
        "critical_finding_pass_rate (>=3)": (df_eval['critical_finding_concordance'] >= 3).mean(),
        "factual_accuracy_pass_rate (>=2)": (df_eval['factual_accuracy'] >= 2).mean(),
        "factual_completeness_pass_rate (>=2)": (df_eval['factual_completeness'] >= 2).mean()
    }
    
    # 4. Correlation Matrix (Spearman for ordinal data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore std dev = 0 warnings
        corr_matrix = df_eval[SCORE_COLUMNS].corr(method='spearman')
    metrics['correlation_matrix'] = corr_matrix.to_dict()
    metrics['internal_corr_matrix_df'] = corr_matrix # For plotting
    
    # 5. Reasoning Keyword Analysis
    all_reasoning_text = " ".join(df_eval['reasoning'].dropna().str.lower().values)
    keyword_counts = Counter()
    for word in REASONING_KEYWORDS:
        keyword_counts[word] = all_reasoning_text.count(word)
    metrics['reasoning_keyword_analysis'] = dict(keyword_counts.most_common())
    
    # 6. Worst Offender Examples
    # Get reasoning for any report that scored '1' on overall_equivalence
    worst_offenders_df = df_eval[df_eval['overall_equivalence'] == 1]
    metrics['worst_offender_reports (overall_equivalence=1)'] = worst_offenders_df[
        ['case_id', 'reasoning']
    ].to_dict('records')

    return metrics

def generate_visualizations(df_eval: pd.DataFrame, metrics: Dict[str, Any], output_path: str, config: argparse.Namespace):
    """Generates and saves all visualizations."""
    
    # 1. Plot Score Distributions
    dist_plot_path = os.path.join(output_path, config.dist_plot_filename)
    plot_score_distributions(df_eval[SCORE_COLUMNS], dist_plot_path)
    
    # 2. Plot Correlation Heatmap
    corr_matrix = metrics.get('internal_corr_matrix_df')
    if corr_matrix is not None:
        corr_plot_path = os.path.join(output_path, config.corr_plot_filename)
        plot_correlation_heatmap(corr_matrix, corr_plot_path)

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
    print("\n--- Average Scores ---")
    avg_scores = metrics.get('descriptive_statistics', {})
    if avg_scores:
        for col in SCORE_COLUMNS:
            print(f"  {col}: {avg_scores[col]['mean']:.2f}")
    
    print("\n--- Clinical Acceptability (Pass Rates) ---")
    for key, val in metrics.get('clinical_acceptability', {}).items():
        print(f"  {key}: {val:.1%}")
        
    print("\n--- Reasoning Keyword Counts (Top 5) ---")
    keywords = metrics.get('reasoning_keyword_analysis', {})
    for key, val in list(keywords.items())[:5]:
        print(f"  {key}: {val}")

    print(f"\n--- Worst Failures (Overall Equivalence = 1) ---")
    worst = metrics.get('worst_offender_reports (overall_equivalence=1)', [])
    print(f"  Found {len(worst)} reports with a score of 1.")
    for i, report in enumerate(worst[:3]): # Print first 3
        print(f"    - Case {report['case_id']}: {report['reasoning'][:100]}...")


    print("\n--- Generating Visualizations ---")
    generate_visualizations(df_eval, metrics, output_path, config)
    
    # Clean up internal data before saving to JSON
    metrics.pop('internal_corr_matrix_df', None)

    # Assemble final results dictionary for JSON output
    final_results = {
        "metadata": {
            "run_timestamp": os.path.basename(output_path).replace("llm_judge_evaluation_", ""),
            "input_file": config.input_json,
        },
        "data_summary": data_summary,
        "metrics": metrics
    }
    
    # Save results to JSON
    results_json_path = os.path.join(output_path, "llm_judge_results.json")
    with open(results_json_path, 'w') as f:
        # Use pandas' to_json for better handling of numpy types
        final_results_df = pd.DataFrame([final_results])
        final_results_df.to_json(f, orient='records', indent=4)
        
    print(f"\nStructured results saved to {results_json_path}")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
