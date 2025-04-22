import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_and_parse_json(raw_text):
    """
    Extract the first complete JSON object from a string, even if it's embedded in natural language.
    """
    start = raw_text.find("{")
    if start == -1:
        raise ValueError("No opening brace found")

    # Track nested braces
    brace_count = 0
    for i in range(start, len(raw_text)):
        if raw_text[i] == "{":
            brace_count += 1
        elif raw_text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                json_str = raw_text[start:i + 1]
                return json.loads(json_str)

    raise ValueError("No complete JSON object found")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fidelity scores from raw LLM evaluation outputs.")
    parser.add_argument("--fidelity_json", type=str, required=True, help="Path to JSON with raw LLM outputs")
    args = parser.parse_args()

    with open(args.fidelity_json, "r") as f:
        raw_data = json.load(f)

    all_overall_scores = []
    per_case_summaries = []
    category_scores = {}

    for entry in raw_data:
        case_id = entry["case_id"]
        raw_output_str = entry["raw_outputs"][0]

        try:
            parsed = extract_and_parse_json(raw_output_str)
        except Exception as e:
            print(f"Skipping {case_id} due to JSON parsing error: {e}")
            continue

        overall_score = parsed.get("overall_fidelity_score", None)
        all_overall_scores.append(overall_score)

        scores = parsed["scores"]
        component_scores = [v["score"] for v in scores.values()]
        component_sum = sum(component_scores)
        component_avg = np.mean(component_scores)
        component_std = np.std(component_scores)

        for category, details in scores.items():
            category_scores.setdefault(category, []).append(details["score"])

        per_case_summaries.append({
            "case_id": case_id,
            "evaluated_case_id": parsed.get("case_id", "UNKNOWN"),
            "overall_fidelity_score": overall_score,
            "component_score_sum": component_sum,
            "component_score_avg": component_avg,
            "component_score_std": component_std
        })

    df_summary = pd.DataFrame(per_case_summaries)

    overall_avg = np.mean(all_overall_scores)
    overall_std = np.std(all_overall_scores)

    print("=== Overall Fidelity Scores ===")
    print(f"Average: {overall_avg:.4f}")
    print(f"Std Dev: {overall_std:.4f}\n")

    print("=== Per-Case Component Score Summary ===")
    print(df_summary.describe(include='all'))

    df_summary.to_csv("fidelity_summary.csv", index=False)

    # ----- Per-Category Stats -----
    df_category = pd.DataFrame.from_dict(category_scores, orient="index").T
    df_melted = df_category.melt(var_name="Category", value_name="Score")

    # Print category stats
    print("\n=== Per-Category Score Stats ===")
    category_means = df_category.mean().sort_values(ascending=False)
    category_stds = df_category.std()
    for cat in category_means.index:
        print(f"{cat:40s}  Mean: {category_means[cat]:.3f}  Std: {category_stds[cat]:.3f}")

    # Save plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x="Category", y="Score")
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribution of Scores per Category")
    plt.tight_layout()
    plt.savefig("category_scores_boxplot.png")
    plt.close()


if __name__ == "__main__":
    main()
