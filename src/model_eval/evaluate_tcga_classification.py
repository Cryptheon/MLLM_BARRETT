import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_multiclass_roc(y_true_bin, y_pred_bin, class_names, save_path):
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f'{name} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate cancer type classification from pathology reports.")
    parser.add_argument("--predictions_json", type=str, required=True, help="Path to model output JSON")
    parser.add_argument("--ground_truth_csv", type=str, required=True, help="CSV with columns 'patient_id,cancer_type'")
    parser.add_argument("--tcga_json", type=str, required=True, help="Path to TCGA abbreviation-to-name mapping JSON")
    args = parser.parse_args()

    with open(args.predictions_json, "r") as f:
        predictions = {x["patient_id"]: x["extracted_label"] for x in json.load(f)}

    with open(args.tcga_json, "r") as f:
        tcga_map = json.load(f)

    # Invert mapping: name -> abbreviation
    name_to_code = {v.lower(): k for k, v in tcga_map.items()}

    df_true = pd.read_csv(args.ground_truth_csv)
    df_true = df_true[df_true["patient_id"].isin(predictions.keys())].copy()
    df_true["predicted"] = df_true["patient_id"].map(predictions)

    y_true = df_true["cancer_type"].fillna("UNKNOWN")
    y_pred_raw = df_true["predicted"].fillna("UNKNOWN").str.lower()
    y_pred = y_pred_raw.map(name_to_code).fillna("UNKNOWN")

    # Map y_true abbreviations to full names for evaluation
    y_true_names = y_true.map(tcga_map).fillna("UNKNOWN")
    y_pred_names = y_pred.map(tcga_map).fillna("UNKNOWN")

    # Classification metrics
    print("Classification Report:\n")
    print(classification_report(y_true_names, y_pred_names, zero_division=0))

    acc = accuracy_score(y_true_names, y_pred_names)
    print(f"Accuracy: {acc:.4f}")

    # # Confusion Matrix
    labels = sorted(set(y_true_names) | set(y_pred_names))
    # cm = confusion_matrix(y_true_names, y_pred_names, labels=labels)
    # plot_confusion_matrix(cm, labels, save_path="confusion_matrix.png")

    # AUC (multiclass)
    y_true_bin = label_binarize(y_true_names, classes=labels)
    y_pred_bin = label_binarize(y_pred_names, classes=labels)
    try:
        auc_macro = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
        print(f"AUC (macro): {auc_macro:.4f}")
    #     plot_multiclass_roc(y_true_bin, y_pred_bin, labels, save_path="roc_curve.png")
    except Exception as e:
        print(f"AUC could not be computed: {e}")

if __name__ == "__main__":
    main()
