# Model Evaluation (`src/model_eval/`)

This directory contains the core logic for assessing the performance of the PathoLlama and multimodal Barrett models. It includes scripts for quantitative metrics (classification), qualitative analysis (t-SNE), and end-to-end report generation from Whole Slide Images (WSIs).

## Overview of Evaluation Scripts

### 1. `compute_tsne_barrett.py`
* **Purpose:** Generates t-SNE (t-Distributed Stochastic Neighbor Embedding) visualizations of the model's latent space for Barrett's Esophagus cases.
* **Functionality:** - Extracts embeddings from the model's vision encoder or multimodal projector.
  - Reduces high-dimensional features to 2D space.
  - Colors points by clinical labels (NDBE, LGD, HGD, EAC) to visualize class separation.
* **Usage:**
  ```bash
  python src/model_eval/compute_tsne_barrett.py --checkpoint /path/to/model --output_dir ./results/plots
  ```

### 2. `evaluate_barrett_classification.py`
* **Purpose:** Performs zero-shot or supervised classification evaluation for Barrett's dysplasia grades.
* **Metrics Provided:** Accuracy, Precision, Recall, F1-Score, and Area Under the ROC Curve (AUC-ROC).
* **Details:** It compares the model's predicted labels against the "ground truth" reference labels stored in `experiments/configs/labels/`.
* **Usage:**
  ```bash
  python src/model_eval/evaluate_barrett_classification.py --config experiments/configs/barrett/eval_config.yaml
  ```

### 3. `evaluate_model_fidelity.py`
* **Purpose:** Measures the "fidelity" or factual accuracy of generated pathology reports.
* **Functionality:**
  - Compares generated reports against original clinician-written reports.
  - Uses NLP metrics (like BLEU, ROUGE) and domain-specific clinical schema extraction.
  - Verifies if the model correctly identified key histological features mentioned in the prompt.

### 4. `evaluate_judge_outputs.py`
* **Purpose:** Interfaces with an "LLM Judge" (typically GPT-4 or a strong open-source model) to provide a semantic score for generated reports.
* **Functionality:** Reads the judge results and aggregates scores for coherence, clinical accuracy, and formatting.

### 5. `generate_reports_from_wsi.py` & `generate_reports_from_single_wsis.py`
* **Purpose:** The primary inference pipeline for turning whole slides into textual diagnostic reports.
* **Logic:**
  - `generate_reports_from_wsi.py`: Processes a batch of slides across multiple patients.
  - `generate_reports_from_single_wsis.py`: Focused utility for generating a report for a specific, single slide path.
* **Expectation:** A JSON or Markdown output containing the generated "Diagnosis" and "Microscopic Description" sections for each slide.

### 6. `conch_evaluation.py`
* **Purpose:** Specifically evaluates the vision-only or zero-shot capabilities of the CONCH encoder within the pipeline, used as a baseline for the full multimodal model.

---

## Configuration

Evaluation behavior is primarily driven by the YAML files in `experiments/configs/barrett/`. Key parameters include:
- `dataset_name`: Which subset to evaluate on.
- `eval_batch_size`: Batch size for feature extraction.
- `temperature`: Randomness for report generation (usually set to `0.0` for deterministic evaluation).

## Expected Outputs

All scripts in this directory generally output to a `results/` folder (configured in the CLI), producing:
1. **JSON Files:** Raw predictions and metric scores.
2. **CSV Files:** Tabulated results for multi-checkpoint comparison.
3. **PNG/PDF:** Visualization plots from t-SNE or Confusion Matrices.