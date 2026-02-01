# MLLM Barrett's Oesophagus

**MLLM Barrett** is a Multimodal Large Language Model (MLLM) designed for the analysis of Barrett's Oesophagus pathology. It integrates Whole Slide Image (WSI) embeddings with clinical text reports to perform tasks such as report generation, classification, and clinical schema extraction.

This repository contains the core model architecture (`src`), training and evaluation experiments (`experiments`), and data processing utilities (`scripts`).

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training (Multimodal)](#1-multimodal-training-wsi--text)
  - [Training (Text-Only)](#2-text-only-pretraining)
  - [Training (MIL Classifier)](#3-mil-classifier-baseline)
  - [Inference](#4-inference)
  - [Evaluation Pipeline](#5-evaluation-pipeline)
- [HPC / Slurm Usage](#-hpc--slurm-usage)

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd MLLM_BARRETT
   ```

2. **Install the package in editable mode:**
   This allows you to modify the code in `src/` and have changes reflected immediately without reinstalling. It also ensures imports work correctly across scripts.
   ```bash
   pip install -e .
   ```

3. **Dependencies:**
   The project requires Python 3.10+ and standard ML libraries (PyTorch, Transformers, etc.) as listed in `pyproject.toml`.

---

## 📂 Project Structure

```text
.
├── experiments/            # Scripts for training and running evaluations
│   ├── configs/            # YAML configuration files for models and training
│   ├── prompts/            # Text prompts for LLM evaluation and extraction
│   ├── train_wsi_lm.py     # Main script for training the Multimodal model
│   ├── train_lm.py         # Script for pretraining the text-only backbone
│   └── ...
├── src/                    # Core source code package
│   ├── data/               # Dataset classes and collators
│   ├── model/              # Model architectures (PathoLlama, TextLlama)
│   ├── trainer/            # Custom HuggingFace Trainer overrides
│   ├── inference/          # Inference scripts for single patients
│   ├── model_eval/         # Evaluation metrics and report generation
│   ├── process_reports/    # LLM-based report processing (vLLM/OpenAI)
│   └── utils/              # General utility functions
├── scripts/                # Data processing and utility scripts
│   └── slurm/              # Job scripts for Snellius HPC
└── pyproject.toml          # Project metadata and dependencies
```

---

## ⚙️ Configuration

All experiments are driven by YAML configuration files located in `experiments/configs/`. These files control dataset paths, model hyperparameters, and training settings.

* **Multimodal Training:** `experiments/configs/barrett/train_config.yaml`
* **Text Pretraining:** `experiments/configs/pubmed/config.yaml`
* **Inference:** `experiments/configs/model_inference/barrett/config.yaml`
* **MIL Baseline:** `experiments/configs/barrett/train_mil_config.yaml`

---

## Data Processing & Experiment Pipeline

### 1. WSI Preprocessing (TRIDENT)
We utilize [TRIDENT](https://github.com/mahmoodlab/TRIDENT) to process Barrett's esophagus whole slide images (WSIs). Following the installation of TRIDENT, the processing pipeline involves three steps:

**A. Extract Patches (CONCH)**
Extract patches using the `conch_v15` encoder.
*Note: We strictly use `--batch_size 1` because some slides contain a massive number of patches, and `--search_nested` ensures all slides in the directory are found.*

```bash
python run_batch_of_slides.py \
  --wsi_dir ../data/raw/HE_revision_24_10_25/RL-0007/ \
  --job_dir ../data/processed/revision_24_10_25/trident_conch_embeddings/ \
  --task all \
  --patch_encoder conch_v15 \
  --mag 20 \
  --patch_size 512 \
  --batch_size 1 \
  --search_nested
```

**B. Extract Slide Features (TITAN)**
Once patches are extracted, generate slide-level features using the `titan` encoder. This loop processes the slides identified in the previous step.

```bash
python run_batch_of_slides.py \
  --task feat \
  --wsi_dir ../data/raw/HE_revision_24_10_25/RL-0007/ \
  --job_dir ../data/processed/revision_24_10_25/trident_conch_embeddings/ \
  --slide_encoder titan \
  --mag 20 \
  --patch_size 512
```

**C. Combine Feature Files**
TRIDENT outputs individual H5 files. We combine these into a single training dataset using our helper script:

```bash
python scripts/combine_h5_files.py \
  --source_dir ../data/processed/revision_24_10_25/trident_conch_embeddings/20x_512px_0px_overlap/slide_features_titan/ \
  --output_file ../data/processed/titan_combined_slide_features/titan_slide_features_conch_v15_revision_24_10_25.h5
```

### 2. Model Training
To train the MLLM using the processed WSI features, run `experiments/train_wsi_lm.py`.

**Configuration:**
Below is the reference configuration (e.g., `experiments/configs/barrett/train_config.yaml`) used for training:

```yaml
model:
  core_model_name: "meta-llama/Meta-Llama-3-8B"
  load_checkpoint_path: "./trained/pretrained_tcga_models/checkpoint-6000-1-layer/"
  vocab_size: 32768
  hidden_size: 768
  num_hidden_layers: 1
  num_attention_heads: 2
  max_position_embeddings: 320
  freeze_config:
    freeze_all: False
    layers_to_unfreeze:
      - "model.layers.1"  # Unfreeze the last hidden layer
      - "lm_head"         # Unfreeze the language model head

tokenizer:
  tokenizer_name: "./tokenizers/trained_tokenizers/32768_pubmed/"
  vocab_size: 32768
  custom_tokenizer: true

dataset:
  train_h5_file_path: "../../data/processed/titan_combined_slide_features/titan_slide_features_conch_v15_revision_24_10_25.h5"
  train_texts_json_path: "../../data/processed/json/cleaned/qwen_235b_tcga_structured_barrett.json"
  val_data_ratio: 0.133
  embeddings_dim_size: 768
  max_seq_length: 320
  random_choice_report: false

training:
  output_dir: "./trained/mid_lr_pretrained_qwen_tcga_structured_revision_1_layer_pretrained"
  eval_strategy: "epoch"
  do_train: true
  do_eval: true
  save_strategy: "steps"
  save_steps: 50
  logging_steps: 1
  lr_scheduler_type: "constant"
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 1
  num_train_epochs: 60
  warmup_ratio: 0.01
  learning_rate: 0.0006
  dataloader_num_workers: 12
  bf16: true
  report_to: ["tensorboard"]
```

**Run Command:**
```bash
python experiments/train_wsi_lm.py --config experiments/configs/barrett/train_config.yaml
```

### 3. Evaluation Pipeline (New Flow)
We have consolidated the inference, schema extraction, and LLM-as-a-Judge evaluation into a single automated pipeline script: `experiments/run_evaluations_vllm.py`.

This pipeline automates the following steps:
1.  **Report Generation:** Generates reports from WSI features using the trained checkpoint.
2.  **Schema Extraction:** Uses a vLLM server to extract structured clinical schemas from the generated reports.
3.  **Ground Truth Transfer:** Transfers label schemas from the reference dataset for comparison.
4.  **LLM Judge:** Uses a specialized prompt (`experiments/prompts/barrett/llm_judge.txt`) to compare the Generated Report vs. the Original Report regarding factual accuracy and critical findings.
5.  **Metrics & Plotting:** Calculates scores and generates distribution plots.

**How to Run:**

First, ensure a vLLM server is running (e.g., on port 8000). Then execute the full loop:

```bash
python experiments/run_evaluations_vllm.py \
  --config experiments/configs/barrett/eval_config.yaml \
  --model trained/mid_lr_pretrained_qwen_tcga_structured_revision_1_layer_pretrained/checkpoint-XXX/model.safetensors \
  --output_base_dir ./evaluation_results \
  --source_report_labels experiments/configs/labels/reference_full_real_report_labels.json \
  --port 8000 \
  --prompt_judge experiments/prompts/barrett/llm_judge.txt \
  --model_name_vllm Qwen/Qwen3-235B-A22B-GPTQ-Int4
```


## 🚀 Usage

**Note:** Ensure you are in the root directory of the project when running these commands.

### 1. Multimodal Training (WSI + Text)
Train the `PathoLlama` model to generate reports from WSI embeddings.

```bash
python experiments/train_wsi_lm.py --config experiments/configs/barrett/train_config.yaml
```

### 2. Text-Only Pretraining
Pretrain the language model backbone on medical text (e.g., PubMed) before multimodal fine-tuning.

```bash
python experiments/train_lm.py --config experiments/configs/pubmed/config.yaml
```

### 3. MIL Classifier Baseline
Train a baseline Multiple Instance Learning (MIL) classifier or Simple MLP on the embeddings.

```bash
python experiments/train_mil_classifier.py --config experiments/configs/barrett/train_mil_config.yaml
```

### 4. Inference
Generate a report for a single patient (or a random validation sample) using a trained model.

```bash
python src/inference/patient_inference.py \
    --config experiments/configs/model_inference/barrett/config.yaml \
    --patient "RL-1234"  # Optional: omit to pick a random sample
```

### 5. Evaluation Pipeline
Run the comprehensive evaluation pipeline using vLLM as a judge. This pipeline generates reports, extracts clinical schemas, and compares them against ground truth using a stronger LLM (e.g., Qwen).

**Prerequisites:**
1.  A running vLLM server (compatible with OpenAI API) on the specified port.
2.  A trained checkpoint (`model.safetensors`).

```bash
python experiments/run_evaluations_vllm.py \
    --config experiments/configs/barrett/eval_config.yaml \
    --model /path/to/checkpoint/model.safetensors \
    --output_base_dir ./evaluation_results \
    --source_report_labels ./path/to/ground_truth_labels.json \
    --port 8000 \
    --model_name_vllm "unsloth/Qwen3-235B-A22B-GGUF"
```

#### Evaluation Metrics Only
If you already have generated reports and want to compute classification metrics (Accuracy, F1, AUC):

```bash
python -m src.model_eval.evaluate_barrett_classification \
    --input_json ./evaluation_results/processed_labels.json \
    --label_source keys \
    --output_dir ./metrics_output
```

---

## ⚡ HPC / Slurm Usage

This project includes specific scripts for running training and evaluation jobs on the **Snellius** national supercomputer. You can find these in `scripts/slurm/`.

### Job Script Locations
* **Evaluation Loop:** `scripts/slurm/run_full_eval_loop.job`
    * Runs generation, schema extraction, and LLM judging in one pipeline.
* **Inference:** `scripts/slurm/run_pathology_inference.job`
    * Runs batched inference on test sets.
* **Preprocessing:** `scripts/slurm/extract_patches_slide_embeddings_trident.job`
    * Handles WSI patching and feature extraction.

### Submitting a Job
To submit a job to the scheduler:

```bash
sbatch scripts/slurm/run_full_eval_loop.job
```

### Typical Snellius Configuration
The job scripts are pre-configured for Snellius partitions (`gpu`). Ensure you have the correct modules loaded or included in the script:

```bash
# Example Snippet
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00

module load 2023
module load Python/3.10.4-GCCcore-11.3.0

source $HOME/venvs/mllm_env/bin/activate
```
