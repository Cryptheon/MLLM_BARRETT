# Pathology Report Batch Processing with Local LLM

This repository contains scripts to batch-process pathology reports using a locally hosted language model. Reports are cleaned and reformatted using a prompt stored in a `.txt` file. The updated script utilizes `llama-cpp-python` for fast inference with GGUF models and supports multi-GPU execution via SLURM.

---

## Scripts Overview

### 1. process_reports_with_llama_cpp.py (Primary Script)

This is the main script for processing pathology reports using a `llama-cpp` model. It supports multi-GPU execution and is designed to be compatible with SLURM batch jobs.

**Functionality:**
- Loads a prompt from a `.txt` file.
- Uses `llama-cpp` to run inference with a local GGUF model.
- Supports generating multiple variations of cleaned reports per input.

**Input CSV format:**
The input CSV must contain at least the following columns:
- `patient_filename`: A unique identifier (e.g., `TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0`)
- `text`: The raw pathology report text to be processed

**Example `text` value:**
```
Date of Receipt: Clinical Diagnosis & History: Incidental 3 cm left upper pole renal mass. Specimens Submitted: 1: Kidney, Left Upper Pole; Partial Nephrectomy. DIAGNOSIS: ...
```

**Running the script manually (single GPU example):**
```bash
python process_reports_with_llama_cpp.py \
  --config path/to/config.yaml \
  --prompt_path path/to/prompt.txt \
  --input_csv path/to/input.csv \
  --output_csv path/to/output.csv \
  --column text \
  --batch_size 1 \
  --num_variations 1 \
  --start_idx 0 \
  --end_idx 100 \
  --gpu 0
```

Outputs are saved incrementally to allow for resuming or parallel GPU processing.

---

**Output:**
A new CSV is created with a column `processed_reports` containing the cleaned reports.

---

### 2. slurm_process_reports.job (Multi-GPU Batch Job with SLURM)

This SLURM script allows you to process a large CSV of pathology reports by launching 4 parallel jobs on separate GPUs, each processing a chunk of the data.

**Functionality:**
- Splits input CSV into N parts based on row indices.
- Assigns each process to one of N GPU.
- Runs `process_reports_with_llama_cpp.py` in parallel.
- Waits for all processes to finish.
- Concatenates partial output files into a final CSV.

**Running the SLURM script:**
```bash
sbatch slurm_process_reports.job
```

**Key Details:**
- `CUDA_VISIBLE_DEVICES`: Controls GPU assignment per process.
- `concatenate_processed_reports.py`: Collects all per-GPU outputs.

**Expected Output Files:**
- `output_reports/histopathology_reports_gpu0.csv` ... `gpu3.csv`: Per-GPU partial outputs.
- `output_reports/histopathology_reports.csv`: Final merged report file.

This setup is ideal for distributed processing of large datasets on a cluster with multiple GPUs.

---

### 3. extract_tcga_labels_with_llm.py (TCGA Label Extraction)

This script extracts standardized TCGA cancer type labels from generated pathology reports using a language model.

**Functionality:**
- Loads a list of standardized TCGA labels.
- Uses a prompt to extract the label from a generated report.
- Normalizes and maps the extracted label to a TCGA code.

**Input JSON format:**
```json
[
  {
    "patient_id": "TCGA-XX-XXXX",
    "generated_report": "..."
  }
]
```

**Running the script:**
Example:

```bash
python extract_tcga_labels_with_llm.py \
  --config ../configs/model_inference/Llama-3.3-70B-instruct-GGUF.yaml \
  --prompt_path ../configs/prompts/extract_tcga_label_with_list.txt \
  --input_json ../data/tcga_data/tcga_generated/generated_reports/tcga_200_2_layers_8192_vocab.json \
  --tcga_json ../configs/tcga/tcga_labels.json \
  --output_json ../data/tcga_data/tcga_generated/extracted_labels/tcga_200_val_2_layers_8192_vocab.json \
  --gpu 0
```

**Output:**
A JSON file with entries including the original patient ID, extracted label, and matched TCGA code.

---

### 4. evaluate_reports_with_llama_cpp.py (Report Evaluation)

This script compares original and generated reports and scores them using a prompt-based rubric via a local LLM.

**Functionality:**
- Loads a prompt rubric.
- Evaluates original vs. generated reports.
- Produces one or more evaluations per report pair.

**Input JSON format:**
```json
[
  {
    "patient_id": "TCGA-XX-XXXX",
    "original_report": "...",
    "generated_report": "..."
  }
]
```

**Running the script:**
Example:

```bash
python python evaluate_reports_with_llama_cpp.py \  
  --config ../configs/model_inference/Llama-3.3-70B-instruct-GGUF.yaml \
  --prompt_path ../configs/prompts/tcga_generated_fidelity_rubric_prompt.txt \   
  --input_json ../data/tcga_data/tcga_generated/tcga_200_val.json \
  --output_json ../data/tcga_data/tcga_generated/llama_70b_eval.json \
  --start_idx 0 \
  --end_idx 20
```

**Output:**
A JSON file where each entry contains the patient ID and a list of raw evaluation outputs.

This script is useful for assessing the fidelity and quality of generated pathology reports.

