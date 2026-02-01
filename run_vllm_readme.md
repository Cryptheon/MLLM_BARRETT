# MLLM Barrett: Pathology Report Evaluation Pipeline (vLLM)

This repository hosts the pipeline for training and evaluating Multi-Modal Large Language Models (MLLMs) on pathology data. The evaluation suite has been modernized to use **vLLM** for high-throughput, batch-processed inference using large MoE models (e.g., Qwen3-235B).

## 🚀 Pipeline Overview

The evaluation pipeline automates the assessment of your trained checkpoints. For every checkpoint in a specified range, it performs the following steps:

1.  **Report Generation:** Generates a pathology report from a Whole Slide Image (WSI) using your trained MLLM adapter.
2.  **Schema Extraction:** Uses a massive "Teacher" LLM (Qwen-235B via vLLM) to extract structured clinical schemas (e.g., diagnoses, grades) from the generated text.
3.  **LLM Judging:** Uses the Teacher LLM to compare the generated report against the ground truth report and provide a fidelity score.
4.  **Metric Calculation:** Computes classification metrics (Accuracy, F1, etc.) based on the extracted schemas.



## 📂 Repository Structure

The relevant files for the vLLM evaluation pipeline are organized as follows:

```text
MLLM_BARRETT/
├── experiments/
│   ├── configs/
│   │   └── eval_config.yaml               # Configuration for WSI report generation
│   ├── prompts/
│   │   ├── clinical_schema_extraction.txt # Prompt for extracting structured labels
│   │   └── llm_judge.txt                  # Prompt for grading report fidelity
│   └── run_evaluations_vllm.py            # [Main Orchestrator] Python script for single-checkpoint eval
├── src/
│   ├── process_reports/
│   │   └── batch_inference_decoder.py     # [vLLM Client] Async script sending requests to server
│   ├── utils/
│   │   └── vllm_utils.py                  # [Helper] Converts JSON <-> JSONL for vLLM
│   └── model_eval/                        # Legacy evaluation scripts (called by orchestrator)
├── scripts/
│   └── slurm/
│       └── run_full_eval_loop.job         # [SLURM Job] Manages Server Lifecycle & Checkpoint Loop
└── ...
```

---

## 🛠️ Environment & Prerequisites

* **Cluster Access:** A SLURM-based cluster (e.g., Snellius) with GPU partitions (H100/A100 required for 235B models).
* **Container:** An Apptainer/Singularity container with `vllm` and `openai` python libraries installed.
    * *Default Path:* `/projects/2/managed_datasets/containers/vllm/cuda-13.0-vllm.sif`
* **Model:** A supported LLM checkpoint (e.g., `Qwen/Qwen3-235B-A22B-GPTQ-Int4`).

---

## 🏃 Usage Guide

The entire process is automated by a single SLURM job script.

### 1. Configure the Job
Open `scripts/slurm/run_full_eval_loop.job` and adjust the configuration variables at the top:

```bash
# Define your model path and evaluation range
START_CHECKPOINT=400
END_CHECKPOINT=1500
BASE_MODEL_PATH="./src/trained/my_experiment_name"
GT_LABELS="./experiments/configs/labels/reference_labels.json"
```

### 2. Submit the Job
Run the following command from the root of the repository:

```bash
sbatch scripts/slurm/run_full_eval_loop.job
```

### 3. Monitor Progress
The job will produce two log files in `slurm_logs/`:
* `eval_loop_[ID].out`: Standard output (progress bars, step completion).
* `eval_loop_[ID].err`: Error logs (vLLM server startup logs, python exceptions).

**What happens inside the job?**
1.  **Server Start:** A vLLM server is launched in the background on a random port. It loads the 235B model across 4 GPUs.
2.  **Wait:** The script waits until the server is healthy (`curl` check).
3.  **Loop:** It iterates through your checkpoints. For each one, it launches `experiments/run_evaluations_vllm.py`.
4.  **Client-Server Interaction:** The python script prepares prompts in JSONL format and sends them to the running local server using `src/process_reports/batch_inference_decoder.py`.
5.  **Cleanup:** The server is killed automatically when the loop finishes.

---

## 🔧 Component Details

### `experiments/run_evaluations_vllm.py`
This is the "brain" of the evaluation for a single checkpoint. It chains together subprocess calls:
1.  **`generate_reports_from_wsi`**: Uses your trained adapter to write reports to `processed_labels.json`.
2.  **`vllm_utils.py prepare`**: Reads `processed_labels.json` and creates `temp_prompts.jsonl`.
3.  **`batch_inference_decoder.py`**: Sends `temp_prompts.jsonl` to the vLLM server and writes `temp_results.jsonl`.
4.  **`vllm_utils.py merge`**: Merges the inference results back into `processed_labels.json`.
5.  **`evaluate_clinical_schemas`**: Calculates F1/Accuracy scores and saves plots.

### `src/utils/vllm_utils.py`
A utility helper that handles data transformation.
* **Prepare:** Injects your raw data (e.g., "Microscopy: ...") into the prompt template (`experiments/prompts/clinical_schema_extraction.txt`).
* **Merge:** Handles the tricky part of mapping async results back to their original JSON objects using a `custom_id`. It also strips out `<think>` tags if using reasoning models.

### `src/process_reports/batch_inference_decoder.py`
The asynchronous OpenAI-compatible client.
* **Concurrency:** It uses `asyncio` to send hundreds of requests in parallel, saturating the vLLM server for maximum throughput.
* **Resumable:** It checks the output file before starting; if the job crashes, it picks up where it left off.

---

## ❓ Troubleshooting

**1. "Server died during startup"**
* Check `slurm_logs/*.err`.
* Common cause: Not enough VRAM. Ensure `--tensor-parallel-size 4` is set and you are on a node with 4x H100/A100 (80GB).
* Common cause: Port conflict. The script picks a random port, but collisions can rarely happen. Resubmit the job.

**2. "FileNotFoundError: experiments/prompts/..."**
* Ensure you are running the sbatch command from the **root** of the repository (`MLLM_BARRETT/`), not from inside `scripts/`.

**3. "Connection Refused" in Python Client**
* The server might have crashed mid-loop. Check the SLURM error log.
* If the server is still running, check if the `$PORT` variable was correctly passed to the python script.
