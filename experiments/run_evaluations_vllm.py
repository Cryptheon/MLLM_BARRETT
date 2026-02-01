import argparse
import subprocess
import sys
from pathlib import Path

# ==========================================
# Path Configuration
# ==========================================
# We assume this script is run from the ROOT directory (MLLM_BARRETT/)
ROOT_DIR = Path(".") 
SCRIPTS_DIR = ROOT_DIR / "scripts"
SRC_DIR = ROOT_DIR / "src"

# Path to the utility script
VLLM_UTILS_PATH = SRC_DIR / "utils" / "vllm_utils.py"
# Path to the batch inference client
INFERENCE_CLIENT_PATH = SRC_DIR / "process_reports" / "batch_inference_decoder.py"

def run_command(command, step_name):
    """Executes a command using subprocess."""
    print(f"\n--- Step: {step_name} ---")
    print(f"CMD: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, text=True)
        print(f"--- Step Succeeded: {step_name} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR in step: {step_name} ---", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    # Report Gen Args
    parser.add_argument("--config", type=str, required=True, help="Path to experiments/configs/x.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint model.safetensors")
    parser.add_argument("--output_base_dir", type=str, required=True)
    parser.add_argument("--source_report_labels", type=str, required=False)
    
    # vLLM Args
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--prompt_schema", type=str, default="experiments/prompts/clinical_schema_extraction.txt")
    parser.add_argument("--prompt_judge", type=str, default="experiments/prompts/llm_judge.txt")
    parser.add_argument("--model_name_vllm", type=str, default="Qwen/Qwen3-235B-A22B-GPTQ-Int4")

    args = parser.parse_args()

    # Determine paths
    model_path = Path(args.model)
    checkpoint_name = model_path.parent.name # e.g. checkpoint-400
    
    # Final Output Dir: .../output_base_dir/checkpoint-X/
    output_dir = Path(args.output_base_dir) / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processing_json = output_dir / "processed_labels.json"
    temp_prompt_jsonl = output_dir / "temp_prompts.jsonl"
    temp_result_jsonl = output_dir / "temp_results.jsonl"

    # ==========================================
    # 1. Generate Reports (Using existing WSI model)
    # ==========================================
    # Assumes module structure: src.model_eval.generate_reports_from_wsi
    cmd_gen = [
        "python", "-m", "src.model_eval.generate_reports_from_wsi",
        "--config", args.config,
        "--output", str(processing_json),
        "--model_path", args.model
    ]
    run_command(cmd_gen, "Generate Reports from WSI")

    # ==========================================
    # 2. Extract Schema (Generated Reports)
    # ==========================================
    # A. Prepare
    run_command([
        "python", str(VLLM_UTILS_PATH), "prepare",
        "--input_json", str(processing_json),
        "--output_jsonl", str(temp_prompt_jsonl),
        "--prompt_file", args.prompt_schema,
        "--input_keys", "generated_report"
    ], "Prepare Prompts: Gen Schema")

    # B. Inference
    if temp_result_jsonl.exists(): temp_result_jsonl.unlink()
    
    run_command([
        "python", str(INFERENCE_CLIENT_PATH),
        "--input", str(temp_prompt_jsonl),
        "--output", str(temp_result_jsonl),
        "--port", args.port,
        "--model_name", args.model_name_vllm,
        "--temperature", "0.6"
    ], "Inference: Gen Schema")

    # C. Merge
    run_command([
        "python", str(VLLM_UTILS_PATH), "merge",
        "--input_json", str(processing_json),
        "--vllm_output_jsonl", str(temp_result_jsonl),
        "--output_key", "gen_clinical_schema",
        "--parse_think"
    ], "Merge: Gen Schema")

    # ==========================================
    # 3. Ground Truth Labels
    # ==========================================
    if args.source_report_labels:
        # Transfer existing labels
        run_command([
            "python", "-m", "src.utils.transfer_labels",
            "--source_json", args.source_report_labels,
            "--label_key", "report_clinical_schema",
            "--target_json", str(processing_json),
            "--output_json", str(processing_json)
        ], "Transfer Ground Truth Labels")

    # ==========================================
    # 4. LLM Judge
    # ==========================================
    # A. Prepare
    run_command([
        "python", str(VLLM_UTILS_PATH), "prepare",
        "--input_json", str(processing_json),
        "--output_jsonl", str(temp_prompt_jsonl),
        "--prompt_file", args.prompt_judge,
        "--input_keys", "original_report", "generated_report"
    ], "Prepare Prompts: Judge")

    # B. Inference
    if temp_result_jsonl.exists(): temp_result_jsonl.unlink()

    run_command([
        "python", str(INFERENCE_CLIENT_PATH),
        "--input", str(temp_prompt_jsonl),
        "--output", str(temp_result_jsonl),
        "--port", args.port,
        "--model_name", args.model_name_vllm,
        "--temperature", "0.6"
    ], "Inference: Judge")

    # C. Merge
    run_command([
        "python", str(VLLM_UTILS_PATH), "merge",
        "--input_json", str(processing_json),
        "--vllm_output_jsonl", str(temp_result_jsonl),
        "--output_key", "llm_judgement",
        "--parse_think"
    ], "Merge: Judge")

    # ==========================================
    # 5. Evaluate/Plot
    # ==========================================
    run_command([
        "python", "-m", "src.model_eval.evaluate_clinical_schemas",
        "--input_json", str(processing_json),
        "--output_dir", str(output_dir),
        "--label_source", "keys"
    ], "Evaluate Schemas")
    
    print(f"Pipeline finished for {checkpoint_name}")

if __name__ == "__main__":
    main()
