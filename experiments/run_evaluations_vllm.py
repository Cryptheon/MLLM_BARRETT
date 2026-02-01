import argparse
import subprocess
import sys
from pathlib import Path

# ==========================================
# Path Configuration
# ==========================================
ROOT_DIR = Path(".") 
# We don't rely on path variables for execution anymore, but on module names
# where possible.

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
    parser.add_argument("--prompt_schema", type=str, default="experiments/prompts/barrett/clinical_schema_extraction.txt")
    parser.add_argument("--prompt_judge", type=str, default="experiments/prompts/barrett/llm_judge.txt")
    parser.add_argument("--model_name_vllm", type=str, default="Qwen/Qwen3-235B-A22B-GPTQ-Int4")

    args = parser.parse_args()

    model_path = Path(args.model)
    checkpoint_name = model_path.parent.name 
    
    output_dir = Path(args.output_base_dir) / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processing_json = output_dir / "processed_labels.json"
    temp_prompt_jsonl = output_dir / "temp_prompts.jsonl"
    temp_result_jsonl = output_dir / "temp_results.jsonl"

    # ==========================================
    # 1. Generate Reports
    # ==========================================
    cmd_gen = [
        "python", "-m", "src.model_eval.generate_reports_from_wsi",
        "--config", args.config,
        "--output", str(processing_json),
        "--model_path", args.model
    ]
    run_command(cmd_gen, "Generate Reports from WSI")

    # ==========================================
    # 2. Extract Schema
    # ==========================================
    # Using -m src.utils.vllm_utils ensures imports inside vllm_utils work
    run_command([
        "python", "-m", "src.utils.vllm_utils", "prepare",
        "--input_json", str(processing_json),
        "--output_jsonl", str(temp_prompt_jsonl),
        "--prompt_file", args.prompt_schema,
        "--input_keys", "generated_report"
    ], "Prepare Prompts: Gen Schema")

    # This script is standalone, but we can call it via file path or module if package exists
    # Using file path is fine if it has no relative imports.
    run_command([
        "python", "src/process_reports/batch_inference_decoder.py",
        "--input", str(temp_prompt_jsonl),
        "--output", str(temp_result_jsonl),
        "--port", args.port,
        "--model_name", args.model_name_vllm,
        "--temperature", "0.6"
    ], "Inference: Gen Schema")

    run_command([
        "python", "-m", "src.utils.vllm_utils", "merge",
        "--input_json", str(processing_json),
        "--vllm_output_jsonl", str(temp_result_jsonl),
        "--output_key", "gen_clinical_schema",
        "--parse_think"
    ], "Merge: Gen Schema")

    # ==========================================
    # 3. Ground Truth Labels
    # ==========================================
    if args.source_report_labels:
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
    run_command([
        "python", "-m", "src.utils.vllm_utils", "prepare",
        "--input_json", str(processing_json),
        "--output_jsonl", str(temp_prompt_jsonl),
        "--prompt_file", args.prompt_judge,
        "--input_keys", "original_report", "generated_report"
    ], "Prepare Prompts: Judge")

    if temp_result_jsonl.exists(): temp_result_jsonl.unlink()

    run_command([
        "python", "src/process_reports/batch_inference_decoder.py",
        "--input", str(temp_prompt_jsonl),
        "--output", str(temp_result_jsonl),
        "--port", args.port,
        "--model_name", args.model_name_vllm,
        "--temperature", "0.6"
    ], "Inference: Judge")

    run_command([
        "python", "-m", "src.utils.vllm_utils", "merge",
        "--input_json", str(processing_json),
        "--vllm_output_jsonl", str(temp_result_jsonl),
        "--output_key", "llm_judgement",
        "--parse_think"
    ], "Merge: Judge")

    # ==========================================
    # 5. Evaluate/Plot
    # ==========================================
    # 5a. Evaluate Clinical Schemas
    run_command([
        "python", "-m", "src.model_eval.evaluate_clinical_schemas",
        "--input_json", str(processing_json),
        "--output_dir", str(output_dir),
        "--label_source", "keys"
    ], "Evaluate Schemas")

    # 5b. Evaluate Judge Outputs (Added to complete the pipeline)
    run_command([
        "python", "-m", "src.model_eval.evaluate_judge_outputs",
        "--input_json", str(processing_json),
        "--output_dir", str(output_dir)
    ], "Evaluate Judge Results")
    
    print(f"Pipeline finished for {checkpoint_name}")

if __name__ == "__main__":
    main()
