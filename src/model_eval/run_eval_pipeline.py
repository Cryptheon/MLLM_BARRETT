import argparse
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
# These paths are fixed as per your request. Modify them here if needed.
#PROMPT_PATH = "./configs/prompts/barrett/extract_barrett_dysplasia_labels.txt"
PROMPT_PATH = "./configs/prompts/barrett/clinical_schema_extraction.txt"
GPU_ID = 0

# Base directory for all outputs (both plots and intermediate JSONs)
#OUTPUT_BASE_DIR = Path("../../data/results/plots")

def run_command(command, step_name):
    """Executes a command using subprocess and handles errors."""
    print(f"--- Step: {step_name} ---")
    print(f"Executing command:\n{' '.join(command)}\n")
    try:
        # Using capture_output=True and text=True to show stdout/stderr on failure
        result = subprocess.run(command, check=True, text=True, capture_output=False)
        print(result.stdout) # Print stdout on success
        print(f"--- Step Succeeded: {step_name} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR in step: {step_name} ---", file=sys.stderr)
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        print("\n--- STDOUT ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1) # Exit the script if any command fails
    except FileNotFoundError as e:
        print(f"--- ERROR in step: {step_name} ---", file=sys.stderr)
        print(f"Error: Command not found. Make sure '{e.filename}' is in your PATH.", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to run the complete evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run the full Barrett evaluation pipeline from report generation to metric calculation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file for report generation (e.g., ./configs/model_inference/barrett/config.yaml)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model safetensors file (e.g., ./pathollama_barrett/checkpoint-1200/model.safetensors)."
    )
    parser.add_argument(
        "--gguf_model",
        type=str,
        required=False,
        help="Path to the gguf yaml (e.g., ./configs/barrett/gguf/qwen3-235b-a22-GGUF.yaml)."
    )

    parser.add_argument(
        "--source_report_labels",
        type=str,
        required=False,
        help="Path to the .json with report_extracted_labels source labels."
    )

    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="../../data/results/plots",
        required=False,
        help="Path to the .json with report_extracted_labels source labels."
    )

    parser.add_argument(
        "--prompt_path_schema",
        type=str,
        default="./configs/prompts/barrett/clinical_schema_extraction.txt",
        required=False,
        help="Path to the .json with the schema prompt."
    )

    parser.add_argument(
        "--prompt_path_judge",
        type=str,
        default="./configs/prompts/barrett/llm_judge.txt",
        required=False,
        help="Path to the .json with the llm judge prompt."
    )


    args = parser.parse_args()

    # --- Infer Paths and Names ---
    model_path = Path(args.model)
    model_base_name = model_path.parent.name # e.g., "checkpoint-1200"
    OUTPUT_BASE_DIR = Path(args.output_base_dir)
    print("OUTPUT BASE DIR SET TO", OUTPUT_BASE_DIR)

    # Create the main output directory if it doesn't exist
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE_DIR / model_base_name).mkdir(parents=True, exist_ok=True)

    # --- Path Definitions ---
    # MODIFIED: A single JSON file used and updated by each step, saved in the main output directory.
    processing_json = OUTPUT_BASE_DIR / model_base_name / "processed_labels.json"
    
    # MODIFIED: A specific subdirectory for the final evaluation, named after the checkpoint.
    evaluation_output_dir = OUTPUT_BASE_DIR / model_base_name

    # --- 1. Generate Reports ---
    cmd_generate_reports = [
        "python", "-m", "model_eval.generate_reports_from_wsi",
        "--config", args.config,
        "--output", str(processing_json),
        "--model_path", args.model
    ]
    run_command(cmd_generate_reports, "Generate Reports from WSI")

    # --- 2. Extract Labels from Generated & Real Reports (if gguf_model is provided) ---
    if args.gguf_model:
        # This script modifies the JSON file in place, adding new keys.
        # We run it twice on the same file, for generated and original reports.
        
        # --- 2a. Extract Labels from Generated Reports ---
        # cmd_extract_gen_labels = [
        #     "python", "-m", "process_reports.process_reports_task",
        #     "--task", "extract_label",
        #     "--input_key", "generated_report",
        #     "--output_key", "gen_extracted_label",
        #     "--config", args.gguf_model,
        #     "--prompt_path", PROMPT_PATH,
        #     "--input_json", str(processing_json),
        #     "--output_json", str(processing_json), # Overwrite the same file
        #     "--batch_size", "1",
        #     "--num_variations", "1",
        #     "--gpu", str(GPU_ID)
        # ]
        cmd_extract_gen_labels = [
            "python", "-m", "process_reports.process_reports_task",
            "--task", "extract_schema",
            "--input_key", "generated_report",
            "--output_key", "gen_clinical_schema",
            "--config", args.gguf_model,
            "--prompt_path", args.prompt_path_schema,
            "--input_json", str(processing_json),
            "--output_json", str(processing_json), # Overwrite the same file
            "--batch_size", "1",
            "--num_variations", "1",
            "--gpu", str(GPU_ID)
        ]
        if "qwen" in args.gguf_model:
            cmd_extract_gen_labels.append("--parse_out_thinking")
        run_command(cmd_extract_gen_labels, "Extract Labels from Generated Reports")

        cmd_extract_judge = [
            "python", "-m", "process_reports.process_reports_task",
            "--task", "judge_reports",
            "--input_key", "original_report", "generated_report",
            "--output_key", "llm_judgement",
            "--config", args.gguf_model,
            "--prompt_path", args.prompt_path_judge,
            "--input_json", str(processing_json),
            "--output_json", str(processing_json), # Overwrite the same file
            "--batch_size", "1",
            "--num_variations", "1",
            "--gpu", str(GPU_ID)
        ]
        run_command(cmd_extract_judge, "Use an LLM to judge both reports.")
        
        if args.source_report_labels:
            # This command assumes you have a file, e.g., 'ground_truth_labels.json', 
            # which contains the correctly extracted labels from a previous run.
            cmd_transfer_labels = [
                "python", "-m", "utils.transfer_labels", # Use the new script
                "--source_json", args.source_report_labels, # IMPORTANT: Set this path correctly
                "--label_key", "report_clinical_schema",
                "--target_json", str(processing_json),
                "--output_json", str(processing_json) # Overwrites the target file with the updated labels
            ]
            run_command(cmd_transfer_labels, "Transfer Ground Truth Labels from Source File")

        
        else:
            # --- 2b. Extract Labels from Real Reports ---
            # cmd_extract_real_labels = [
            #     "python", "-m", "process_reports.process_reports_task",
            #     "--task", "extract_label",
            #     "--input_key", "original_report",
            #     "--output_key", "report_extracted_label",
            #     "--config", args.gguf_model,
            #     "--prompt_path", PROMPT_PATH,
            #     "--input_json", str(processing_json), # Use the same, now-updated, file
            #     "--output_json", str(processing_json), # Overwrite it again
            #     "--batch_size", "1",
            #     "--num_variations", "1",
            #     "--gpu", str(GPU_ID)
            # ]
            cmd_extract_real_labels = [
                "python", "-m", "process_reports.process_reports_task",
                "--task", "extract_schema",
                "--input_key", "original_report",
                "--output_key", "report_clinical_schema",
                "--config", args.gguf_model,
                "--prompt_path", args.prompt_path_schema,
                "--input_json", str(processing_json), # Use the same, now-updated, file
                "--output_json", str(processing_json), # Overwrite it again
                "--batch_size", "1",
                "--num_variations", "1",
                "--gpu", str(GPU_ID)
            ]
            if "qwen" in args.gguf_model:
                cmd_extract_real_labels.append("--parse_out_thinking")
            run_command(cmd_extract_real_labels, "Extract Labels from Real Reports")     


    # --- 3. Evaluate and Generate Plots ---
    if not processing_json.exists():
        print(f"--- FATAL ERROR ---", file=sys.stderr)
        print(f"The input file for the evaluation step was not found: {processing_json}", file=sys.stderr)
        sys.exit(1)

    # cmd_evaluate = [
    #     "python", "model_eval/evaluate_barrett_classification.py",
    #     "--input_json", str(processing_json),
    #     "--output_dir", str(evaluation_output_dir), # Use the new checkpoint-specific directory
    #     "--label_source", "keys" if args.gguf_model else "text"
    # ]
    cmd_evaluate = [
        "python", "model_eval/evaluate_clinical_schemas.py",
        "--input_json", str(processing_json),
        "--output_dir", str(evaluation_output_dir), # Use the new checkpoint-specific directory
        "--label_source", "keys" if args.gguf_model else "text"
    ]
    run_command(cmd_evaluate, "Evaluate Classification and Generate Plots")

    cmd_evaluate_judge = [
        "python", "model_eval/evaluate_judge_outputs.py",
        "--input_json", str(processing_json),
        "--output_dir", str(evaluation_output_dir), # Use the new checkpoint-specific directory
        "--label_source", "keys" if args.gguf_model else "text"
    ]
    run_command(cmd_evaluate_judge, "Evaluate LLM Judge outputs and Generate Plots")

    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()