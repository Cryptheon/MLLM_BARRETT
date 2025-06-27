import json
import argparse
import time
import os
from typing import Dict, Any, List, Callable, Tuple

# We still use the same powerful, unchanged ModelProcessor
from process_reports.llm_processor import ModelProcessor
# Assuming load_config exists in utils
from utils.util_functions import load_config

# --- Data Handler Functions ---
# These functions define the unique logic for each pipeline step.

def extract_for_translation(case_item: Dict[str, Any]) -> Dict[str, Any]:
    """Extractor for the 'translate' task."""
    return case_item.get('case', {})

def format_for_translation(original_item: Dict[str, Any], model_output: List[Dict]) -> Dict[str, Any]:
    """Formatter for the 'translate' task."""
    return {
        "original_case": original_item.get('case', {}),
        "translated_reports": model_output
    }

def extract_for_cleaning(case_item: Dict[str, Any]) -> Dict[str, Any]:
    """Extractor for the 'clean' task."""
    # Takes the first report from the 'translated_reports' list
    if 'translated_reports' in case_item and case_item['translated_reports']:
        return case_item['translated_reports'][0]
    return {} # Return empty dict if not found, to be skipped later

def format_for_cleaning(original_item: Dict[str, Any], model_output: List[Dict]) -> Dict[str, Any]:
    """Formatter for the 'clean' task."""
    return {
        "original_case": original_item.get('original_case', {}),
        "translated_reports": original_item.get('translated_reports', []),
        "cleaned_reports": model_output # Use a new key for the cleaned output
    }

# --- Main Processing Logic ---

def format_case_to_json_string(case_data: Dict[str, Any]) -> str:
    """Serializes a case dictionary into a JSON-formatted string."""
    return json.dumps(case_data, indent=4, ensure_ascii=False)

def main():
    # A dictionary mapping task names to their specific data handler functions
    TASK_HANDLERS = {
        "translate": (extract_for_translation, format_for_translation),
        "clean": (extract_for_cleaning, format_for_cleaning)
    }

    parser = argparse.ArgumentParser(description="Run a modular step in the pathology report processing pipeline.")
    parser.add_argument("--task", type=str, required=True, choices=TASK_HANDLERS.keys(), help="The processing task to perform ('translate' or 'clean').")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the file containing the base prompt for this task.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of cases to process in a single batch.")
    parser.add_argument("--num_variations", type=int, default=1, help="Number of variations to generate for each case.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index of cases to process.")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending index of cases to process (exclusive).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use.")

    args = parser.parse_args()
    config = load_config(args.config)

    # Select the correct data handlers based on the --task argument
    extractor_func, formatter_func = TASK_HANDLERS[args.task]

    print(f"Running task '{args.task}' on GPU {args.gpu}")
    os.environ["LLAMA_VISIBLE_DEVICES"] = str(args.gpu)

    # Initialize the processor
    processor = ModelProcessor(config=config, prompt_path=args.prompt_path)

    # Load the input data
    with open(args.input_json, 'r', encoding='utf-8') as f:
        all_cases = json.load(f)

    print("Number of cases available:", len(all_cases))

    end_idx = args.end_idx if args.end_idx is not None else len(all_cases)
    cases_to_process = all_cases[args.start_idx:end_idx]

    out_path = args.output_json.replace(".json", f"_gpu{args.gpu}.json")
    all_results = []
    
    # Resume logic
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"Resuming. Loaded {len(all_results)} existing results from {out_path}")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing output file. Starting over.")
            all_results = []

    start_time_total = time.time()

    for i in range(0, len(cases_to_process), args.batch_size):
        batch_slice = cases_to_process[i:i + args.batch_size]
        current_batch_index = args.start_idx + i
        print(f"Processing batch starting at index {current_batch_index}/{end_idx}")

        json_texts_to_process = []
        valid_cases_in_batch = []
        for case_item in batch_slice:
            # Use the selected extractor function
            report_to_process = extractor_func(case_item)
            if report_to_process: # Ensure we don't process empty objects
                json_texts_to_process.append(format_case_to_json_string(report_to_process))
                valid_cases_in_batch.append(case_item)
            else:
                print(f"Warning: Skipping item at index {i} as it could not be processed by the '{args.task}' extractor.")
        
        if not json_texts_to_process:
            print("No valid cases to process in this batch. Skipping.")
            continue

        # Process the batch using the same robust processor
        batch_results = processor.process_batch(
            json_texts=json_texts_to_process,
            num_variations=args.num_variations
        )
        print(batch_results)

        # Structure and store results using the selected formatter function
        for original_case_item, model_output in zip(valid_cases_in_batch, batch_results):
            result_entry = formatter_func(original_case_item, model_output)
            all_results.append(result_entry)

        # Save results to the output file after each batch
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(all_results)} total results to {out_path}")

    print(f"\nFinished GPU {args.gpu}: processed cases {args.start_idx}–{end_idx} in {(time.time() - start_time_total):.2f}s")

if __name__ == "__main__":
    main()
