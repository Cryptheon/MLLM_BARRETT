import json
import argparse
import time
import os
from typing import Dict, Any, List, Callable
import functools

from process_reports.llm_processor import ModelProcessor
# Assuming load_config exists in utils
from utils.util_functions import load_config

def get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """
    Retrieves a value from a nested dict using a dot-separated path.
    Example: get_nested_value(case, 'cleaned_reports.0.Diagnose')
    """
    try:
        return functools.reduce(
            lambda d, key: d[int(key)] if key.isdigit() and isinstance(d, list) else d.get(key),
            path.split('.'),
            data
        )
    except (KeyError, TypeError, IndexError):
        return None

def set_nested_value(data: Dict[str, Any], path: str, value: Any):
    """
    Sets a value in a nested dict using a dot-separated path.
    Example: set_nested_value(case, 'cleaned_reports.0.barrett_label', ['LGD'])
    """
    if "." in path:
        keys = path.split('.')
        parent = get_nested_value(data, '.'.join(keys[:-1]))

        final_key = keys[-1]
        parent[final_key] = value

    else:
        data[path] = value

def main():
    parser = argparse.ArgumentParser(description="Run a generic, modular step in the pathology report processing pipeline.")
    parser.add_argument("--task", type=str, required=True, help="A descriptive name for the task being performed (e.g., 'label', 'evaluate').")

    parser.add_argument(
        "--input_key",
        type=str,
        required=True,
        nargs='+',
        help="One or more keys in the input JSON to be processed. Use dot-notation for nested fields. If multiple keys are provided, their values will be concatenated with a newline."
    )

    parser.add_argument("--output_key", type=str, required=True, help="The new key to add to the JSON where the output will be stored. Use dot-notation for nested fields (e.g., 'cleaned_reports.0.barrett_label').")

    parser.add_argument("--config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the file containing the base prompt for this task.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of cases to process in a single batch.")
    parser.add_argument("--num_variations", type=int, default=1, help="Number of variations to generate for each case.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index of cases to process.")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending index of cases to process (exclusive).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use.")
    parser.add_argument('--parse_out_thinking', action='store_true', default=False, required=False)

    args = parser.parse_args()
    config = load_config(args.config)
    print(config)

    # Note: args.input_key is now a list of strings
    print(f"Running task '{args.task}' on GPU {args.gpu}: processing '{' + '.join(args.input_key)}' -> '{args.output_key}'")
    os.environ["LLAMA_VISIBLE_DEVICES"] = str(args.gpu)

    processor = ModelProcessor(config=config, prompt_path=args.prompt_path)

    with open(args.input_json, 'r', encoding='utf-8') as f:
        all_cases = json.load(f)

    print("Number of cases available:", len(all_cases))

    end_idx = args.end_idx if args.end_idx is not None else len(all_cases)
    cases_to_process = all_cases[args.start_idx:end_idx]

    #out_path = args.output_json.replace(".json", f"_{args.task}_gpu{args.gpu}.json")
    out_path = args.output_json
    # TODO
    # Check if the output file already exists to append results
    # if os.path.exists(out_path):
    #     with open(out_path, 'r', encoding='utf-8') as f:
    #         all_results = json.load(f)
    # else:
    
    all_results = []

    start_time_total = time.time()

    for i in range(len(all_results), len(cases_to_process), args.batch_size):
        batch_slice = cases_to_process[i:i + args.batch_size]
        current_batch_index = args.start_idx + i
        print(f"Processing batch starting at index {current_batch_index}/{end_idx}")

        cases_in_batch = []
        for case_item in batch_slice:

            # 1. Retrieve the value for each input key provided
            values_to_combine = []
            for key_path in args.input_key:
                value = get_nested_value(case_item, key_path)
                if value: # Ensure value is not None or empty
                    values_to_combine.append(str(value).strip())

            # 2. Join the found values with a newnewline
            if values_to_combine:
                report_to_process = "\n\n".join(values_to_combine)
                cases_in_batch.append(report_to_process)
            else:
                # This case happens if none of the provided keys were found
                print(f"Warning: None of the input keys {args.input_key} found in case at index {args.start_idx + i + len(cases_in_batch)}. Skipping.")
                cases_in_batch.append("")

        batch_results = processor.process_batch(
            texts=cases_in_batch,
            num_variations=args.num_variations
        )

        print(batch_results)
        
        # This logic seems to handle single-item batches that might be nested in an extra list
        if len(batch_results) == 1 and args.batch_size == 1 and isinstance(batch_results[0], list):
             batch_results = batch_results[0]
        
        if args.parse_out_thinking:
            batch_results = [batch_results[0].split("<think>\n\n</think>\n\n")[1]]

        for original_case_item, model_output in zip(batch_slice, batch_results):
            result_entry = original_case_item.copy()
            
            # --- FIX STARTS HERE ---
            # The model is likely returning a JSON-formatted string. We parse it into a
            # Python dictionary so it gets saved as a proper JSON object.
            parsed_output = model_output
            try:
                # Attempt to parse the string into a Python object (dict or list)
                parsed_output = json.loads(model_output)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails (e.g., it's not a valid JSON string or not a string at all),
                # print a warning and store the raw output.
                print(f"Warning: Could not parse model output as JSON. Storing as raw output.")
            # --- FIX ENDS HERE ---

            set_nested_value(result_entry, args.output_key, parsed_output)
            all_results.append(result_entry)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(all_results)} total results to {out_path}")

    print(f"\nFinished GPU {args.gpu}: processed cases {args.start_idx}–{end_idx} in {(time.time() - start_time_total):.2f}s")

if __name__ == "__main__":
    main()