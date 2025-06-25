import json
import argparse
import time
import os
from llama_cpp import Llama
from typing import List, Dict, Any

# Assuming load_config exists in utils
from utils.util_functions import load_config

def get_prompt(prompt_path: str) -> str:
    """Reads the base prompt from a file."""
    with open(prompt_path, "r", encoding='utf-8') as file:
        return file.read()

def format_case_to_json_string(case_data: Dict[str, Any]) -> str:
    """Serializes a case dictionary into a JSON-formatted string."""
    return json.dumps(case_data, indent=4, ensure_ascii=False)

def process_batch(model: Llama, json_texts: List[str], base_prompt: str, config: Any, num_variations: int) -> List[List[Dict]]:
    """
    Processes a batch of JSON strings using the language model.
    The model's output is parsed back into a JSON object.
    """
    processed_batch_results = []
    start_time = time.time()
    total_tokens = 0

    for json_text in json_texts:
        single_input_variations = []
        print(json_text)
        messages = [
            {"role": "system", "content": "You are an expert pathology report translator outputting only valid JSON."},
            {"role": "user", "content": base_prompt.replace("__JSON_TO_TRANSLATE__", json_text)}
        ]
        
        for i in range(num_variations):
            try:
                output = model.create_chat_completion(
                    messages=messages,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    repeat_penalty=config.repetition_penalty,
                    stream=False,
                    # Enforce JSON output if the model supports it
                    response_format={"type": "json_object"},
                )
                
                result_text = output['choices'][0]['message']['content']
                # Parse the output string back into a dictionary
                parsed_result = json.loads(result_text)
                single_input_variations.append(parsed_result)
                
                total_tokens += output.get('usage', {}).get('completion_tokens', 0)

            except json.JSONDecodeError:
                print(f"Warning: Could not parse LLM output as JSON. Saving raw text.\nRaw output: {result_text}")
                single_input_variations.append({"error": "Failed to parse JSON", "raw_output": result_text})
            except Exception as e:
                print(f"An unexpected error occurred during attempt {i+1}/{num_variations}: {e}")
                single_input_variations.append({"error": str(e)})

        processed_batch_results.append(single_input_variations)

    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
    print(f"\nThroughput: {tokens_per_second:.2f} tokens/sec | Total: {total_tokens} tokens in {elapsed_time:.2f}s for {len(json_texts)} items.")
    return processed_batch_results

def main():
    parser = argparse.ArgumentParser(description="Process pathology JSON data using Llama-cpp.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the file containing the base prompt.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of cases to process in a single batch.")
    parser.add_argument("--num_variations", type=int, default=1, help="Number of translated variations to generate for each case.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index of cases to process.")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending index of cases to process (exclusive).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use.")

    args = parser.parse_args()
    config = load_config(args.config)
    base_prompt = get_prompt(args.prompt_path)

    print(f"Running on GPU {args.gpu}")
    os.environ["LLAMA_VISIBLE_DEVICES"] = str(args.gpu)

    # Initialize the model
    model = Llama(model_path=config.model_path, 
                  n_gpu_layers=config.n_gpu_layers,
                  flash_attn=True,
                  n_ctx=16384 * 2,  # Context window size
                  chat_format=config.chat_format,
                  verbose=True)

    # Load the JSON data
    with open(args.input_json, 'r', encoding='utf-8') as f:
        all_cases = json.load(f)

    print("Number of cases available:", len(all_cases))
    
    end_idx = args.end_idx if args.end_idx is not None else len(all_cases)
    cases_to_process = all_cases[args.start_idx:end_idx]

    # Prepare output path
    out_path = args.output_json.replace(".json", f"_gpu{args.gpu}.json")
    all_results = []
    
    # Load existing results if the output file already exists to allow resuming
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

        # Format each case in the batch into a JSON string
        json_texts_to_process = [format_case_to_json_string(case_item['case']) for case_item in batch_slice]
        
        # Process the batch
        batch_results = process_batch(model, json_texts_to_process, base_prompt, config, args.num_variations)

        # Structure and store results
        for original_case_item, translated_variations in zip(batch_slice, batch_results):
            result_entry = {
                "original_case": original_case_item['case'],
                "translated_reports": translated_variations
            }
            all_results.append(result_entry)

        # Save results to the output file after each batch
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(all_results)} total results to {out_path}")

    print(f"\nFinished GPU {args.gpu}: processed cases {args.start_idx}–{end_idx} in {(time.time() - start_time_total):.2f}s")

if __name__ == "__main__":
    main()
