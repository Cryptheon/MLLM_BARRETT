import json
import argparse
import re
import sys
from pathlib import Path

# Define project root relative to this file: src/utils/vllm_utils.py -> root is 2 levels up
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def get_nested_value(data, path):
    """Retrieves value from nested dict (e.g., 'case.report')."""
    keys = path.split('.')
    val = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        elif isinstance(val, list) and k.isdigit():
            try:
                val = val[int(k)]
            except IndexError:
                return None
        else:
            return None
    return val

def set_nested_value(data, path, value):
    """Sets value in nested dict, creating intermediate dicts if needed."""
    keys = path.split('.')
    ref = data
    for k in keys[:-1]:
        if k not in ref:
            ref[k] = {}
        ref = ref[k]
    ref[keys[-1]] = value

def prepare_prompts(args):
    """Converts Main JSON -> vLLM JSONL inputs."""
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Robust path handling for prompts
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
         # Fallback: check inside experiments/prompts/ relative to root
         fallback = ROOT_DIR / "experiments/prompts" / args.prompt_file
         if fallback.exists():
             prompt_path = fallback
         else:
             # Just in case the user provided a full relative path that didn't match cwd
             # try relative to root directly
             fallback_root = ROOT_DIR / args.prompt_file
             if fallback_root.exists():
                 prompt_path = fallback_root
             else:
                 raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(data):
            # Extract inputs (can be multiple keys joined)
            extracted_texts = []
            for k in args.input_keys:
                val = get_nested_value(item, k)
                if val:
                    extracted_texts.append(str(val))
            
            # If input keys are missing/empty, we skip adding a prompt for this index
            if not extracted_texts:
                continue

            input_text = "\n\n".join(extracted_texts)
            
            # Template replacement
            if "__INPUT_TEXT__" in base_prompt:
                full_prompt = base_prompt.replace("__INPUT_TEXT__", input_text)
            else:
                full_prompt = base_prompt + "\n\n" + input_text

            # Create vLLM entry. 'custom_id' tracks the original index.
            entry = {
                "custom_id": idx,
                "prompt": full_prompt
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Prepared prompts in {args.output_jsonl}")

def merge_results(args):
    """Merges vLLM JSONL outputs -> Main JSON."""
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results_map = {}
    if Path(args.vllm_output_jsonl).exists():
        with open(args.vllm_output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    val = res.get('response') or res.get('report') or ""
                    results_map[res['custom_id']] = val
                except:
                    continue
    else:
        print(f"Warning: Output file {args.vllm_output_jsonl} not found. No merging done.")
        return

    count = 0
    for idx, item in enumerate(data):
        if idx in results_map:
            raw_output = results_map[idx]
            
            # Qwen <think> tag parsing
            if args.parse_think and raw_output:
                clean_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
                final_val = clean_output
            else:
                final_val = raw_output

            # Try to parse as JSON object
            if final_val and (final_val.strip().startswith('{') or final_val.strip().startswith('[')):
                try:
                    final_val = json.loads(final_val)
                except json.JSONDecodeError:
                    pass 

            set_nested_value(item, args.output_key, final_val)
            count += 1

    with open(args
