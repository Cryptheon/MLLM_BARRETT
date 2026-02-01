import json
import argparse
import os

def load_prompt_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="Prepare JSONL for vLLM by injecting data into a prompt.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input data JSONL.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to .txt file with __INPUT_TEXT__ placeholder.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save ready-for-inference JSONL.")
    parser.add_argument("--input_keys", nargs='+', required=True, help="List of keys inside 'case' to join and insert.")

    args = parser.parse_args()

    base_prompt = load_prompt_template(args.prompt_file)
    
    with open(args.input_jsonl, 'r', encoding='utf-8') as fin, \
         open(args.output_jsonl, 'w', encoding='utf-8') as fout:
        
        count = 0
        for line in fin:
            if not line.strip(): continue
            
            data = json.loads(line)
            
            # Handle your specific nesting structure ({"case": {...}})
            case_data = data.get("case", data)

            # Combine requested fields (e.g., Microscopy + Conclusion)
            combined_text_parts = []
            for key in args.input_keys:
                val = case_data.get(key, "")
                if val:
                    combined_text_parts.append(f"{key}: {val}")
            
            full_text = "\n\n".join(combined_text_parts)

            # Replace placeholder
            # Assuming your prompt text file uses "__INPUT_TEXT__" or similar
            if "__INPUT_TEXT__" in base_prompt:
                final_prompt = base_prompt.replace("__INPUT_TEXT__", full_text)
            else:
                # Fallback: just append
                final_prompt = base_prompt + "\n\n" + full_text

            # Construct entry for vLLM decoder
            # We preserve the original data for reference
            out_entry = {
                "prompt": final_prompt,
                "original_data": case_data
            }
            
            fout.write(json.dumps(out_entry, ensure_ascii=False) + '\n')
            count += 1

    print(f"Prepared {count} prompts. Saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()
