import json
import argparse
import time
import os
from llama_cpp import Llama
from utils.util_functions import load_config

def get_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as file:
        return file.read()

def process_batch(model: Llama, examples: list, base_prompt: str, config, num_variations: int) -> list:
    results = []
    start_time = time.time()
    total_tokens = 0

    for item in examples:
        single_output_variations = []
        original_report = item["original_report"]
        generated_report = item["generated_report"]
        prompt_text = (
            "Original report:\n"
            f"{original_report}\n\n"
            "Generated report:\n"
            f"{generated_report}\n\n"
            "## JSON output:"
        )

        messages = [
            {"role": "system", "content": "You are an advanced medical language model trained to evaluate pathology reports."},
            {"role": "user", "content": base_prompt+prompt_text}
        ]

        for _ in range(num_variations):
            output = model.create_chat_completion(
                messages=messages,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repeat_penalty=config.repetition_penalty,
                stream=False
            )
            result_text = output['choices'][0]['message']['content']
            single_output_variations.append(result_text)
            total_tokens += output['usage']['completion_tokens']

        results.append({
            "case_id": item.get("patient_id", f"case_{int(time.time()*1000)}"),
            "raw_outputs": single_output_variations
        })

    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
    print(f"\nThroughput: {tokens_per_second:.2f} tokens/sec | Total: {total_tokens} tokens in {elapsed_time:.2f}s\n")
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated pathology reports using Llama-cpp.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--num_variations", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    config = load_config(args.config)
    base_prompt = get_prompt(args.prompt_path)

    print(f"Running on GPU {args.gpu}")
    os.environ["LLAMA_VISIBLE_DEVICES"] = str(args.gpu)

    model = Llama(model_path=config.model_path,
                  n_gpu_layers=config.n_gpu_layers,
                  flash_attn=True,
                  n_ctx=8192*2,
                  chat_format=config.chat_format)

    with open(args.input_json, "r") as infile:
        data = json.load(infile)

    end_idx = args.end_idx if args.end_idx is not None else len(data)
    data_slice = data[args.start_idx:end_idx]

    all_outputs = process_batch(model, data_slice, base_prompt, config, args.num_variations)

    with open(args.output_json, "w") as outfile:
        json.dump(all_outputs, outfile, indent=2)

    print(f"\nFinished GPU {args.gpu}: processed {len(data_slice)} items and saved to {args.output_json}")

if __name__ == "__main__":
    main()

