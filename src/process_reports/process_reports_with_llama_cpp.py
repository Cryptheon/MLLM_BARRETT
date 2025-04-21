import torch
import json
import math
import argparse
import pandas as pd
import time
import os
from llama_cpp import Llama
from utils.util_functions import load_config

def get_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as file:
        return file.read()

def process_batch(model: Llama, texts: list, base_prompt: str, config, num_variations: int) -> list:
    processed_texts = []
    start_time = time.time()
    total_tokens = 0

    for text in texts:
        single_input_variations = []
        messages = [{"role": "system", "content": "You are an advanced medical language model trained to process pathology-related text with accuracy and clarity."},
                    {"role": "user", "content": base_prompt.format(text)}]
        
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
            single_input_variations.append(result_text)
            total_tokens += output['usage']['completion_tokens']

        processed_texts.append(single_input_variations)

    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
    print(f"\nThroughput: {tokens_per_second:.2f} tokens/sec | Total: {total_tokens} tokens in {elapsed_time:.2f}s\n")
    return processed_texts

def main():
    parser = argparse.ArgumentParser(description="Process pathology text data using Llama-cpp.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--column", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
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

    df = pd.read_csv(args.input_csv)
    if args.column not in df.columns:
        raise ValueError(f"Input CSV must contain a '{args.column}' column.")

    end_idx = args.end_idx if args.end_idx is not None else len(df)
    df_slice = df.iloc[args.start_idx:end_idx].copy()

    start_time = time.time()
    
    out_path = args.output_csv.replace(".csv", f"_gpu{args.gpu}.csv")
    write_header = not os.path.exists(out_path)

    for i in range(0, len(df_slice), args.batch_size):
        print(f"Processing row {args.start_idx + i}/{end_idx}")

        row_text = df_slice.iloc[i][args.column]
        batch_results = process_batch(model, [row_text], base_prompt, config, args.num_variations)

        variations = batch_results[0]
        row_df = df_slice.iloc[[i]][["patient_filename", args.column]].copy()
        row_df["processed_reports"] = [json.dumps(variations)]

        # Ensure correct column order
        row_df = row_df[["patient_filename", args.column, "processed_reports"]]

        row_df.to_csv(out_path, index=False, mode='a', header=write_header)
        write_header = False  # Only write header for the first batch

        print(f"Saved batch {i} to {out_path}")

    print(f"\nFinished GPU {args.gpu}: processed rows {args.start_idx}–{end_idx} in {(time.time()-start_time):.2f}s")

if __name__ == "__main__":
    main()
