import torch
import argparse
import pandas as pd
import time
from llama_cpp import Llama
from utils.util_functions import load_config

def get_prompt(prompt_path: str) -> str:
    """Reads the prompt template from a file."""
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt

def process_batch(model: Llama, 
                  texts: list, 
                  base_prompt: str, 
                  config, 
                  num_variations: int) -> list:
    """Processes a batch of texts using the Llama language model, generating multiple variations per input."""
    processed_texts = []
    start_time = time.time()
    total_tokens = 0

    for text in texts:
        single_input_variations = []
        user_content = base_prompt.format(text)

        messages = [
            {"role": "system", "content": "You are an advanced medical language model trained to process pathology-related text with accuracy and clarity."},
            {"role": "user", "content": user_content}
        ]

        # format for Gemma-3 27b it
        # messages = [
        #     [
        #         {"role": "system", "content": [{"type": "text", "text":"You are an advanced medical language model trained to process pathology-related text with accuracy and clarity."}]},
        #         {"role": "user", "content": [{"type": "text", "text":user_content}]}
        #     ]
        #     for text in texts
        # ]
        
        for _ in range(num_variations):
            output = model.create_chat_completion(
                messages=messages,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repeat_penalty=config.repetition_penalty,
                #stop=["<|eot_id|>"],  # Optional, depending on model's behavior
                stream=False
            )
            result_text = output['choices'][0]['message']['content']
            single_input_variations.append(result_text)
            total_tokens += output['usage']['completion_tokens']

        processed_texts.append(single_input_variations)

    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

    print(f"\nThroughput Results:")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}\n")

    return processed_texts

def main():
    parser = argparse.ArgumentParser(description="Process text data using Llama-cpp.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt template file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file containing text data.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the processed CSV file.")
    parser.add_argument("--column", type=str, required=True, help="Column name containing the text to process.")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of texts to process in a single batch.")
    parser.add_argument("--num_variations", type=int, required=True, help="Number of different versions to generate per input.")
    
    args = parser.parse_args()
    config = load_config(args.config)
    base_prompt = get_prompt(args.prompt_path)

    print("Loading Llama model...")
    model = Llama(model_path=config.model_path, chat_format="gemma")
    
    df = pd.read_csv(args.input_csv)
    if args.column not in df.columns:
        raise ValueError(f"The input CSV file must contain a '{args.column}' column with text data.")
    
    results = []
    for i in range(0, len(df), args.batch_size):
        batch_texts = df[args.column].iloc[i:i + args.batch_size].tolist()
        batch_results = process_batch(model, batch_texts, base_prompt, config, args.num_variations)
        results.extend(batch_results)
    
    df["processed_outputs"] = results
    df.to_csv(args.output_csv, index=False)
    print(f"Processed data saved to {args.output_csv}")

if __name__ == "__main__":
    main()
