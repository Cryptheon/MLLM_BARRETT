import torch
import argparse
import pandas as pd
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import time
from utils.util_functions import load_config

def get_prompt(prompt_path: str) -> str:
    """Reads the prompt template from a file."""
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt

def process_batch(model: FastLanguageModel, 
                  tokenizer: AutoTokenizer, 
                  texts: list, 
                  base_prompt: str, 
                  config, 
                  num_variations: int) -> list:
    """Processes a batch of texts using the given language model, generating multiple variations per input."""
    
    # format for Gemma-3 27b it
    messages = [
        [
            {"role": "system", "content": [{"type": "text", "text":"You are an advanced medical language model trained to process pathology-related text with accuracy and clarity."}]},
            {"role": "user", "content": [{"type": "text", "text":base_prompt.format(text)}]}
        ]
        for text in texts
    ]
    
    prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_ids = inputs["input_ids"]
    input_num_tokens = input_ids.shape[1]
    
    start_time = time.time()
    
    generated_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"],
        use_cache=True,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        min_p=config.min_p,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        do_sample=True,
        num_return_sequences=num_variations
    )
    
    end_time = time.time()
    
    total_tokens = generated_outputs[:, input_num_tokens:].numel()
    elapsed_time = end_time - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    print(f"\nThroughput Results:")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}\n")
    
    processed_texts = [
        [tokenizer.decode(output[input_num_tokens:], skip_special_tokens=True) 
         for output in generated_outputs[i * num_variations : (i + 1) * num_variations]]
        for i in range(len(texts))
    ]
    return processed_texts

def main():
    parser = argparse.ArgumentParser(description="Process text data using an LLM.")
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
    
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length
    )
    
    FastLanguageModel.for_inference(model)
    
    df = pd.read_csv(args.input_csv)
    if args.column not in df.columns:
        raise ValueError(f"The input CSV file must contain a '{args.column}' column with text data.")
    
    results = []
    for i in range(0, len(df), args.batch_size):
        batch_texts = df[args.column].iloc[i:i + args.batch_size].tolist()
        batch_results = process_batch(model, tokenizer, batch_texts, base_prompt, config, args.num_variations)
        results.extend(batch_results)
    
    df["processed_outputs"] = results
    df.to_csv(args.output_csv, index=False)
    print(f"Processed data saved to {args.output_csv}")

if __name__ == "__main__":
    main()

