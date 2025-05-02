import argparse
import yaml
import logging
import torch
import json
import time
import random
from transformers import AutoTokenizer
from model.patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig
from data.datasets import PathoMultiModalDataset
from utils.util_functions import print_model_size
from safetensors.torch import load_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_model_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token

    model_config = PathoLlamaConfig(**config["model"])
    model = PathoLlamaForCausalLM(model_config)
    model.training = False
    state_dict = load_file(config["inference"]["model_path"])
    model.load_state_dict(state_dict)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print_model_size(model)
    return model, tokenizer

def load_data(config, tokenizer):
    dataset = PathoMultiModalDataset(
        pickle_file=config["dataset"]["val_pickle_file_path"],
        max_seq_length=config["dataset"]["max_seq_length"],
        embeddings_dim_size=config["dataset"]["embeddings_dim_size"],
        tokenizer=tokenizer,
        random_choice_report=config["dataset"]["random_choice_report"]
    )
    return dataset

def generate(model, tokenizer, wsi_embeddings, config):
    input_tokens = tokenizer("", return_tensors="pt", truncation=True,
                             max_length=config["dataset"]["max_seq_length"])
    input_ids = input_tokens["input_ids"].to(model.device)
    # patch the first BOS token to 0th index if we're using our own trained tokenizer based on Llama's
    if config["tokenizer"]["custom_tokenizer"]:
        input_ids[0] = 0
    wsi_embeddings = wsi_embeddings.to(model.device)

    with torch.no_grad():
        generated = model.generate(
            inputs=input_ids,
            wsi_embeddings=wsi_embeddings,
            max_new_tokens=config["inference"]["max_new_tokens"],
            do_sample=config["inference"]["do_sample"],
            temperature=config["inference"]["temperature"],
            stop_strings="<|end_of_text|>",
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer
        )

    return tokenizer.decode(generated[0], skip_special_tokens=False), generated.shape[-1]

def main():
    parser = argparse.ArgumentParser(description="Run batch inference on PathoLlama model")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--output', type=str, default="predictions.json", help='Output JSON file')
    args = parser.parse_args()

    config = load_config(args.config)
    model, tokenizer = load_model_tokenizer(config)
    dataset = load_data(config, tokenizer)

    results = []
    total_tokens = 0
    start_time = time.time()

    for i, patient_id in enumerate(dataset.patient_ids):
        if (i+1)%10==0:
            logger.info("Processing patient number %d", i)
        
        patient_data = dataset.data[patient_id]
        text_variations = eval(patient_data["reports"])
        original_text = (random.choice(text_variations)
                         if config["dataset"]["random_choice_report"]
                         else text_variations[0])

        wsi_embeddings = torch.tensor(patient_data["embeddings"]).unsqueeze(0)
        generated_text, num_tokens = generate(model, tokenizer, wsi_embeddings, config)

        total_tokens += num_tokens

        results.append({
            "patient_id": patient_id,
            "original_report": original_text,
            "generated_report": generated_text
        })

    duration = time.time() - start_time
    tokens_per_second = total_tokens / duration

    with open(args.output, 'w') as outfile:
        json.dump(results, outfile, indent=2)

    logger.info("Saved %d predictions to %s", len(results), args.output)
    logger.info("Saved %d predictions to %s", len(results), args.output)
    logger.info("Total tokens generated: %d", total_tokens)
    logger.info("Time taken: %.2f seconds", duration)
    logger.info("Tokens per second: %.2f", tokens_per_second)


if __name__ == "__main__":
    main()

