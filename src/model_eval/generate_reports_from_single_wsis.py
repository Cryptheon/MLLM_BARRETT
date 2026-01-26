import argparse
import yaml
import logging
import torch
import json
import time
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from safetensors.torch import load_file

from model.patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig
from data.datasets import MultiModalBarrettSingleSlide
from utils.util_functions import print_model_size

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_model_tokenizer(config: dict, model_path: str = None):
    """Loads the model and tokenizer based on the provided configuration."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded.")

    logger.info("Loading model...")
    model_config = PathoLlamaConfig(**config["model"])
    model = PathoLlamaForCausalLM(model_config)

    if model_path is None:
        model_path = config["inference"]["model_path"]
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Inference model file not found at: {model_path}")
        
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model loaded onto {model.device} and set to evaluation mode.")

    print_model_size(model)
    return model, tokenizer

def load_validation_data(config: dict, tokenizer: AutoTokenizer) -> MultiModalBarrettSingleSlide:
    """Loads the validation dataset using the MultiModalBarrettSingleSlide class."""
    logger.info("Loading validation dataset...")
    data_config = config["dataset"]
    
    val_dataset = MultiModalBarrettSingleSlide(
        json_file=data_config["train_texts_json_path"],
        embeddings_file=data_config["train_h5_file_path"], 
        tokenizer=tokenizer,
        phase="val",
        val_data_ratio=data_config["val_data_ratio"],
        max_seq_length=data_config["max_seq_length"]
    )
    logger.info(f"Validation dataset loaded with {len(val_dataset)} samples.")
    return val_dataset

def generate_report(model: PathoLlamaForCausalLM, tokenizer: AutoTokenizer, wsi_embeddings: torch.Tensor, config: dict) -> tuple[str, int]:
    """Generates a text report from WSI embeddings."""
    wsi_embeddings = wsi_embeddings.unsqueeze(0).to(model.device)
    input_ids = tokenizer("", return_tensors="pt").input_ids.to(model.device)

    # Note: Patching for custom tokenizer is handled in the dataset now, but kept for safety.
    if "custom_tokenizer" in config["tokenizer"] and config["tokenizer"]["custom_tokenizer"]:
        input_ids[0][0] = 0

    with torch.no_grad():
        generated_ids = model.generate(
            inputs=input_ids,
            wsi_embeddings=wsi_embeddings,
            max_new_tokens=config["inference"]["max_new_tokens"],
            do_sample=config["inference"]["do_sample"],
            temperature=config["inference"]["temperature"],
            top_p=config["inference"]["top_p"],
            top_k=config["inference"]["top_k"],
            stop_strings=["<|end_of_text|>", "END OF REPORT"],
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer
        )

    decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    num_tokens = generated_ids.shape[-1]
    
    return decoded_text, num_tokens

def main():
    parser = argparse.ArgumentParser(description="Run batch inference to generate reports from WSI embeddings.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--output', type=str, default="generated_reports.json", help='Path to the output JSON file for predictions.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file safetensors.')
    args = parser.parse_args()

    config = load_config(args.config)
    model, tokenizer = load_model_tokenizer(config, args.model_path)
    dataset = load_validation_data(config, tokenizer)

    # --- MODIFICATION START ---
    # The lookup map is no longer needed. We will get the ground truth text directly from the dataset item.
    
    results = []
    total_tokens = 0
    start_time = time.time()

    logger.info(f"Starting generation for {len(dataset)} validation samples...")
    for item in tqdm(dataset, desc="Generating Reports"):
        case_id = item["case_id"]
        wsi_embeddings = item["wsi_embeddings"]
        ground_truth_labels = item["labels"]

        # 1. Generate the report from the WSI embeddings
        generated_text, num_tokens = generate_report(model, tokenizer, wsi_embeddings, config)
        total_tokens += num_tokens

        # 2. Decode the ground truth 'labels' from the dataset to get the original sub-report text
        original_text = tokenizer.decode(ground_truth_labels, skip_special_tokens=True).strip()

        results.append({
            "case_id": case_id,
            "original_report": original_text,
            "generated_report": generated_text.strip()
        })
    # --- MODIFICATION END ---

    duration = time.time() - start_time
    tokens_per_second = total_tokens / duration if duration > 0 else 0

    output_path = args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info("--------------------")
    logger.info("Generation Complete! ✅")
    logger.info(f"Saved {len(results)} generated reports to {output_path}")
    logger.info(f"Total tokens generated: {total_tokens}")
    logger.info(f"Total time taken: {duration:.2f} seconds")
    logger.info(f"Performance: {tokens_per_second:.2f} tokens/second")
    logger.info("--------------------")

if __name__ == "__main__":
    main()
