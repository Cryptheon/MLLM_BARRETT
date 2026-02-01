import argparse
import yaml
import logging
import torch
import json
import time
import random
import numpy as np
import os
import sys

# Ensure src is in path if running from root
sys.path.append(os.path.join(os.getcwd(), 'src'))

from transformers import AutoTokenizer
from src.model.patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig
from src.data.datasets import MultiModalBarrett
from src.utils.util_functions import print_model_size
from safetensors.torch import load_file
from tqdm import tqdm

# ... rest of the file logic remains the same ...
# (Only imports needed changing)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_model_tokenizer(config: dict, model_path: str = None):
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
    logger.info(f"Trained Main Model loaded onto {model.device} and set to evaluation mode.")

    print_model_size(model)
    return model, tokenizer

def load_validation_data(config: dict, tokenizer: AutoTokenizer) -> MultiModalBarrett:
    logger.info("Loading validation dataset...")
    data_config = config["dataset"]
    
    val_dataset = MultiModalBarrett(
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
    wsi_embeddings = wsi_embeddings.unsqueeze(0).to(model.device)
    input_ids = tokenizer("", return_tensors="pt").input_ids.to(model.device)

    if "custom_tokenizer" in config["tokenizer"] and config["tokenizer"]["custom_tokenizer"]:
        input_ids[0] = 0

    with torch.no_grad():
        generated_ids = model.generate(
            inputs=input_ids,
            wsi_embeddings=wsi_embeddings,
            max_new_tokens=config["inference"]["max_new_tokens"],
            do_sample=config["inference"]["do_sample"],
            temperature=config["inference"]["temperature"],
            top_p=config["inference"]["top_p"],
            top_k=config["inference"]["top_k"],
            repetition_penalty=config["inference"]["repetition_penalty"],
            exponential_decay_length_penalty=config["inference"]["exponential_decay_length_penalty"],
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

    set_seed(42)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = load_config(args.config)
    model, tokenizer = load_model_tokenizer(config, args.model_path)
    dataset = load_validation_data(config, tokenizer)

    case_id_to_data = {
        sample["case_data"].get("original_case", {}).get("An_Number"): sample["case_data"]
        for sample in dataset.samples
    }

    results = []
    total_tokens = 0
    start_time = time.time()
    
    nl_to_eng_sections = {"Microscopie": "microscopy", 
                          "Conclusie": "conclusion", 
                          "barrett_label": "barrett_label", 
                          "KlinischeVraagstelling": "clinicalQuery",
                          "KlinischeGegevens": "clinicalData",
                          "An_Number": "an_number",
                          "Diagnose": "diagnosis",
                          "report": "report"}

    logger.info(f"Starting generation for {len(dataset)} validation samples...")
    
    for item in tqdm(dataset, desc="Generating Reports"):
        case_id = item["case_id"]
        wsi_embeddings = item["wsi_embeddings"]
        generated_text, num_tokens = generate_report(model, tokenizer, wsi_embeddings, config)
        total_tokens += num_tokens

        original_case_data = case_id_to_data.get(case_id, {})
        cleaned_reports = original_case_data.get("tcga_structured", [{}])
        text_parts = [f"{nl_to_eng_sections[k]}: " + cleaned_reports.get(k, "") for k in ["report"]]
        original_text = "\n\n".join(filter(None, text_parts))

        results.append({
            "case_id": case_id,
            "original_report": original_text,
            "generated_report": generated_text
        })

    duration = time.time() - start_time
    tokens_per_second = total_tokens / duration if duration > 0 else 0

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info("--------------------")
    logger.info("Generation Complete!")
    logger.info(f"Saved {len(results)} generated reports to {args.output}")
    logger.info(f"Performance: {tokens_per_second:.2f} tokens/second")
    logger.info("--------------------")

if __name__ == "__main__":
    main()
