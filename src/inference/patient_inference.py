import argparse
import yaml
import logging
import random
import torch
import sys
import os

# Ensure src is reachable if run from anywhere
sys.path.append(os.path.join(os.getcwd(), 'src'))

from transformers import AutoTokenizer
from src.model.patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig 
from src.data.datasets import MultiModalBarrett
from src.utils.util_functions import print_model_size
from safetensors.torch import load_file
from rich.console import Console
from rich.panel import Panel
from rich import box

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_model_tokenizer(config):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded and pad token set.")

    # Load model
    model_config = PathoLlamaConfig(**config["model"])
    model = PathoLlamaForCausalLM(model_config)
    model.training = False
    
    # Check if model path exists
    model_path = config["inference"]["model_path"]
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Model loaded from %s", model_path)

    trainable_params, total_params, total_mb = print_model_size(model)
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Approx. model size: %.2f MB", total_mb)
    return model, tokenizer

def load_data(config, tokenizer):
    # Load dataset using MultiModalBarrett
    # Note: random_choice_report argument removed as it's not in the __init__ of MultiModalBarrett
    dataset =  MultiModalBarrett(json_file=config["dataset"]["train_texts_json_path"], # Using val path for inference
                                      embeddings_file=config["dataset"]["train_h5_file_path"], # Using val path for inference
                                      tokenizer=tokenizer,
                                      phase="val",
                                      val_data_ratio=config["dataset"]["val_data_ratio"],
                                      max_seq_length=config["dataset"]["max_seq_length"])
    
    logger.info("Validation dataset loaded from JSON and HDF5 files.")

    return dataset

def get_data(config, tokenizer, dataset, args):
    """
    Selects a sample from the MultiModalBarrett dataset, retrieves the processed
    WSI embeddings and the corresponding original text.
    """
    idx = -1
    # If a specific patient is requested, search for its index in the dataset
    if args.patient:
        try:
            # The patient ID is stored as 'An_Number' in the case data
            idx = next(i for i, sample in enumerate(dataset.samples) 
                       if sample['case_data'].get('original_case', {}).get('An_Number') == args.patient)
            logger.info(f"Found patient ID '{args.patient}' at index {idx}.")
        except StopIteration:
            logger.warning(f"Patient ID '{args.patient}' not found. A random sample will be used instead.")
            # idx remains -1, triggering random selection

    # If no patient was specified or found, pick a random sample
    if idx == -1:
        idx = random.randint(0, len(dataset) - 1)
        logger.info(f"Randomly selected sample at index: {idx}")

    # Use the dataset's __getitem__ to get the processed tensors
    processed_sample = dataset[idx]
    wsi_embeddings = processed_sample["wsi_embeddings"]
    patient_id = processed_sample["case_id"]

    # For display purposes, we need to reconstruct the original text.
    # We access the raw sample from the .samples list to get the text data.
    raw_sample = dataset.samples[idx]
    case_data = raw_sample["case_data"]
    # print(case_data) # Debug print removed
    
    # Handle report structure (TCGA structured or legacy cleaned_reports)
    if "tcga_structured" in case_data:
        report_data = case_data["tcga_structured"]
        # Assuming we just want the 'report' field for TCGA structured
        report_text = report_data.get("report", "")
        original_text = f"report: {report_text}"
    else:
        # Fallback to old structure if needed
        translated_reports = case_data.get("cleaned_reports", [])
        if isinstance(translated_reports, list) and translated_reports:
             if config["dataset"].get("random_choice_report", False) and len(translated_reports) > 1:
                 report = random.choice(translated_reports)
             else:
                 report = translated_reports[0]
             text_parts = [report.get(k, "") for k in ["Microscopie", "Conclusie", "Diagnose"]]
             original_text = "\n".join(filter(None, text_parts))
        else:
            original_text = "No text report available."

    # Add the batch dimension required for model inference
    wsi_embeddings = wsi_embeddings.unsqueeze(0)
    # print("embeddings shape", wsi_embeddings.shape) # Debug print removed

    return wsi_embeddings, original_text, patient_id


def generate(model, tokenizer, wsi_embeddings, config):
    # We start generation from an empty prompt, as the context comes from WSI embeddings
    input_tokens = tokenizer("", return_tensors="pt")
    input_ids = input_tokens["input_ids"].to(model.device)

    # Patch the first BOS token to 0th index if using a custom tokenizer
    if config["tokenizer"]["custom_tokenizer"]:
        if input_ids.shape[1] > 0:
            input_ids[0, 0] = 0 # BOS token
        else: # Handle empty input case
            input_ids = torch.tensor([[0]], device=model.device)

    wsi_embeddings = wsi_embeddings.to(model.device)

    # Generate prediction
    logger.info("Running inference...")
    with torch.no_grad():
        generated = model.generate(
            inputs=input_ids,
            wsi_embeddings=wsi_embeddings,
            max_new_tokens=config["inference"]["max_new_tokens"],
            do_sample=config["inference"]["do_sample"],
            temperature=config["inference"]["temperature"],
            stop_strings=["<|end_of_text|>", "END OF REPORT"],
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer
        )
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Run inference on PathoLlama model with multimodal data")
    # Updated default config path
    parser.add_argument('--config', type=str, default="experiments/configs/model_inference/barrett/config.yaml", help='Path to config YAML file')
    parser.add_argument('--patient', type=str, default=None, help='Patient ID to use (optional, will sample if not provided)')
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Configuration loaded from %s", args.config)
    
    console.print(Panel(yaml.dump(config, sort_keys=False), title="[bold yellow]Configuration", expand=False))

    model, tokenizer = load_model_tokenizer(config)
    dataset = load_data(config, tokenizer)
    wsi_embeddings, original_text, patient_id = get_data(config, tokenizer, dataset, args)

    # Display input
    console.print(Panel(original_text, title=f"[bold green]Original Report: {patient_id}", box=box.DOUBLE))

    generated_text = generate(model, tokenizer, wsi_embeddings, config)

    # Display output
    console.print(Panel(generated_text, title="[bold blue]Generated Text", box=box.DOUBLE))

if __name__ == "__main__":
    main()
