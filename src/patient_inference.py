import argparse
import yaml
import logging
import random
import torch
from transformers import AutoTokenizer
from model.patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig
from data.datasets import PathoMultiModalDataset
from utils.util_functions import print_model_size
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


def main():
    parser = argparse.ArgumentParser(description="Run inference on PathoLlama model with multimodal data")
    parser.add_argument('--config', type=str, default="./configs/tcga/config.yaml", help='Path to config YAML file')
    parser.add_argument('--patient', type=str, default=None, help='Patient ID to use (optional, will sample if not provided)')
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Full config:\n%s", yaml.dump(config, sort_keys=False))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded and pad token set.")

    # Load model
    model_config = PathoLlamaConfig(**config["model"])
    model = PathoLlamaForCausalLM(model_config)
    model.training = False
    state_dict = load_file(config["inference"]["model_path"])
    model.load_state_dict(state_dict)#, map_location="cuda"), weights_only=False)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Model loaded from %s", config["inference"]["model_path"])

    trainable_params, total_params, total_mb = print_model_size(model)
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Approx. model size: %.2f MB", total_mb)

    # Load dataset
    dataset = PathoMultiModalDataset(
        pickle_file=config["dataset"]["pickle_file_path"],
        max_seq_length=config["dataset"]["max_seq_length"],
        embeddings_dim_size=config["dataset"]["embeddings_dim_size"],
        tokenizer=tokenizer
    )
    logger.info("Dataset loaded from %s", config["dataset"]["pickle_file_path"])

    if args.patient and args.patient in dataset.patient_ids:
        patient_id = args.patient
    else:
        patient_id = random.choice(dataset.patient_ids)
        logger.info("Randomly sampled patient ID: %s", patient_id)

    patient_data = dataset.data[patient_id]
    original_text = patient_data["report_text"]
    wsi_embeddings = torch.tensor(patient_data["embeddings"]).unsqueeze(0)  # Add batch dimension

    # Tokenize prompt
    input_tokens = tokenizer("", 
                             return_tensors="pt", 
                             truncation=True, 
                             max_length=config["dataset"]["max_seq_length"])
    
    input_ids = input_tokens["input_ids"].to(model.device)
    wsi_embeddings = wsi_embeddings.to(model.device)

    # Display input
    console.print(Panel(original_text, title=f"[bold green]Original Report: {patient_id}", box=box.DOUBLE))

    # Generate prediction
    logger.info("Running inference...")
    with torch.no_grad():
        generated = model.generate(
            inputs=input_ids,
            wsi_embeddings=wsi_embeddings,
            max_new_tokens=config["inference"]["max_new_tokens"],
            do_sample=config["inference"]["do_sample"],
            #top_p=config["inference"]["top_p"],
            temperature=config["inference"]["temperature"],
            stop_strings="<|end_of_text|>",
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer
        )
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)

    # Display output
    console.print(Panel(generated_text, title="[bold blue]Generated Text", box=box.DOUBLE))


if __name__ == "__main__":
    main()

