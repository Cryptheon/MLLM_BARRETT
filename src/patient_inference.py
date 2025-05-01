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

def load_model_tokenizer(config):
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
    return model, tokenizer

def load_data(config, tokenizer):
    # Load dataset
    dataset = PathoMultiModalDataset(
        pickle_file=config["dataset"]["val_pickle_file_path"],
        max_seq_length=config["dataset"]["max_seq_length"],
        embeddings_dim_size=config["dataset"]["embeddings_dim_size"],
        tokenizer=tokenizer,
        random_choice_report=config["dataset"]["random_choice_report"]

    )
    logger.info("Dataset loaded from %s", config["dataset"]["val_pickle_file_path"])

    return dataset

def get_data(config, tokenizer, dataset, args):
    
    if args.patient and args.patient in dataset.patient_ids:
        patient_id = args.patient
    else:
        patient_id = random.choice(dataset.patient_ids)
        logger.info("Randomly sampled patient ID: %s", patient_id)

    patient_data = dataset.data[patient_id]
    text_variations = patient_data["reports"]
    text_variations = eval(text_variations)
    if config["dataset"]["random_choice_report"]:
        original_text: str = random.choice(text_variations)
    else:
        original_text = text_variations[0]
    
    wsi_embeddings = torch.tensor(patient_data["embeddings"]).unsqueeze(0)  # Add batch dimension

    return wsi_embeddings, original_text, patient_id

def generate(model, tokenizer, wsi_embeddings, input_ids, config):

    # Tokenize prompt
    input_tokens = tokenizer("", 
                             return_tensors="pt", 
                             truncation=True, 
                             max_length=config["dataset"]["max_seq_length"])
    
    input_ids = input_tokens["input_ids"].to(model.device)
    wsi_embeddings = wsi_embeddings.to(model.device)

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
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Run inference on PathoLlama model with multimodal data")
    parser.add_argument('--config', type=str, default="./configs/tcga/config.yaml", help='Path to config YAML file')
    parser.add_argument('--patient', type=str, default=None, help='Patient ID to use (optional, will sample if not provided)')
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Full config:\n%s", yaml.dump(config, sort_keys=False))

    model, tokenizer = load_model_tokenizer(config)

    dataset = load_data(config, tokenizer)

    wsi_embeddings, original_text, patient_id = get_data(config, tokenizer, dataset, args)

    # Display input
    console.print(Panel(original_text, title=f"[bold green]Original Report: {patient_id}", box=box.DOUBLE))

    generated_text = generate(model, tokenizer, wsi_embeddings, original_text, config)

    # Display output
    console.print(Panel(generated_text, title="[bold blue]Generated Text", box=box.DOUBLE))


if __name__ == "__main__":
    main()

