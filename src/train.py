import argparse
import yaml
import logging
from utils.util_functions import print_model_size
from transformers import AutoTokenizer, TrainingArguments
from model.patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig
from data.datasets import PathoMultiModalDataset
from multimodal_trainer import MultiModalTrainer
from data.collator import MultiModalCollator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Train PathoLlama model with multimodal data")
    parser.add_argument('--config', type=str, default="./configs/tcga/config.yaml", help='Path to the config YAML file')
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Full config:\n%s", yaml.dump(config, sort_keys=False))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["core_model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded and pad token set.")

    model_config = PathoLlamaConfig(**config["model"])
    model = PathoLlamaForCausalLM(model_config)
    trainable_params, total_params, total_mb = print_model_size(model)
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Approx. model size: %.2f MB", total_mb)

    train_dataset = PathoMultiModalDataset(
        pickle_file=config["dataset"]["pickle_file_path"],
        max_seq_length=config["dataset"]["max_seq_length"],
        embeddings_dim_size=config["dataset"]["embeddings_dim_size"],
        tokenizer=tokenizer
    )
    logger.info("Training dataset loaded from %s", config["dataset"]["pickle_file_path"])

    collator = MultiModalCollator(tokenizer)
    logger.info("Data collator initialized.")

    training_args = TrainingArguments(**config["training"])
    logger.info("Training arguments set.")

    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=collator
    )
    logger.info("Trainer initialized. Starting training...")

    trainer.train()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()