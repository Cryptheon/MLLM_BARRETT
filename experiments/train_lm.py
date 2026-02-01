import argparse
import yaml
import logging
import sys
import os

# Ensure src is in path if running from root
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.utils.util_functions import print_model_size 
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling 
)

from src.model.text_llama import TextLlamaForCausalLM, TextLlamaConfig 
from src.data.datasets import PubMedTextDataset 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Pretrain TextLlama model on text data")
    # Updated default config path
    parser.add_argument('--config', type=str, default="experiments/configs/pubmed/config.yaml", 
                        help='Path to the text pretraining config YAML file')
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Full config:\n%s", yaml.dump(config, default_flow_style=False, sort_keys=False))

    tokenizer_config = config["tokenizer"]
    tokenizer_name = tokenizer_config["tokenizer_name"]
    if not tokenizer_name:
        raise ValueError("Tokenizer name ('tokenizer_name') not specified in config.")
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer pad_token not set. Using eos_token ({tokenizer.eos_token}) as pad_token.")
    
    logger.info("Tokenizer '%s' loaded.", tokenizer_name)

    model_params = config["model"]
    pretrained_path = model_params.pop("pretrained_model_name_or_path", None) 
    
    if pretrained_path:
        logger.info(f"Loading model weights from pretrained path: {pretrained_path}")
        model_config = TextLlamaConfig.from_pretrained(pretrained_path, **model_params)
        model = TextLlamaForCausalLM.from_pretrained(pretrained_path, config=model_config)
    else:
        logger.info("Initializing model from scratch using config.")
        model_config = TextLlamaConfig(**model_params) 
        model = TextLlamaForCausalLM(model_config)

    trainable_params, total_params, total_mb = print_model_size(model)
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Approx. model size: %.2f MB", total_mb)

    dataset_config = config["dataset"]
    train_json_path = dataset_config["train_json_path"]
    max_seq_length = dataset_config["max_seq_length"]

    if not max_seq_length:
         raise ValueError("max_seq_length not specified in dataset config.")

    logger.info("Loading training dataset...")
    train_dataset = PubMedTextDataset(
        json_path=train_json_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        shuffle_on_init=True, 
        custom_tokenizer=config["tokenizer"]["custom_tokenizer"]
    )
    logger.info("Training dataset loaded from %s (%d samples)", train_json_path, len(train_dataset))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    training_params = config["training"]
    training_args = TrainingArguments(**training_params)
    logger.info("Training arguments set.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer 
    )
    logger.info("Standard Trainer initialized. Starting pretraining...")

    if training_args.resume_from_checkpoint:
         logger.info(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
         trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
         trainer.train()
         
    logger.info("Pretraining complete.")

    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model() 
    trainer.save_state()
    logger.info("Final model and trainer state saved.")

if __name__ == "__main__":
    main()
