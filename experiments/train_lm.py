import argparse
import yaml
import logging
from utils.util_functions import print_model_size 
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling # Use standard data collator for LM
)

from model.text_llama import TextLlamaForCausalLM, TextLlamaConfig 
from data.datasets import PubMedTextDataset 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    """Loads configuration from a YAML file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Pretrain TextLlama model on text data")
    # Update default config path if needed
    parser.add_argument('--config', type=str, default="./configs/pubmed/config_pretrain.yaml", 
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
    # Decide whether to load from pretrained or initialize from config
    pretrained_path = model_params.pop("pretrained_model_name_or_path", None) 
    
    if pretrained_path:
        logger.info(f"Loading model weights from pretrained path: {pretrained_path}")
        model_config = TextLlamaConfig.from_pretrained(pretrained_path, **model_params)
        model = TextLlamaForCausalLM.from_pretrained(pretrained_path, config=model_config)
    else:
        logger.info("Initializing model from scratch using config.")
        # Ensure necessary Llama config params are present in model_params
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
        shuffle_on_init=True, # Shuffle training data when loaded
        custom_tokenizer=config["tokenizer"]["custom_tokenizer"]
        # num_data_points can be added here if specified in config
    )
    logger.info("Training dataset loaded from %s (%d samples)", train_json_path, len(train_dataset))

    # For Causal LM pretraining (predicting the next token), mlm=False.
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    logger.info("Data collator for Causal LM initialized (mlm=False).")

    training_params = config["training"]
    if not training_params["output_dir"]:
         raise ValueError("output_dir must be specified in training config.")
         
    training_args = TrainingArguments(**training_params)
    logger.info("Training arguments set.")
    # Log effective batch size
    effective_batch_size = (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps *
                            training_args.world_size) # world_size is 1 if not distributed
    logger.info(f"Effective batch size: {effective_batch_size}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer # Pass tokenizer for saving purposes
    )
    logger.info("Standard Trainer initialized. Starting pretraining...")

    # Check for resuming from checkpoint
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if resume_from_checkpoint:
         logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
         trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
         trainer.train()
         
    logger.info("Pretraining complete.")

    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model() 
    trainer.save_state()
    logger.info("Final model and trainer state saved.")

if __name__ == "__main__":
    main()
