import argparse
import yaml
from transformers import AutoTokenizer, TrainingArguments
from src.model.patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig
from src.data.datasets import PathoMultiModalDataset
from src.multimodal_trainer import MultiModalTrainer

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Train PathoLlama model with multimodal data")
    parser.add_argument('--config', type=str, default="./configs/tcga/config.yaml", help='Path to the config YAML file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["core_model_name"])
    tokenizer.pad_token = "<|eot_id|>"

    model_config = PathoLlamaConfig(**config["model"])
    model = PathoLlamaForCausalLM(model_config)

    train_dataset = PathoMultiModalDataset(pickle_file=config["dataset"]["pickle_file_path"],
                                           max_seq_length=config["dataset"]["max_seq_length"],
                                           embeddings_hidden_size=config["dataset"]["embeddings_dim_size"],
                                           tokenizer=tokenizer)

    training_args = TrainingArguments(**config["training"])

    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset
    )

    trainer.train()

if __name__ == "__main__":
    main()
