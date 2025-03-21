import torch
import yaml
from transformers import AutoTokenizer, TrainingArguments
from patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig
from dummy_dataset import DummyMultiModalDataset
from multimodal_trainer import MultiModalTrainer

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config("config.yaml")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = "<pad>"
    tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>"})

    model_config = PathoLlamaConfig(**config["model"])
    model = PathoLlamaForCausalLM(model_config)

    train_dataset = DummyMultiModalDataset(
        tokenizer=tokenizer,
        length=config["dataset"]["length"],
        seq_len=config["dataset"]["seq_len"],
        hidden_size=config["model"]["hidden_size"]
    )

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

