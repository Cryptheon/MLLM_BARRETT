# test_patho_dataset.py
import torch
from transformers import AutoTokenizer
from data.datasets import PathoMultiModalDataset  # Adjust the import path

def test_patho_dataset(pickle_file_path: str):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = "<|eot_id|>"

    # might create issues in the predefined vocabulary for Llama-3-8B.
    #tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "<|eot_id|>"})

    dataset = PathoMultiModalDataset(pickle_file=pickle_file_path, tokenizer=tokenizer)

    sample = dataset[1]
    print("Testing PathoMultiModalDataset...\n")
    print("Input IDs:", sample["input_ids"].shape)
    print("Labels:", sample["labels"].shape)
    print("WSI Embeddings:", sample["wsi_embeddings"].shape)

    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)
    assert isinstance(sample["wsi_embeddings"], torch.Tensor)

    assert sample["input_ids"].shape == torch.Size([dataset.max_length])
    assert sample["labels"].shape == torch.Size([dataset.max_length])
    assert sample["wsi_embeddings"].shape[1] == 768  # (N_patches, hidden_size)

    print("\nPatho dataset test passed!")

if __name__ == "__main__":
    pickle_path = "./tcga_data/tcga_titan_embeddings_reports.pkl"  # <- Change this to your pickle file path
    test_patho_dataset(pickle_path)
