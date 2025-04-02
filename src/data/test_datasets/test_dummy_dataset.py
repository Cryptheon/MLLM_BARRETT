# test_dummy_dataset.py
import torch
from transformers import AutoTokenizer
from data.datasets import DummyMultiModalDataset  # Adjust the import path

def test_dummy_dataset():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    dataset = DummyMultiModalDataset(tokenizer=tokenizer, length=10, seq_len=32, hidden_size=768)

    sample = dataset[0]
    print(sample)

    print("Testing DummyMultiModalDataset...\n")
    print("Input IDs:", sample["input_ids"].shape)
    print("Labels:", sample["labels"].shape)
    print("WSI Embeddings:", sample["wsi_embeddings"].shape)

    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)
    assert isinstance(sample["wsi_embeddings"], torch.Tensor)

    assert sample["input_ids"].shape == torch.Size([32])
    assert sample["labels"].shape == torch.Size([32])
    assert sample["wsi_embeddings"].shape[1] == 768  # (N_patches, hidden_size)

    print("\nDummy dataset test passed!")

if __name__ == "__main__":
    test_dummy_dataset()

