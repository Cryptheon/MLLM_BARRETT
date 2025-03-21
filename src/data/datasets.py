from typing import Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class DummyMultiModalDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        length: int = 128,
        seq_len: int = 32,
        hidden_size: int = 768
    ) -> None:
        self.tokenizer = tokenizer
        self.length = length
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        num_patches = torch.randint(2, 5, (1,)).item()
        wsi_embeddings = torch.randn(num_patches, self.hidden_size)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "wsi_embeddings": wsi_embeddings
        }
        
class PathoMultiModalDataset(Dataset):
    def __init__(
        self,
        pickle_file: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        hidden_size: int = 768
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hidden_size = hidden_size

        if not os.path.exists(pickle_file):
            raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

        self.patient_ids = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self.patient_ids[idx]
        patient = self.data[patient_id]

        text: str = patient["report_text"]
        embeddings = patient["embeddings"]  # List of tensors

        # Truncate to max number of embeddings if needed
        if len(embeddings) > self.max_length:
            embeddings = embeddings[:self.max_length]

        # Stack WSI embeddings (convert list to tensor)
        wsi_embeddings = torch.stack(embeddings)  # shape: (N_patches, hidden_size)

        # Tokenize text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)  # remove batch dim
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "wsi_embeddings": wsi_embeddings
        }
