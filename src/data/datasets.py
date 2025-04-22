import os
import pickle
from typing import Dict, Any
import random
import numpy as np
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
        max_seq_length: int = 1024,
        embeddings_dim_size: int = 768,
        random_choice_report: bool = False
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # max length refers to max length of the whole
        # sequence (WSI embeddings + textual embeddings)
        self.max_length = max_seq_length
        self.hidden_size = embeddings_dim_size
        self.random_choice_report = random_choice_report

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
        text_variations: str = patient["reports"]
        text_variations = eval(text_variations)

        if self.random_choice_report:
            text: str = random.choice(text_variations)
        else:
            text = text_variations[0]

        wsi_embeddings = torch.tensor(np.array(patient["embeddings"]))  # List of tensors
        
        if text is None or len(wsi_embeddings)==0 :
            # Resample if shuffling, otherwise get next sequential
            new_idx = random.randint(0, len(self.patient_ids) - 1) if getattr(self, "shuffle", True) else (idx + 1) % len(self)
            return self.__getitem__(new_idx)

        # Truncate to max number of embeddings if needed.
        # Although, we can skip this as we're not exceeding the 
        # max length with only WSI embeddings.
        if len(wsi_embeddings) > self.max_length:
            wsi_embeddings = wsi_embeddings[:self.max_length]

        # Tokenize text
        tokenized = self.tokenizer(
            text=text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)  # remove batch dim

        # We clone the labels without shifting it for CausalLM.
        # We explicitly construct the shifted labels for next token prediction in the model
        # The dataset should return the cloned input ids only for flexiblity.
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "wsi_embeddings": wsi_embeddings
        }
