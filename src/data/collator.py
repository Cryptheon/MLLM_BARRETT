from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List
import torch

class MultiModalCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, padding_side: str = "right"):
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
         input_ids = torch.stack([f["input_ids"] for f in features])
         labels = torch.stack([f["labels"] for f in features])
         wsi_embeddings = [f["wsi_embeddings"] for f in features]  # leave as list

         return {
             "input_ids": input_ids,
             "labels": labels,
             "wsi_embeddings": wsi_embeddings
         }