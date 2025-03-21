from typing import Any, Dict
from transformers import Trainer

class MultiModalTrainer(Trainer):
    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False):
        wsi_embeddings = inputs.pop("wsi_embeddings", None)
        outputs = model(**inputs, wsi_embeddings=wsi_embeddings)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
