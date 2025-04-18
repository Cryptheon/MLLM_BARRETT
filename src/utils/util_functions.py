import yaml
from types import SimpleNamespace
import torch
from collections import OrderedDict

from transformers import (
    AutoModelForCausalLM
)

def move_optimizer_to_device(optimizer: torch.optim.AdamW, device: torch.device) -> None:
    """Move optimizer states to the specified device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def load_state_dict(checkpoint_path: str) -> OrderedDict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    total_parameters = sum(value.numel() for value in checkpoint.values())
    print(f"Loaded Model! Total Parameters in this model: {total_parameters}")
    return checkpoint

def load_checkpoint(checkpoint_path: str, 
                    model: AutoModelForCausalLM, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler,
                    device: torch.device) -> int:
    """
    Load the model, optimizer, and scheduler states from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model: Hugging Face model instance.
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance.

    Returns:
        int: The last training step (batch_step) saved in the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    batch_step = checkpoint.get('batch_step', 0)
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from batch step {batch_step}")
    move_optimizer_to_device(optimizer, device)

    return batch_step

def print_model_size(model: torch.nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    buffer_bytes = sum(b.element_size() * b.numel() for b in model.buffers())
    total_bytes += buffer_bytes
    total_mb = total_bytes / (1024 ** 2)

    return trainable_params, total_params, total_mb

def load_config(config_path: str):
    """
    Load configuration from a YAML file and convert it to a SimpleNamespace
    for attribute-style access.
    """
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    print(config_dict)  # Check loaded values
    return SimpleNamespace(**config_dict)
