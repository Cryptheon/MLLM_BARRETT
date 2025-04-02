import torch

def print_model_size(model: torch.nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    buffer_bytes = sum(b.element_size() * b.numel() for b in model.buffers())
    total_bytes += buffer_bytes
    total_mb = total_bytes / (1024 ** 2)

    return trainable_params, total_params, total_mb

