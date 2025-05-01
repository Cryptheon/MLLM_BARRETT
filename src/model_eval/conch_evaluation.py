import argparse
import json
import torch
import numpy as np
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch.nn.functional as F

def load_model(model_cfg: str, checkpoint_path: str, device: torch.device):
    """Loads the CONCH model."""
    model, _ = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
    model.eval()
    return model

def compute_cosine_similarity(model, text1: str, text2: str, tokenizer, device: torch.device) -> float:
    """Computes cosine similarity between two texts."""
    tokenized = tokenize(texts=[text1, text2], tokenizer=tokenizer).to(device)

    with torch.inference_mode():
        embeddings = model.encode_text(tokenized)
        embeddings = F.normalize(embeddings, dim=-1)
        similarity = torch.matmul(embeddings[0], embeddings[1]).item()

    return similarity

def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarity between original and generated reports using CONCH.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON file with [{'patient_id': ..., 'original_report': ..., 'generated_report': ...}, ...]")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/CONCH/pytorch_model.bin", help="Path to the model checkpoint")
    parser.add_argument("--model_cfg", type=str, default="conch_ViT-B-16", help="Model config name")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run the model on")

    args = parser.parse_args()
    device = torch.device(args.device)

    with open(args.input_json, 'r') as f:
        entries = json.load(f)

    if not entries:
        print("The input JSON is empty.")
        return

    model = load_model(args.model_cfg, args.checkpoint_path, device)
    tokenizer = get_tokenizer()

    similarities = []
    for entry in entries:
        pid = entry.get("patient_id")
        text1 = entry.get("original_report")
        text2 = entry.get("generated_report")

        if not text1 or not text2:
            print(f"Skipping {pid or '[unknown ID]'}: missing one of the reports.")
            continue

        similarity = compute_cosine_similarity(model, text1, text2, tokenizer, device)
        similarities.append(similarity)

    if not similarities:
        print("No valid report pairs processed.")
        return

    similarities_np = np.array(similarities)
    mean_score = similarities_np.mean()
    std_score = similarities_np.std()

    print(f"\nProcessed {len(similarities)} report pairs.")
    print(f"Mean cosine similarity: {mean_score:.4f}")
    print(f"Standard deviation:     {std_score:.4f}")

if __name__ == "__main__":
    main()
