import os
import json
import argparse
import logging
import random
import re
import h5py
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import List, Dict, Tuple, Any

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Hybrid Dataset Class (Kept in-script as it's experiment-specific)
# -----------------------------------------------------------------------------
class BarrettEmbeddingDataset(Dataset):
    def __init__(self, json_file: str, embeddings_file: str, phase: str = "train", 
                 val_ratio: float = 0.2, allowed_classes: List[str] = ["ND", "LGD", "HGD"],
                 single_instance_mode: bool = False):
        
        self.json_file = json_file
        self.embeddings_file = embeddings_file
        self.allowed_classes = allowed_classes
        self.class_map = {label: i for i, label in enumerate(allowed_classes)}
        self.single_instance_mode = single_instance_mode
        
        # Load Raw Data
        all_samples = self._load_data(json_file, embeddings_file)
        
        # Split logic (Shuffle once before splitting to ensure random distribution)
        # We sort by An_Number for deterministic behavior across runs
        all_samples.sort(key=lambda x: str(x.get("case_id")))
        
        split_idx = int(len(all_samples) * (1 - val_ratio))
        if phase == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
            
        # Count stats
        label_counts = {k: 0 for k in allowed_classes}
        for s in self.samples:
            # Reverse lookup for logging
            lbl_str = [k for k, v in self.class_map.items() if v == s['label_int']][0]
            label_counts[lbl_str] += 1
            
        logger.info(f"[{phase.upper()}] Mode: {'Single Instance' if self.single_instance_mode else 'MIL (Bag)'}")
        logger.info(f"Samples: {len(self.samples)} | Distribution: {label_counts}")

    def _load_data(self, json_file, embeddings_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        
        with h5py.File(embeddings_file, 'r') as hf:
            embedding_keys = list(hf.keys())

        valid_samples = []

        for case in text_data:
            an_number = case.get("original_case", {}).get("An_Number")
            if not an_number: continue

            # --- MODE A: SINGLE INSTANCE LEARNING (MLP) ---
            if self.single_instance_mode:
                divided = case.get("divided_reports")
                # Ensure divided_reports exists and is a dict
                if not isinstance(divided, dict): 
                    continue
                
                labels_dict = divided.get("barrett_label")
                if not isinstance(labels_dict, dict):
                    continue

                # Iterate through parts (I, II, III...)
                for part_id, label_str in labels_dict.items():
                    if label_str not in self.allowed_classes:
                        continue
                    
                    # FIND MATCHING KEY logic:
                    matching_key = None
                    for key in embedding_keys:
                        if key.startswith(an_number):
                            # Pattern: delimiter + part_id + delimiter/end
                            if re.search(f"[^a-zA-Z0-9]{part_id}[^a-zA-Z0-9]", key + "_"):
                                matching_key = key
                                break
                    
                    if matching_key:
                        valid_samples.append({
                            "case_id": f"{an_number}_{part_id}",
                            "embedding_keys": [matching_key], # List of 1 key
                            "label_int": self.class_map[label_str]
                        })

            # --- MODE B: MIL LEARNING (Bag) ---
            else:
                raw_report = case.get("cleaned_reports")
                if not isinstance(raw_report, dict): continue
                
                label_str = raw_report.get("barrett_label")
                if label_str not in self.allowed_classes: continue

                # Collect ALL keys for this patient
                patient_keys = [k for k in embedding_keys if k.startswith(an_number)]
                
                if patient_keys:
                    valid_samples.append({
                        "case_id": an_number,
                        "embedding_keys": patient_keys,
                        "label_int": self.class_map[label_str]
                    })

        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        keys = sample["embedding_keys"]
        label = sample["label_int"]
        
        wsi_list = []
        try:
            with h5py.File(self.embeddings_file, 'r') as hf:
                for k in keys:
                    if k in hf:
                        feats = torch.tensor(hf[k]['features'][:], dtype=torch.float32)
                        if feats.shape[0] > 0:
                            wsi_list.append(feats)
        except Exception as e:
            logger.warning(f"Error reading h5 for index {idx}: {e}")

        # Handle missing data
        if not wsi_list:
            # Return dummy: (1, 768)
            return torch.zeros(1, 768), torch.tensor(label, dtype=torch.long)
            
        features = torch.vstack(wsi_list) # Shape: (Total_Patches, 768)

        # If Single Instance Mode, we generally expect 1 WSI. 
        # However, a WSI is still a "bag" of patches.
        # If you want to Average Pool the patches for a Simple MLP:
        if self.single_instance_mode:
            # Mean Pooling over patches to get (1, 768) -> (768,)
            features = torch.mean(features, dim=0) 
        
        return features, torch.tensor(label, dtype=torch.long)

# -----------------------------------------------------------------------------
# 2. Models
# -----------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=3, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits, None 

class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=3, dropout=0.25, gated=True):
        super().__init__()
        self.L = hidden_dim
        self.D = hidden_dim
        self.K = 1 
        self.gated = gated

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.gated:
            self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
            self.attention_weights = nn.Linear(self.D, self.K)
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.K)
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(0) # Handle batch_size=1
        
        H = self.feature_extractor(x) 

        if self.gated:
            A_V = self.attention_V(H)
            A_U = self.attention_U(H)
            A = self.attention_weights(A_V * A_U) 
        else:
            A = self.attention(H)
        
        A = torch.transpose(A, 1, 0) # K x N
        A = F.softmax(A, dim=1)       
        M = torch.mm(A, H)           # K x L
        logits = self.classifier(M)  # K x C
        
        return logits.squeeze(0), A

# -----------------------------------------------------------------------------
# 3. Training & Util Functions
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    counts = np.zeros(len(dataset.allowed_classes))
    for s in dataset.samples:
        counts[s["label_int"]] += 1
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)

def get_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    use_mil = not cfg['model']['use_single_instance_learning']
    batch_size = 1 if use_mil else cfg['training']['batch_size']
    
    train_ds = BarrettEmbeddingDataset(
        json_file=cfg['data']['json_file'],
        embeddings_file=cfg['data']['embeddings_file'],
        phase="train",
        val_ratio=cfg['data']['val_ratio'],
        allowed_classes=cfg['data']['classes_to_use'],
        single_instance_mode=cfg['model']['use_single_instance_learning']
    )
    
    val_ds = BarrettEmbeddingDataset(
        json_file=cfg['data']['json_file'],
        embeddings_file=cfg['data']['embeddings_file'],
        phase="val",
        val_ratio=cfg['data']['val_ratio'],
        allowed_classes=cfg['data']['classes_to_use'],
        single_instance_mode=cfg['model']['use_single_instance_learning']
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    weights = compute_class_weights(train_ds)
    
    return train_loader, val_loader, weights

def train_epoch(model, loader, optimizer, criterion, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (features, label) in enumerate(loader):
        features, label = features.to(device), label.to(device)

        logits, _ = model(features)
        loss = criterion(logits, label)
        
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accumulation_steps

    return running_loss / len(loader)

def validate(model, loader, criterion, device, target_names):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, label in loader:
            features, label = features.to(device), label.to(device)

            logits, _ = model(features)
            loss = criterion(logits, label)
            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    metrics = {
        "loss": running_loss / len(loader),
        "acc": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average='macro'),
        "f1_weighted": f1_score(all_labels, all_preds, average='weighted'),
        "report_dict": classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0),
        "report_str": classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    }
    return metrics

def save_plots(history: Dict, output_dir: str):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], label='Val F1 (Macro)', color='orange')
    plt.title('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training
