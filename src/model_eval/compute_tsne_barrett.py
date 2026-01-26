import argparse
import yaml
import logging
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Assumes your project structure allows this import ---
# You should have a file like ./data/datasets.py with the MultiModalBarrett class
try:
    from data.datasets import MultiModalBarrett
except ImportError:
    print("Error: Could not import 'MultiModalBarrett' from 'data.datasets'.")
    print("Please ensure your Python path is set up correctly and the file 'data/datasets.py' exists.")
    exit()

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Utility and Core Functions ---

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_data_from_dataset(config: dict, max_samples: int = None) -> tuple[torch.Tensor, list]:
    """
    Loads WSI embeddings and Barrett's labels using the MultiModalBarrett dataset.
    This function filters for cases that have exactly one associated WSI embedding.

    Args:
        config (dict): The configuration dictionary.
        max_samples (int, optional): Maximum number of cases to load. Defaults to None.

    Returns:
        tuple[torch.Tensor, list]: A tuple containing the concatenated embeddings tensor
                                   and a list of corresponding Barrett's labels.
    """
    logger.info("Loading data using the imported MultiModalBarrett dataset class...")
    data_config = config["dataset"]
    
    # 1. Instantiate tokenizer (path is expected in the config)
    tokenizer_name = config.get("tokenizer", {}).get("tokenizer_name", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer '{tokenizer_name}' loaded.")

    # 2. Instantiate your dataset for the validation phase
    dataset = MultiModalBarrett(
        json_file=data_config["train_texts_json_path"],
        embeddings_file=data_config["train_h5_file_path"],
        tokenizer=tokenizer,
        phase="train", # We visualize the validation set
        val_data_ratio=data_config.get("val_data_ratio", 0.1),
        max_seq_length=data_config.get("max_seq_length", 1024)
    )

    all_embeddings = []
    all_labels = []

    # 3. Filter for cases with a single embedding key by inspecting the dataset's internal list
    logger.info("Filtering for cases with a single embedding key...")
    single_embedding_indices = [
        i for i, sample in enumerate(dataset.samples) 
        if len(sample.get("embedding_keys", [])) == 1
    ]
    logger.info(f"Found {len(single_embedding_indices)} cases with a single embedding.")

    # Apply max_samples limit if provided
    if max_samples:
        single_embedding_indices = single_embedding_indices[:max_samples]
        logger.info(f"Limiting to a maximum of {max_samples} slides for t-SNE.")

    # 4. Iterate through the filtered samples and get data using the dataset's __getitem__
    for idx in tqdm(single_embedding_indices, desc="Loading Filtered Embeddings"):
        try:
            sample_data = dataset[idx] # Use the dataset object to get the processed sample
            embeddings_tensor = sample_data.get("wsi_embeddings")
            barrett_label = sample_data.get("barrett_label")

            if embeddings_tensor is not None and embeddings_tensor.numel() > 0 and barrett_label:
                all_embeddings.append(embeddings_tensor)
                # Each patch in the tensor gets the same label
                all_labels.extend([barrett_label] * embeddings_tensor.shape[0])
            else:
                logger.warning(f"Skipping sample at index {idx} due to missing embedding or label.")
        except Exception as e:
            logger.warning(f"Could not process sample at index {idx}. Error: {e}. Skipping.")

    if not all_embeddings:
        raise ValueError("No valid embeddings were loaded. Check dataset filtering, file paths, or labels.")

    logger.info("Concatenating all embedding tensors...")
    concatenated_embeddings = torch.cat(all_embeddings, dim=0)
    
    logger.info(f"Final concatenated embeddings shape: {concatenated_embeddings.shape}")
    logger.info(f"Total number of patch labels: {len(all_labels)}")

    return concatenated_embeddings, all_labels

def main():
    parser = argparse.ArgumentParser(description="Compute and plot t-SNE of WSI embeddings labeled by Barrett's classification.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--output', type=str, default="tsne_barrett_label_visualization.png", help='Path to save the output plot PNG file.')
    parser.add_argument('--max_samples', type=int, default=2000, help='Maximum number of slides (with single embeddings) to process.')
    parser.add_argument('--perplexity', type=int, default=100, help='Perplexity value for the t-SNE algorithm.')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 1. Load data using the new function
    embeddings, labels = load_data_from_dataset(config, args.max_samples)

    # 2. Compute t-SNE
    logger.info(f"Starting t-SNE computation with perplexity={args.perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=42,
        max_iter=2000,
        verbose=1,
    )
    tsne_results = tsne.fit_transform(embeddings.numpy())
    logger.info("t-SNE computation complete.")

    # 3. Create and save the plot
    logger.info(f"Generating plot and saving to {args.output}...")
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['label'] = labels
    
    # Define a specific color palette for better interpretation
    label_order = ["ND", "LGD", "HGD"] # Defines order in legend
    color_palette = {
        "ND": "#2ca02c",  # Green
        "LGD": "#ff7f0e", # Orange
        "HGD": "#d62728",  # Red
        "IND": "#004bbb",
        "C": "#b200c2"
    }
    
    num_unique_cases = len(pd.unique(labels))

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        hue_order=label_order, # Ensure consistent legend order
        palette=color_palette,
        data=df,
        legend="full",
        alpha=0.6,
        s=50 # Smaller points for better visibility of clusters
    )
    
    plt.legend(title="Barrett's Label", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(f"t-SNE of WSI Embeddings from {args.max_samples} Slides ({num_unique_cases} Labels)")
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    
    logger.info("--------------------")
    logger.info("Processing Complete!")
    logger.info(f"t-SNE plot saved successfully to {args.output}")
    logger.info("--------------------")

if __name__ == "__main__":
    main()