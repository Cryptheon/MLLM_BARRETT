import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_embeddings(file_path: str) -> dict:
    """Loads the embeddings file from a local path."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_data_info(data: dict):
    """Outputs basic information about the dataset."""
    embeddings = data['embeddings']
    filenames = data['filenames']
    print("Dataset Information:")
    print(f"- Number of entries: {len(filenames)}")
    print(f"- Shape of embedding matrix: {embeddings.shape}")

def show_sample_data(data: dict):
    """Displays the first data point with its corresponding filename."""
    embeddings = data['embeddings']
    filenames = data['filenames']
    print("\nSample Data:")
    print(f"Filename: {filenames[0]}")
    print(f"Embeddings: {embeddings[0].shape}")

def compute_statistics(data: dict):
    """Computes global statistics on the dataset."""
    embeddings = data['embeddings']
    mean_value = np.mean(embeddings)
    std_dev_value = np.std(embeddings)
    print("\nData Statistics:")
    print(f"- Global Mean: {mean_value}")
    print(f"- Global Std Dev: {std_dev_value}")

def visualize_tsne(data: dict, sample_size=500, output_path="tsne_visualization.png"):
    """Visualizes a subset of embeddings using t-SNE and saves the plot."""
    embeddings = data['embeddings']
    sample_indices = np.random.choice(len(embeddings), min(sample_size, len(embeddings)), replace=False)
    sampled_embeddings = embeddings[sample_indices]
    
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(sampled_embeddings)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(output_path)
    print(f"t-SNE visualization saved to {output_path}")

def main():
    """Main function to execute the script."""
    parser = argparse.ArgumentParser(description="Analyze TITAN embeddings.")
    parser.add_argument("--file_path", type=str, default="src/data/tcga_titan_features", help="Path to the embedding file")
    parser.add_argument("--output_path", type=str, default="tsne_visualization.png", help="Path to save the t-SNE plot")
    args = parser.parse_args()
    
    data = load_embeddings(args.file_path)
    get_data_info(data)
    show_sample_data(data)
    compute_statistics(data)
    visualize_tsne(data, output_path=args.output_path)

if __name__ == "__main__":
    main()
