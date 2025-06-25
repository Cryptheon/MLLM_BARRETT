import pickle
import argparse

def load_pickle(file_path: str):
    """Loads a pickle file from a local path."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def show_patient(merged_file: str):
    """Displays the first patient's data in the merged dataset."""
    
    # Load merged pickle file
    data = load_pickle(merged_file)

    if not data:
        print("\nThe merged dataset is empty!")
        return

    # Get an example patient
    patient_id = list(iter(data))[1]
    patient_data = data[patient_id]

    print(f"\nFirst Patient in Merged Dataset: {patient_id}")
    print(f"Report Filename: {patient_data['report_filename']}")
    print(f"Report Text: {patient_data['reports'][:100]}...")  # Show only first 100 chars
    print(f"Total Embeddings: {len(patient_data['embeddings'])}")

    for i, (filename, emb) in enumerate(zip(patient_data["embedding_filenames"], patient_data["embeddings"])):
        print(f"  Embedding {i+1}: {filename}, Shape: {emb.shape}")

def main():
    parser = argparse.ArgumentParser(description="Check the first patient data from the merged TCGA dataset.")
    parser.add_argument("--merged_file_path", type=str, help="Path to the merged .pkl file")

    args = parser.parse_args()

    show_patient(args.merged_file_path)

if __name__ == "__main__":
    main()
