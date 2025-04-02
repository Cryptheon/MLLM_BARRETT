import pandas as pd
import pickle
import argparse
from collections import defaultdict

def load_embeddings(file_path: str) -> dict:
    """Loads the embeddings file from a local path."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_reports(csv_path: str) -> pd.DataFrame:
    """Loads the clinical reports CSV file."""
    return pd.read_csv(csv_path)

def extract_patient_id(filename: str) -> str:
    """Extracts the patient ID from a TCGA filename."""
    return filename.split('.')[0]  # Keep only 'TCGA-XX-YYYY'

def extract_patient_id_from_embedding(filename: str) -> str:
    """Extracts the patient ID from an embedding filename that contains additional fields."""
    splitted = filename.split("-")
    return '-'.join(splitted[:3])  # Extract 'TCGA-XX-YYYY'

def merge_embeddings_with_reports(embeddings_file: str, reports_file: str, output_file: str):
    """Merges the embeddings with the clinical reports and saves the result as a pickle file, grouped per patient."""
    
    # Load data
    embeddings_data = load_embeddings(embeddings_file)
    reports_df = load_reports(reports_file)

    # Extract filenames and patient IDs for embeddings
    embedding_filenames = embeddings_data['filenames']
    embedding_vectors = embeddings_data['embeddings']

    # Create a dictionary that maps patient IDs to multiple embeddings
    patient_data = defaultdict(lambda: {
        "report_filename": None,
        "report_text": None,
        "embedding_filenames": [],
        "embeddings": []
    })

    # Extract patient IDs from reports
    reports_df["patient_id"] = reports_df["patient_filename"].apply(extract_patient_id)

    # Store report data per patient
    for _, row in reports_df.iterrows():
        patient_id = row["patient_id"]
        patient_data[patient_id]["report_filename"] = row["patient_filename"]
        patient_data[patient_id]["report_text"] = row["text"]

    # Store embeddings per patient
    for filename, embedding in zip(embedding_filenames, embedding_vectors):
        patient_id = extract_patient_id_from_embedding(filename)
        # should have a check if the patient_id exists in the reports_df, else
        # we can be storing embeddings for a patient for which there is no report.
        patient_data[patient_id]["embedding_filenames"].append(filename)
        patient_data[patient_id]["embeddings"].append(embedding)

    # Save merged data as a pickle file
    with open(output_file, 'wb') as file:
        pickle.dump(dict(patient_data), file)  # Convert defaultdict to dict before saving

    print(f"\nMerged dataset saved to {output_file}")
    print(f"Total unique patients: {len(patient_data)}")

def main():
    parser = argparse.ArgumentParser(description="Merge TCGA clinical reports with embeddings, grouped per patient.")
    parser.add_argument("--embeddings_file_path", type=str, required=True, help="Path to the embeddings .pkl file")
    parser.add_argument("--reports_file_path", type=str, required=True, help="Path to the clinical reports .csv file")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to save the merged output .pkl file")

    args = parser.parse_args()

    merge_embeddings_with_reports(args.embeddings_file_path, args.reports_file_path, args.output_file_path)

if __name__ == "__main__":
    main()
