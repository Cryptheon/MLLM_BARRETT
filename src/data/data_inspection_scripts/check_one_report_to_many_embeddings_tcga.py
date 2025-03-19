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
    return splitted[0] + '-' + splitted[1] + '-' + splitted[2]  # 'TCGA-XX-YYYY'

def check_multiple_embeddings(embeddings_file: str, reports_file: str):
    """Checks if multiple embeddings exist for a single patient-level clinical report."""
    
    # Load data
    embeddings_data = load_embeddings(embeddings_file)
    reports_df = load_reports(reports_file)

    # Extract filenames
    embedding_filenames = embeddings_data['filenames']
    report_filenames = reports_df['patient_filename'].tolist()

    # Extract patient IDs from both sources
    embeddings_by_patient = defaultdict(list)
    for filename in embedding_filenames:
        patient_id = extract_patient_id_from_embedding(filename)
        embeddings_by_patient[patient_id].append(filename)

    report_patients = {extract_patient_id(f) for f in report_filenames}

    # Find patients with multiple embeddings for the same report
    multiple_embeddings_patients = {
        patient: files for patient, files in embeddings_by_patient.items() 
        if patient in report_patients and len(files) > 1
    }

    # Display results
    if multiple_embeddings_patients:
        print("\nPatients with multiple embeddings for the same clinical report:")
        for patient, files in multiple_embeddings_patients.items():
            print(f"{patient}: {len(files)} embeddings → {files}")
    else:
        print("\nNo patients have multiple embeddings for the same clinical report.")

def main():
    parser = argparse.ArgumentParser(description="Check for multiple embeddings per clinical report in TCGA dataset.")
    parser.add_argument("--embeddings_file_path", type=str, required=True, help="Path to the embeddings .pkl file")
    parser.add_argument("--reports_file_path", type=str, required=True, help="Path to the clinical reports .csv file")

    args = parser.parse_args()

    check_multiple_embeddings(args.embeddings_file_path, args.reports_file_path)

if __name__ == "__main__":
    main()
