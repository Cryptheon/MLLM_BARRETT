import pickle
import argparse
import random

def load_pickle(file_path: str):
    """Loads a pickle file from a local path."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(data, file_path: str):
    """Saves a dictionary to a pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def split_dataset(merged_file: str, train_file: str, val_file: str, validation_amount: int):
    """Splits the dataset into train and validation sets."""
    data = load_pickle(merged_file)

    if not data:
        print("\nThe merged dataset is empty!")
        return

    all_patient_ids = list(data.keys())
    if len(all_patient_ids) < validation_amount:
        print("\nNot enough data points to create a validation set of 200 patients.")
        return

    # Shuffle and select validation IDs
    random.shuffle(all_patient_ids)
    val_ids = set(all_patient_ids[:validation_amount])
    train_ids = set(all_patient_ids[validation_amount:])

    # Split the data
    val_data = {pid: data[pid] for pid in val_ids}
    train_data = {pid: data[pid] for pid in train_ids}

    # Save to pickle files
    save_pickle(train_data, train_file)
    save_pickle(val_data, val_file)

    print(f"\nSaved {len(train_data)} training patients to: {train_file}")
    print(f"Saved {len(val_data)} validation patients to: {val_file}")

def main():
    parser = argparse.ArgumentParser(description="Split merged dataset into train and validation pickles.")
    parser.add_argument("--merged_file_path", type=str, help="Path to the merged .pkl file")
    parser.add_argument("--train_file_path", type=str, default="train.pkl", help="Output path for the train .pkl file")
    parser.add_argument("--val_file_path", type=str, default="val.pkl", help="Output path for the val .pkl file")
    parser.add_argument("--validation_amount", type=int, default=200, help="Output path for the val .pkl file")

    args = parser.parse_args()

    split_dataset(args.merged_file_path, args.train_file_path, args.val_file_path)

if __name__ == "__main__":
    main()

