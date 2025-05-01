"""
Trains a new tokenizer based on an existing one using data from a local CSV file.

Example usage:
python train_tokenizer_csv.py \
    --csv_path path/to/your/data.csv \
    --text_column processed_report \
    --base_tokenizer gpt2 \
    --vocab_size 25000 \
    --output_dir my-new-csv-tokenizer
"""

import argparse
import logging
import pandas as pd
import sys 
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a new tokenizer from an existing one on data from a CSV file.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="processed_report",
        help="The name of the column in the CSV containing the text data."
    )
    
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Name or path of the base pre-trained hf tokenizer. Needs to be a fast tokenizer."
        )
    
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="The desired vocabulary size for the new tokenizer."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for iterating over the dataset during training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the newly trained tokenizer will be saved."
    )
    return parser.parse_args()

def load_data_from_csv(csv_path, text_column):
    """Loads data from the specified CSV file, returning a DataFrame."""
    logging.info(f"Loading data from CSV file: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"CSV loaded successfully. Number of rows: {len(df)}")

        if text_column not in df.columns:
            logging.error(f"Text column '{text_column}' not found in CSV columns: {df.columns.tolist()}")
            sys.exit(1) 

        # Optional: Handle potential missing values in the text column
        initial_rows = len(df)
        df.dropna(subset=[text_column], inplace=True)
        if len(df) < initial_rows:
            logging.warning(f"Dropped {initial_rows - len(df)} rows with missing values in '{text_column}'.")

        # Ensure the text column is of string type
        df[text_column] = df[text_column].astype(str)

        logging.info(f"Using text data from column '{text_column}'.")
        return df

    except FileNotFoundError:
        logging.error(f"CSV file not found at path: {csv_path}")
        sys.exit(1) 
    except Exception as e:
        logging.error(f"Failed to load or process CSV: {e}")
        sys.exit(1) 


def create_batch_iterator(data_frame, batch_size, text_column):
    """Creates a Python generator to iterate over the DataFrame in batches."""
    logging.info(f"Creating batch iterator with batch size {batch_size}...")
    # Assume text_column exists as it was checked in load_data_from_csv
    num_rows = len(data_frame)

    def batch_iterator():
        for i in range(0, num_rows, batch_size):
            batch = data_frame.iloc[i : i + batch_size][text_column].tolist()
            yield batch
    return batch_iterator

def load_base_tokenizer(model_name):
    """Loads the base tokenizer and checks if it's a fast tokenizer."""
    logging.info(f"Loading base tokenizer '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.is_fast:
            logging.error(
                f"The base tokenizer '{model_name}' is not a 'fast' tokenizer. "
                "Training requires a fast tokenizer backed by the Hugging Face Tokenizers library."
            )
            sys.exit(1) 
        logging.info("Base tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load base tokenizer '{model_name}': {e}")
        sys.exit(1) 


def train_tokenizer(base_tokenizer, iterator_fn, vocab_size):
    """Trains a new tokenizer using the base tokenizer's methods."""
    logging.info(f"Starting tokenizer training with vocab size {vocab_size}...")
   
    new_tokenizer = base_tokenizer.train_new_from_iterator(iterator_fn(), vocab_size=vocab_size)
    logging.info("Tokenizer training completed.")
    return new_tokenizer

def save_tokenizer(tokenizer, output_dir):
    """Saves the trained tokenizer to the specified directory."""
    logging.info(f"Saving tokenizer to '{output_dir}'...")
    tokenizer.save_pretrained(output_dir)
    logging.info("Tokenizer saved successfully.")

       
def main():
    args = parse_arguments()

    data_frame = load_data_from_csv(args.csv_path, args.text_column)
    batch_iterator_fn = create_batch_iterator(data_frame, args.batch_size, args.text_column)
    base_tokenizer = load_base_tokenizer(args.base_tokenizer)
    new_tokenizer = train_tokenizer(base_tokenizer, batch_iterator_fn, args.vocab_size)
    save_tokenizer(new_tokenizer, args.output_dir)

    logging.info("Tokenizer training process finished.")

if __name__ == "__main__":
    main()