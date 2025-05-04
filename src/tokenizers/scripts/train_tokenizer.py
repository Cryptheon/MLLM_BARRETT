# train_tokenizer.py (updated to handle CSV or JSON)

import argparse
import logging
import pandas as pd
import json # Added for JSON loading
from pathlib import Path # Added for robust path handling
import sys
from typing import List, Dict, Any, Iterator # Added typing

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a new tokenizer from an existing one using data from a CSV or JSON file.")
    
    # --- Input Data Arguments (Mutually Exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--csv_path",
        type=Path,
        help="Path to the input CSV file. Use with --text_column."
    )
    input_group.add_argument(
        "--json_path",
        type=Path,
        help="Path to the input JSON file (PubMed format). Use with --json_*_key args."
    )

    # --- CSV Specific Arguments ---
    parser.add_argument(
        "--text_column",
        type=str,
        default="processed_report",
        help="[CSV only] The name of the column in the CSV containing the text data."
    )

    # --- Tokenizer Arguments ---
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
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the newly trained tokenizer will be saved."
    )

    # --- Training Process Arguments ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for iterating over the dataset during training."
    )
    
    return parser.parse_args()


def load_data_from_csv(csv_path: Path, text_column: str) -> pd.DataFrame:
    """Loads data from the specified CSV file, returning a DataFrame."""
    logging.info(f"Loading data from CSV file: {csv_path}...")
    if not csv_path.is_file():
        logging.error(f"CSV file not found at path: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        logging.info(f"CSV loaded successfully. Number of rows: {len(df)}")

        if text_column not in df.columns:
            logging.error(f"Text column '{text_column}' not found in CSV columns: {df.columns.tolist()}")
            sys.exit(1)

        # Handle potential missing values in the text column
        initial_rows = len(df)
        df.dropna(subset=[text_column], inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logging.warning(f"Dropped {rows_dropped} rows with missing values in '{text_column}'.")

        # Ensure the text column is of string type
        df[text_column] = df[text_column].astype(str)

        logging.info(f"Using text data from column '{text_column}'.")
        return df

    except Exception as e:
        logging.error(f"Failed to load or process CSV {csv_path}: {e}")
        sys.exit(1)

def load_data_from_json(json_path: Path, title_key: str, abstract_key: str) -> List[str]:
    """
    Loads PubMed-style JSON data and extracts text according to the specified format.
    Returns a list of text strings.
    """
    logging.info(f"Loading data from JSON file: {json_path}...")
    if not json_path.is_file():
        logging.error(f"JSON file not found at path: {json_path}")
        sys.exit(1)

    try:
        with open(json_path, "r", encoding='utf-8') as f:
            # Data is stored as {pmid: {details}}
            raw_data: Dict[str, Dict[str, Any]] = json.load(f)
        logging.info(f"JSON loaded successfully. Number of entries: {len(raw_data)}")

        text_corpus: List[str] = []
        processed_count = 0
        skipped_count = 0

        for pmid, details in raw_data.items():
            title = details.get(title_key, "")
            abstract = details.get(abstract_key, "")

            text = abstract.strip()

            if text: # Only add non-empty text
                text_corpus.append(text)
                processed_count += 1
            else:
                skipped_count += 1

        if skipped_count > 0:
             logging.warning(f"Skipped {skipped_count} entries due to empty resulting text.")
        
        if not text_corpus:
             logging.error("No text data could be extracted from the JSON file. Exiting.")
             sys.exit(1)
             
        return text_corpus

    except json.JSONDecodeError as e:
         logging.error(f"Error decoding JSON from {json_path}: {e}")
         sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load or process JSON {json_path}: {e}")
        sys.exit(1)

def create_batch_iterator(data: pd.Series, batch_size: int) -> Iterator[List[str]]:
    """Creates a Python generator to iterate over text data (Pandas Series) in batches."""
    logging.info(f"Creating batch iterator with batch size {batch_size}...")
    num_rows = len(data)
    if num_rows == 0:
        logging.error("Cannot create iterator from empty data.")
        sys.exit(1)

    def batch_iterator():
        for i in range(0, num_rows, batch_size):
            # Ensure we yield a list of strings
            batch = data.iloc[i : i + batch_size].tolist()
            yield batch
    return batch_iterator

def load_base_tokenizer(model_name: str) -> AutoTokenizer:
    """Loads the base tokenizer and checks if it's a fast tokenizer."""
    logging.info(f"Loading base tokenizer '{model_name}'...")
    try:
        # Explicitly request the fast version if available
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not tokenizer.is_fast:
            # Fallback or error if fast version not found/supported by from_pretrained
            logging.warning(f"Could not load '{model_name}' as a fast tokenizer directly. Trying without use_fast=True.")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if not tokenizer.is_fast:
                logging.error(
                    f"The base tokenizer '{model_name}' is not a 'fast' tokenizer. "
                    "Training requires a fast tokenizer backed by the Hugging Face Tokenizers library."
                )
                sys.exit(1)
            else:
                 logging.info("Loaded non-fast version, but it reports is_fast=True.") # Should not happen often

        logging.info("Base fast tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load base tokenizer '{model_name}': {e}")
        sys.exit(1)


def train_tokenizer(base_tokenizer: AutoTokenizer, iterator_fn: Iterator[List[str]], vocab_size: int) -> AutoTokenizer:
    """Trains a new tokenizer using the base tokenizer's methods."""
    logging.info(f"Starting tokenizer training with vocab size {vocab_size}...")
    
    # The iterator needs to be a function that returns an iterator,
    # create_batch_iterator returns a function, so we call it here.
    training_iterator = iterator_fn() 
    
    # Use the train_new_from_iterator method of the base fast tokenizer
    new_tokenizer = base_tokenizer.train_new_from_iterator(training_iterator, vocab_size=vocab_size)
    logging.info("Tokenizer training completed.")
    return new_tokenizer

def save_tokenizer(tokenizer: AutoTokenizer, output_dir: Path):
    """Saves the trained tokenizer to the specified directory."""
    logging.info(f"Saving tokenizer to '{output_dir}'...")
    try:
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        tokenizer.save_pretrained(str(output_dir)) # save_pretrained expects string path
        logging.info(f"Tokenizer saved successfully to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save tokenizer to {output_dir}: {e}")
        sys.exit(1)
        
def main():
    args = parse_arguments()
    text_data_series = None # To hold the text data as a Pandas Series

    # --- Load Data based on Input Type ---
    if args.csv_path:
        logging.info("Processing CSV input.")
        data_frame = load_data_from_csv(args.csv_path, args.text_column)
        text_data_series = data_frame[args.text_column]
    elif args.json_path:
        logging.info("Processing JSON input.")
        text_list = load_data_from_json(
            args.json_path, 
            "abstract_title", 
            "abstract_text", 
        )
        # Convert list of strings to a Pandas Series for the iterator
        text_data_series = pd.Series(text_list)
    else:
        # This case should be caught by argparse mutually exclusive group
        logging.error("No input data path specified.")
        sys.exit(1)
        
    # --- Prepare for Training ---
    batch_iterator_fn = create_batch_iterator(text_data_series, args.batch_size)
    base_tokenizer = load_base_tokenizer(args.base_tokenizer)
    
    # --- Train and Save ---
    new_tokenizer = train_tokenizer(base_tokenizer, batch_iterator_fn, args.vocab_size)
    save_tokenizer(new_tokenizer, args.output_dir)

    logging.info("Tokenizer training process finished.")

if __name__ == "__main__":
    main()
