import os
import pickle
import json
import h5py
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DummyMultiModalDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        length: int = 128,
        seq_len: int = 32,
        hidden_size: int = 768
    ) -> None:
        self.tokenizer = tokenizer
        self.length = length
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        num_patches = torch.randint(2, 5, (1,)).item()
        wsi_embeddings = torch.randn(num_patches, self.hidden_size)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "wsi_embeddings": wsi_embeddings
        }
        

class MultiModalBarrett(Dataset):
    """
    A PyTorch dataset for multimodal pathology data, loading text from a JSON
    file and all WSI embeddings from a single, combined HDF5 file.
    """
    def __init__(
        self,
        json_file: str,
        embeddings_file: str, 
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
        random_choice_report: bool = False,
        custom_tokenizer: bool = True,
        phase: str = "train",
        val_data_ratio: float = 0.2
    ) -> None:
        """
        Initializes the dataset.

        Args:
            json_file (str): Path to the JSON file with text reports.
            embeddings_file (str): Path to the single combined HDF5 file.
            tokenizer (PreTrainedTokenizer): The tokenizer for text processing.
            max_seq_length (int): The maximum sequence length for tokenized text.
            random_choice_report (bool): If True, randomly selects one of the
                                         translated reports. Otherwise, uses the first.
            custom_tokenizer (bool): Flag to handle special tokens for a custom tokenizer.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_seq_length
        self.random_choice_report = random_choice_report
        self.custom_tokenizer = custom_tokenizer
        self.embeddings_file = embeddings_file

        # Validate file paths
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings HDF5 file not found: {embeddings_file}")

        # Load and process data
        self.samples = self._load_and_pair_data(json_file, embeddings_file)
        self.samples.sort()

        if phase=="train":
            self.samples = self.samples[:int(len(self.samples)*(1-val_data_ratio))]

        elif phase=="val":
            self.samples = self.samples[-int(len(self.samples)*val_data_ratio):]

        if not self.samples:
            logger.warning("No valid data samples were found after processing.")

    def _load_and_pair_data(self, json_file: str, embeddings_file: str) -> List[Dict[str, Any]]:
        """Loads data from JSON and pairs it with keys from the combined HDF5 file."""
        logger.info(f"Loading JSON data from {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        logger.info(f"Scanning keys from combined HDF5 file: {embeddings_file}")
        with h5py.File(embeddings_file, 'r') as hf:
            embedding_keys = list(hf.keys())
        
        paired_samples = []
        for case in text_data:
            an_number = case.get("original_case", {}).get("An_Number")
            if not an_number:
                logger.warning("Skipping case due to missing 'An_Number'.")
                continue

            # Find all matching embedding keys for the An_Number
            matching_keys = [key for key in embedding_keys if key.startswith(an_number)]
            
            if not matching_keys:
                logger.warning(f"No embedding keys found for An_Number: {an_number}")
                continue

            for key in matching_keys:
                paired_samples.append({
                    "case_data": case,
                    "embedding_key": key # Store the key instead of a path
                })
        
        logger.info(f"Successfully paired {len(paired_samples)} text-embedding samples.")
        return paired_samples

    def __len__(self) -> int:
        """Returns the total number of paired samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single paired sample of text and WSI embeddings.
        Opens the H5 file here to be compatible with multi-worker DataLoader.
        """
        sample = self.samples[idx]
        case_data = sample["case_data"]
        embedding_key = sample["embedding_key"]

        # 1. Process Text (same as before)
        translated_reports = case_data.get("translated_reports", [])
        if not translated_reports:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        report = random.choice(translated_reports) if self.random_choice_report else translated_reports[0]
        text_parts = [report.get(k, "") for k in ["KlinischeGegevens", "Macroscopie", "Microscopie", "Conclusie", "Diagnose"]]
        text = "\n".join(filter(None, text_parts))

        if not text:
            return self.__getitem__(random.randint(0, len(self) - 1))

        tokenized = self.tokenizer(text=text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)

        if self.custom_tokenizer and input_ids.numel() > 0:
            input_ids[0] = 0

        labels = input_ids.clone()

        # 2. Load WSI Embeddings from the single H5 file
        try:
            with h5py.File(self.embeddings_file, 'r') as hf:
                # Access the group using the key, then the 'features' dataset
                wsi_embeddings = torch.tensor(hf[embedding_key]['features'][:], dtype=torch.float32)
        except Exception as e:
            logger.error(f"Failed to load embeddings for key '{embedding_key}': {e}. Resampling.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        if wsi_embeddings.shape[0] == 0:
            logger.warning(f"Empty embeddings for key '{embedding_key}'. Resampling.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        if wsi_embeddings.shape[0] > self.max_length:
            wsi_embeddings = wsi_embeddings[:self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "wsi_embeddings": wsi_embeddings,
            "case_id": case_data.get("original_case", {}).get("An_Number")
        }

class PathoMultiModalDataset(Dataset):
    def __init__(
        self,
        pickle_file: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
        embeddings_dim_size: int = 768,
        random_choice_report: bool = False,
        custom_tokenizer: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # max length refers to max length of the whole
        # sequence (WSI embeddings + textual embeddings)
        self.max_length = max_seq_length
        self.hidden_size = embeddings_dim_size
        self.random_choice_report = random_choice_report
        self.custom_tokenizer = custom_tokenizer

        if not os.path.exists(pickle_file):
            raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

        self.patient_ids = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self.patient_ids[idx]
        patient = self.data[patient_id]
        text_variations: str = patient["reports"]
        text_variations = eval(text_variations)

        if self.random_choice_report:
            text: str = random.choice(text_variations)
        else:
            text = text_variations[0]

        wsi_embeddings = torch.tensor(np.array(patient["embeddings"])) 
        
        if text is None or len(wsi_embeddings)==0 :
            # Resample if shuffling, otherwise get next sequential
            new_idx = random.randint(0, len(self.patient_ids) - 1) if getattr(self, "shuffle", True) else (idx + 1) % len(self)
            return self.__getitem__(new_idx)

        # Truncate to max number of embeddings if needed.
        # Although, we can skip this as we're not exceeding the 
        # max length with only WSI embeddings.
        if len(wsi_embeddings) > self.max_length:
            wsi_embeddings = wsi_embeddings[:self.max_length]

        # Tokenize text
        tokenized = self.tokenizer(
            text=text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        #torch.set_printoptions(threshold=10_000)

        input_ids = tokenized["input_ids"].squeeze(0)  # remove batch dim
        # If we trained our own tokenizer based on Llama's we need to patch the first BOS token
        # Somehow Llama's tokenizer persists with index 128000 as starting token
        if self.custom_tokenizer:
            # <|begin_of_text|> token is 0th index
            input_ids[0] = 0

        # We clone the labels without shifting it for CausalLM.
        # We explicitly construct the shifted labels for next token prediction in the model
        # The dataset should return the cloned input ids only for flexiblity.
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "wsi_embeddings": wsi_embeddings
        }


class PubMedTextDataset(Dataset):
    """
    Loads pre-filtered PubMed abstract data from JSON, tokenizes text,
    and prepares input_ids and labels for language model training,
    following the style of PathoMultiModalDataset.
    """
    def __init__(
        self,
        json_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
        num_data_points: Optional[int] = None,
        shuffle_on_init: bool = False, # Renamed for clarity, shuffle happens once on load
        custom_tokenizer: bool = False, 
    ) -> None:
        """
        Initializes the PubMedTextDataset dataset loader.

        Args:
            json_path (Union[str, Path]): Path to the JSON file containing filtered PubMed data.
                                           Expected format: {pmid: {"year":..., "abstract_text":..., "abstract_title":...}}.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for text processing.
            max_seq_length (int): The maximum total sequence length after tokenization.
                                  Sequences longer than this will be truncated. Defaults to 1024.
            num_data_points (Optional[int]): Maximum number of data points to load.
                                             If None, loads all data points found in the file. Defaults to None.
            shuffle_on_init (bool): If True, shuffles the loaded data points once during initialization.
                                    This affects the order accessed by index but __getitem__ resampling
                                    logic uses its own randomization if needed. Defaults to False.
            custom_tokenizer (bool): If True, applies specific adjustments assuming a custom tokenizer
                                     (e.g., patching BOS token). Defaults to False.
        """
        super().__init__() 
        
        self.json_path: Path = Path(json_path)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = max_seq_length 
        self.num_data_points: Optional[int] = num_data_points
        self.shuffle: bool = shuffle_on_init
        self.custom_tokenizer: bool = custom_tokenizer

        self.data: List[Dict[str, Any]] = self._load_data()

        # Shuffle the data *after* loading and potential slicing if requested
        if self.shuffle:
            logger.info("Shuffling the loaded dataset during initialization.")
            random.shuffle(self.data)
            
        if not self.data:
             logger.warning(f"Loaded dataset from {self.json_path} is empty!")


    def _load_data(self) -> List[Dict[str, Any]]:
        """Loads PubMed data from the specified JSON file."""
        if not self.json_path.is_file():
            logger.error(f"JSON file not found at: {self.json_path}")
            raise FileNotFoundError(f"JSON file not found at: {self.json_path}")

        logger.info(f"Loading PubMed data from: {self.json_path}")
        
        with open(self.json_path, "r", encoding='utf-8') as f:
            # Data is stored as {pmid: {details}}
            raw_data: Dict[str, Dict[str, Any]] = json.load(f)
        
        # Convert the dictionary values into a list of dictionaries
        # Add the pmid back into the dictionary for potential future use
        loaded_data = []
        for pmid, details in raw_data.items():
            details['pmid'] = pmid 
            loaded_data.append(details)
            
        logger.info(f"Successfully loaded {len(loaded_data)} entries.")

        # Apply num_data_points limit if specified
        if self.num_data_points is not None and self.num_data_points > 0:
            if self.num_data_points < len(loaded_data):
                logger.info(f"Limiting dataset to the first {self.num_data_points} entries (before potential shuffle).")
                return loaded_data[:self.num_data_points]
            else:
                logger.warning(f"num_data_points ({self.num_data_points}) is >= total loaded data ({len(loaded_data)}). Using all loaded data.")
                return loaded_data
        else:
            return loaded_data 

    def __len__(self) -> int:
        """Returns the total number of loaded data points."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves an item by index, processes it, and returns tokenized data.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "input_ids": Token IDs (shape: [max_seq_length])
                - "labels": Labels for language modeling, identical to input_ids
                            (shape: [max_seq_length])
        Raises:
            IndexError: If the index is out of bounds and dataset is empty.
            KeyError: If expected keys ('abstract_text', 'abstract_title') are missing in source data.
        """
        if not self.data:
            raise IndexError("Dataset is empty, cannot retrieve item.")
            
        if not 0 <= idx < len(self.data):
             raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.data)}")

        item = self.data[idx]
        
        title = item.get("abstract_title", "")
        abstract = item.get("abstract_text", "")
  
        text = abstract.strip()

        # Resample if text is essentially empty
        if not text:
            pmid = item.get("pmid", "N/A")
            logger.warning(f"Empty text content for index {idx} (PMID: {pmid}). Resampling...")
            # Use self.shuffle attribute to decide resampling strategy
            # Note: self.shuffle refers to shuffle_on_init here. If True, random sample.
            # If False, sequential sample (consistent with PathoMultiModalDataset example)
            new_idx = random.randint(0, self.__len__() - 1) if self.shuffle else (idx + 1) % self.__len__()
            return self.__getitem__(new_idx)

        try:
            # Ensure text is not None before passing to tokenizer
            text_to_tokenize = text if text is not None else ""
            
            tokenized = self.tokenizer(
                text=text_to_tokenize,
                truncation=True,
                max_length=self.max_length,
                padding="max_length", # Pad to max_length
                return_tensors="pt"  # Return PyTorch tensors
            )
        except Exception as e:
             pmid = item.get("pmid", "N/A")
             logger.error(f"Tokenization failed for index {idx} (PMID: {pmid}): {e}. Text snippet: '{text_to_tokenize[:100]}...'. Resampling.")
             new_idx = random.randint(0, self.__len__() - 1) if self.shuffle else (idx + 1) % self.__len__()
             return self.__getitem__(new_idx)


        input_ids = tokenized["input_ids"].squeeze(0)

        if self.custom_tokenizer:
            # Assuming 0 is the index for <|begin_of_text|> or equivalent BOS
            if input_ids.numel() > 0: # Check if tensor is not empty
                 input_ids[0] = 0 # Adjust BOS token ID if necessary
            else:
                 logger.warning(f"Empty input_ids tensor after tokenization for index {idx}. Resampling.")
                 new_idx = random.randint(0, self.__len__() - 1) if self.shuffle else (idx + 1) % self.__len__()
                 return self.__getitem__(new_idx)

        # For standard Causal LM training, labels are typically the same as input_ids
        # (model handles shifting internally or in the training loop)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }