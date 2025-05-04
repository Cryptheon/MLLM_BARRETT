import os
import pickle
import json
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
        self.max_length: int = max_seq_length # max_length now refers to token length
        self.num_data_points: Optional[int] = num_data_points
        self.shuffle: bool = shuffle_on_init # Store shuffle setting for resampling logic
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

        # Example: Patching the first token if using a specific custom Llama tokenizer
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