import os
import json
import h5py
from typing import List, Dict, Any, Optional, Union
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModalBarrettSingleSlide(Dataset):
    """
    A PyTorch dataset for multimodal pathology data, loading text and a single
    WSI embedding per sample. Each sample corresponds to a specific slide.
    """
    def __init__(
        self,
        json_file: str,
        embeddings_file: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
        phase: str = "train",
        val_data_ratio: float = 0.2,
        allowed_labels: List[str] = ["HGD", "LGD", "ND"],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_seq_length
        self.custom_tokenizer = True
        self.embeddings_file = embeddings_file

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings HDF5 file not found: {embeddings_file}")

        self.samples = self._load_and_pair_data(json_file, embeddings_file)
        self.samples.sort(key=lambda x: x["embedding_key"])

        if phase == "train":
            self.samples = self.samples[:int(len(self.samples) * (1 - val_data_ratio))]
        elif phase == "val":
            self.samples = self.samples[-int(len(self.samples) * val_data_ratio):]

        if not self.samples:
            logger.warning("No valid data samples were found after processing.")

        self.allowed_labels = allowed_labels
        self.text_sections = ["Microscopie", "Conclusie", "barrett_label"]

        logging.info(f"Loaded {len(self.samples)} single-slide samples for phase: {phase}.")

    def _load_and_pair_data(self, json_file: str, embeddings_file: str) -> List[Dict[str, Any]]:
        logger.info(f"Loading JSON data from {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text_data = list(data.values()) if isinstance(data, dict) else data

        logger.info(f"Scanning keys from combined HDF5 file: {embeddings_file}")
        with h5py.File(embeddings_file, 'r') as hf:
            embedding_keys_set = set(hf.keys())

        paired_samples = []
        for case in text_data:
            an_number = case.get("original_case", {}).get("An_Number")
            divided_reports = case.get("divided_reports")

            if not isinstance(divided_reports, dict) or not an_number:
                continue

            sub_report_keys = divided_reports.get("barrett_label", {}).keys()
            for roman_numeral in sub_report_keys:
                base_key_prefix = f"{an_number}-{roman_numeral}"
                matching_full_keys = [key for key in embedding_keys_set if key.startswith(base_key_prefix)]

                for full_key in matching_full_keys:
                    paired_samples.append({
                        "case_data": case,
                        "roman_numeral": roman_numeral,
                        "embedding_key": full_key
                    })
        return paired_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        case_data = sample["case_data"]
        roman_numeral = sample["roman_numeral"]
        embedding_key = sample["embedding_key"]

        divided_reports = case_data.get("divided_reports", {})
        barrett_label = divided_reports.get("barrett_label", {}).get(roman_numeral)
        
        if not barrett_label or barrett_label not in self.allowed_labels:
            return self.__getitem__(random.randint(0, len(self) - 1))

        text_parts = []
        for section in self.text_sections:
            section_key = section.lower().replace(" ", "")
            text_content = divided_reports.get(section, {}).get(roman_numeral, "")
            text_parts.append(f"{section_key}: {text_content}")

        text = "\n\n".join(filter(None, text_parts))

        if not text.strip():
            return self.__getitem__(random.randint(0, len(self) - 1))

        tokenized = self.tokenizer(text=text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)

        if self.custom_tokenizer:
            input_ids[0] = 0

        labels = input_ids.clone()

        try:
            with h5py.File(self.embeddings_file, 'r') as hf:
                wsi_embeddings = torch.tensor(hf[embedding_key]['features'][:], dtype=torch.float32)
                if wsi_embeddings.shape[0] == 0:
                    return self.__getitem__(random.randint(0, len(self) - 1))
        except (IOError, KeyError):
            return self.__getitem__(random.randint(0, len(self) - 1))        

        wsi_embeddings = wsi_embeddings.unsqueeze(0)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "wsi_embeddings": wsi_embeddings,
            "case_id": embedding_key,
            "barrett_label": barrett_label
        }

class MultiModalBarrett(Dataset):
    """
    A PyTorch dataset for multimodal pathology data, loading text from a JSON
    file and all corresponding WSI embeddings from a single, combined HDF5 file.
    """
    def __init__(
        self,
        json_file: str,
        embeddings_file: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
        phase: str = "train",
        val_data_ratio: float = 0.2
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_seq_length
        self.custom_tokenizer = True
        self.embeddings_file = embeddings_file

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings HDF5 file not found: {embeddings_file}")

        self.samples = self._load_and_pair_data(json_file, embeddings_file)
        self.samples.sort(key=lambda x: str(x["case_data"].get("original_case", {}).get("An_Number")))

        if phase == "train":
            self.samples = self.samples[:int(len(self.samples) * (1 - val_data_ratio))]
        elif phase == "val":
            self.samples = self.samples[-int(len(self.samples) * val_data_ratio):]

        if not self.samples:
            logger.warning("No valid data samples were found after processing.")

        self.allowed_labels = ["HGD", "LGD", "ND"]
        self.nl_to_eng_sections = {
            "Microscopie": "microscopy", 
            "Conclusie": "conclusion", 
            "barrett_label": "barrett_label", 
            "KlinischeVraagstelling": "clinicalQuery",
            "KlinischeGegevens": "clinicalData",
            "An_Number": "an_number",
            "Diagnose": "diagnosis",
            "report": "report"
        }
        logging.info(f"Loaded {len(self.samples)} samples for phase: {phase}.")

    def _load_and_pair_data(self, json_file: str, embeddings_file: str) -> List[Dict[str, Any]]:
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
                continue
            matching_keys = [key for key in embedding_keys if key.startswith(an_number)]
            if not matching_keys:
                continue
            paired_samples.append({
                "case_data": case,
                "embedding_keys": matching_keys
            })
        return paired_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        case_data = sample["case_data"]
        embedding_keys = sample["embedding_keys"]
       
        report = case_data.get("tcga_structured", [])
        if not report or isinstance(report, str):
            return self.__getitem__(random.randint(0, len(self) - 1))

        text_parts = [f"{self.nl_to_eng_sections[k]}: " + report.get(k, "") for k in ["report"]]
        text = "\n\n".join(filter(None, text_parts))

        if not text.strip():
            return self.__getitem__(random.randint(0, len(self) - 1))

        tokenized = self.tokenizer(text=text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)

        if self.custom_tokenizer:
            input_ids[0] = 0

        labels = input_ids.clone()
        wsi_embeddings_list = []
        try:
            with h5py.File(self.embeddings_file, 'r') as hf:
                for key in embedding_keys:
                    try:
                        embeddings = torch.tensor(hf[key]['features'][:], dtype=torch.float32)
                        if embeddings.shape[0] > 0:
                            wsi_embeddings_list.append(embeddings)
                    except KeyError:
                        pass
        except IOError:
            return self.__getitem__(random.randint(0, len(self) - 1))

        if not wsi_embeddings_list:
            return self.__getitem__(random.randint(0, len(self) - 1))

        wsi_embeddings = torch.vstack(wsi_embeddings_list)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "wsi_embeddings": wsi_embeddings,
            "case_id": case_data.get("original_case", {}).get("An_Number"),
        }


class PubMedTextDataset(Dataset):
    """
    Loads pre-filtered PubMed abstract data from JSON.
    """
    def __init__(
        self,
        json_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
        num_data_points: Optional[int] = None,
        shuffle_on_init: bool = False,
        custom_tokenizer: bool = False, 
    ) -> None:
        super().__init__() 
        
        self.json_path: Path = Path(json_path)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = max_seq_length 
        self.num_data_points: Optional[int] = num_data_points
        self.shuffle: bool = shuffle_on_init
        self.custom_tokenizer: bool = custom_tokenizer

        self.data: List[Dict[str, Any]] = self._load_data()

        if self.shuffle:
            logger.info("Shuffling the loaded dataset during initialization.")
            random.shuffle(self.data)
            
        if not self.data:
             logger.warning(f"Loaded dataset from {self.json_path} is empty!")

    def _load_data(self) -> List[Dict[str, Any]]:
        if not self.json_path.is_file():
            raise FileNotFoundError(f"JSON file not found at: {self.json_path}")

        logger.info(f"Loading PubMed data from: {self.json_path}")
        with open(self.json_path, "r", encoding='utf-8') as f:
            raw_data: Dict[str, Dict[str, Any]] = json.load(f)
        
        loaded_data = []
        for pmid, details in raw_data.items():
            details['pmid'] = pmid 
            loaded_data.append(details)
            
        logger.info(f"Successfully loaded {len(loaded_data)} entries.")

        if self.num_data_points is not None and self.num_data_points > 0:
             return loaded_data[:self.num_data_points]
        return loaded_data 

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.data:
            raise IndexError("Dataset is empty.")
        
        item = self.data[idx]
        abstract = item.get("abstract_text", "")
        text = abstract.strip()

        if not text:
            new_idx = random.randint(0, self.__len__() - 1) if self.shuffle else (idx + 1) % self.__len__()
            return self.__getitem__(new_idx)

        try:
            tokenized = self.tokenizer(
                text=text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length", 
                return_tensors="pt" 
            )
        except Exception:
             new_idx = random.randint(0, self.__len__() - 1) if self.shuffle else (idx + 1) % self.__len__()
             return self.__getitem__(new_idx)

        input_ids = tokenized["input_ids"].squeeze(0)

        if self.custom_tokenizer and input_ids.numel() > 0:
             input_ids[0] = 0 
        elif self.custom_tokenizer:
             new_idx = random.randint(0, self.__len__() - 1) if self.shuffle else (idx + 1) % self.__len__()
             return self.__getitem__(new_idx)

        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
