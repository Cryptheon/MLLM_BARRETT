Tokenizers Documentation
Below is the Markdown source code for the tokenizers/ directory README.

# Tokenizers Directory

This directory manages the training and storage of custom tokenizers used by the PathoLlama and multimodal models. Standard LLM tokenizers (like Llama-3's default) are often suboptimal for pathology-specific terminology; these tools allow for the creation of vocabularies that better represent medical and histological text.

## Directory Structure

- `scripts/`: Python scripts for training new tokenizers from scratch or extending existing ones.
- `trained_tokenizers/`: Serialized tokenizer artifacts compatible with the Hugging Face `transformers` and `tokenizers` libraries.
  - `32768_pubmed/`: A tokenizer trained on filtered PubMed abstracts with a vocabulary size of 32,768.

---

## Training Scripts

### `scripts/train_tokenizer.py`

* **What it does:** Trains a Byte-Level BPE (Byte Pair Encoding) or SentencePiece tokenizer using the `tokenizers` library. It processes raw text files (e.g., the output from `scripts/download_pubmed_abstracts_filtered.py`) to build a vocabulary that captures frequent medical subwords and terms.

* **How to run:**
  ```bash
  python tokenizers/scripts/train_tokenizer.py \
      --data_path ./data/pubmed_corpus.txt \
      --vocab_size 32768 \
      --save_dir ./tokenizers/trained_tokenizers/32768_pubmed \
      --tokenizer_type bpe
  ```

* **What to expect:**
  The script will iterate through the provided text corpus and save a directory containing the following standard files:
  - `tokenizer.json`: The full serialized state of the tokenizer.
  - `tokenizer_config.json`: Configuration settings (padding, truncation, etc.).
  - `special_tokens_map.json`: Mappings for specific tokens like `<|endoftext|>`, `[PAD]`, `[CLS]`, etc.

---

## Usage in Project

When initializing the model or dataset in `src/data/datasets.py` or `experiments/train_lm.py`, you can point to these local directories instead of a Hugging Face Hub ID:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tokenizers/trained_tokenizers/32768_pubmed")
```

## Maintenance

If you update the medical keyword filters in `src/data/keywords/`, it is recommended to re-run the PubMed downloader and subsequently re-train the tokenizer to ensure the vocabulary remains aligned with the latest domain-specific data.
Copy Markdown