# TCGA Data Processing Scripts

This repository contains two scripts for merging and inspecting TCGA clinical reports and image embeddings, ensuring they are correctly structured and validated.

## Scripts Overview

### 1. `merge_embeddings_reports_files.py`
This script merges TCGA clinical reports with image embeddings, grouping data by patient.

#### Functionality:
- Loads TCGA embeddings from a `.pkl` file and clinical reports from a `.csv` file.
- Extracts patient IDs from filenames.
- Groups embeddings per patient while keeping a single clinical report.
- Saves the merged dataset as a `.pkl` file.

#### Running the script:
```sh
python ./data_utils_scripts/merge_embeddings_reports_files.py \
  --embeddings_file_path ./tcga_data/tcga_titan_features/TCGA_TITAN_features.pkl \
  --reports_file_path ./tcga_data/tcga_reports/tcga_processed_reports.csv \
  --output_file_path tcga_titan_embeddings_reports.pkl \
  --report_text_column processed_report
```

#### Output:
- `tcga_titan_embeddings_reports.pkl`: A pickle file where each key is a patient ID, and values contain:
  - Report filename and text
  - List of embeddings and corresponding filenames

---

### 2. `read_merged_tcga_pkl.py`
This script reads and inspects the merged dataset to verify correctness.

#### Functionality:
- Loads the merged dataset from a `.pkl` file.
- Displays one patient's report and associated embeddings.

#### Running the script:
```sh
python data_utils_scripts/read_merged_tcga_pkl.py --merged_file_path tcga_titan_embeddings_reports.pkl
```

#### Example Output:
```
First Patient in Merged Dataset: TCGA-3L-AA1B
Report Filename: TCGA-3L-AA1B.36121b4d-6dde-4223-9f7a-5ca50ad4f7b5
Report Text: Patient history with left temporal lesion biopsied...
Total Embeddings: 2
  Embedding 1: TCGA-3L-AA1B-01Z-00-DX2.17CE3683, Shape: (768,)
  Embedding 2: TCGA-3L-AA1B-01Z-00-DX1.8923A1, Shape: (768,)
```


### 3. `split_data_train_val.py`
This script splits the merged dataset into training and validation sets.

#### Functionality:
- Loads the merged `.pkl` dataset.
- Randomly selects 200 patients for validation.
- Saves the remaining patients as the training set.
- Outputs two new `.pkl` files: one for training and one for validation.

#### Running the script:
```sh
python data_utils_scripts/split_data_train_val.py \
  --merged_file_path tcga_titan_embeddings_reports.pkl \
  --train_file_path tcga_train.pkl \
  --val_file_path tcga_val.pkl
```


