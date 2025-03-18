# TCGA Data Inspection Scripts

This repository contains two scripts for inspecting and analyzing TCGA dataset embeddings and clinical reports. These scripts help ensure data integrity and provide insights into the dataset's structure and statistics.

## Scripts Overview

### 1. `check_one_report_to_many_embeddings_tcga.py`
This script verifies whether multiple embeddings exist for the same clinical report at the patient level. It helps ensure that each patient report corresponds to a single embedding.

#### Functionality:
- Loads embeddings from a `.pkl` file and clinical reports from a `.csv` file.
- Extracts patient IDs from filenames.
- Identifies patients with multiple embeddings linked to the same report.
- Displays a summary of patients with multiple embeddings, if any.

#### Running the script:
```sh
python data_inspection_scripts/check_one_report_to_many_embeddings_tcga.py \
    --embeddings_file_path ./tcga_titan_features/TCGA_TITAN_features.pkl \
    --reports_file_path ./tcga_reports/TCGA_Reports.csv
```

#### Example Output:
```
No patients have multiple embeddings for the same clinical report.
```

### 2. `inspect_tcga_titan_features.py`
This script provides insights into TCGA TITAN feature embeddings, including dataset statistics and visualization using t-SNE.

#### Functionality:
- Loads embeddings from a `.pkl` file.
- Displays dataset information, including the number of entries and embedding shape.
- Prints sample data details.
- Computes global mean and standard deviation of the embeddings.
- Generates a t-SNE visualization of the embeddings and saves it as an image.

#### Running the script:
```sh
python data_inspection_scripts/inspect_tcga_titan_features.py \
    --file_path ./tcga_titan_features/TCGA_TITAN_features.pkl
```

#### Example Output:
```
Dataset Information:
- Number of entries: 11658
- Shape of embedding matrix: (11658, 768)

Sample Data:
Filename: TCGA-06-1087-01Z-00-DX2.1f91f05a-f277-4c98-9955-37e0c83b745f
Embeddings: (768,)

Data Statistics:
- Global Mean: 0.006781017407774925
- Global Std Dev: 0.5708754658699036

t-SNE visualization saved to tsne_visualization.png
```

#### Custom Output Path for t-SNE Visualization:
```sh
python data_inspection_scripts/inspect_tcga_titan_features.py \
    --file_path ./tcga_titan_features/TCGA_TITAN_features.pkl \
    --output_path custom_tsne_plot.png
```



