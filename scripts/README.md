# Data Inspection Utilities

This repository contains a collection of Python scripts designed to inspect and understand common data formats used in computational pathology and general data archiving. These tools help you quickly view the contents, structure, and metadata of complex files like HDF5 (`.h5`) and Tape Archives (`.tar`).

## Requirements

These scripts are written in Python 3. To install all the necessary libraries, run the following command in your terminal:

```bash
pip install numpy torch h5py huggingface-hub pillow humanize
```

## Scripts

### 1. HDF5 File Inspector (`inspect_tcga_titan_demo_data.py`)

#### Purpose

This script is designed to download and/or inspect HDF5 (`.h5`) files, specifically those structured for computational pathology workflows like `MahmoodLab/TITAN`. These files typically store patch-based features, coordinates, tissue masks, and other metadata extracted from whole-slide images (WSI).

#### Usage

There are two primary ways to use this script:

##### A) Inspect the Default Demo File

Simply run the script from your terminal without any arguments. It will download the TCGA demo file from Hugging Face and print its inspection report.

```bash
python inspect_tcga_titan_demo_data.py
```

#### Example Output

```
==================================================
INSPECTING H5 FILE: /home/user/.cache/huggingface/hub/models--MahmoodLab--TITAN/snapshots/.../TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5
==================================================

Keys found in H5 file: ['coords', 'features']

--- Inspecting Key: 'coords' ---
Shape: (4675, 2)
Data Type (dtype): int64
Sample Data (first 5 rows):
[[24576 53248]
 [25088 53248]
 [25600 53248]
 [26112 53248]
 [26624 53248]]

------------------------------

--- Inspecting Key: 'features' ---
Shape: (4675, 768)
Data Type (dtype): float32
Sample Data (first 5 rows):
[[ 0.04690327 -0.1245088   0.2789139  ... -0.01234567  0.156789   -0.24581234]
 [ 0.13245678 -0.05891234  0.31124567 ...  0.07890123  0.05432109 -0.19876543]
 ...]

------------------------------
```

---

### 2. TAR Archive Inspector (`inspect_tar.py`)

#### Purpose

This script provides a detailed look inside Tape Archive (`.tar`) files. It lists all contained files and directories and gives special attention to `.png` images, inspecting their properties without extracting them to the hard drive. This is useful for quickly verifying the contents of large archives.

#### Usage

This script is intended to be run from the command line. You must provide the path to the `.tar` file you wish to inspect using the `--tar_path` argument.

**Basic Command:**

```bash
python inspect_tar.py --tar_path /path/to/your/archive.tar
```

**Example:**

```bash
python inspect_tar.py --tar_path LMM/RL-0006.tar
```

#### Example Output

```
--- Inspecting Archive: LMM/RL-0006.tar ---
Name                                               | Type       | Size
--------------------------------------------------------------------------------
RL-0006/                                           | Directory  | 0 B
RL-0006/RL-0006_23.png                             | File       | 1.2 MB
 └─ Inspecting PNG details...
   ├─ Format: PNG
   ├─ Dimensions: 4096x4096 pixels
   ├─ Color Mode: RGB
   └─ Metadata:
       - dpi: (96, 96)
       - creation_time: 2024-10-28T14:30:00Z
RL-0006/RL-0006_24.png                             | File       | 987.5 kB
 └─ Inspecting PNG details...
   ├─ Format: PNG
   ├─ Dimensions: 4096x4096 pixels
   ├─ Color Mode: L
   └─ No metadata found.
RL-0006/README.txt                                 | File       | 256 B

--- End of Archive ---
```

---
