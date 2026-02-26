
---

# Textile Defect Detection

An automated pipeline for textile surface defect classification using deep learning, featuring MD5-based data purification and GPU acceleration.

## 1. Repository Structure

```text
CHE1148_Defect_Detecting/
├── main.py                     # Master script
├── environment.yml             # Conda environment config
├── best_textile_baseline.pth   # Output: Saved weights of the best performing model
├── README.md                   # Project documentation
└── data/
    ├── raw/
    │   └── textile/            # Source Directory
    │       ├── train64.h5      # Original training images
    │       ├── train64.csv     # Original training labels
    │       ├── test64.h5       # Original testing images
    │       └── test64.csv      # Original testing labels
    └── processed/              # Workflow Outputs
        ├── full64.h5           # Unified HDF5 dataset for the entire project
        ├── full64.csv          # Unified metadata with absolute physical pointers
        ├── duplicates_report.csv # Duplicate groups found
        ├── train_split.csv     # Cleaned training set (Deduplicated)
        ├── val_split.csv       # Cleaned validation set (Stratified from train)
        └── test_split.csv      # Cleaned testing set (Deduplicated)

```

## 2. Environment Setup

Build and activate the environment using Conda:

```bash
conda env create -f environment.yml
conda activate CHE1148_Defect_Detecting

```

## 3. Data Setup

1. Create the folder: `data/raw/textile/`
2. Place your raw Kaggle `.h5` and `.csv` files inside. From: https://www.kaggle.com/datasets/belkhirnacim/textiledefectdetection.
3. The pipeline will automatically handle the creation of the `data/processed/` directory and its contents.

## 4. How to Run

Execute the complete pipeline with a single command:

```bash
python main.py

```

## 5. Key Pipeline Features

* **MD5 Deduplication**: Conducts pixel-level hashing to identify and remove identical images, ensuring no redundant data biases the model.
* **Leakage Prevention**: Validates that no identical images exist in both training and testing splits.
* **TextileBaselineCNN**: A 3-layer Convolutional Neural Network (Conv-BN-ReLU-Pool) designed for surface anomaly extraction.
* **Automatic Early Stopping**: Monitors validation loss with a patience of 7 epochs and automatically restores the best model weights.


---