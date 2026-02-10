# CHE1148 Textile Defect Detecting

Team project repo for textile surface defect detection (proxy for SEM/TEM surface anomalies).

## Project Structure
CHE1148_Defect_Detecting/
  main.py
  environment.yml
  .gitignore
  data/ # ignored by git (local only)
    raw/
      textile/ # put downloaded .h5 here
    processed/ # generated folders/images go here

## Setup (Conda)

## Create environment:
conda env create -f environment.yml
conda activate CHE1148_Defect_Detecting

## (If needed) install basic deps
conda install -y numpy h5py pillow matplotlib scikit-learn tqdm

## Dataset Placement
Put Kaggle .h5 file(s) here:

data/raw/textile/
  matchingtDATASET_test_32.h5
  (optional) matchingtDATASET_train_32.h5

## Quick Check (H5 Readability)
Run main.py after adding a small H5 inspection snippet (print top-level keys and one sample shape).
If it prints class keys + dataset shape without errors, the dataset is ready.

## Git Notes
Do not commit data/ (it is ignored by .gitignore).
Commit code + environment.yml so teammates can reproduce the same environment.