````md
# CHE1148 Textile Defect Detection (Textile → SEM/TEM Proxy)

This repo contains our CHE1148 team project on **surface defect detection** using a textile defect dataset as a proxy for **SEM/TEM-like** surface anomalies.

---

## Repository Layout

```text
CHE1148_Defect_Detecting/
├─ main.py
├─ environment.yml
├─ README.md
├─ .gitignore
└─ data/                      # local only (ignored by git)
   ├─ raw/
   │  └─ textile/             # put downloaded .h5 files here
   └─ processed/              # generated folders/images go here
````

> **Important:** `data/` is **not** pushed to GitHub. Each teammate stores the dataset locally.

---

## Setup (Conda)

### 1) Create the environment

From the project root:

```bash
conda env create -f environment.yml
conda activate CHE1148_Defect_Detecting
```

### 2) Install basic dependencies (if needed)

These are used for H5 inspection / preprocessing:

```bash
conda install -y numpy h5py pillow matplotlib scikit-learn tqdm
```

---

## Dataset Placement

Put Kaggle `.h5` file(s) here:

```text
data/raw/textile/
├─ matchingtDATASET_test_32.h5
└─ matchingtDATASET_train_32.h5   (if available)
```

---

## Quick Check (H5 Readability)

Run a small H5 inspection script (in `main.py`) to confirm:

* the file opens without error
* top-level keys (classes) are readable
* a sample dataset has a valid shape (e.g., `(N, 32, 32, 1)`)

If this works, the dataset is ready for preprocessing and training.

---

## Git Workflow Notes

* **Do not commit** `data/` (ignored by `.gitignore`).
* Commit only code/config files (e.g., `main.py`, `environment.yml`, `README.md`).
* Use branches + PRs for collaboration when possible.

---