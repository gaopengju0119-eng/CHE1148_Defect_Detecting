# %% [markdown]
# ### Project: Textile Defect Detection Baseline
# **Task:** End-to-end pipeline including data merging, MD5-based deduplication,
# stratified splitting, and CNN training with Early Stopping.

import copy
import hashlib
# %% [markdown]
# ### 1. Imports and Configuration
import os
from collections import defaultdict
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Disable HDF5 file locking for better compatibility
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Path configuration
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "textile"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# File paths
TRAIN_H5, TRAIN_CSV = RAW / "train64.h5", RAW / "train64.csv"
TEST_H5, TEST_CSV = RAW / "test64.h5", RAW / "test64.csv"
OUT_H5, OUT_CSV = PROCESSED / "full64.h5", PROCESSED / "full64.csv"

# Device setup
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    try:
        import torch_directml

        device = torch_directml.device()
    except ImportError:
        pass


# %% [markdown]
# ### 2. Data Merging and Preprocessing
def merge_data():
    """Merge separate train/test H5 and CSV files into a unified dataset."""
    if OUT_H5.exists() and OUT_CSV.exists():
        print("Dataset already merged. Skipping.")
        return

    # Merge CSVs
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    df_train['original_split'] = 'train'
    df_test['original_split'] = 'test'
    full_df = pd.concat([df_train, df_test], ignore_index=True)
    full_df.to_csv(OUT_CSV, index=False)
    print(f"Saved merged CSV: {OUT_CSV}")

    # Merge H5s
    with h5py.File(OUT_H5, 'w') as f_out:
        with h5py.File(TRAIN_H5, 'r') as f_tr, h5py.File(TEST_H5, 'r') as f_te:
            tr_imgs = f_tr['images']
            te_imgs = f_te['images']
            total_shape = (tr_imgs.shape[0] + te_imgs.shape[0], *tr_imgs.shape[1:])
            dset = f_out.create_dataset('images', shape=total_shape, dtype='f')
            dset[:tr_imgs.shape[0]] = tr_imgs[:]
            dset[tr_imgs.shape[0]:] = te_imgs[:]
    print(f"Saved merged H5: {OUT_H5}")


# %% [markdown]
# ### 3. Integrity and Deduplication Analysis
def get_h5_hashes(h5_path, total_images, chunk_size=5000):
    """Generate MD5 fingerprints for all images in the H5 file."""
    hashes = [None] * total_images
    print(f"Generating MD5 fingerprints for {total_images} images...")
    with h5py.File(h5_path, 'r') as f:
        for start in range(0, total_images, chunk_size):
            end = min(start + chunk_size, total_images)
            chunk = f['images'][start:end]
            for i, img in enumerate(chunk):
                h = hashlib.md5(img.tobytes()).hexdigest()
                hashes[start + i] = h
    return hashes


def analyze_duplicates():
    """Identify duplicate images and check for data leakage."""
    df = pd.read_csv(OUT_CSV)
    with h5py.File(OUT_H5, "r") as f:
        total = f["images"].shape[0]

    all_hashes = get_h5_hashes(OUT_H5, total)
    hash_map = defaultdict(list)
    for idx, h in enumerate(all_hashes):
        hash_map[h].append(idx)

    duplicate_indices = [idx for indices in hash_map.values() if len(indices) > 1 for idx in indices]

    if duplicate_indices:
        report_df = df.iloc[duplicate_indices].copy()
        report_df.to_csv(PROCESSED / "duplicates_report.csv", index=False)
        leakage = report_df.groupby('index')['original_split'].nunique()
        if (leakage > 1).any():
            print("[WARNING] Data leakage detected across splits!")
        else:
            print("[SAFE] No leakage found among duplicates.")
    return all_hashes


# %% [markdown]
# ### 4. Dataset Splitting with Per-Split Deduplication
def create_clean_split(all_hashes):
    """Split dataset and remove internal duplicates from train and test sets."""
    df = pd.read_csv(OUT_CSV)
    df['abs_ptr'] = range(len(df))
    df['md5'] = all_hashes

    # Separate original train and test portions
    tr_df = df[df['original_split'] == 'train'].copy()
    te_df = df[df['original_split'] == 'test'].copy()

    # Deduplicate within each portion
    tr_df = tr_df.drop_duplicates(subset='md5', keep='first')
    te_df = te_df.drop_duplicates(subset='md5', keep='first')

    # Stratified Split (Training -> Train/Val)
    unique_indices = tr_df['index'].unique()
    strat_labels = tr_df.drop_duplicates('index')['indication_type']

    train_idx, val_idx = train_test_split(
        unique_indices, test_size=0.1, random_state=42, stratify=strat_labels
    )

    df_train = tr_df[tr_df['index'].isin(train_idx)].sample(frac=1, random_state=42)
    df_val = tr_df[tr_df['index'].isin(val_idx)].sample(frac=1, random_state=42)

    df_train.to_csv(PROCESSED / "train_split.csv", index=False)
    df_val.to_csv(PROCESSED / "val_split.csv", index=False)
    te_df.to_csv(PROCESSED / "test_split.csv", index=False)

    print(f"Datasets finalized: Train({len(df_train)}), Val({len(df_val)}), Test({len(te_df)})")


# %% [markdown]
# ### 5. PyTorch Dataset and Model Definition
class TextileDataset(Dataset):
    def __init__(self, csv_path, h5_path):
        self.df = pd.read_csv(csv_path)
        self.h5_path = h5_path

        # Define a fixed mapping for the 6 defect categories
        # This ensures 'metal_contamination' consistently becomes an integer
        self.label_map = {
            'good': 0,
            'hole': 1,
            'metal_contamination': 2,
            'oil_spot': 3,
            'thread': 4,
            'wrinkle': 5
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with h5py.File(self.h5_path, 'r') as f:
            # Use abs_ptr to fetch the specific image from merged H5
            img = f['images'][int(row['abs_ptr'])]

        # Image Preprocessing
        img = torch.from_numpy(img).float()
        if img.max() > 1.0:
            img /= 255.0

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.shape[-1] == 1:
            img = img.permute(2, 0, 1)

        # --- FIX: Handle string labels ---
        label_raw = row['indication_type']
        if isinstance(label_raw, str):
            # Convert string name to integer index using our map
            label = self.label_map.get(label_raw, 0)
        else:
            # If it's already a number, just ensure it's an int
            label = int(label_raw)

        return img, torch.tensor(label, dtype=torch.long)


class TextileBaselineCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# %% [markdown]
# ### 6. Training Utilities and Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved. Saving model weights.")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def run_step(model, loader, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


# %% [markdown]
# ### 7. Main Execution Flow
if __name__ == "__main__":
    # Parameters
    cfg = {"batch": 64, "lr": 0.001, "epochs": 30, "patience": 7}

    # Pipeline
    merge_data()
    hashes = analyze_duplicates()
    create_clean_split(hashes)

    # Loaders
    train_ds = TextileDataset(PROCESSED / "train_split.csv", OUT_H5)
    val_ds = TextileDataset(PROCESSED / "val_split.csv", OUT_H5)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch"])

    # Initialization
    model = TextileBaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], foreach=False)
    early_stop = EarlyStopping(patience=cfg["patience"])

    print(f"Starting training on: {device}")
    for epoch in range(cfg["epochs"]):
        t_loss, t_acc = run_step(model, train_loader, criterion, optimizer, device, True)
        v_loss, v_acc = run_step(model, val_loader, criterion, optimizer, device, False)

        print(
            f"Epoch [{epoch + 1:02d}] | Train: {t_acc:.2f}% (Loss: {t_loss:.4f}) | Val: {v_acc:.2f}% (Loss: {v_loss:.4f})")

        early_stop(v_loss, model)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            model.load_state_dict(early_stop.best_model_state)
            break

    torch.save(model.state_dict(), "best_textile_baseline.pth")
    print("Training Complete.")