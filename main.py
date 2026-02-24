# %% [markdown]
# ### Data Merging Pipeline (Refined)
# **Task:** Merge train/test H5 and CSV files while preserving original split information.
# **Key Changes:**
# 1. Renamed original 'split' column to 'original_split'.
# 2. Removed 'global_index' and 'source_split' to maintain the original column count.
# **Outputs:** `full64.h5` and `full64.csv` in the `data/processed` directory.

# %% Data Preprocessing Implementation
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

# Path configuration using pathlib for cross-platform compatibility
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "textile"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Define source file paths
TRAIN_H5 = RAW / "train64.h5"
TRAIN_CSV = RAW / "train64.csv"
TEST_H5 = RAW / "test64.h5"
TEST_CSV = RAW / "test64.csv"

# Define output file paths
OUT_H5 = OUT / "full64.h5"
OUT_CSV = OUT / "full64.csv"

DSET = "images"      # Dataset name inside the H5 file
CHUNK = 2048         # Processing batch size to manage memory usage


def merge_csv():
    """
    Merges metadata CSV files.
    Renames the 'split' column to 'original_split' and preserves all original data
    without adding new columns or indices.
    """
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)

    # Concatenate the dataframes
    full = pd.concat([tr, te], ignore_index=True)

    # Rename 'split' to 'original_split' to track origin while keeping column count constant
    full = full.rename(columns={"split": "original_split"})

    # Save to CSV without the pandas default index column
    full.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)


def merge_h5():
    """
    Merges H5 image data using a chunking strategy to prevent memory overflow.
    Verifies that image dimensions and data types match before merging.
    """
    with h5py.File(TRAIN_H5, "r") as ftr, h5py.File(TEST_H5, "r") as fte:
        xtr = ftr[DSET]
        xte = fte[DSET]

        # Integrity check: Ensure spatial dimensions and dtypes match
        if xtr.shape[1:] != xte.shape[1:]:
            raise ValueError(f"Shape mismatch: train {xtr.shape} vs test {xte.shape}")
        if xtr.dtype != xte.dtype:
            raise ValueError(f"Dtype mismatch: train {xtr.dtype} vs test {xte.dtype}")

        ntr, nte = xtr.shape[0], xte.shape[0]
        ntotal = ntr + nte

        with h5py.File(OUT_H5, "w") as fout:
            # Create an empty dataset with the combined total shape
            y = fout.create_dataset(DSET, shape=(ntotal, *xtr.shape[1:]), dtype=xtr.dtype)

            # Copy training data in chunks
            for i in range(0, ntr, CHUNK):
                j = min(i + CHUNK, ntr)
                y[i:j] = xtr[i:j]

            # Copy test data in chunks with an offset
            offset = ntr
            for i in range(0, nte, CHUNK):
                j = min(i + CHUNK, nte)
                y[offset + i: offset + j] = xte[i:j]

    print("Saved:", OUT_H5)


if __name__ == "__main__":
    merge_csv()
    merge_h5()
    print("Done. Combined dataset created successfully.")

# %% End of merging cell

# %% [markdown]
# ### Dataset Integrity and Duplicate Check
# **Task:** Verify the merged `full64.h5` file, count total images, and detect identical duplicates.
# **Method:** Uses MD5 hashing to efficiently compare image data without overloading memory.

# %% Integrity Check Implementation
from pathlib import Path

# Path configuration (must match your previous merging script)
ROOT = Path(__file__).resolve().parent
OUT_H5 = ROOT / "data" / "processed" / "full64.h5"
DSET = "images"
CHUNK_SIZE = 1000  # Number of images to process at a time to save RAM


def check_h5_integrity():
    if not OUT_H5.exists():
        print(f"Error: File not found at {OUT_H5}")
        return

    print(f"Opening file: {OUT_H5}")
    try:
        with h5py.File(OUT_H5, "r") as f:
            if DSET not in f:
                print(f"Error: Dataset '{DSET}' not found in the H5 file.")
                return

            dataset = f[DSET]
            total_images = dataset.shape[0]
            img_shape = dataset.shape[1:]

            print(f"--- Dataset Statistics ---")
            print(f"Total Number of Images: {total_images}")
            print(f"Image Dimensions: {img_shape}")
            print(f"Data Type: {dataset.dtype}")
            print("-" * 30)

            # Duplicate Check using MD5 Hashing
            print("Scanning for duplicate images (processing in chunks)...")
            hashes = set()
            duplicate_count = 0

            for i in range(0, total_images, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, total_images)
                batch = dataset[i:end]

                for img in batch:
                    # Convert the image array to bytes and generate a unique hash
                    img_hash = hashlib.md5(img.tobytes()).hexdigest()

                    if img_hash in hashes:
                        duplicate_count += 1
                    else:
                        hashes.add(img_hash)

            if duplicate_count == 0:
                print(f"Verified: No duplicates found. All {total_images} images are unique.")
            else:
                print(f"Result: Found {duplicate_count} duplicate images.")
                print(f"Total Unique Images: {len(hashes)}")

    except Exception as e:
        print(f"Integrity check failed: {e}")


if __name__ == "__main__":
    check_h5_integrity()

# %% End of Integrity Check Cell

# %% [markdown]
# ### Duplicate Images Origin Analysis
# **Task:** Identify the 391 duplicates and export their metadata to CSV.
# **Purpose:** Check if duplicates cross the Train/Test boundary (Data Leakage).

# %% Implementation
import hashlib
from pathlib import Path
from collections import defaultdict

# Configuration
ROOT = Path(__file__).resolve().parent
H5_PATH = ROOT / "data" / "processed" / "full64.h5"
CSV_PATH = ROOT / "data" / "processed" / "full64.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "duplicates_report.csv"


def analyze_duplicates():
    # 1. Load Metadata
    print("Loading metadata...")
    df = pd.read_csv(CSV_PATH)

    # 2. Scan H5 for hashes
    print("Scanning H5 images for duplicates (this may take a minute)...")
    hash_map = defaultdict(list)  # Stores {md5_hash: [list_of_row_indices]}

    with h5py.File(H5_PATH, "r") as f:
        images = f["images"]
        total = images.shape[0]

        # We process in small batches to save memory
        chunk_size = 1000
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            batch = images[start:end]

            for i, img in enumerate(batch):
                global_idx = start + i
                # Generate unique fingerpint for the image
                img_hash = hashlib.md5(img.tobytes()).hexdigest()
                hash_map[img_hash].append(global_idx)

    # 3. Identify all duplicate groups
    # A group is a duplicate if it has more than 1 index for the same hash
    duplicate_indices = []
    hash_group_id = []
    group_counter = 0

    for img_hash, indices in hash_map.items():
        if len(indices) > 1:
            duplicate_indices.extend(indices)
            # Assign a group ID so we can see which images are identical to each other
            for _ in indices:
                hash_group_id.append(f"Group_{group_counter}")
            group_counter += 1

    # 4. Extract metadata and save
    if duplicate_indices:
        # Create a report dataframe
        report_df = df.iloc[duplicate_indices].copy()
        report_df['duplicate_group_id'] = hash_group_id

        # Sort by group ID to see identical images next to each other
        report_df = report_df.sort_values(by=['duplicate_group_id', 'original_split'])

        # Save to CSV
        report_df.to_csv(OUTPUT_PATH, index=False)

        print("-" * 30)
        print(f"Analysis Complete!")
        print(f"Total rows involved in duplicates: {len(report_df)}")
        print(f"Number of unique duplicate groups: {group_counter}")
        print(f"Report saved to: {OUTPUT_PATH}")

        # Cross-split check (The most important part for PhD rigor)
        leakage = report_df.groupby('duplicate_group_id')['original_split'].nunique()
        cross_split_count = (leakage > 1).sum()
        if cross_split_count > 0:
            print(f"\n[WARNING] Found {cross_split_count} groups that cross the Train/Test boundary!")
            print("This indicates minor Data Leakage.")
        else:
            print("\n[SAFE] All duplicates are contained within their own splits. No leakage found.")
    else:
        print("No duplicates found.")


if __name__ == "__main__":
    analyze_duplicates()

# %% End of Duplicate Images Origin Analysis

# %% [markdown]
# ### Official Split Implementation
# **Task:** Sub-split author's 'train64' into Train/Val, and keep 'test64' as final Test.
# **Integrity:** Grouping by 'index' is preserved to prevent rotation leakage.

# %% Splitting Implementation
from sklearn.model_selection import train_test_split
from pathlib import Path

# Path configuration (Point to your RAW folder)
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "textile"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def create_official_tri_split():
    # 1. Load original metadata from RAW directory
    train_df_full = pd.read_csv(RAW / "train64.csv")
    test_df_full = pd.read_csv(RAW / "test64.csv")

    # --- Core Fix: Establish absolute pointers for full64.h5 ---
    # Training data takes the first chunk of full64.h5
    train_df_full['abs_ptr'] = range(len(train_df_full))
    # Test data follows training data
    test_offset = len(train_df_full)
    test_df_full['abs_ptr'] = range(test_offset, test_offset + len(test_df_full))

    # 2. Group by 'index' to avoid rotation leakage
    unique_train_indices = train_df_full['index'].unique()
    stratify_labels = train_df_full.drop_duplicates('index')['indication_type']

    # 3. Sub-split author's 'train' into our local Train and Val
    train_indices, val_indices = train_test_split(
        unique_train_indices,
        test_size=0.10,
        random_state=42,
        stratify=stratify_labels
    )

    # 4. Map back
    df_train = train_df_full[train_df_full['index'].isin(train_indices)].copy()
    df_val = train_df_full[train_df_full['index'].isin(val_indices)].copy()
    df_test = test_df_full.copy()

    # --- 关键插入点：全局打乱 (Global Shuffling) ---
    # frac=1 表示抽取100%的数据，即全量打乱顺序
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True)
    # ----------------------------------------------

    # 5. Save to CSV
    df_train.to_csv(PROCESSED / "train_split.csv", index=False)
    df_val.to_csv(PROCESSED / "val_split.csv", index=False)
    df_test.to_csv(PROCESSED / "test_split.csv", index=False)

    print(f"✅ Split and Shuffling complete. Train: {len(df_train)}, Val: {len(df_val)}")


if __name__ == "__main__":
    create_official_tri_split()
# %% End of splitting


# %% [markdown]
# ### Custom PyTorch Dataset Implementation
# **Task:** Develop a memory-efficient data loader to fetch textile images from H5 files using CSV metadata.
# **Integrity:** Normalizes pixel ranges and ensures dimension alignment (CHW) for PyTorch model compatibility.

# %% Dataset Implementation

class TextileDataset(Dataset):
    """
    Standard PyTorch Dataset class for textile defect detection.
    Connects the processed CSV metadata with the binary H5 image storage.
    """

    def __init__(self, csv_path, h5_path, transform=None):
        # Load the split metadata (train_split.csv, val_split.csv, or test_split.csv)
        self.df = pd.read_csv(csv_path)
        self.h5_path = h5_path
        self.transform = transform

        # Label Encoding: Map categorical strings to numerical integers
        # Sort labels to ensure consistent mapping across different runs
        self.label_map = {
            'color': 0,
            'cut': 1,
            'good': 2,
            'hole': 3,
            'metal_contamination': 4,
            'thread': 5
        }

    def __len__(self):
        # Return the total number of samples in the current split
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset with automatic scaling detection.
        """
        # 1. Retrieve metadata using the absolute pointer
        row = self.df.iloc[idx]
        global_idx = int(row['abs_ptr'])
        label_str = row['indication_type']

        # 2. Access the image from H5
        with h5py.File(self.h5_path, 'r') as f:
            img = f['images'][global_idx]

        # 3. --- 核心修复：智能缩放逻辑 ---
        # 如果数据是 uint8 (0-255)，则除以 255 转为 0-1
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            # 如果已经是 float32，说明已经是 0-1 范围，直接转换即可
            img = img.astype(np.float32)
        # --------------------------------

        # 4. Dimension Permutation (H,W,C) -> (C,H,W)
        img = np.transpose(img, (2, 0, 1))

        # 5. Label Mapping & Tensorization
        label = self.label_map[label_str]
        image_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

# %% [markdown]
# ### Dataset Instantiation Example
# **Note:** Replace the paths below with your actual local file locations.
# The 'train64.h5' and 'test64.h5' should be the original files provided by the author.
# %% [markdown]
# ### Baseline CNN Model Architecture
# **Task:** Define a standard Convolutional Neural Network (CNN) for 6-class grayscale image classification.
# **Design:** Uses 3 convolutional blocks followed by fully connected layers to extract and classify defect features.

# %% Model Implementation

class TextileBaselineCNN(nn.Module):
    """
    A standard CNN baseline for 64x64 grayscale images.
    Consists of three convolutional blocks and a classification head.
    """

    def __init__(self, num_classes=6):
        super(TextileBaselineCNN, self).__init__()

        # Block 1: Input (1, 64, 64) -> Output (32, 32, 32)
        # Extracts low-level features like edges and basic textures
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Input (32, 32, 32) -> Output (64, 16, 16)
        # Extracts mid-level geometric patterns (e.g., holes, threads)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: Input (64, 16, 16) -> Output (128, 8, 8)
        # Extracts high-level semantic features specific to defect types
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification Head
        # Flattened size: 128 channels * 8 width * 8 height = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)  # Reduces overfitting
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        # Block 1 execution
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2 execution
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Block 3 execution
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten for the fully connected layers
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer (Returns raw logits for CrossEntropyLoss)
        x = self.fc2(x)

        return x

# %% [markdown]
# ### Model Summary and Initialization
# **Note:** The model expects input tensors of shape (Batch_Size, 1, 64, 64).

# %% Initialization Example
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TextileBaselineCNN(num_classes=6).to(device)
# print(model)

# %% [markdown]
# ### Streamlined Training Pipeline
# **Task:** High-performance training with modular design and auto-hardware detection.
# **Refactoring:** Encapsulated logic to minimize code redundancy while maintaining multi-GPU support.

# %% Execution Implementation
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path


# 1. Compact Device Selection (NVIDIA -> AMD -> CPU)
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda"), "NVIDIA GPU"
    try:
        import torch_directml
        return torch_directml.device(), "AMD GPU (DirectML)"
    except:
        return torch.device("cpu"), "CPU"


device, dev_name = get_device()
print(f"🚀 Training on: {dev_name}")

# 2. Setup & Hyperparameters
cfg = {"batch": 64, "lr": 0.001, "epochs": 10, "h5": Path("data/processed/full64.h5")}
train_loader = DataLoader(TextileDataset("data/processed/train_split.csv", cfg["h5"]),
                          batch_size=cfg["batch"], shuffle=True)
val_loader = DataLoader(TextileDataset("data/processed/val_split.csv", cfg["h5"]),
                        batch_size=cfg["batch"], shuffle=True)

train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

print(f"训练集像素最大值: {train_batch[0].max().item():.2f}")
print(f"验证集像素最大值: {val_batch[0].max().item():.2f}")
print(f"训练集标签样本: {train_batch[1][:5].tolist()}")
print(f"验证集标签样本: {val_batch[1][:5].tolist()}")

# 3. Model & Engine Initialization
model = TextileBaselineCNN(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], foreach=False)


# 4. Helper for Training/Evaluation Step
def run_step(loader, is_train=True):
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


# 5. Optimized Main Loop
for epoch in range(cfg["epochs"]):
    train_loss, train_acc = run_step(train_loader, is_train=True)
    val_loss, val_acc = run_step(val_loader, is_train=False)

    print(f"Epoch [{epoch + 1:02d}/{cfg['epochs']}] | "
          f"Train: {train_acc:>5.2f}% (Loss: {train_loss:.4f}) | "
          f"Val: {val_acc:>5.2f}% (Loss: {val_loss:.4f})")

print(f"✅ Training completed on {dev_name}.")