# %% [markdown]
# ### Data Merging Pipeline (Refined)
# **Task:** Merge train/test H5 and CSV files while preserving original split information.
# **Key Changes:**
# 1. Renamed original 'split' column to 'original_split'.
# 2. Removed 'global_index' and 'source_split' to maintain the original column count.
# **Outputs:** `full64.h5` and `full64.csv` in the `data/processed` directory.

# %% Data Preprocessing Implementation
import hashlib
from pathlib import Path
import h5py
import pandas as pd

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
import h5py
import hashlib
import pandas as pd
import numpy as np
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