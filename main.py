# %% [markdown]
# # Textile Defect Detection (Kaggle TextileDefectDetection) — Clean Baseline
# This script keeps the original pipeline and paths, but is reorganized for clarity.
#
# Pipeline:
# 1) Merge raw train/test CSV + H5 into a single processed dataset
# 2) Compute MD5 hashes to analyze exact duplicates / leakage
# 3) Deduplicate within each original split, then stratified Train/Val split
# 4) Train a small CNN classifier with Early Stopping

# %% [markdown]
# ## 1. Imports & Configuration

import copy
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, cast

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Disable HDF5 file locking for better compatibility
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Path configuration (DO NOT CHANGE)
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "textile"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# File paths (DO NOT CHANGE)
TRAIN_H5, TRAIN_CSV = RAW / "train64.h5", RAW / "train64.csv"
TEST_H5, TEST_CSV = RAW / "test64.h5", RAW / "test64.csv"
OUT_H5, OUT_CSV = PROCESSED / "full64.h5", PROCESSED / "full64.csv"

# Keep False by default for broader local compatibility.
# Set to True only when you explicitly require NVIDIA CUDA.
REQUIRE_CUDA = False
GLOBAL_SEED = 42


def _print_torch_runtime() -> None:
    cuda_version = _torch_cuda_version()
    print(f"[Runtime] torch={torch.__version__}")
    print(f"[Runtime] torch.version.cuda={cuda_version}")
    print(f"[Runtime] cuda.is_available={torch.cuda.is_available()}")
    print(f"[Runtime] cuda.device_count={torch.cuda.device_count()}")


def _torch_cuda_version() -> Optional[str]:
    version_module = getattr(torch, "version", None)
    return cast(Optional[str], getattr(version_module, "cuda", None))


def select_device(require_cuda: bool = False):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"cuda ({gpu_name})"

    if require_cuda:
        cuda_version = _torch_cuda_version()
        if cuda_version is None:
            reason = (
                "Detected CPU-only PyTorch build. "
                "Install CUDA-enabled PyTorch in this environment."
            )
        else:
            reason = "CUDA build detected, but no usable CUDA device is visible to PyTorch."
        raise RuntimeError(
            "REQUIRE_CUDA=True but CUDA is unavailable.\n"
            f"Reason: {reason}\n"
            "Quick fix (conda): conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia"
        )

    try:
        import torch_directml  # type: ignore

        dml_device = torch_directml.device()
        return dml_device, "directml (AMD/Intel/NVIDIA)"
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] DirectML detected but unavailable: {e}")

    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps (Apple Silicon)"

    return torch.device("cpu"), "cpu"


def set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = (GLOBAL_SEED + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_dataloader_generator(seed: int = GLOBAL_SEED) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


_print_torch_runtime()
device, device_name = select_device(require_cuda=REQUIRE_CUDA)
set_seed(GLOBAL_SEED)
print(f"Using device: {device_name}")


def _require_file(path: Path) -> None:
    """Fail fast when a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def _normalize_label(x) -> str:
    return str(x).strip()


def print_class_counts(df: pd.DataFrame, title: str) -> None:
    """Print total rows and per-class distribution for the given dataframe."""
    if "indication_type" not in df.columns:
        print(f"[{title}] Missing column: indication_type")
        return

    vc = df["indication_type"].astype(str).str.strip().value_counts()
    print(f"\n[{title}] total_images={len(df)}")
    for k, v in vc.items():
        print(f"  {k}: {v}")


# %% [markdown]
# ## 2. Merge Raw Train/Test into Processed Full Dataset

def merge_data() -> None:
    """
    Merge separate train/test H5 and CSV files into a unified dataset.

    Outputs:
      - data/processed/full64.csv
      - data/processed/full64.h5
    """
    if OUT_H5.exists() and OUT_CSV.exists():
        print("Dataset already merged. Skipping merge.")
        return

    _require_file(TRAIN_CSV)
    _require_file(TEST_CSV)
    _require_file(TRAIN_H5)
    _require_file(TEST_H5)

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # Keep original split info
    df_train["original_split"] = "train"
    df_test["original_split"] = "test"

    full_df = pd.concat([df_train, df_test], ignore_index=True)
    full_df.to_csv(OUT_CSV, index=False)
    print(f"Saved merged CSV: {OUT_CSV}")

    with h5py.File(OUT_H5, "w") as f_out:
        with h5py.File(TRAIN_H5, "r") as f_tr, h5py.File(TEST_H5, "r") as f_te:
            tr_imgs = f_tr["images"]
            te_imgs = f_te["images"]

            total_shape = (tr_imgs.shape[0] + te_imgs.shape[0], *tr_imgs.shape[1:])
            dset = f_out.create_dataset("images", shape=total_shape, dtype="f")  # keep original dtype choice

            dset[: tr_imgs.shape[0]] = tr_imgs[:]
            dset[tr_imgs.shape[0] :] = te_imgs[:]

    print(f"Saved merged H5: {OUT_H5}")


# %% [markdown]
# ## 3. MD5 Hashing & Duplicate Analysis

def get_h5_hashes(h5_path: Path, total_images: int, chunk_size: int = 5000) -> List[str]:
    """Generate MD5 fingerprints for all images in the H5 file."""
    hashes: List[str] = [""] * total_images
    print(f"Generating MD5 fingerprints for {total_images} images...")

    with h5py.File(h5_path, "r") as f:
        images = f["images"]
        for start in range(0, total_images, chunk_size):
            end = min(start + chunk_size, total_images)
            chunk = images[start:end]
            for i, img in enumerate(chunk):
                hashes[start + i] = hashlib.md5(img.tobytes()).hexdigest()

    return hashes


def analyze_duplicates() -> List[str]:
    """
    Identify exact duplicates via MD5 and check leakage across original splits.

    Output:
      - data/processed/duplicates_report.csv (if duplicates exist)
    """
    _require_file(OUT_CSV)
    _require_file(OUT_H5)

    df = pd.read_csv(OUT_CSV)
    with h5py.File(OUT_H5, "r") as f:
        total = int(f["images"].shape[0])

    all_hashes = get_h5_hashes(OUT_H5, total)

    hash_map: Dict[str, List[int]] = defaultdict(list)
    for idx, h in enumerate(all_hashes):
        hash_map[h].append(idx)

    dup_groups = {h: idxs for h, idxs in hash_map.items() if len(idxs) > 1}
    dup_rows = sum(len(idxs) for idxs in dup_groups.values())

    print(f"Duplicate groups={len(dup_groups)} | duplicate_rows={dup_rows}")

    if dup_groups:
        dup_indices = [i for idxs in dup_groups.values() for i in idxs]
        report_df = df.iloc[dup_indices].copy()
        report_df["md5"] = [all_hashes[i] for i in dup_indices]
        report_path = PROCESSED / "duplicates_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"Saved duplicates report: {report_path}")

        # Leakage check: same md5 appears in both original train and original test
        leakage = report_df.groupby("md5")["original_split"].nunique()
        if (leakage > 1).any():
            print("[WARNING] Data leakage detected across original splits (train/test)!")
        else:
            print("[SAFE] No leakage found among duplicates across original splits.")

    return all_hashes


# %% [markdown]
# ## 4. Split Generation (Dedup per original split + Stratified Train/Val)

def create_clean_split(all_hashes: List[str]) -> None:
    """
    Remove internal duplicates within each original split and generate Train/Val/Test CSVs.

    Outputs:
      - data/processed/train_split.csv
      - data/processed/val_split.csv
      - data/processed/test_split.csv
    """
    df = pd.read_csv(OUT_CSV).copy()
    df["abs_ptr"] = range(len(df))  # pointer into full64.h5
    df["md5"] = all_hashes
    df["indication_type"] = df["indication_type"].astype(str).str.strip()

    tr_df_raw = df[df["original_split"] == "train"].copy()
    te_df_raw = df[df["original_split"] == "test"].copy()

    # Deduplicate within each portion
    tr_before, te_before = len(tr_df_raw), len(te_df_raw)
    tr_df = tr_df_raw.drop_duplicates(subset="md5", keep="first")
    te_df = te_df_raw.drop_duplicates(subset="md5", keep="first")
    tr_removed, te_removed = tr_before - len(tr_df), te_before - len(te_df)
    total_removed = tr_removed + te_removed

    print(f"Duplicates removed (within split): train={tr_removed}, test={te_removed}, total={total_removed}")

    # Stratified split (Train -> Train/Val) based on unique image index
    unique_df = tr_df.drop_duplicates("index")[["index", "indication_type"]].copy()
    train_idx, val_idx = train_test_split(
        unique_df["index"],
        test_size=0.1,
        random_state=42,
        stratify=unique_df["indication_type"],
    )

    df_train = tr_df[tr_df["index"].isin(train_idx)].sample(frac=1, random_state=42)
    df_val = tr_df[tr_df["index"].isin(val_idx)].sample(frac=1, random_state=42)

    train_path = PROCESSED / "train_split.csv"
    val_path = PROCESSED / "val_split.csv"
    test_path = PROCESSED / "test_split.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    te_df.to_csv(test_path, index=False)

    print(f"Datasets finalized: Train({len(df_train)}), Val({len(df_val)}), Test({len(te_df)})")

    # Requested reporting
    print_class_counts(df, "FULL (merged)")
    print_class_counts(tr_df_raw, "ORIG TRAIN (raw)")
    print_class_counts(te_df_raw, "ORIG TEST (raw)")
    print_class_counts(tr_df, "ORIG TRAIN (deduped)")
    print_class_counts(te_df, "ORIG TEST (deduped)")
    print_class_counts(df_train, "TRAIN SPLIT")
    print_class_counts(df_val, "VAL SPLIT")
    print_class_counts(te_df, "TEST SPLIT")


# %% [markdown]
# ## 5. Label Map

LABEL_MAP_JSON = PROCESSED / "label_map.json"
EXPECTED_CLASSES = [
    "good",
    "color",
    "cut",
    "hole",
    "thread",
    "metal_contamination",
]


def _validate_labels(observed: List[str], label_map: Dict[str, int]) -> None:
    unknown = sorted(set(observed) - set(label_map.keys()))
    if unknown:
        raise ValueError(
            "CSV contains unknown class names (not in label_map).\n"
            f"unknown_labels={unknown}\n"
            f"label_map_keys={sorted(label_map.keys())}"
        )


def build_label_map_from_full_csv(full_csv_path: Path) -> Dict[str, int]:
    """
    Build a stable label map.
    We read the CSV only to verify labels; the mapping order is fixed (EXPECTED_CLASSES).
    """
    df = pd.read_csv(full_csv_path)
    labels = set(df["indication_type"].astype(str).str.strip().unique().tolist())

    expected = set(EXPECTED_CLASSES)
    missing = sorted(expected - labels)
    extra = sorted(labels - expected)
    if missing or extra:
        raise ValueError(
            "full CSV labels do not match EXPECTED_CLASSES.\n"
            f"missing={missing}\n"
            f"extra={extra}\n"
            f"full_labels={sorted(labels)}"
        )

    return {name: i for i, name in enumerate(EXPECTED_CLASSES)}


def load_or_create_label_map() -> Dict[str, int]:
    """Create label_map.json once and reuse it for all splits."""
    if LABEL_MAP_JSON.exists():
        return json.loads(LABEL_MAP_JSON.read_text(encoding="utf-8"))

    label_map = build_label_map_from_full_csv(OUT_CSV)
    LABEL_MAP_JSON.write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return label_map


def validate_split_labels(csv_path: Path, label_map: Dict[str, int]) -> None:
    df = pd.read_csv(csv_path)
    observed = df["indication_type"].astype(str).str.strip().unique().tolist()
    _validate_labels([_normalize_label(x) for x in observed], label_map)


# %% [markdown]
# ## 6. PyTorch Dataset & Model

class TextileDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Read images from full64.h5 using pointers stored in the split CSV."""

    def __init__(
        self,
        csv_path: Path,
        h5_path: Path,
        *,
        label_map: Optional[Dict[str, int]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        strict_labels: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.h5_path = str(h5_path)
        self.transform = transform

        if "abs_ptr" not in self.df.columns:
            raise ValueError(
                "CSV missing abs_ptr. Please use create_clean_split() outputs: train_split.csv/val_split.csv/test_split.csv."
            )
        if "indication_type" not in self.df.columns:
            raise ValueError("CSV missing indication_type.")

        if label_map is None:
            raise ValueError(
                "label_map is required. Call load_or_create_label_map() then pass it into TextileDataset(..., label_map=label_map)."
            )
        self.label_map = dict(label_map)

        labels = [_normalize_label(x) for x in self.df["indication_type"].tolist()]
        if strict_labels:
            _validate_labels(labels, self.label_map)
        self.df["indication_type"] = labels

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        abs_ptr = int(row["abs_ptr"])

        with h5py.File(self.h5_path, "r") as f:
            img = f["images"][abs_ptr]

        img_t = torch.from_numpy(img).float()
        if img_t.max() > 1.0:
            img_t /= 255.0

        # Ensure channel-first (1, H, W)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        elif img_t.ndim == 3 and img_t.shape[-1] == 1:
            img_t = img_t.permute(2, 0, 1)

        if self.transform is not None:
            img_t = self.transform(img_t)

        label = self.label_map[row["indication_type"]]
        return img_t, torch.tensor(label, dtype=torch.long)


class TextileBaselineCNN(nn.Module):
    """Small CNN baseline for 64x64 grayscale images."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# %% [markdown]
# ## 7. Training Utilities (Early Stopping)

class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print("Validation loss improved. Saving model weights.")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def run_step(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    is_train: bool = True,
) -> Tuple[float, float]:
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            preds = outputs.argmax(dim=1)
            total += int(labels.size(0))
            correct += int((preds == labels).sum().item())

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


# %% [markdown]
# ## 8. Main

if __name__ == "__main__":
    # Parameters (DO NOT CHANGE)
    cfg = {"batch": 64, "lr": 0.001, "epochs": 30, "patience": 7}

    # Build processed dataset
    merge_data()
    hashes = analyze_duplicates()
    create_clean_split(hashes)

    # Frozen label map
    label_map = load_or_create_label_map()
    validate_split_labels(PROCESSED / "train_split.csv", label_map)
    validate_split_labels(PROCESSED / "val_split.csv", label_map)
    validate_split_labels(PROCESSED / "test_split.csv", label_map)
    print("\nlabel_map:", label_map)

    # Loaders
    train_ds = TextileDataset(PROCESSED / "train_split.csv", OUT_H5, label_map=label_map)
    val_ds = TextileDataset(PROCESSED / "val_split.csv", OUT_H5, label_map=label_map)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch"],
        shuffle=True,
        worker_init_fn=_seed_worker,
        generator=_build_dataloader_generator(GLOBAL_SEED),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch"],
        shuffle=False,
        worker_init_fn=_seed_worker,
        generator=_build_dataloader_generator(GLOBAL_SEED),
    )

    # Model & training setup
    model = TextileBaselineCNN(num_classes=len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], foreach=False)
    early_stop = EarlyStopping(patience=cfg["patience"])

    print(f"\nStarting training on: {device}")
    for epoch in range(cfg["epochs"]):
        t_loss, t_acc = run_step(model, train_loader, criterion, optimizer, device, True)
        v_loss, v_acc = run_step(model, val_loader, criterion, optimizer, device, False)

        print(
            f"Epoch [{epoch + 1:02d}] | "
            f"Train: {t_acc:.2f}% (Loss: {t_loss:.4f}) | "
            f"Val: {v_acc:.2f}% (Loss: {v_loss:.4f})"
        )

        early_stop(v_loss, model)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            model.load_state_dict(early_stop.best_model_state)
            break

    torch.save(model.state_dict(), "best_textile_baseline.pth")
    print("Training Complete.")
