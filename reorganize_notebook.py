import json

# Read the original notebook
with open('/workspace/main.ipynb', 'r') as f:
    notebook = json.load(f)

cells = notebook['cells']

# Helper function to find cell by content
def find_cell_by_content(content_str):
    for cell in cells:
        if cell['cell_type'] == 'code' and content_str in ''.join(cell['source']):
            return cell
    return None

# Build new organized notebook
new_cells = []

# === SECTION 1: SETUP ===
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# Textile Defect Detection\n", "## Deep Learning Pipeline for Automated Quality Control\n"]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 1. Environment Setup\n"]
})

# Drive mount and paths
drive_cell = find_cell_by_content('drive.mount')
if drive_cell:
    drive_cell['source'] = [
        "# Mount Google Drive and set paths\n",
        "from google.colab import drive\n",
        "from pathlib import Path\n",
        "import os\n",
        "\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "PROJECT_ROOT = Path(\"/content/drive/MyDrive/Colab_Notebooks/CHE1148/project_code/CHE1148_Defect_Detecting\")\n",
        "if not PROJECT_ROOT.exists():\n",
        "    raise FileNotFoundError(f\"Project root not found: {PROJECT_ROOT}\")\n",
        "\n",
        "os.chdir(PROJECT_ROOT)\n",
        "print(f\"Working directory: {os.getcwd()}\")\n"
    ]
    new_cells.append(drive_cell)

# Imports
imports_cell = find_cell_by_content('import hashlib')
if imports_cell:
    imports_cell['source'] = [
        "# Import libraries\n",
        "import copy, hashlib, json, math, random\n",
        "from collections import defaultdict\n",
        "from pathlib import Path\n",
        "from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast\n",
        "\n",
        "import h5py, matplotlib.pyplot as plt, numpy as np, pandas as pd\n",
        "import torch, torch.nn as nn, torch.optim as optim\n",
        "import torchvision.models as models, tqdm\n",
        "from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,\n",
        "                             confusion_matrix, f1_score)\n",
        "from sklearn.model_selection import train_test_split\n",
        "from torch.utils.data import DataLoader, Dataset\n",
        "\n",
        "# Optional imports\n",
        "try:\n",
        "    import seaborn as sns\n",
        "except ImportError:\n",
        "    sns = None\n",
        "\n",
        "try:\n",
        "    import torchinfo\n",
        "except ImportError:\n",
        "    torchinfo = None\n"
    ]
    new_cells.append(imports_cell)

# Configuration
config_cell = find_cell_by_content('IN_COLAB = True')
if config_cell:
    config_cell['source'] = [
        "# Configuration and device setup\n",
        "from pathlib import Path\n",
        "import torch, random, numpy as np\n",
        "\n",
        "# Paths\n",
        "ROOT = Path(\"/content/drive/MyDrive/Colab_Notebooks/CHE1148/project_code/CHE1148_Defect_Detecting\")\n",
        "DATA_ROOT = ROOT / \"data\"\n",
        "DATA_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "RAW = DATA_ROOT / \"raw\" / \"textile\"\n",
        "PROCESSED = DATA_ROOT / \"processed\"\n",
        "PROCESSED.mkdir(parents=True, exist_ok=True)\n",
        "OUTPUT_DIR = PROCESSED / \"output\"\n",
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# File paths\n",
        "TRAIN_H5, TRAIN_CSV = RAW / \"train64.h5\", RAW / \"train64.csv\"\n",
        "TEST_H5, TEST_CSV = RAW / \"test64.h5\", RAW / \"test64.csv\"\n",
        "OUT_H5, OUT_CSV = PROCESSED / \"full64.h5\", PROCESSED / \"full64.csv\"\n",
        "TRAIN_SPLIT_CSV = PROCESSED / \"train_split.csv\"\n",
        "VAL_SPLIT_CSV = PROCESSED / \"val_split.csv\"\n",
        "TEST_SPLIT_CSV = PROCESSED / \"test_split.csv\"\n",
        "\n",
        "# Device selection\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(f\"Using device: {device}\")\n",
        "\n",
        "# Reproducibility\n",
        "def set_seed(seed=42):\n",
        "    random.seed(seed)\n",
        "    np.random.seed(seed)\n",
        "    torch.manual_seed(seed)\n",
        "    if torch.cuda.is_available():\n",
        "        torch.cuda.manual_seed_all(seed)\n",
        "        torch.backends.cudnn.deterministic = True\n",
        "        torch.backends.cudnn.benchmark = False\n",
        "\n",
        "set_seed(42)\n"
    ]
    new_cells.append(config_cell)

# Global config
global_config_cell = find_cell_by_content('FULL_CLASSES = ')
if global_config_cell:
    global_config_cell['source'] = [
        "# Global configuration\n",
        "FULL_CLASSES = [\"good\", \"color\", \"cut\", \"hole\", \"thread\", \"metal_contamination\"]\n",
        "\n",
        "TRAIN_CFG = {\"batch\": 512, \"epochs\": 30, \"patience\": 5}\n",
        "OPTIM_CFG = {\"name\": \"adam\", \"lr\": 0.001, \"foreach\": False}\n",
        "EVAL_CFG = {\"f1_average\": \"macro\", \"auprc_average\": \"macro\",\n",
        "            \"zero_division\": 0, \"early_stop_metric\": \"f1\", \"early_stop_mode\": \"max\"}\n",
        "\n",
        "SPLIT_SCENARIOS = {\n",
        "    \"all_training\": {\"defect_classes\": FULL_CLASSES, \"train_size\": 47843,\n",
        "                     \"defect_frac\": 0.0, \"desc\": \"Use all deduplicated training data\"},\n",
        "    \"fifty_fifty\": {\"defect_classes\": FULL_CLASSES, \"train_size\": 20000,\n",
        "                    \"defect_frac\": 0.10, \"desc\": \"50% good + 50% defects (balanced)\"},\n",
        "    \"exclude_two_classes\": {\"defect_classes\": [\"good\", \"color\", \"cut\", \"hole\"],\n",
        "                            \"train_size\": 20000, \"defect_frac\": 0.0,\n",
        "                            \"desc\": \"Exclude thread and metal_contamination\"},\n",
        "    \"imbalanced\": {\"defect_classes\": FULL_CLASSES, \"train_size\": 20000,\n",
        "                   \"defect_frac\": 0.02, \"desc\": \"Strong imbalance toward good samples\"},\n",
        "}\n",
        "\n",
        "BASELINE_SCENARIO = \"all_training\"\n"
    ]
    new_cells.append(global_config_cell)

# === SECTION 2: DATA PROCESSING ===
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["\n", "### 2. Data Processing\n"]
})

# Find utility functions
for cell in cells:
    if cell['cell_type'] == 'code' and '_require_file' in ''.join(cell['source']):
        # Extract only utility functions
        source = ''.join(cell['source'])
        if '_require_file' in source and '_normalize_label' in source:
            cell['source'] = [
                "# Utility functions\n",
                "def _require_file(path: Path) -> None:\n",
                "    if not path.exists():\n",
                "        raise FileNotFoundError(f\"Missing required file: {path}\")\n",
                "\n",
                "def _normalize_label(x) -> str:\n",
                "    return str(x).strip()\n",
                "\n",
                "def print_class_counts(df: pd.DataFrame, title: str) -> None:\n",
                "    if \"indication_type\" not in df.columns:\n",
                "        print(f\"[{title}] Missing column: indication_type\")\n",
                "        return\n",
                "    vc = df[\"indication_type\"].astype(str).str.strip().value_counts()\n",
                "    print(f\"\\n[{title}] total_images={len(df)}\")\n",
                "    for k, v in vc.items():\n",
                "        print(f\"  {k}: {v}\")\n"
            ]
            new_cells.append(cell)
            break

# Merge data function
for cell in cells:
    if cell['cell_type'] == 'code' and 'def merge_data' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# MD5 hashing
for cell in cells:
    if cell['cell_type'] == 'code' and 'def get_h5_hashes' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# Create clean split
for cell in cells:
    if cell['cell_type'] == 'code' and 'def create_clean_split' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# Execute data pipeline
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["#### Execute Data Pipeline\n"]
})

for cell in cells:
    if cell['cell_type'] == 'code' and 'merge_data()' in ''.join(cell['source']):
        new_cells.append(cell)
        break

for cell in cells:
    if cell['cell_type'] == 'code' and 'analyze_duplicates()' in ''.join(cell['source']) and 'hashes =' in ''.join(cell['source']):
        new_cells.append(cell)
        break

for cell in cells:
    if cell['cell_type'] == 'code' and 'create_clean_split(' in ''.join(cell['source']) and 'BASELINE_SPLIT_CFG' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# === SECTION 3: DATASET AND MODEL ===
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["\n", "### 3. Dataset and Model Classes\n"]
})

# Dataset class
for cell in cells:
    if cell['cell_type'] == 'code' and 'class TextileDataset' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# Baseline CNN
for cell in cells:
    if cell['cell_type'] == 'code' and 'class TextileBaselineCNN' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# ResNet model
for cell in cells:
    if cell['cell_type'] == 'code' and 'class TextileResNet' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# Training utilities
for cell in cells:
    if cell['cell_type'] == 'code' and 'class EarlyStopping' in ''.join(cell['source']):
        new_cells.append(cell)
        break

for cell in cells:
    if cell['cell_type'] == 'code' and 'def run_step' in ''.join(cell['source']) and 'model.train()' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# === SECTION 4: BASELINE CNN TRAINING ===
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["\n", "### 4. Baseline CNN Training\n"]
})

for cell in cells:
    if cell['cell_type'] == 'code' and 'TextileBaselineCNN(' in ''.join(cell['source']) and 'torchinfo.summary' in ''.join(cell['source']):
        new_cells.append(cell)
        break

for cell in cells:
    if cell['cell_type'] == 'code' and 'Starting training on:' in ''.join(cell['source']):
        # Clean up verbose output
        source = ''.join(cell['source'])
        if 'verbose=False' not in source:
            # Modify to suppress verbose early stopping messages
            pass
        new_cells.append(cell)
        break

# === SECTION 5: RESNET TRAINING ===
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["\n", "### 5. ResNet-18 Training (All Scenarios)\n"]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["#### Train ResNet-18 on all 4 split scenarios\n"]
})

for cell in cells:
    if cell['cell_type'] == 'code' and 'prepare_resnet_split_and_loaders' in ''.join(cell['source']):
        new_cells.append(cell)
        break

for cell in cells:
    if cell['cell_type'] == 'code' and 'Run ResNet for all 4 split scenarios' in ''.join(cell['source']):
        # Modify to suppress verbose output
        new_source = []
        for line in cell['source']:
            if 'verbose=False' in line or 'stopper_curr = EarlyStopping(' in line:
                new_source.append(line.replace('verbose=False', 'verbose=False'))
            elif 'verbose=True' in line:
                new_source.append(line.replace('verbose=True', 'verbose=False'))
            else:
                new_source.append(line)
        cell['source'] = new_source
        new_cells.append(cell)
        break

# === SECTION 6: RESULTS ===
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["\n", "### 6. Results and Evaluation\n"]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["#### Test Set Performance\n"]
})

for cell in cells:
    if cell['cell_type'] == 'code' and 'Test-set evaluation' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# Visualization
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["\n", "#### Visualizations\n"]
})

for cell in cells:
    if cell['cell_type'] == 'code' and 'def plot_history' in ''.join(cell['source']):
        new_cells.append(cell)
        break

for cell in cells:
    if cell['cell_type'] == 'code' and 'def show_confusion_matrix' in ''.join(cell['source']):
        new_cells.append(cell)
        break

# Write new notebook
new_notebook = {
    "cells": new_cells,
    "metadata": notebook['metadata'],
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('/workspace/main_reorganized.ipynb', 'w') as f:
    json.dump(new_notebook, f, indent=2)

print(f"Created reorganized notebook with {len(new_cells)} cells")
print("Saved to: /workspace/main_reorganized.ipynb")
