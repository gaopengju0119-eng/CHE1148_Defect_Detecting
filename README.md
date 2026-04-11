# Defect Detecting Learned from Textile Image and Using for TEM Images

Target: Train models for detecting defects within TEM images (nanostructure of materials/catalysts). However, because of lacking high quality, well established dataset, we chose to use the Textile Defect Detection (TDD) dataset (public dataset by the MVTec company) as a structural prox to train model. And then checked their performance using real TEM images.

The main implementation is in `main.ipynb`. The current notebook is centered on ViT, but the model factory is designed so the active model can be changed to other architectures, including the CNN baseline, ResNet-18, and EfficientNet/

## Repository Layout

```text
CHE1148_Defect_Detecting/
|-- main.ipynb
|-- environment.yml
|-- README.md
|-- data/
|   |-- raw/
|   |   |-- textile/
|   |   |   |-- train64.h5
|   |   |   |-- train64.csv
|   |   |   |-- test64.h5
|   |   |   |-- test64.csv
|   |   |-- TEM/
|   |       |-- TEM images for reference
|   |-- processed/
|       |-- full64.h5
|       |-- full64.csv
|       |-- train_split.csv
|       |-- val_split.csv
|       |-- test_split.csv
|       |-- label_map.json
|       |-- output/
|           |-- model checkpoints, histories, metrics, and figures
```

## Notebook Overview

`main.ipynb` is organized as a full experiment pipeline:

1. Imports, runtime path setup, and device selection.
2. Global training and evaluation configuration.
3. Dataset merge, duplicate analysis, and train/validation/test split generation.
4. Label-map construction and validation.
5. DataLoader utilities and model-specific image transforms.
6. Dataset class for loading textile images from H5 files.
7. Model architecture factory (ViT, CNN baseline, ResNet, EfficientNet).
8. Training or checkpoint-loading mode.
9. Multi-scenario (conducted 38 scenarios) training and model checkpoint saving.
10. Test evaluation and confusion-matrix visualization.
11. IG (Integrated Gradients) utilities.
12. Trained-vs-untrained IG interpretability comparison.
13. TEM image inference and TEM result visualization.
14. TEM IG interpretation.

## Environment Setup

Create the Conda environment from the project root:

```bash
conda env create -f environment.yml
conda activate CHE1148_Defect_Detecting
```

## Running the Notebook

The notebook supports two main execution modes:

```python
USE_EXISTING_CHECKPOINT = True
```

This loads an existing checkpoint and runs evaluation/inference.

```python
USE_EXISTING_CHECKPOINT = False
```

This trains the active model on the configured scenario and saves a new checkpoint/history file under `data/processed/output/`.

When checkpoint mode is enabled, make sure `EXISTING_CHECKPOINT`, `EXISTING_SCENARIO_NAME`, and `EXISTING_SCENARIO_CONFIG` match the model and class setup used during training.

## Model Selection

The model is selected in the `# --- Model Architecture Factory ---` cell:

```python
ACTIVE_MODEL_NAME = "vit"
```

The model registry currently includes:

```python
MODEL_SPECS = {
    "vit": ...,
    "basic_cnn": ...,
    "resnet18": ...,
    "efficientnet_b2": ...,
}
```

Model branch references:

- ViT: available in Matt's branch and the main branch.
- CNN baseline: available in Pengju's branch.
- ResNet-18: available in Pengju's branch.
- EfficientNet: available in Sebastian's branch.

## Training Scenarios

Training scenarios are configured in the common data setup cell:

```python
scenarios = [
    {
        "train_factor": 0,
        "defect_ratio": 1,
        "defect_classes": ["good", "cut", "color", "metal_contamination", "hole", "thread"],
    }
]
```
