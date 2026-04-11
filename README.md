# Defect Detecting Learned from Textile Image and Using for TEM Images

Target: Train models for detecting defects within TEM images (nanostructure of materials/catalysts). However, because of lacking high quality, well established dataset, we chose to use the Textile Defect Detection (TDD) dataset (public dataset by the MVTec company) as a structural prox to train model. And then checked their performance using real TEM images.

This branch contains CNN baseline model and ResNet model (including test evaluation and IG interperation), and training ResNet model under 38 different scenarios.

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