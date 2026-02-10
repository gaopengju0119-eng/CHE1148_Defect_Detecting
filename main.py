from pathlib import Path
import h5py
import numpy as np

# 1) Set dataset path
ROOT = Path(__file__).resolve().parent
H5_PATH = ROOT / "data" / "raw" / "textile" / "matchingtDATASET_test_32.h5"

print("Dataset path:", H5_PATH)
print("Exists:", H5_PATH.exists())
if not H5_PATH.exists():
    raise FileNotFoundError("Cannot find the .h5 file. Check filename and location.")

print("File size (MB):", round(H5_PATH.stat().st_size / (1024**2), 2))

# 2) Open the H5 file
with h5py.File(H5_PATH, "r") as hf:
    # 3) Inspect top-level keys (usually class names)
    class_keys = list(hf.keys())
    print("\nTop-level keys (classes):")
    print(class_keys)

    # 4) Pick one class and look at what's inside
    first_class = class_keys[0]
    obj = hf[first_class]
    print(f"\nInside class '{first_class}':")
    if hasattr(obj, "keys"):
        print("Children keys (first 10):", list(obj.keys())[:10])
    else:
        print("This class is not a group (unexpected).")

    # 5) Find one dataset and show its shape
    def find_first_dataset(g):
        if isinstance(g, h5py.Dataset):
            return g
        if isinstance(g, h5py.Group):
            for name in g.keys():
                ds = find_first_dataset(g[name])
                if ds is not None:
                    return ds
        return None

    ds = find_first_dataset(obj)
    if ds is None:
        print("\nNo dataset found under this class. Structure may differ.")
    else:
        print("\nFound dataset:")
        print("Dataset shape:", ds.shape, "dtype:", ds.dtype)

        sample = np.array(ds[0])  # first image
        print("One sample shape:", sample.shape)
        print("Sample min/max:", float(sample.min()), float(sample.max()))

print("\n✅ H5 opened successfully. Dataset import/inspection step is done.")