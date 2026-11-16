"""Dataset Acquisition & Organization Pipeline

Axlero AI Diagnostic System
Downloads and organizes the Kermany/Mooney Chest X-Ray Pneumonia dataset
into data/raw/chest_xray with train, val, and test splits.
"""

import os
import shutil
import sys
from pathlib import Path


def download_and_organize_dataset() -> None:
    """Acquire and structure raw clinical radiograph dataset."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    target_dir = raw_dir / "chest_xray"

    print("=" * 65)
    print("Axlero Diagnostic Lab: Dataset Acquisition & Verification")
    print(f"Target Directory: {target_dir}")
    print("=" * 65)

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[STATUS] Raw dataset already exists in {target_dir}")
        splits = [d.name for d in target_dir.iterdir() if d.is_dir()]
        print(f"[STATUS] Found partitions: {splits}")
        for s in ["train", "val", "test"]:
            s_dir = target_dir / s
            if s_dir.exists():
                classes = {c.name: len(list(c.glob("*.jpeg"))) for c in s_dir.iterdir() if c.is_dir()}
                print(f"  - {s.upper()}: {classes}")
        return

    try:
        import kagglehub
        print("[INFO] Initiating automated download via kagglehub...")
        cache_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"[SUCCESS] Download completed to cache: {cache_path}")

        cache_path = Path(cache_path)
        roots = list(cache_path.glob("**/chest_xray"))
        source = roots[0] if roots else cache_path
        if (source / "chest_xray").exists():
            source = source / "chest_xray"

        print(f"[INFO] Transferring dataset into {target_dir}...")
        shutil.copytree(source, target_dir, dirs_exist_ok=True)
        print(f"[SUCCESS] Dataset successfully organized into {target_dir}")

    except Exception as exc:
        print(f"[ERROR] Automated acquisition error: {exc}")
        print("\nManual Acquisition Instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Download archive.zip (~1.18 GB).")
        print(f"3. Extract folder structure into: {target_dir.resolve()}")
        sys.exit(1)


if __name__ == "__main__":
    download_and_organize_dataset()
