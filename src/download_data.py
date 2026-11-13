"""Dataset Download & Extraction Script for Chest X-Ray Pneumonia

Downloads the Paul Mooney Chest X-Ray Pneumonia dataset from Kaggle
and organizes it into data/raw/chest_xray.
"""

import os
import shutil
import sys
from pathlib import Path


def download_and_organize_dataset():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    target_dir = raw_dir / "chest_xray"

    print("=" * 60)
    print("Chest X-Ray Pneumonia Dataset Download & Organization")
    print("Target Directory:", target_dir)
    print("=" * 60)

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[INFO] Dataset directory already exists and is not empty: {target_dir}")
        return

    try:
        import kagglehub

        print("[INFO] Attempting download via kagglehub...")
        download_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"[SUCCESS] Download completed to cache: {download_path}")

        download_path = Path(download_path)

        # Look for the root of chest_xray inside downloaded folder
        potential_roots = list(download_path.glob("**/chest_xray"))
        if not potential_roots:
            potential_roots = [download_path]

        source_chest_xray = potential_roots[0]
        # In some nested downloads, there is a chest_xray inside chest_xray
        if (source_chest_xray / "chest_xray").exists():
            source_chest_xray = source_chest_xray / "chest_xray"

        print(f"[INFO] Copying files from {source_chest_xray} to {target_dir}...")
        shutil.copytree(source_chest_xray, target_dir, dirs_exist_ok=True)
        print(f"[SUCCESS] Dataset successfully organized into: {target_dir}")

    except Exception as e:
        print(f"[ERROR] Automatic download encountered an error: {e}")
        print("\nManual Download Instructions:")
        print("1. Visit https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Click 'Download' (archive.zip ~1.15 GB).")
        print(f"3. Extract the contents into: {target_dir.resolve()}")
        print("   The folder should contain 'train', 'test', and 'val' subdirectories.")
        sys.exit(1)


if __name__ == "__main__":
    download_and_organize_dataset()

