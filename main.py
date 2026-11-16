"""Healthcare & Medical - Diagnostic Lab Image Classification (Computer Vision)

Organization: Axlero
Project: AI-Assisted Chest X-Ray Pneumonia Diagnostic System
Branch: healthcare-cnn-development
"""

import os
import sys
from pathlib import Path


def verify_project_structure() -> bool:
    """Verify presence of all project directories."""
    base_dir = Path(__file__).resolve().parent
    required_dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "interim",
        base_dir / "data" / "processed",
        base_dir / "notebooks",
        base_dir / "src",
        base_dir / "models",
        base_dir / "reports" / "figures",
        base_dir / "backend",
        base_dir / "frontend",
        base_dir / "tests",
    ]

    print("\n[1/3] Verifying Complete Project Architecture:")
    all_ok = True
    for directory in required_dirs:
        rel = directory.relative_to(base_dir)
        if directory.exists():
            print(f"  [OK] {rel}/")
        else:
            print(f"  [MISSING] {rel}/")
            all_ok = False
    return all_ok


def check_dependencies() -> None:
    """Check and display installed technology stack components."""
    print("\n[2/3] Checking Technology Stack & Compute Environment:")
    try:
        import torch
        import torchvision
        print(f"  [OK] PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
        print(f"  [OK] Torchvision: {torchvision.__version__}")
        if torch.cuda.is_available():
            print(f"       GPU Device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"       Compute Mode: Optimized CPU Inference / Training")
    except ImportError as e:
        print(f"  [WARN] PyTorch/Torchvision: {e}")

    try:
        import cv2
        print(f"  [OK] OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"  [WARN] OpenCV: {e}")

    try:
        from PIL import Image
        print(f"  [OK] Pillow: {Image.__version__}")
    except ImportError as e:
        print(f"  [WARN] Pillow: {e}")

    try:
        import numpy as np
        import pandas as pd
        print(f"  [OK] NumPy: {np.__version__} | Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"  [WARN] Data Science Libraries: {e}")

    try:
        import fastapi
        import uvicorn
        print(f"  [OK] FastAPI: {fastapi.__version__} | Uvicorn: {uvicorn.__version__}")
    except ImportError as e:
        print(f"  [WARN] Web Backend: {e}")


def check_dataset_status() -> None:
    """Inspect local dataset status."""
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw" / "chest_xray"
    print("\n[3/3] Inspecting Dataset Status:")
    if raw_dir.exists():
        splits = [s.name for s in raw_dir.iterdir() if s.is_dir()]
        print(f"  [OK] Raw Dataset Found at: {raw_dir.relative_to(base_dir)}")
        print(f"       Available Subdirectories: {splits}")
    else:
        print(f"  [INFO] Raw Dataset Directory not yet initialized at: {raw_dir.relative_to(base_dir)}")


def main() -> None:
    print("=" * 70)
    print(" Axlero - Diagnostic Lab Image Classification (Computer Vision)")
    print(" Complete AI Diagnostic System: Data -> CNN -> Grad-CAM -> API -> UI")
    print("=" * 70)

    struct_ok = verify_project_structure()
    check_dependencies()
    check_dataset_status()

    print("\n" + "=" * 70)
    if struct_ok:
        print(" System Status: Ready for Development & Training.")
    else:
        print(" System Status: Please initialize missing directories.")
    print("=" * 70)


if __name__ == "__main__":
    main()
