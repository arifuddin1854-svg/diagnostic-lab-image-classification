"""Main Entry Point - Diagnostic Lab Image Classification

Member 1: Data Collection & Preprocessing Engineer
Organization: Axlero
"""

import os
import sys
from pathlib import Path


def verify_project_structure() -> bool:
    """Verify that all standard directories exist."""
    base_dir = Path(__file__).resolve().parent
    required_dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "interim",
        base_dir / "data" / "processed",
        base_dir / "notebooks",
        base_dir / "src",
        base_dir / "reports" / "figures",
        base_dir / "tests",
    ]

    all_exist = True
    print("Verifying Project Directory Structure:")
    for directory in required_dirs:
        rel_path = directory.relative_to(base_dir)
        if directory.exists():
            print(f"  [OK] {rel_path}/")
        else:
            print(f"  [MISSING] {rel_path}/")
            all_exist = False

    return all_exist


def verify_python_environment() -> None:
    """Print Python environment details."""
    print("\nEnvironment Information:")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    print(f"  Working Directory: {os.getcwd()}")


def main() -> None:
    """Run initial sanity checks for Member 1."""
    print("=" * 60)
    print("Axlero - Diagnostic Lab Image Classification")
    print("Member 1: Data Collection & Preprocessing Engineer")
    print("=" * 60)

    verify_python_environment()
    print()
    structure_ok = verify_project_structure()

    print("\n" + "=" * 60)
    if structure_ok:
        print("Status: Project structure initialized successfully.")
    else:
        print("Status: Some directories are missing. Please initialize them.")
    print("=" * 60)


if __name__ == "__main__":
    main()
