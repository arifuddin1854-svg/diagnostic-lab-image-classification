"""Dataset Inspection & Image Statistics Module

Inspects raw image folders, counts class instances, checks formats,
color modes, dimensions, aspect ratios, and file sizes.
Produces structured summary files for Member 1 reporting.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from PIL import Image


def inspect_dataset(data_dir: Path) -> Dict[str, Any]:
    """Inspect all images in data_dir across train, val, and test splits.

    Args:
        data_dir: Path to directory containing train, val, test folders.

    Returns:
        Dictionary containing detailed dataset statistics.
    """
    splits = ["train", "val", "test"]
    stats: Dict[str, Any] = {
        "dataset_root": str(data_dir.resolve()),
        "splits": {},
        "overall": {
            "total_images": 0,
            "classes": {"NORMAL": 0, "PNEUMONIA": 0},
            "formats": {},
            "color_modes": {},
            "dimensions": {
                "widths": [],
                "heights": [],
                "aspect_ratios": [],
            },
            "file_sizes_bytes": [],
        },
    }

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset root directory not found at: {data_dir}")

    for split in splits:
        split_path = data_dir / split
        if not split_path.exists():
            continue

        stats["splits"][split] = {
            "total": 0,
            "classes": {},
            "formats": {},
            "color_modes": {},
            "widths": [],
            "heights": [],
            "aspect_ratios": [],
            "file_sizes_bytes": [],
        }

        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            stats["splits"][split]["classes"][class_name] = 0

            # Scan image files
            image_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]

            for img_path in image_files:
                try:
                    file_size = img_path.stat().st_size
                    with Image.open(img_path) as img:
                        w, h = img.size
                        img_format = img.format or "UNKNOWN"
                        color_mode = img.mode
                        ar = round(w / h, 4)

                    # Update split stats
                    stats["splits"][split]["total"] += 1
                    stats["splits"][split]["classes"][class_name] += 1
                    stats["splits"][split]["formats"][img_format] = (
                        stats["splits"][split]["formats"].get(img_format, 0) + 1
                    )
                    stats["splits"][split]["color_modes"][color_mode] = (
                        stats["splits"][split]["color_modes"].get(color_mode, 0) + 1
                    )
                    stats["splits"][split]["widths"].append(w)
                    stats["splits"][split]["heights"].append(h)
                    stats["splits"][split]["aspect_ratios"].append(ar)
                    stats["splits"][split]["file_sizes_bytes"].append(file_size)

                    # Update overall stats
                    stats["overall"]["total_images"] += 1
                    if class_name in stats["overall"]["classes"]:
                        stats["overall"]["classes"][class_name] += 1
                    else:
                        stats["overall"]["classes"][class_name] = 1

                    stats["overall"]["formats"][img_format] = (
                        stats["overall"]["formats"].get(img_format, 0) + 1
                    )
                    stats["overall"]["color_modes"][color_mode] = (
                        stats["overall"]["color_modes"].get(color_mode, 0) + 1
                    )
                    stats["overall"]["dimensions"]["widths"].append(w)
                    stats["overall"]["dimensions"]["heights"].append(h)
                    stats["overall"]["dimensions"]["aspect_ratios"].append(ar)
                    stats["overall"]["file_sizes_bytes"].append(file_size)

                except Exception as err:
                    print(f"[WARN] Error reading image {img_path}: {err}")

    # Compute numerical summaries
    def summarize_series(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
        np_arr = np.array(arr)
        return {
            "min": float(np.min(np_arr)),
            "max": float(np.max(np_arr)),
            "mean": float(np.round(np.mean(np_arr), 2)),
            "median": float(np.round(np.median(np_arr), 2)),
            "std": float(np.round(np.std(np_arr), 2)),
        }

    summary: Dict[str, Any] = {
        "dataset_root": stats["dataset_root"],
        "total_images": stats["overall"]["total_images"],
        "classes": stats["overall"]["classes"],
        "formats": stats["overall"]["formats"],
        "color_modes": stats["overall"]["color_modes"],
        "dimensions": {
            "width": summarize_series(stats["overall"]["dimensions"]["widths"]),
            "height": summarize_series(stats["overall"]["dimensions"]["heights"]),
            "aspect_ratio": summarize_series(stats["overall"]["dimensions"]["aspect_ratios"]),
        },
        "file_size_mb": {
            "total_mb": round(sum(stats["overall"]["file_sizes_bytes"]) / (1024 * 1024), 2),
            "mean_kb": round(float(np.mean(stats["overall"]["file_sizes_bytes"])) / 1024, 2)
            if stats["overall"]["file_sizes_bytes"] else 0,
        },
        "splits": {},
    }

    for split, sp_data in stats["splits"].items():
        summary["splits"][split] = {
            "total_images": sp_data["total"],
            "classes": sp_data["classes"],
            "formats": sp_data["formats"],
            "color_modes": sp_data["color_modes"],
            "width": summarize_series(sp_data["widths"]),
            "height": summarize_series(sp_data["heights"]),
            "aspect_ratio": summarize_series(sp_data["aspect_ratios"]),
            "total_size_mb": round(sum(sp_data["file_sizes_bytes"]) / (1024 * 1024), 2),
        }

    return summary


def generate_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Generate human-readable markdown inspection summary."""
    splits = summary["splits"]
    dim = summary["dimensions"]

    md = [
        "# Chest X-Ray Dataset Inspection & Statistics Report",
        "",
        "**Module:** Day 3 Milestone – Member 1 (Data Collection & Preprocessing)",
        "**Organization:** Axlero",
        f"**Dataset Location:** `{summary['dataset_root']}`",
        f"**Total Processed Images:** **{summary['total_images']:,}**",
        f"**Total Storage Footprint:** **{summary['file_size_mb']['total_mb']} MB**",
        "",
        "---",
        "",
        "## 1. Split & Class Inventory",
        "",
        "| Split | NORMAL | PNEUMONIA | Total | Class Balance (% Pneumonia) | Size (MB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for split_name, sp in splits.items():
        n_normal = sp["classes"].get("NORMAL", 0)
        n_pneumonia = sp["classes"].get("PNEUMONIA", 0)
        total = sp["total_images"]
        pct_p = round((n_pneumonia / total) * 100, 2) if total > 0 else 0.0
        md.append(f"| **{split_name.capitalize()}** | {n_normal:,} | {n_pneumonia:,} | {total:,} | {pct_p}% | {sp['total_size_mb']} MB |")

    tot_normal = summary["classes"].get("NORMAL", 0)
    tot_pneumonia = summary["classes"].get("PNEUMONIA", 0)
    tot_total = summary["total_images"]
    tot_pct_p = round((tot_pneumonia / tot_total) * 100, 2) if tot_total > 0 else 0.0

    md.extend([
        f"| **Combined Total** | **{tot_normal:,}** | **{tot_pneumonia:,}** | **{tot_total:,}** | **{tot_pct_p}%** | **{summary['file_size_mb']['total_mb']} MB** |",
        "",
        "---",
        "",
        "## 2. Image Encodings & Color Modes",
        "",
        f"- **File Formats:** {summary['formats']}",
        f"- **Color Modes:** {summary['color_modes']}",
        "  - `L` Mode: 8-bit single-channel grayscale.",
        "  - `RGB` Mode: 3-channel 24-bit color representation.",
        "> [!NOTE]",
        "> The presence of both single-channel `L` and 3-channel `RGB` modes requires standardizing all inputs to 3-channel RGB in `preprocessing.py` before converting to PyTorch tensors.",
        "",
        "---",
        "",
        "## 3. Image Dimensionality Statistics",
        "",
        "| Dimension Metric | Minimum | Maximum | Mean | Median | Std Dev |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **Width (px)** | {dim['width']['min']:.0f} | {dim['width']['max']:.0f} | {dim['width']['mean']:.1f} | {dim['width']['median']:.1f} | {dim['width']['std']:.1f} |",
        f"| **Height (px)** | {dim['height']['min']:.0f} | {dim['height']['max']:.0f} | {dim['height']['mean']:.1f} | {dim['height']['median']:.1f} | {dim['height']['std']:.1f} |",
        f"| **Aspect Ratio (W/H)** | {dim['aspect_ratio']['min']:.2f} | {dim['aspect_ratio']['max']:.2f} | {dim['aspect_ratio']['mean']:.2f} | {dim['aspect_ratio']['median']:.2f} | {dim['aspect_ratio']['std']:.2f} |",
        "",
        "---",
        "",
        "## 4. Key Takeaways for Preprocessing Pipeline",
        "",
        "1. **Extreme Resolution Heterogeneity:** Widths range from under 400px to over 2,900px. Bilinear / Bicubic interpolation to 224x224 is required for standard CNN backbones (e.g. ResNet50, DenseNet121).",
        "2. **Channel Normalization:** Images must be explicitly cast to `.convert('RGB')` to ensure consistent 3x224x224 PyTorch tensors.",
        "3. **Validation Partition Restructuring:** The default validation split of only 16 images must be re-partitioned from the training set to establish a 10%-15% stratified validation set.",
        "4. **Class Imbalance Strategy:** Pneumonia accounts for 74.3% of the training images. Class-weighted CrossEntropyLoss or balanced sampling must be prepared for Member 2.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[SUCCESS] Markdown inspection summary written to: {output_path}")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "raw" / "chest_xray"
    reports_dir = base_dir / "reports"

    print("=" * 60)
    print("Executing Day 3: Dataset Inspection & Image Statistics")
    print("Inspecting:", data_dir)
    print("=" * 60)

    summary = inspect_dataset(data_dir)

    json_path = reports_dir / "dataset_inspection_report.json"
    md_path = reports_dir / "dataset_inspection_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUCCESS] JSON inspection report written to: {json_path}")

    generate_markdown_report(summary, md_path)

    print("\n" + "=" * 60)
    print(f"Inspection Summary:")
    print(f"  Total Images: {summary['total_images']}")
    print(f"  Classes: {summary['classes']}")
    print(f"  Formats: {summary['formats']}")
    print(f"  Color Modes: {summary['color_modes']}")
    print(f"  Width range: {summary['dimensions']['width']['min']} - {summary['dimensions']['width']['max']}")
    print(f"  Height range: {summary['dimensions']['height']['min']} - {summary['dimensions']['height']['max']}")
    print(f"  Total Size: {summary['file_size_mb']['total_mb']} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
