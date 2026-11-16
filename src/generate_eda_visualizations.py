"""Day 5: Exploratory Data Analysis & Visualization Pipeline

Axlero AI Diagnostic Systems
Generates publication-quality diagnostic visualizations:
1. Class distribution across splits (bar + pie chart).
2. Clinical sample image grid (NORMAL, BACTERIAL, VIRAL pneumonia).
3. Image dimensions & aspect ratio distributions.
4. Pixel intensity distribution (NORMAL vs PNEUMONIA).
Saves figures directly to reports/figures/.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

# Avoid Matplotlib cache warnings
os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial", "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

COLORS = {
    "NORMAL": "#2b5c8f",      # Clinical navy
    "PNEUMONIA": "#d95f02",    # Alert amber/orange
    "BACTERIAL": "#c0392b",    # Crimson red
    "VIRAL": "#e67e22",        # Golden orange
}


def plot_class_distribution(data_dir: Path, output_dir: Path) -> Path:
    """Generate split-wise class distribution bar and pie charts."""
    splits = ["train", "val", "test"]
    data = {"Split": [], "Class": [], "Count": []}

    for split in splits:
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = data_dir / split / cls
            count = len(list(cls_dir.glob("*.jpeg"))) if cls_dir.exists() else 0
            data["Split"].append(split.capitalize())
            data["Class"].append(cls)
            data["Count"].append(count)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Grouped Bar Chart
    x = np.arange(len(splits))
    width = 0.35
    normal_counts = [data["Count"][i] for i in range(len(data["Count"])) if data["Class"][i] == "NORMAL"]
    pneumonia_counts = [data["Count"][i] for i in range(len(data["Count"])) if data["Class"][i] == "PNEUMONIA"]

    bars1 = axes[0].bar(x - width/2, normal_counts, width, label="NORMAL", color=COLORS["NORMAL"], edgecolor="black", alpha=0.9)
    bars2 = axes[0].bar(x + width/2, pneumonia_counts, width, label="PNEUMONIA", color=COLORS["PNEUMONIA"], edgecolor="black", alpha=0.9)

    axes[0].set_title("Class Frequency by Dataset Split", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Partition", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of Radiographs", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    axes[0].legend(frameon=True, fontsize=10)

    for bar in bars1 + bars2:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Overall Class Balance Pie Chart
    tot_normal = sum(normal_counts)
    tot_pneumonia = sum(pneumonia_counts)
    axes[1].pie(
        [tot_normal, tot_pneumonia],
        labels=[f"NORMAL\n({tot_normal:,})", f"PNEUMONIA\n({tot_pneumonia:,})"],
        autopct="%1.1f%%",
        startangle=140,
        colors=[COLORS["NORMAL"], COLORS["PNEUMONIA"]],
        explode=(0.05, 0),
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "black", "linewidth": 1.2},
    )
    axes[1].set_title(f"Overall Class Distribution (Total: {tot_normal + tot_pneumonia:,})", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    output_path = output_dir / "class_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved: {output_path}")
    return output_path


def plot_sample_images_grid(data_dir: Path, output_dir: Path) -> Path:
    """Generate clinical sample images comparison grid (Normal, Bacterial, Viral)."""
    train_dir = data_dir / "train"
    normal_dir = train_dir / "NORMAL"
    pneumonia_dir = train_dir / "PNEUMONIA"

    normal_samples = sorted(list(normal_dir.glob("*.jpeg")))[:3]
    bacterial_samples = sorted([f for f in pneumonia_dir.glob("*.jpeg") if "bacteria" in f.name.lower()])[:3]
    viral_samples = sorted([f for f in pneumonia_dir.glob("*.jpeg") if "virus" in f.name.lower()])[:3]

    rows = [
        ("NORMAL (Healthy Cohort)", normal_samples, COLORS["NORMAL"]),
        ("PNEUMONIA (Bacterial - Focal Consolidation)", bacterial_samples, COLORS["BACTERIAL"]),
        ("PNEUMONIA (Viral - Interstitial Infiltrate)", viral_samples, COLORS["VIRAL"]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=300)

    for row_idx, (row_label, samples, color) in enumerate(rows):
        for col_idx, img_path in enumerate(samples):
            ax = axes[row_idx, col_idx]
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                w, h = img.size
                ax.imshow(img_rgb)
            ax.set_title(f"{img_path.name[:18]}...\n({w}x{h} px)", fontsize=9)
            ax.axis("off")

        axes[row_idx, 0].text(
            -0.15, 0.5, row_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=11, fontweight="bold", color=color,
            va="center", ha="right", rotation=90
        )

    plt.suptitle("Clinical Radiograph Comparison: Normal vs Bacterial vs Viral Pneumonia", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path = output_dir / "sample_images_grid.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved: {output_path}")
    return output_path


def plot_dimension_distribution(data_dir: Path, output_dir: Path) -> Path:
    """Generate image resolution and aspect ratio distributions."""
    widths = []
    heights = []

    for cls in ["NORMAL", "PNEUMONIA"]:
        for img_path in (data_dir / "train" / cls).glob("*.jpeg"):
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = widths / heights

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    sns.histplot(widths, kde=True, ax=axes[0], color=COLORS["NORMAL"], bins=30)
    axes[0].axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean: {np.mean(widths):.0f}px")
    axes[0].set_title("Image Width Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Width (pixels)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    axes[0].legend()

    sns.histplot(heights, kde=True, ax=axes[1], color=COLORS["PNEUMONIA"], bins=30)
    axes[1].axvline(np.mean(heights), color="red", linestyle="--", label=f"Mean: {np.mean(heights):.0f}px")
    axes[1].set_title("Image Height Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Height (pixels)", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].legend()

    sns.histplot(aspect_ratios, kde=True, ax=axes[2], color="#27ae60", bins=30)
    axes[2].axvline(np.mean(aspect_ratios), color="red", linestyle="--", label=f"Mean AR: {np.mean(aspect_ratios):.2f}")
    axes[2].set_title("Aspect Ratio (Width / Height)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Aspect Ratio", fontsize=10)
    axes[2].set_ylabel("Frequency", fontsize=10)
    axes[2].legend()

    plt.suptitle("Dataset Resolution & Geometric Heterogeneity (Training Set)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = output_dir / "dimension_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved: {output_path}")
    return output_path


def plot_pixel_intensity_distribution(data_dir: Path, output_dir: Path) -> Path:
    """Generate pixel intensity distribution histogram comparing Normal vs Pneumonia."""
    normal_pixels = []
    pneumonia_pixels = []

    normal_samples = list((data_dir / "train" / "NORMAL").glob("*.jpeg"))[:100]
    pneumonia_samples = list((data_dir / "train" / "PNEUMONIA").glob("*.jpeg"))[:100]

    for p in normal_samples:
        with Image.open(p) as img:
            arr = np.array(img.convert("L").resize((128, 128)))
            normal_pixels.extend(arr.flatten()[::10])

    for p in pneumonia_samples:
        with Image.open(p) as img:
            arr = np.array(img.convert("L").resize((128, 128)))
            pneumonia_pixels.extend(arr.flatten()[::10])

    plt.figure(figsize=(9, 5), dpi=300)
    sns.kdeplot(normal_pixels, label="NORMAL (Clear Lungs)", color=COLORS["NORMAL"], fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(pneumonia_pixels, label="PNEUMONIA (Consolidation/Infiltrate)", color=COLORS["PNEUMONIA"], fill=True, alpha=0.35, linewidth=2)

    plt.title("Pixel Intensity Distribution Comparison (Grayscale [0 - 255])", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Pixel Intensity Value", fontsize=11, fontweight="bold")
    plt.ylabel("Probability Density", fontsize=11, fontweight="bold")
    plt.xlim(0, 255)
    plt.legend(frameon=True, fontsize=10)

    output_path = output_dir / "pixel_intensity_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved: {output_path}")
    return output_path


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "raw" / "chest_xray"
    output_dir = base_dir / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Executing Day 5: EDA & Visualization Generation")
    print("Input Directory:", data_dir)
    print("Output Directory:", output_dir)
    print("=" * 65)

    plot_class_distribution(data_dir, output_dir)
    plot_sample_images_grid(data_dir, output_dir)
    plot_dimension_distribution(data_dir, output_dir)
    plot_pixel_intensity_distribution(data_dir, output_dir)

    print("\n" + "=" * 65)
    print("Day 5 Visualizations Successfully Generated in reports/figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()
