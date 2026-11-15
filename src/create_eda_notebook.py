"""Script to generate notebooks/1.0-data-exploration.ipynb in valid nbformat v4."""

import json
from pathlib import Path


def build_notebook():
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "1.0-data-exploration.ipynb"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 1.0 Exploratory Data Analysis: Pediatric Chest X-Ray Dataset\n",
                "\n",
                "**Project:** Diagnostic Lab Image Classification (Computer Vision)  \n",
                "**Organization:** Axlero  \n",
                "**Author:** Member 1 – Data Collection & Preprocessing Engineer  \n",
                "**Milestone:** Day 5 – Initial EDA Visualizations & Data Analysis  \n",
                "\n",
                "---\n",
                "\n",
                "## 1. Objectives & Scope\n",
                "1. Load and verify raw Chest X-Ray dataset across `train`, `val`, and `test` partitions.\n",
                "2. Analyze class frequency, distribution, and class imbalance.\n",
                "3. Interactively display representative radiographs for **NORMAL**, **BACTERIAL**, and **VIRAL** pneumonia.\n",
                "4. Analyze dimensional heterogeneity (resolutions, aspect ratios, color channels).\n",
                "5. Establish quantitative directives for Member 2's deep learning pipeline.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "from collections import Counter\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from PIL import Image\n",
                "\n",
                "# Configure plotting\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "%matplotlib inline\n",
                "\n",
                "# Locate project data directory\n",
                "project_root = Path('..').resolve() if Path('..').resolve().name != 'notebooks' else Path('.').resolve()\n",
                "data_dir = project_root / 'data' / 'raw' / 'chest_xray'\n",
                "print(f\"Project Root: {project_root}\")\n",
                "print(f\"Dataset Directory: {data_dir}\")\n",
                "assert data_dir.exists(), f\"Data directory not found at {data_dir}\"\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Inventory & Class Frequencies\n",
                "We scan the dataset directory structure to count images in each split and class."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "splits = ['train', 'val', 'test']\n",
                "records = []\n",
                "\n",
                "for split in splits:\n",
                "    split_path = data_dir / split\n",
                "    for cls_path in sorted(split_path.iterdir()):\n",
                "        if cls_path.is_dir():\n",
                "            img_files = list(cls_path.glob('*.jpeg'))\n",
                "            for img_f in img_files:\n",
                "                records.append({\n",
                "                    'split': split,\n",
                "                    'class': cls_path.name,\n",
                "                    'filename': img_f.name,\n",
                "                    'path': str(img_f)\n",
                "                })\n",
                "\n",
                "df = pd.DataFrame(records)\n",
                "print(f\"Total Radiographs Indexed: {len(df):,}\")\n",
                "print(\"\\n--- Split & Class Summary Table ---\")\n",
                "summary_table = pd.crosstab(df['split'], df['class'], margins=True)\n",
                "summary_table['% Pneumonia'] = (summary_table['PNEUMONIA'] / summary_table['All'] * 100).round(2)\n",
                "display(summary_table)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Class Distribution Visualizations\n",
                "Visualizing class frequencies and proportions across splits."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Grouped Bar Plot\n",
                "split_counts = df.groupby(['split', 'class']).size().unstack(fill_value=0)\n",
                "split_counts.plot(kind='bar', ax=axes[0], color=['#2b5c8f', '#d95f02'], edgecolor='black', width=0.7)\n",
                "axes[0].set_title('Radiograph Frequency by Split', fontsize=12, fontweight='bold')\n",
                "axes[0].set_xlabel('Dataset Split')\n",
                "axes[0].set_ylabel('Number of Images')\n",
                "axes[0].legend(title='Diagnosis')\n",
                "\n",
                "# Overall Distribution Pie\n",
                "total_counts = df['class'].value_counts()\n",
                "axes[1].pie(\n",
                "    total_counts.values,\n",
                "    labels=[f\"{cls}\\n({cnt:,})\" for cls, cnt in total_counts.items()],\n",
                "    autopct='%1.1f%%',\n",
                "    colors=['#d95f02', '#2b5c8f'],\n",
                "    startangle=140,\n",
                "    explode=(0.04, 0),\n",
                "    wedgeprops={'edgecolor': 'black', 'linewidth': 1.2}\n",
                ")\n",
                "axes[1].set_title(f'Overall Class Balance (N = {len(df):,})', fontsize=12, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Clinical Visual Comparison: NORMAL vs BACTERIAL vs VIRAL\n",
                "- **NORMAL:** Clear bilateral lung aeration, distinct diaphragmatic contours.\n",
                "- **BACTERIAL PNEUMONIA:** Dense focal consolidation (often lobar) with air bronchograms.\n",
                "- **VIRAL PNEUMONIA:** Diffuse bilateral interstitial or peribronchial opacities."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_normal = df[(df['split'] == 'train') & (df['class'] == 'NORMAL')]['path'].tolist()[:3]\n",
                "train_bacterial = [p for p in df[(df['split'] == 'train') & (df['class'] == 'PNEUMONIA')]['path'] if 'bacteria' in p.lower()][:3]\n",
                "train_viral = [p for p in df[(df['split'] == 'train') & (df['class'] == 'PNEUMONIA')]['path'] if 'virus' in p.lower()][:3]\n",
                "\n",
                "categories = [\n",
                "    ('NORMAL (Clear Lungs)', train_normal, '#2b5c8f'),\n",
                "    ('PNEUMONIA (Bacterial - Focal)', train_bacterial, '#c0392b'),\n",
                "    ('PNEUMONIA (Viral - Interstitial)', train_viral, '#e67e22')\n",
                "]\n",
                "\n",
                "fig, axes = plt.subplots(3, 3, figsize=(12, 12))\n",
                "for row_idx, (title, sample_paths, col) in enumerate(categories):\n",
                "    for col_idx, p in enumerate(sample_paths):\n",
                "        ax = axes[row_idx, col_idx]\n",
                "        with Image.open(p) as img:\n",
                "            ax.imshow(img.convert('RGB'))\n",
                "            ax.set_title(f\"{Path(p).name[:15]}... ({img.size[0]}x{img.size[1]})\", fontsize=9)\n",
                "        ax.axis('off')\n",
                "    axes[row_idx, 0].text(-0.1, 0.5, title, transform=axes[row_idx, 0].transAxes,\n",
                "                          fontsize=11, fontweight='bold', color=col, va='center', ha='right', rotation=90)\n",
                "\n",
                "plt.suptitle('Clinical Radiograph Visual Comparison', fontsize=14, fontweight='bold', y=0.98)\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Resolution & Aspect Ratio Analysis\n",
                "Measuring pixel dimension distributions to determine optimal image sizing for CNN backbones."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Subsample for fast interactive execution\n",
                "sample_df = df.sample(min(500, len(df)), random_state=42).copy()\n",
                "dims = []\n",
                "for p in sample_df['path']:\n",
                "    with Image.open(p) as img:\n",
                "        dims.append(img.size)\n",
                "\n",
                "sample_df['width'] = [d[0] for d in dims]\n",
                "sample_df['height'] = [d[1] for d in dims]\n",
                "sample_df['aspect_ratio'] = sample_df['width'] / sample_df['height']\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n",
                "sns.histplot(sample_df['width'], kde=True, ax=axes[0], color='#2b5c8f')\n",
                "axes[0].set_title(f\"Width (Mean: {sample_df['width'].mean():.0f}px)\", fontweight='bold')\n",
                "\n",
                "sns.histplot(sample_df['height'], kde=True, ax=axes[1], color='#d95f02')\n",
                "axes[1].set_title(f\"Height (Mean: {sample_df['height'].mean():.0f}px)\", fontweight='bold')\n",
                "\n",
                "sns.histplot(sample_df['aspect_ratio'], kde=True, ax=axes[2], color='#27ae60')\n",
                "axes[2].set_title(f\"Aspect Ratio (Mean: {sample_df['aspect_ratio'].mean():.2f})\", fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Key Takeaways & Preprocessing Pipeline Requirements\n",
                "Based on the exploratory data analysis, the Member 1 preprocessing pipeline must implement:\n",
                "1. **Uniform Dimension Standardization:** Resize all images to standard `224x224` pixels for ResNet50/DenseNet121 compatibility.\n",
                "2. **Channel Standardization:** Convert all images (both `L` mode and `RGB` mode) to 3-channel RGB.\n",
                "3. **ImageNet Normalization:** Standardize pixel values using mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.\n",
                "4. **Class Imbalance Remediation:** Implement class-weighted loss and minority data augmentation.\n",
                "5. **Stratified Split Reconstitution:** Repartition raw data into a reliable 10%-15% validation split without patient leakage.\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[SUCCESS] Created Jupyter Notebook: {nb_path}")


if __name__ == "__main__":
    build_notebook()
