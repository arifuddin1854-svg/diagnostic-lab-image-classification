"""Build notebooks/3_model_training.ipynb in valid nbformat v4."""

import json
from pathlib import Path


def build_notebook():
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "3_model_training.ipynb"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 3.0 Deep Learning Model Training & Transfer Learning\n",
                "\n",
                "**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  \n",
                "**Organization:** Axlero  \n",
                "**Milestone:** Phase 2 (Day 9) – Initial Model Training & Convergence Evaluation  \n",
                "**Branch:** `Syed-Abdus-Sabur`  \n",
                "\n",
                "---\n",
                "\n",
                "## 1. Objectives & Training Setup\n",
                "1. Load chest radiograph batches via stratified `create_dataloaders()`.\n",
                "2. Apply class-weighted Cross-Entropy Loss to penalize false negatives (prioritizing recall > 90%).\n",
                "3. Instantiate pretrained CNN transfer learning architectures (`ResNet50` and `DenseNet121`).\n",
                "4. Train models using AdamW with weight decay regularization.\n",
                "5. Track loss, accuracy, sensitivity (recall), and F1-score across epochs.\n",
                "6. Persist model checkpoints to `models/`.\n"
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
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.optim import AdamW\n",
                "\n",
                "project_root = Path('..').resolve() if Path('..').resolve().name != 'notebooks' else Path('.').resolve()\n",
                "sys.path.insert(0, str(project_root))\n",
                "\n",
                "from src.dataset import create_dataloaders\n",
                "from src.model import build_model\n",
                "from src.utils import get_device, set_seed, count_parameters\n",
                "\n",
                "set_seed(42)\n",
                "device = get_device()\n",
                "print(f\"Active Compute Device: {device}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. DataLoader & Class Weights Inspection\n",
                "Inspecting stratified train, validation, and test splits with inverse-frequency class weighting."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_dir = project_root / 'data' / 'raw' / 'chest_xray'\n",
                "loaders = create_dataloaders(data_dir=data_dir, batch_size=32, random_state=42)\n",
                "\n",
                "train_loader = loaders['train']\n",
                "val_loader = loaders['val']\n",
                "class_weights = loaders['class_weights'].to(device)\n",
                "\n",
                "print(f\"Training Batches:   {len(train_loader)} ({len(loaders['train_dataset']):,} images)\")\n",
                "print(f\"Validation Batches: {len(val_loader)} ({len(loaders['val_dataset']):,} images)\")\n",
                "print(f\"Class Weights:      NORMAL={class_weights[0]:.3f}, PNEUMONIA={class_weights[1]:.3f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. ResNet50 Transfer Learning Architecture\n",
                "Initializing ResNet50 backbone with custom 2-class regularized classification head."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "resnet_model = build_model('resnet50', num_classes=2, pretrained=True, freeze_backbone=True)\n",
                "resnet_model = resnet_model.to(device)\n",
                "\n",
                "total_p, trainable_p = count_parameters(resnet_model)\n",
                "print(f\"ResNet50 Total Parameters:     {total_p:,}\")\n",
                "print(f\"ResNet50 Trainable Parameters: {trainable_p:,} (Classification Head)\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. DenseNet121 Transfer Learning Architecture\n",
                "Initializing DenseNet121 backbone known for parameter efficiency and dense feature propagation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "densenet_model = build_model('densenet121', num_classes=2, pretrained=True, freeze_backbone=True)\n",
                "densenet_model = densenet_model.to(device)\n",
                "\n",
                "total_d, trainable_d = count_parameters(densenet_model)\n",
                "print(f\"DenseNet121 Total Parameters:     {total_d:,}\")\n",
                "print(f\"DenseNet121 Trainable Parameters: {trainable_d:,} (Classification Head)\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Loss Function & Optimization Strategy\n",
                "- **Weighted CrossEntropyLoss:** Compensates for class imbalance.\n",
                "- **AdamW:** Decouples weight decay (`1e-3`) from gradient updates for superior generalization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "criterion = nn.CrossEntropyLoss(weight=class_weights)\n",
                "optimizer = AdamW(filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=5e-4, weight_decay=1e-3)\n",
                "print(\"Loss and Optimizer Configured Successfully.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Training Execution & Checkpointing\n",
                "Execute initial training session via `src/run_initial_training.py` and inspect learning curves in `reports/figures/training_curves.png`."
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.5"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[SUCCESS] Created notebook: {nb_path}")


if __name__ == "__main__":
    build_notebook()
