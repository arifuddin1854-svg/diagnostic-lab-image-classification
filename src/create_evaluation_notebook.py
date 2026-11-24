"""Interactive Model Evaluation Notebook Generator (Day 14)

Axlero AI Diagnostic System
Generates notebooks/4_model_evaluation.ipynb with comprehensive cells for:
- Data cohort loading and distribution
- Inference and probability generation
- Diagnostic metric computation
- Confusion matrix and ROC-PR curves
- Clinical threshold tuning
"""

import json
from pathlib import Path


def create_evaluation_notebook(output_path: Path) -> Path:
    """Generate clean Jupyter Notebook for model evaluation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 4. Model Evaluation & Clinical Diagnostic Assessment\n",
                "\n",
                "**Axlero Diagnostic Systems**  \n",
                "**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  \n",
                "**Phase 3:** Model Evaluation, ROC-AUC, Precision-Recall, Confusion Matrix, and Threshold Tuning."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Environment setup and imports\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "import torch\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from src.dataset import create_dataloaders\n",
                "from src.model import build_model\n",
                "from src.evaluate import evaluate_predictions, find_optimal_threshold, plot_confusion_matrix, plot_roc_pr_curves, predict_dataloader\n",
                "from src.run_evaluation import plot_threshold_tuning_curve\n",
                "from src.utils import get_device, set_seed\n",
                "\n",
                "set_seed(42)\n",
                "device = get_device()\n",
                "print(f\"Device: {device}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Test Cohort Dataloaders\n",
                "Loads the official held-out test cohort consisting of 624 pediatric chest X-rays (234 Normal, 390 Pneumonia)."
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
                "test_loader = loaders['test']\n",
                "val_loader = loaders['val']\n",
                "print(f\"Test Batches: {len(test_loader)} (624 images)\")\n",
                "print(f\"Validation Batches: {len(val_loader)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Production Model (DenseNet-121)\n",
                "Instantiates DenseNet-121 with tuned hyperparameters and evaluates validation data for threshold calibration."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = build_model(architecture='densenet121', num_classes=2, pretrained=True, freeze_backbone=True, dropout_rate=0.3)\n",
                "model = model.to(device)\n",
                "\n",
                "# Calibration on validation cohort\n",
                "val_y_true, val_y_prob, _ = predict_dataloader(model, val_loader, device=device, max_batches=10)\n",
                "opt_thresh, val_metrics = find_optimal_threshold(val_y_true, val_y_prob, target_recall=0.90)\n",
                "print(f\"Optimal Operating Threshold: {opt_thresh:.2f} (Validation Sensitivity: {val_metrics['recall']*100:.1f}%)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Threshold Calibration Curve\n",
                "Visualizes the trade-off between Sensitivity, Specificity, and F1 across classification thresholds."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "thresh_fig = project_root / 'reports' / 'figures' / 'threshold_tuning_curve.png'\n",
                "plot_threshold_tuning_curve(val_y_true, val_y_prob, opt_thresh, thresh_fig)\n",
                "print(f\"Saved: {thresh_fig}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Full Inference on Held-Out Test Cohort\n",
                "Runs predictions across all 624 test images and evaluates at default (0.50) vs calibrated threshold."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_y_true, test_y_prob, _ = predict_dataloader(model, test_loader, device=device)\n",
                "test_default = evaluate_predictions(test_y_true, test_y_prob, threshold=0.50)\n",
                "test_calibrated = evaluate_predictions(test_y_true, test_y_prob, threshold=opt_thresh)\n",
                "\n",
                "print(\"--- Test Results (Default 0.50) ---\")\n",
                "print(f\"Recall: {test_default['recall']*100:.1f}% | Accuracy: {test_default['accuracy']*100:.1f}% | F1: {test_default['f1']:.4f}\")\n",
                "\n",
                "print(f\"--- Test Results (Calibrated {opt_thresh:.2f}) ---\")\n",
                "print(f\"Recall: {test_calibrated['recall']*100:.1f}% | Accuracy: {test_calibrated['accuracy']*100:.1f}% | F1: {test_calibrated['f1']:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Confusion Matrix & ROC-PR Curves"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cm_path = project_root / 'reports' / 'figures' / 'confusion_matrix.png'\n",
                "plot_confusion_matrix(test_calibrated['confusion_matrix'], cm_path, architecture='DenseNet121')\n",
                "\n",
                "roc_path = project_root / 'reports' / 'figures' / 'roc_pr_curves.png'\n",
                "plot_roc_pr_curves(test_y_true, test_y_prob, roc_path, architecture='DenseNet121')\n",
                "print(\"Evaluation artifacts saved successfully.\")"
            ]
        }
    ]

    notebook_content = {
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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    return output_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks" / "4_model_evaluation.ipynb"
    create_evaluation_notebook(out)
    print(f"Created notebook: {out}")
