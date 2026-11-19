"""Training Progress & Learning Curve Visualizer

Axlero AI Diagnostic System
Generates publication-quality dual/triple panel learning curves:
1. Cross-Entropy Loss vs Epochs (Train vs Val)
2. Classification Accuracy vs Epochs (Train vs Val)
3. Clinical Sensitivity (Recall) & F1-Score vs Epochs
"""

from pathlib import Path
from typing import Any, Dict, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curves(
    history: Dict[str, list],
    output_path: Union[str, Path],
    architecture: str = "ResNet50",
) -> Path:
    """Plot and save training and validation metrics progression."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # 1. Loss Curve
    axes[0].plot(epochs, history["train_loss"], "o-", color="#2b5c8f", lw=2, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "s--", color="#d95f02", lw=2, label="Validation Loss")
    axes[0].set_title(f"{architecture}: Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss (Weighted Cross-Entropy)", fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Accuracy Curve
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#2b5c8f", lw=2, label="Train Accuracy")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="#27ae60", lw=2, label="Validation Accuracy")
    axes[1].set_title(f"{architecture}: Classification Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Recall & F1-Score (Clinical Sensitivity)
    axes[2].plot(epochs, [r * 100 for r in history["val_recall"]], "o-", color="#c0392b", lw=2, label="Val Recall (Sensitivity)")
    axes[2].plot(epochs, [f * 100 for f in history["val_f1"]], "^-.", color="#8e44ad", lw=2, label="Val F1-Score")
    axes[2].axhline(90.0, color="black", linestyle=":", lw=1.5, label="Target Recall (90%)")
    axes[2].set_title(f"{architecture}: Clinical Sensitivity & F1", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Score (%)", fontsize=11)
    axes[2].legend(frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out
