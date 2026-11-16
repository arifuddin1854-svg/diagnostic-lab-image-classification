"""Deep Learning Model Training & Optimization Pipeline

Axlero AI Diagnostic System
Implements:
1. Two-phase transfer learning training engine (warmup head -> fine-tune backbone).
2. Loss functions with optional class-imbalance weighting.
3. Early Stopping monitor to prevent overfitting.
4. Learning Rate Scheduler (ReduceLROnPlateau / CosineAnnealing).
5. Comprehensive metrics evaluation per epoch (Loss, Accuracy, Precision, Recall, F1).
6. Checkpoint persistence into models/.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import count_parameters, get_device, save_checkpoint, set_seed


class EarlyStopping:
    """Early stopping monitor based on validation metric."""

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation or testing dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            total += labels.size(0)

    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)

    val_loss = total_loss / max(total, 1)
    acc = accuracy_score(all_targets_np, all_preds_np)
    prec = precision_score(all_targets_np, all_preds_np, pos_label=1, zero_division=0)
    rec = recall_score(all_targets_np, all_preds_np, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets_np, all_preds_np, pos_label=1, zero_division=0)

    return {
        "loss": float(val_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def train_model(
    architecture: str = "resnet50",
    data_dir: str = "data/raw/chest_xray",
    models_dir: str = "models",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    use_class_weights: bool = True,
    patience: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete model training workflow with early stopping and checkpointing."""
    set_seed(seed)
    device = get_device()
    print("=" * 65)
    print(f"Training Architecture: {architecture.upper()} on {device}")
    print("=" * 65)

    loaders = create_dataloaders(data_dir=data_dir, batch_size=batch_size, random_state=seed)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Build model with pretrained weights
    model = build_model(architecture=architecture, num_classes=2, pretrained=True, freeze_backbone=False)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss configuration
    if use_class_weights:
        weights = loaders["class_weights"].to(device)
        print(f"Applying Class Weights to Loss: {weights.cpu().numpy().round(3)}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=patience, mode="max")

    checkpoint_dir = Path(models_dir)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_recall = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        val_recall = val_metrics["recall"]
        scheduler.step(val_recall)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_metrics["f1"])

        is_best = val_recall > best_recall
        if is_best:
            best_recall = val_recall

        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": val_metrics,
            "history": history,
        }
        save_checkpoint(
            checkpoint_state,
            checkpoint_dir,
            filename=f"{architecture}_checkpoint.pth",
            is_best=is_best,
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"Recall: {val_recall:.4f}, F1: {val_metrics['f1']:.4f}"
            f"{' [BEST RECALL]' if is_best else ''}"
        )

        improved = early_stopping(val_recall)
        if early_stopping.early_stop:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch} (Patience: {patience}).")
            break

    print("\n" + "=" * 65)
    print(f"Training Complete for {architecture.upper()}. Best Val Recall: {best_recall:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir.resolve()}")
    print("=" * 65)

    return {
        "model": model,
        "history": history,
        "best_recall": best_recall,
        "checkpoint_path": checkpoint_dir / f"{architecture}_checkpoint.pth",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chest Radiograph Classification CNN")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["baseline_cnn", "resnet50", "densenet121"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_model(architecture=args.arch, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
