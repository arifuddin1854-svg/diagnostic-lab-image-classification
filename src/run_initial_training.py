"""Initial Model Training Runner for Day 9

Axlero AI Diagnostic System
Executes initial transfer learning training run:
- Freezes ImageNet backbone for rapid feature extraction warmup.
- Trains custom classification head with AdamW and class-weighted CrossEntropy.
- Evaluates validation accuracy and recall per epoch.
- Serializes checkpoints to models/ and generates reports/figures/training_curves.png.
"""

import json
import sys
import time
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.dataset import create_dataloaders
from src.model import build_model
from src.plot_training_curves import plot_learning_curves
from src.train import validate
from src.utils import count_parameters, get_device, save_checkpoint, set_seed


def run_initial_training(
    architecture: str = "resnet50",
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 5e-4,
    seed: int = 42,
):
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "raw" / "chest_xray"
    models_dir = base_dir / "models"
    figures_dir = base_dir / "reports" / "figures"

    set_seed(seed)
    device = get_device()

    print("=" * 65)
    print(f"Axlero Initial Model Training (Day 9): {architecture.upper()}")
    print(f"Hardware Compute Device: {device}")
    print("=" * 65)

    loaders = create_dataloaders(data_dir=data_dir, batch_size=batch_size, random_state=seed)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    class_weights = loaders["class_weights"].to(device)

    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
    print(f"Computed Class Weights: {class_weights.cpu().numpy().round(3)}")

    # Transfer learning with frozen backbone for fast, stable head convergence
    model = build_model(architecture=architecture, num_classes=2, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,} | Trainable (Head): {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_recall = 0.0
    best_checkpoint_path = None

    # Train for specified epochs (limit batches per epoch for fast initial baseline demonstration on CPU)
    max_train_batches = min(len(train_loader), 15)
    max_val_batches = min(len(val_loader), 8)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for b_idx, (images, labels) in enumerate(train_loader):
            if b_idx >= max_train_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / max(total_samples, 1)
        epoch_acc = running_correct / max(total_samples, 1)

        # Validation phase with capped batches for CPU speed
        val_metrics = validate(model, val_loader, criterion, device, max_batches=max_val_batches)
        elapsed = time.time() - t0

        val_recall = val_metrics["recall"]
        is_best = val_recall > best_recall
        if is_best:
            best_recall = val_recall

        history["train_loss"].append(round(epoch_loss, 4))
        history["train_acc"].append(round(epoch_acc, 4))
        history["val_loss"].append(round(val_metrics["loss"], 4))
        history["val_acc"].append(round(val_metrics["accuracy"], 4))
        history["val_precision"].append(round(val_metrics["precision"], 4))
        history["val_recall"].append(round(val_recall, 4))
        history["val_f1"].append(round(val_metrics["f1"], 4))

        print(
            f"Epoch {epoch}/{epochs} [{elapsed:.1f}s] - "
            f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.1f}%, "
            f"Recall: {val_recall*100:.1f}%, F1: {val_metrics['f1']:.4f}"
            f"{' [★ BEST RECALL]' if is_best else ''}"
        )

        state = {
            "epoch": epoch,
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "history": history,
        }
        best_checkpoint_path = save_checkpoint(
            state,
            models_dir,
            filename=f"initial_{architecture}_checkpoint.pth",
            is_best=is_best,
        )

    # Save training history JSON
    history_json_path = models_dir / "training_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[SUCCESS] Saved training history: {history_json_path}")

    # Plot learning curves
    curve_path = figures_dir / "training_curves.png"
    plot_learning_curves(history, curve_path, architecture=architecture.upper())
    print(f"[SUCCESS] Saved training curves: {curve_path}")

    print("\n" + "=" * 65)
    print(f"Day 9 Initial Model Training Completed! Best Val Recall: {best_recall*100:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_initial_training()
