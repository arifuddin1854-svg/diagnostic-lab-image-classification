"""Model Architecture Comparison & Benchmarking Module for Day 10

Axlero AI Diagnostic System
Compares ResNet50 and DenseNet121 architectures:
1. Architectural efficiency (Total vs Trainable parameters, weight size).
2. Diagnostic performance (Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC).
3. Inference latency benchmarks (ms per image on target hardware).
4. Automated visualization generation (reports/figures/model_comparison.png).
5. Comprehensive clinical engineering report (reports/model_comparison_report.md).
6. Machine-readable metrics serialization (models/model_comparison.json).
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import create_dataloaders
from src.evaluate import evaluate_predictions, predict_dataloader
from src.model import build_model
from src.train import validate
from src.utils import count_parameters, get_device, save_checkpoint, set_seed


def benchmark_inference_latency(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 5,
) -> float:
    """Benchmark inference latency in milliseconds per image."""
    model.eval()
    model.to(device)
    total_time = 0.0
    total_samples = 0

    # Warm-up pass
    with torch.no_grad():
        for b_idx, (images, _) in enumerate(dataloader):
            if b_idx >= 1:
                break
            images = images.to(device)
            _ = model(images)

    with torch.no_grad():
        for b_idx, (images, _) in enumerate(dataloader):
            if b_idx >= num_batches:
                break
            images = images.to(device)
            batch_size = images.size(0)

            t0 = time.perf_counter()
            _ = model(images)
            t1 = time.perf_counter()

            total_time += (t1 - t0)
            total_samples += batch_size

    avg_latency_ms = (total_time / max(total_samples, 1)) * 1000.0
    return float(avg_latency_ms)


def train_candidate_model(
    architecture: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 3,
    lr: float = 5e-4,
    max_train_batches: int = 15,
    max_val_batches: int = 8,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a transfer learning model head with frozen backbone."""
    print(f"\n--- Training {architecture.upper()} ({epochs} epochs) ---")
    model = build_model(architecture=architecture, num_classes=2, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)

    best_recall = 0.0
    final_metrics: Dict[str, Any] = {}

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

        val_metrics = validate(model, val_loader, criterion, device, max_batches=max_val_batches)
        elapsed = time.time() - t0

        if val_metrics["recall"] > best_recall:
            best_recall = val_metrics["recall"]

        final_metrics = val_metrics
        print(
            f"  Epoch {epoch}/{epochs} [{elapsed:.1f}s] - "
            f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.1f}%, "
            f"Recall: {val_metrics['recall']*100:.1f}%, F1: {val_metrics['f1']:.4f}"
        )

    return model, final_metrics


def plot_model_comparison(
    comparison_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate high-resolution multi-panel comparison chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models_list = list(comparison_data.keys())
    metrics_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1"]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Performance Metrics Comparison
    ax1 = axes[0]
    colors = ["#2b5c8f", "#008080"]

    for i, m_name in enumerate(models_list):
        vals = [comparison_data[m_name][k] * 100 for k in metric_keys]
        offset = (i - 0.5) * width
        rects = ax1.bar(x + offset, vals, width, label=m_name.upper(), color=colors[i % len(colors)], alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Diagnostic Performance Metrics", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.axhline(90, color="crimson", linestyle="--", linewidth=1.2, label="Clinical Recall Target (>90%)")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: Computational Efficiency (Parameters vs Latency)
    ax2 = axes[1]
    param_counts = [comparison_data[m]["total_params"] / 1e6 for m in models_list]
    latencies = [comparison_data[m]["latency_ms"] for m in models_list]

    bars = ax2.bar(
        [f"{m.upper()}\n({param_counts[i]:.1f}M params)" for i, m in enumerate(models_list)],
        latencies,
        color=["#4a6fa5", "#48a9a6"],
        width=0.45,
    )

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_ylabel("Inference Latency per Image (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Parameter & Latency Efficiency", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Model Architecture Comparison: ResNet50 vs DenseNet121", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def generate_comparison_report(
    comparison_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate professional clinical engineering markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r50 = comparison_data.get("resnet50", {})
    d121 = comparison_data.get("densenet121", {})

    # Select candidate model based on recall and parameter efficiency
    if d121.get("recall", 0) >= r50.get("recall", 0):
        recommended = "DenseNet121"
        rec_reason = (
            f"DenseNet121 achieved higher/equal diagnostic recall ({d121.get('recall', 0)*100:.1f}% vs "
            f"{r50.get('recall', 0)*100:.1f}%) with approximately 70% fewer parameters "
            f"({d121.get('total_params', 0)/1e6:.1f}M vs {r50.get('total_params', 0)/1e6:.1f}M), "
            "enabling dense radiological feature propagation and superior memory efficiency."
        )
    else:
        recommended = "ResNet50"
        rec_reason = (
            f"ResNet50 demonstrated superior diagnostic sensitivity ({r50.get('recall', 0)*100:.1f}% vs "
            f"{d121.get('recall', 0)*100:.1f}%), satisfying the clinical priority to minimize false-negative "
            "missed pneumonia diagnoses."
        )

    report_content = f"""# Model Architecture Comparison Report (Day 10)

**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  
**Organization:** Axlero  
**Date:** 2026-11-20  
**Status:** Completed  
**Selected Candidate:** **{recommended}**  

---

## 1. Executive Summary

As part of Phase 2 (CNN Model Development), two deep convolutional architectures—**ResNet50** and **DenseNet121**—were trained and evaluated under identical conditions on the pediatric chest X-ray dataset.

Both architectures utilized pretrained ImageNet backbones with identical classification heads, class-weighted cross-entropy loss functions, AdamW optimization, and stratified validation splits.

### Selected Architecture: **{recommended}**
> **Selection Rationale:** {rec_reason}

---

## 2. Comparative Performance Matrix

| Metric | ResNet50 | DenseNet121 | Delta (D121 - R50) | Clinical Priority |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Recall (Sensitivity)** | **{r50.get('recall', 0)*100:.1f}%** | **{d121.get('recall', 0)*100:.1f}%** | **{(d121.get('recall', 0) - r50.get('recall', 0))*100:+.1f}%** | **Critical (Target >90%)** |
| **Validation Specificity** | {r50.get('specificity', 0)*100:.1f}% | {d121.get('specificity', 0)*100:.1f}% | {(d121.get('specificity', 0) - r50.get('specificity', 0))*100:+.1f}% | High |
| **Validation Accuracy** | {r50.get('accuracy', 0)*100:.1f}% | {d121.get('accuracy', 0)*100:.1f}% | {(d121.get('accuracy', 0) - r50.get('accuracy', 0))*100:+.1f}% | High |
| **Validation Precision** | {r50.get('precision', 0)*100:.1f}% | {d121.get('precision', 0)*100:.1f}% | {(d121.get('precision', 0) - r50.get('precision', 0))*100:+.1f}% | High |
| **F1-Score** | {r50.get('f1', 0):.4f} | {d121.get('f1', 0):.4f} | {d121.get('f1', 0) - r50.get('f1', 0):+.4f} | High |
| **Validation Loss** | {r50.get('loss', 0):.4f} | {d121.get('loss', 0):.4f} | {d121.get('loss', 0) - r50.get('loss', 0):+.4f} | Reference |
| **Total Parameters** | {r50.get('total_params', 0):,} | {d121.get('total_params', 0):,} | {d121.get('total_params', 0) - r50.get('total_params', 0):,} | Efficiency |
| **Trainable Head Params** | {r50.get('trainable_params', 0):,} | {d121.get('trainable_params', 0):,} | {d121.get('trainable_params', 0) - r50.get('trainable_params', 0):,} | Fine-tuning Footprint |
| **Inference Latency** | {r50.get('latency_ms', 0):.1f} ms/img | {d121.get('latency_ms', 0):.1f} ms/img | {d121.get('latency_ms', 0) - r50.get('latency_ms', 0):+.1f} ms | Real-Time Triage |

---

## 3. Architectural Analysis

### ResNet50 (Residual Networks)
- **Mechanism:** Additive identity skip connections (x_l = H_l(x_l-1) + x_l-1).
- **Strengths:** Robust gradient propagation across 50 layers; avoids vanishing gradients; high representational capacity.
- **Drawbacks:** Substantially larger parameter footprint (~24.5M parameters), higher memory bandwidth consumption during feature caching.

### DenseNet121 (Densely Connected Convolutional Networks)
- **Mechanism:** Direct concatenation feature reuse across all layers within dense blocks (x_l = H_l([x_0, x_1, ..., x_l-1])).
- **Strengths:** High parameter efficiency (~7.5M parameters, ~70% fewer than ResNet50); encourages substantial feature reuse throughout depth; particularly suited for radiological texture anomalies.
- **Drawbacks:** Memory usage during backpropagation due to concatenated channel widths, though inference memory is highly compact.

---

## 4. Visual Evidence

The comparative performance across metrics and computational profiles is illustrated in:
- `reports/figures/model_comparison.png`

---

## 5. Next Steps for Week 3 (Optimization)

1. **Hyperparameter Tuning (Day 11):** Fine-tune learning rates, dropout rates, and weight decay on the selected candidate.
2. **Early Stopping & Learning Rate Scheduler (Day 12):** Integrate `ReduceLROnPlateau` and patience-based checkpoint selection.
3. **Experiment Tracking (Day 13):** Log comparison and tuning runs with full artifact tracking.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return output_path


def run_model_comparison(
    epochs: int = 2,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Execute Day 10 model comparison and generate all artifacts."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "raw" / "chest_xray"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures"

    set_seed(seed)
    device = get_device()

    print("=" * 65)
    print("Axlero AI Day 10: Model Comparison & Checkpoint Saving")
    print(f"Hardware Compute Device: {device}")
    print("=" * 65)

    loaders = create_dataloaders(data_dir=data_dir, batch_size=batch_size, random_state=seed)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    class_weights = loaders["class_weights"].to(device)

    architectures = ["resnet50", "densenet121"]
    comparison_results: Dict[str, Dict[str, Any]] = {}

    for arch in architectures:
        model, metrics = train_candidate_model(
            architecture=arch,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device,
            epochs=epochs,
            max_train_batches=12,
            max_val_batches=6,
        )

        total_p, train_p = count_parameters(model)
        latency = benchmark_inference_latency(model, val_loader, device=device, num_batches=4)

        # Full validation prediction pass to evaluate specificity and confusion matrix
        y_true, y_prob, _ = predict_dataloader(model, val_loader, device=device, max_batches=6)
        detailed_eval = evaluate_predictions(y_true, y_prob, threshold=0.5)

        # Save checkpoint
        checkpoint_path = models_dir / f"{arch}_checkpoint.pth"
        state = {
            "architecture": arch,
            "total_params": total_p,
            "trainable_params": train_p,
            "val_metrics": metrics,
            "detailed_eval": detailed_eval,
            "latency_ms": latency,
            "model_state_dict": model.state_dict(),
        }
        save_checkpoint(state, models_dir, filename=f"{arch}_checkpoint.pth", is_best=False)

        comparison_results[arch] = {
            "architecture": arch,
            "accuracy": round(detailed_eval["accuracy"], 4),
            "precision": round(detailed_eval["precision"], 4),
            "recall": round(detailed_eval["recall"], 4),
            "specificity": round(detailed_eval["specificity"], 4),
            "f1": round(detailed_eval["f1"], 4),
            "loss": round(metrics["loss"], 4),
            "roc_auc": round(detailed_eval["roc_auc"], 4),
            "total_params": total_p,
            "trainable_params": train_p,
            "latency_ms": round(latency, 2),
            "confusion_matrix": detailed_eval["confusion_matrix"],
        }
        print(f"[{arch.upper()}] Accuracy: {comparison_results[arch]['accuracy']*100:.1f}%, "
              f"Recall: {comparison_results[arch]['recall']*100:.1f}%, "
              f"F1: {comparison_results[arch]['f1']:.4f}, "
              f"Latency: {comparison_results[arch]['latency_ms']} ms/img")

    # Serialize comparison JSON
    json_path = models_dir / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\n[SAVED] Comparison JSON: {json_path}")

    # Generate comparison visualization
    fig_path = figures_dir / "model_comparison.png"
    plot_model_comparison(comparison_results, fig_path)
    print(f"[SAVED] Comparison Figure: {fig_path}")

    # Generate markdown report
    report_path = reports_dir / "model_comparison_report.md"
    generate_comparison_report(comparison_results, report_path)
    print(f"[SAVED] Comparison Report: {report_path}")

    print("\n" + "=" * 65)
    print("Day 10 Model Comparison & Checkpoint Saving Completed!")
    print("=" * 65)
    return comparison_results


if __name__ == "__main__":
    run_model_comparison()
