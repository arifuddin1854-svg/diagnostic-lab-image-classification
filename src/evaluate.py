"""Model Evaluation, Metrics & Clinical Diagnostic Assessment Module

Axlero AI Diagnostic System
Implements:
1. Multi-metric evaluation (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC).
2. Confusion matrix computation and visualization (NORMAL vs PNEUMONIA).
3. ROC curve and Precision-Recall curve generation.
4. Validation threshold tuning to prioritize clinical sensitivity (>90% recall).
5. Comprehensive evaluation report generation in reports/evaluation_report.md.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import get_device

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive clinical diagnostic metrics.

    Args:
        y_true: 1D array of binary ground-truth labels (0=NORMAL, 1=PNEUMONIA).
        y_prob: 1D array of predicted probabilities for positive class (PNEUMONIA).
        threshold: Classification decision threshold (default 0.5).

    Returns:
        Dict of metrics including Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, CM.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),  # Sensitivity
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90,
) -> Tuple[float, Dict[str, Any]]:
    """Determine decision threshold achieving target recall while maximizing specificity.

    In clinical triage, false negatives (missed pneumonia) carry severe medical risk.
    This function searches thresholds [0.05, 0.95] on validation data to identify the
    optimal operating threshold meeting or exceeding target_recall.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

    candidates = []
    for th in thresholds:
        m = evaluate_predictions(y_true, y_prob, threshold=th)
        if m["recall"] >= target_recall:
            candidates.append((th, m))

    if candidates:
        # Choose candidate with highest F1 score / specificity among those meeting target recall
        candidates.sort(key=lambda x: (x[1]["f1"], x[1]["specificity"]), reverse=True)
        best_thresh, best_metrics = candidates[0]

    return float(best_thresh), best_metrics


def predict_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run model inference across a DataLoader.

    Returns:
        (y_true, y_prob, mean_loss)
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    all_targets = []
    all_probs = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for b_idx, (images, labels) in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # Softmax probabilities for positive class (PNEUMONIA)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_samples += labels.size(0)

    mean_loss = total_loss / max(total_samples, 1)
    return np.array(all_targets), np.array(all_probs), float(mean_loss)


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    output_path: Union[str, Path],
    architecture: str = "DenseNet121",
    normalize: bool = False,
) -> Path:
    """Plot and save clinical confusion matrix heatmap."""
    cm_arr = np.array(cm)
    if normalize:
        cm_display = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
    else:
        cm_display = cm_arr
        fmt = "d"

    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        square=True,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"Clinical Confusion Matrix: {architecture.upper()}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Clinical Category", fontsize=11, fontweight="bold")
    plt.ylabel("Reference Diagnosis (Ground Truth)", fontsize=11, fontweight="bold")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path],
    architecture: str = "DenseNet121",
) -> Path:
    """Plot and save dual ROC and Precision-Recall diagnostic curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance Level (AUC = 0.500)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Receiver Operating Characteristic: {architecture}", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True)

    # 2. Precision-Recall Curve
    axes[1].plot(rec, prec, color="#d95f02", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    axes[1].set_title(f"Precision-Recall Diagnostic Curve: {architecture}", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left", frameon=True)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def generate_evaluation_report(
    eval_results: Dict[str, Any],
    output_path: Union[str, Path],
    architecture: str = "DenseNet121",
) -> Path:
    """Generate Markdown evaluation report for model comparison and stakeholder review."""
    m = eval_results
    cm = m["confusion_matrix"]

    md = [
        f"# Model Evaluation & Clinical Diagnostic Report: {architecture.upper()}",
        "",
        "**Organization:** Axlero Diagnostic Systems  ",
        f"**Evaluated Architecture:** `{architecture.upper()}`  ",
        "**Domain:** Pediatric Chest Radiograph Pneumonia Detection  ",
        f"**Decision Threshold:** `{m['threshold']:.2f}`  ",
        "",
        "---",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Evaluation Metric | Score | Clinical Interpretation |",
        "| :--- | :---: | :--- |",
        f"| **Sensitivity (Recall)** | **{m['recall']:.2%}** | Critical triage metric: Proportion of pneumonia scans detected. |",
        f"| **Specificity** | **{m['specificity']:.2%}** | Proportion of healthy normal scans correctly classified. |",
        f"| **Accuracy** | **{m['accuracy']:.2%}** | Overall proportion of correct predictions across all cases. |",
        f"| **Precision (PPV)** | **{m['precision']:.2%}** | Reliability of positive pneumonia alert. |",
        f"| **F1-Score** | **{m['f1']:.2%}** | Harmonic mean of precision and recall. |",
        f"| **ROC-AUC** | **{m['roc_auc']:.4f}** | Overall discriminative power across all thresholds. |",
        "",
        "---",
        "",
        "## 2. Clinical Confusion Matrix",
        "",
        "| | Predicted NORMAL | Predicted PNEUMONIA | Total Ground Truth |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Actual NORMAL** | **{cm[0][0]:,}** (TN) | **{cm[0][1]:,}** (FP) | **{cm[0][0] + cm[0][1]:,}** |",
        f"| **Actual PNEUMONIA** | **{cm[1][0]:,}** (FN) | **{cm[1][1]:,}** (TP) | **{cm[1][0] + cm[1][1]:,}** |",
        "",
        "> [!IMPORTANT]",
        f"> **Sensitivity Assessment:** The model achieved **{m['recall']:.2%}** recall, {'successfully meeting the >90% project sensitivity requirement' if m['recall'] >= 0.90 else 'nearing the target sensitivity threshold'}.",
        "",
        "---",
        "",
        "## 3. Radiologist Triage Recommendation",
        "",
        f"- Scans with predicted pneumonia probability **≥ {m['threshold']:.2f}** are flagged as high risk for priority review.",
        "- Scans exhibiting focal dense consolidation or diffuse interstitial infiltration trigger automated Grad-CAM saliency mapping for expedited verification.",
    ]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return out
