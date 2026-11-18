"""Unit Tests for Evaluation and Clinical Metrics Pipeline

Tests:
1. Multi-metric evaluation calculation (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC).
2. Clinical decision threshold optimization for target recall.
3. Confusion matrix plot generation.
4. ROC and PR curves plot generation.
5. Markdown report generation.
"""

from pathlib import Path
import numpy as np
import pytest
from src.evaluate import (
    evaluate_predictions,
    find_optimal_threshold,
    generate_evaluation_report,
    plot_confusion_matrix,
    plot_roc_pr_curves,
)


def test_evaluate_predictions():
    # Perfectly separated predictions
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.3, 0.8, 0.9, 0.7, 0.85])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[4, 0], [0, 4]]


def test_find_optimal_threshold():
    # Synthetic case where lower threshold is required to catch all positive cases
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.8, 0.9])

    best_th, metrics = find_optimal_threshold(y_true, y_prob, target_recall=1.0)

    # Threshold must be <= 0.45 to achieve 100% recall
    assert best_th <= 0.45
    assert metrics["recall"] == 1.0


def test_plot_generation(tmp_path):
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.3, 0.8, 0.7, 0.9, 0.2, 0.65, 0.4])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

    # 1. Confusion Matrix
    cm_path = tmp_path / "cm_test.png"
    out_cm = plot_confusion_matrix(metrics["confusion_matrix"], cm_path, architecture="DenseNet121")
    assert out_cm.exists() and out_cm.stat().st_size > 0

    # 2. ROC / PR curves
    curves_path = tmp_path / "curves_test.png"
    out_curves = plot_roc_pr_curves(y_true, y_prob, curves_path, architecture="DenseNet121")
    assert out_curves.exists() and out_curves.stat().st_size > 0

    # 3. Evaluation Report
    report_path = tmp_path / "eval_report.md"
    out_rep = generate_evaluation_report(metrics, report_path, architecture="DenseNet121")
    assert out_rep.exists() and out_rep.stat().st_size > 0
