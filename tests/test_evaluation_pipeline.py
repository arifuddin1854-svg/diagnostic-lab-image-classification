"""Unit Tests for Evaluation Pipeline and Threshold Calibration (Day 14)

Tests:
1. Threshold tuning curve generation.
2. Comprehensive evaluation report generation.
3. Confusion matrix structure and clinical metric bounds.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.evaluate import evaluate_predictions, find_optimal_threshold
from src.run_evaluation import (
    generate_comprehensive_evaluation_report,
    plot_threshold_tuning_curve,
)


@pytest.fixture
def dummy_predictions():
    # 100 samples: 40 normal (0), 60 pneumonia (1)
    np.random.seed(42)
    y_true = np.array([0] * 40 + [1] * 60)
    # Give higher probabilities to pneumonia with some noise
    y_prob = np.concatenate([
        np.random.beta(2, 5, size=40),  # skewed toward 0
        np.random.beta(5, 2, size=60),  # skewed toward 1
    ])
    return y_true, y_prob


def test_find_optimal_threshold(dummy_predictions):
    y_true, y_prob = dummy_predictions
    thresh, metrics = find_optimal_threshold(y_true, y_prob, target_recall=0.90)
    assert 0.05 <= thresh <= 0.95
    assert metrics["recall"] >= 0.90
    assert 0.0 <= metrics["specificity"] <= 1.0


def test_plot_threshold_tuning_curve(dummy_predictions):
    y_true, y_prob = dummy_predictions
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = Path(tmp_dir) / "test_thresh_curve.png"
        res = plot_threshold_tuning_curve(y_true, y_prob, optimal_threshold=0.35, output_path=out_fig)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_generate_comprehensive_evaluation_report(dummy_predictions):
    y_true, y_prob = dummy_predictions
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_rep = Path(tmp_dir) / "test_eval_report.md"
        val_eval = evaluate_predictions(y_true, y_prob, threshold=0.35)
        test_def = evaluate_predictions(y_true, y_prob, threshold=0.50)
        test_cal = evaluate_predictions(y_true, y_prob, threshold=0.35)

        res = generate_comprehensive_evaluation_report(val_eval, test_def, test_cal, 0.35, out_rep)
        assert res.exists()
        content = res.read_text(encoding="utf-8")
        assert "DenseNet-121" in content
        assert "Confusion Matrix" in content
        assert "Calibrated Threshold" in content
