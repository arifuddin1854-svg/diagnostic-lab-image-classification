"""Unit Tests for Optimization, Early Stopping and LR Scheduling (Day 12)

Tests:
1. EnhancedEarlyStopping improvement tracking and patience countdown.
2. EnhancedEarlyStopping weight restoration.
3. Plot optimization curves generation.
4. Optimization report generation.
"""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from src.optimization import (
    EnhancedEarlyStopping,
    generate_optimization_report,
    plot_optimization_curves,
)


@pytest.fixture
def dummy_history():
    return {
        "train_loss": [0.65, 0.45, 0.35, 0.30],
        "train_acc": [0.65, 0.80, 0.85, 0.88],
        "val_loss": [0.55, 0.40, 0.38, 0.42],
        "val_accuracy": [0.70, 0.82, 0.85, 0.84],
        "val_precision": [0.75, 0.86, 0.88, 0.87],
        "val_recall": [0.88, 0.94, 0.96, 0.93],
        "val_f1": [0.81, 0.90, 0.92, 0.90],
        "learning_rate": [1e-4, 1e-4, 5e-5, 5e-5],
    }


def test_early_stopping_improvement():
    model = nn.Linear(10, 2)
    es = EnhancedEarlyStopping(patience=2, mode="min", min_delta=1e-3)

    assert es(0.50, model, epoch=1) is True
    assert es.counter == 0
    assert es.best_epoch == 1

    # Score improves
    assert es(0.40, model, epoch=2) is True
    assert es.counter == 0
    assert es.best_epoch == 2

    # Score worsens
    assert es(0.45, model, epoch=3) is False
    assert es.counter == 1
    assert es.early_stop is False

    # Score worsens again -> triggers patience = 2
    assert es(0.46, model, epoch=4) is False
    assert es.counter == 2
    assert es.early_stop is True


def test_early_stopping_restore():
    model = nn.Linear(5, 1)
    orig_weight = model.weight.clone()

    es = EnhancedEarlyStopping(patience=2, mode="min", restore_best_weights=True)
    es(0.30, model, epoch=1)

    # Change weights
    with torch.no_grad():
        model.weight.fill_(99.0)

    # Restore
    es.restore(model)
    assert torch.allclose(model.weight, orig_weight)


def test_plot_optimization_curves(dummy_history):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = Path(tmp_dir) / "test_opt_curves.png"
        res = plot_optimization_curves(dummy_history, best_epoch=3, output_path=out_fig)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_generate_optimization_report(dummy_history):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_rep = Path(tmp_dir) / "test_opt_report.md"
        summary = {
            "best_epoch": 3,
            "best_val_loss": 0.38,
            "peak_recall": 0.96,
            "peak_f1": 0.92,
            "best_accuracy": 0.85,
            "final_learning_rate": 5e-5,
            "early_stopped": False,
        }
        res = generate_optimization_report(dummy_history, summary, output_path=out_rep)
        assert res.exists()
        content = res.read_text(encoding="utf-8")
        assert "DenseNet-121" in content
        assert "ReduceLROnPlateau" in content
        assert "Early Stopping" in content
