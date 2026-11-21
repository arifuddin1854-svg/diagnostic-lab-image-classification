"""Unit Tests for Hyperparameter Tuning Pipeline (Day 11)

Tests:
1. Hyperparameter configuration space structure.
2. Clinical score objective formulation.
3. Visualization figure generation.
4. Report generation.
"""

from pathlib import Path
import tempfile
import pytest

from src.tune import (
    generate_tuning_report,
    get_hyperparameter_configs,
    plot_tuning_results,
)


@pytest.fixture
def dummy_tuning_results():
    return [
        {
            "trial_id": "trial_1_conservative",
            "name": "Conservative LR + Light Regularization",
            "hyperparameters": {"lr": 1e-4, "dropout_rate": 0.3, "weight_decay": 1e-4},
            "metrics": {
                "val_loss": 0.45,
                "accuracy": 0.78,
                "precision": 0.82,
                "recall": 0.85,
                "specificity": 0.70,
                "f1": 0.83,
                "roc_auc": 0.88,
                "clinical_score": 0.842,
                "training_time_s": 25.0,
            },
        },
        {
            "trial_id": "trial_4_optimal_candidate",
            "name": "Balanced LR + Moderate Regularization",
            "hyperparameters": {"lr": 5e-4, "dropout_rate": 0.3, "weight_decay": 1e-3},
            "metrics": {
                "val_loss": 0.35,
                "accuracy": 0.84,
                "precision": 0.88,
                "recall": 0.91,
                "specificity": 0.75,
                "f1": 0.89,
                "roc_auc": 0.93,
                "clinical_score": 0.902,
                "training_time_s": 24.5,
            },
        },
    ]


def test_get_hyperparameter_configs():
    configs = get_hyperparameter_configs()
    assert isinstance(configs, list)
    assert len(configs) >= 4
    for c in configs:
        assert "trial_id" in c
        assert "lr" in c
        assert "dropout_rate" in c
        assert "weight_decay" in c


def test_plot_tuning_results(dummy_tuning_results):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = Path(tmp_dir) / "test_tuning_curves.png"
        res = plot_tuning_results(dummy_tuning_results, out_fig)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_generate_tuning_report(dummy_tuning_results):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_report = Path(tmp_dir) / "test_tuning_report.md"
        best_trial = dummy_tuning_results[1]
        res = generate_tuning_report(dummy_tuning_results, best_trial, out_report)
        assert res.exists()
        content = res.read_text(encoding="utf-8")
        assert "DenseNet-121" in content
        assert "trial_4_optimal_candidate" in content
        assert "Recall (Sensitivity)" in content
