"""Unit Tests for MLflow and Experiment Tracking (Day 13)

Tests:
1. Tracker initialization and run lifecycle (start_run, log_params, log_metrics, end_run).
2. Local JSON store persistence and retrieval.
3. Dashboard visualization generation.
4. Experiment tracking report generation.
"""

from pathlib import Path
import tempfile
import pytest

from src.tracking import (
    ExperimentTracker,
    generate_tracking_report,
    plot_experiment_dashboard,
)


@pytest.fixture
def dummy_runs():
    return [
        {
            "run_id": "ResNet50_20261123_100000",
            "run_name": "ResNet50_Baseline",
            "params": {"architecture": "resnet50", "learning_rate": 5e-4},
            "metrics": {"recall": 0.88, "accuracy": 0.85, "f1": 0.89, "val_loss": 0.35},
        },
        {
            "run_id": "DenseNet121_20261123_110000",
            "run_name": "DenseNet121_Optimized",
            "params": {"architecture": "densenet121", "learning_rate": 1e-4},
            "metrics": {"recall": 0.95, "accuracy": 0.90, "f1": 0.93, "val_loss": 0.28},
        },
    ]


def test_tracker_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "test_runs.json"
        tracker = ExperimentTracker(experiment_name="test_exp", runs_json_path=json_path)

        run_id = tracker.start_run("Unit_Test_Run", tags={"env": "test"})
        assert run_id.startswith("Unit_Test_Run")

        tracker.log_params({"lr": 1e-4, "batch_size": 32})
        tracker.log_metric("val_loss", 0.42, step=1)
        tracker.log_metric("recall", 0.92, step=1)

        completed = tracker.end_run()
        assert completed["run_id"] == run_id
        assert completed["params"]["lr"] == 1e-4
        assert completed["metrics"]["recall"] == 0.92

        # Check persistence
        loaded = tracker.load_runs()
        assert len(loaded) == 1
        assert loaded[0]["run_id"] == run_id


def test_plot_experiment_dashboard(dummy_runs):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = Path(tmp_dir) / "test_dash.png"
        res = plot_experiment_dashboard(dummy_runs, out_fig)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_generate_tracking_report(dummy_runs):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_rep = Path(tmp_dir) / "test_rep.md"
        res = generate_tracking_report(dummy_runs, out_rep)
        assert res.exists()
        content = res.read_text(encoding="utf-8")
        assert "MLflow" in content
        assert "ResNet50_Baseline" in content
        assert "DenseNet121_Optimized" in content
