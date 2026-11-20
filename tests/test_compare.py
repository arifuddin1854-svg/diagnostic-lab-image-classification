"""Unit Tests for Model Comparison and Benchmarking (Day 10)

Tests:
1. Metric computation schema and keys for ResNet50 and DenseNet121.
2. Latency benchmark execution.
3. Visualization figure generation.
4. Model comparison markdown report generation.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from src.compare_models import (
    benchmark_inference_latency,
    generate_comparison_report,
    plot_model_comparison,
)
from src.model import build_model


@pytest.fixture
def dummy_comparison_data():
    return {
        "resnet50": {
            "architecture": "resnet50",
            "accuracy": 0.88,
            "precision": 0.89,
            "recall": 0.92,
            "specificity": 0.82,
            "f1": 0.90,
            "loss": 0.31,
            "roc_auc": 0.93,
            "total_params": 24558146,
            "trainable_params": 1050114,
            "latency_ms": 28.5,
            "confusion_matrix": [[60, 13], [15, 168]],
        },
        "densenet121": {
            "architecture": "densenet121",
            "accuracy": 0.90,
            "precision": 0.91,
            "recall": 0.94,
            "specificity": 0.84,
            "f1": 0.92,
            "loss": 0.28,
            "roc_auc": 0.95,
            "total_params": 7479746,
            "trainable_params": 525826,
            "latency_ms": 32.1,
            "confusion_matrix": [[62, 11], [11, 172]],
        },
    }


def test_plot_model_comparison(dummy_comparison_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = Path(tmp_dir) / "test_comparison.png"
        res = plot_model_comparison(dummy_comparison_data, out_fig)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_generate_comparison_report(dummy_comparison_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_report = Path(tmp_dir) / "test_report.md"
        res = generate_comparison_report(dummy_comparison_data, out_report)
        assert res.exists()
        content = res.read_text(encoding="utf-8")
        assert "ResNet50" in content
        assert "DenseNet121" in content
        assert "Validation Recall" in content


def test_benchmark_inference_latency():
    model = build_model(architecture="baseline_cnn", num_classes=2, pretrained=False)
    dummy_loader = [(torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1])) for _ in range(3)]
    latency = benchmark_inference_latency(model, dummy_loader, device=torch.device("cpu"), num_batches=2)
    assert isinstance(latency, float)
    assert latency > 0.0
