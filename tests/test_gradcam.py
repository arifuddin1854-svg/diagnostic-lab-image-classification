"""Unit Tests for Grad-CAM Explainable AI Module (Day 16)

Tests:
1. Hook registration and removal.
2. Heatmap generation with correct output shape and value bounds [0, 1].
3. Overlay blending with original image.
4. Side-by-side clinical figure generation.
"""

from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch

from src.gradcam import GradCAM
from src.model import build_model, get_gradcam_target_layer


@pytest.fixture
def dummy_gradcam():
    model = build_model(architecture="baseline_cnn", num_classes=2, pretrained=False)
    target_layer = get_gradcam_target_layer(model, architecture="baseline_cnn")
    gc = GradCAM(model=model, target_layer=target_layer)
    yield gc
    gc.remove_hooks()


def test_gradcam_generate_heatmap(dummy_gradcam):
    input_tensor = torch.randn(1, 3, 224, 224)
    heatmap, pred_idx, conf = dummy_gradcam.generate_heatmap(input_tensor, target_class=1)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0
    assert pred_idx in [0, 1]
    assert 0.0 <= conf <= 1.0


def test_gradcam_overlay():
    img_rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
    heatmap = np.ones((224, 224), dtype=np.float32)

    overlay = GradCAM.overlay_heatmap(img_rgb, heatmap, alpha=0.5)
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_visualize_side_by_side(dummy_gradcam):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_img_path = Path(tmp_dir) / "dummy_xray.jpeg"
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(dummy_img_path)

        out_fig = Path(tmp_dir) / "test_side_by_side.png"
        res = dummy_gradcam.visualize_side_by_side(dummy_img_path, out_fig, true_label="NORMAL")

        assert out_fig.exists()
        assert out_fig.stat().st_size > 1000
        assert "predicted_class" in res
        assert "confidence" in res
        assert "risk_tier" in res
