"""Unit Tests for Deep Learning Architectures

Tests:
1. Model instantiation across all three architectures.
2. Forward pass tensor dimensions on batches of chest X-ray tensors.
3. Parameter freezing and trainable parameter counting.
4. Grad-CAM target layer resolution.
"""

import pytest
import torch
from src.model import BaselineCNN, ResNet50Binary, DenseNet121Binary, build_model, get_gradcam_target_layer
from src.utils import count_parameters


def test_baseline_cnn_forward():
    model = BaselineCNN(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
    total, trainable = count_parameters(model)
    assert total > 0 and trainable == total, "All parameters should be trainable in BaselineCNN"


def test_resnet50_forward():
    model = ResNet50Binary(num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
    target_layer = get_gradcam_target_layer(model, "resnet50")
    assert target_layer is not None


def test_resnet50_freeze():
    model = ResNet50Binary(num_classes=2, pretrained=False, freeze_backbone=True)
    total, trainable = count_parameters(model)

    assert trainable < total, "Trainable parameters must be strictly less than total when backbone is frozen"
    assert trainable == (2048 * 512 + 512) + (512 * 2 + 2), "Only custom head should be trainable"


def test_densenet121_forward():
    model = DenseNet121Binary(num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
    target_layer = get_gradcam_target_layer(model, "densenet121")
    assert target_layer is not None


def test_build_model_factory():
    cnn = build_model("baseline_cnn", num_classes=2, pretrained=False)
    assert isinstance(cnn, BaselineCNN)

    resnet = build_model("resnet50", num_classes=2, pretrained=False)
    assert isinstance(resnet, ResNet50Binary)

    densenet = build_model("densenet121", num_classes=2, pretrained=False)
    assert isinstance(densenet, DenseNet121Binary)

    with pytest.raises(ValueError):
        build_model("unsupported_architecture")
