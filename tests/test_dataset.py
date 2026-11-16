"""Unit Tests for Dataset and DataLoaders

Tests:
1. ChestXRayDataset instantiation and indexing.
2. Tensor shape and type verification.
3. Class weights calculation.
4. Stratified DataLoader batch generation.
"""

from pathlib import Path
import pytest
import torch
from src.dataset import ChestXRayDataset, compute_class_weights, create_dataloaders


def test_dataset_unit():
    # Test with dummy paths
    dummy_paths = [Path("dummy1.jpeg"), Path("dummy2.jpeg")]
    dummy_labels = [0, 1]
    dataset = ChestXRayDataset(dummy_paths, dummy_labels)
    assert len(dataset) == 2
    assert dataset.labels == [0, 1]


def test_class_weights_computation():
    # 75% class 1, 25% class 0
    labels = [0] * 25 + [1] * 75
    weights = compute_class_weights(labels)

    assert len(weights) == 2
    # Class 0 should receive higher weight (approx 2.0 vs 0.67)
    assert weights[0] > weights[1]
    assert pytest.approx(weights[0].item(), rel=1e-2) == 2.0
    assert pytest.approx(weights[1].item(), rel=1e-2) == 0.667


def test_dataloaders_real_data():
    data_dir = Path("data/raw/chest_xray")
    if not data_dir.exists():
        pytest.skip("Dataset directory not found on disk")

    loaders = create_dataloaders(data_dir, batch_size=4, val_split_from_train=0.15)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    # Fetch one batch
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32
