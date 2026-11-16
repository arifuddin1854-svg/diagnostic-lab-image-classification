"""PyTorch Dataset & DataLoader Pipeline for Chest X-Ray Radiographs

Axlero AI Diagnostic System
Implements:
1. ChestXRayDataset: Robust PyTorch Dataset with RGB conversion and transform pipelines.
2. Stratified train/val/test partitioning resolving the 16-image Kaggle validation bottleneck.
3. Class weight calculation for imbalanced loss penalization.
4. WeightedRandomSampler creation for balanced batch mini-batch sampling.
5. DataLoader factory function create_dataloaders().
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

from src.preprocessing import DEFAULT_IMAGE_SIZE, get_eval_transforms, get_train_transforms

CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}


class ChestXRayDataset(Dataset):
    """PyTorch Dataset for Pediatric Chest Radiographs.

    Args:
        image_paths: List of file paths to images.
        labels: List of integer class labels (0 for NORMAL, 1 for PNEUMONIA).
        transform: PyTorch torchvision transforms pipeline.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: List[int],
        transform: Optional[T.Compose] = None,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = [int(lbl) for lbl in labels]
        self.transform = transform

        if len(self.image_paths) != len(self.labels):
            raise ValueError("image_paths and labels must have identical length.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")

        if self.transform:
            tensor = self.transform(img_rgb)
        else:
            tensor = T.ToTensor()(img_rgb)

        return tensor, label

    def get_class_counts(self) -> Dict[str, int]:
        """Return sample counts per class."""
        counts = {0: 0, 1: 0}
        for lbl in self.labels:
            counts[lbl] += 1
        return {IDX_TO_CLASS[k]: v for k, v in counts.items()}


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """Compute balanced class weights for CrossEntropyLoss.

    Formula: N / (num_classes * count_c)
    """
    labels_np = np.array(labels)
    total_samples = len(labels_np)
    n_classes = len(np.unique(labels_np))
    counts = np.bincount(labels_np, minlength=n_classes)

    weights = total_samples / (n_classes * np.maximum(counts, 1).astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def create_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to draw class-balanced batches."""
    labels_np = np.array(labels)
    class_counts = np.bincount(labels_np)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels_np]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def discover_image_files(data_dir: Path) -> Tuple[List[Path], List[int]]:
    """Recursively collect image paths and binary labels from NORMAL and PNEUMONIA folders."""
    paths: List[Path] = []
    labels: List[int] = []

    for cls_name, cls_idx in CLASS_TO_IDX.items():
        cls_dir = data_dir / cls_name
        if cls_dir.exists():
            for img_file in sorted(cls_dir.glob("*.jpeg")):
                paths.append(img_file)
                labels.append(cls_idx)

    return paths, labels


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    num_workers: int = 0,
    val_split_from_train: float = 0.15,
    random_state: int = 42,
    use_balanced_sampler: bool = False,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders with stratified train/val reconstitution.

    Addresses the Kaggle raw validation bottleneck (16 images) by drawing a
    statistically robust stratified validation subset from the training corpus.

    Args:
        data_dir: Path to data/raw/chest_xray containing train, val, test.
        batch_size: Mini-batch size.
        image_size: Target resolution (H, W).
        num_workers: Number of DataLoader subprocesses (0 recommended on Windows).
        val_split_from_train: Fraction of training corpus to allocate to validation.
        random_state: Random seed for stratified splitting.
        use_balanced_sampler: Whether to use WeightedRandomSampler for training.

    Returns:
        Dict with keys 'train', 'val', 'test', 'class_weights'.
    """
    root = Path(data_dir)
    train_dir = root / "train"
    raw_val_dir = root / "val"
    test_dir = root / "test"

    # 1. Collect training + raw val data to form a unified pool
    train_paths, train_labels = discover_image_files(train_dir)
    val_paths, val_labels = discover_image_files(raw_val_dir)

    all_train_paths = train_paths + val_paths
    all_train_labels = train_labels + val_labels

    # 2. Re-partition into stratified train and validation sets
    stratified_train_paths, stratified_val_paths, stratified_train_labels, stratified_val_labels = train_test_split(
        all_train_paths,
        all_train_labels,
        test_size=val_split_from_train,
        stratify=all_train_labels,
        random_state=random_state,
    )

    # 3. Collect held-out test data
    test_paths, test_labels = discover_image_files(test_dir)

    # 4. Instantiate datasets with proper transform pipelines
    train_dataset = ChestXRayDataset(
        stratified_train_paths,
        stratified_train_labels,
        transform=get_train_transforms(image_size),
    )
    val_dataset = ChestXRayDataset(
        stratified_val_paths,
        stratified_val_labels,
        transform=get_eval_transforms(image_size),
    )
    test_dataset = ChestXRayDataset(
        test_paths,
        test_labels,
        transform=get_eval_transforms(image_size),
    )

    # 5. Compute class weights for loss function
    class_weights = compute_class_weights(stratified_train_labels)

    # 6. Configure DataLoader samplers
    if use_balanced_sampler:
        train_sampler = create_balanced_sampler(stratified_train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": class_weights,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }
