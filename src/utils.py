"""Utilities & Helper Functions Module

Axlero AI Diagnostic System
Implements:
1. Hardware device detection and selection (CUDA / CPU).
2. Deterministic seed management for reproducible training.
3. Checkpoint serialization and deserialization.
4. Model parameter counting and reporting.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return optimal available compute device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Calculate total and trainable parameter counts for a PyTorch module.

    Returns:
        (total_parameters, trainable_parameters)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    filename: str = "checkpoint.pth",
    is_best: bool = False,
) -> Path:
    """Save model checkpoint dictionary to disk.

    Args:
        state: Dict containing 'epoch', 'model_state_dict', 'optimizer_state_dict', 'val_recall', etc.
        checkpoint_dir: Directory where checkpoint will be saved.
        filename: Checkpoint filename.
        is_best: If True, also saves a copy as 'best_model.pth'.

    Returns:
        Path to saved checkpoint.
    """
    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    filepath = chk_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_path = chk_dir / "best_model.pth"
        torch.save(state, best_path)

    return filepath


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model and optimizer states from a saved checkpoint.

    Returns:
        Checkpoint dictionary containing saved metrics, epoch, and metadata.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
