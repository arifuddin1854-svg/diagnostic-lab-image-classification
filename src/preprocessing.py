"""Image Preprocessing & Augmentation Pipeline

Axlero AI Diagnostic System
Implements:
1. Image loading and channel standardization (RGB conversion).
2. Resolution standardization (224x224 pixels).
3. Training-only dynamic data augmentation (rotation, horizontal flip, color jitter).
4. Deterministic validation and testing transformations.
5. ImageNet normalization and inverse normalization for Grad-CAM visualization.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Standard ImageNet parameters for transfer learning backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = (224, 224)


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an image from disk and standardize to 3-channel RGB.

    Handles single-channel grayscale ('L'), RGBA, and palette-based formats.
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
    return img_rgb


def get_train_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """Return PyTorch data augmentation pipeline for training data only.

    Includes clinical-safe spatial transformations:
    - Random Resized Crop (scale: 0.8 to 1.0)
    - Random Horizontal Flip (simulates symmetric anatomical projection)
    - Gentle Rotation (+/- 10 degrees, mimicking slight patient posture tilt)
    - Subtle Color Jitter (brightness & contrast variance in radiograph exposure)
    - Tensor conversion and ImageNet normalization
    """
    return T.Compose([
        T.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """Return deterministic transformations for validation, testing, and production inference.

    Strictly applies NO random transformations to guarantee evaluation reproducibility.
    """
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalization for Grad-CAM overlays and clinical visualization.

    Args:
        tensor: Normalized PyTorch tensor of shape (3, H, W) or (1, 3, H, W).

    Returns:
        numpy uint8 array of shape (H, W, 3) in range [0, 255].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    t = tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)

    # Convert to (H, W, C) numpy array [0, 255]
    np_img = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return np_img


def preprocess_pil_image(img: Image.Image, image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Preprocess a single PIL Image for model inference.

    Returns:
        PyTorch tensor of shape (1, 3, H, W).
    """
    transform = get_eval_transforms(image_size)
    tensor = transform(img.convert("RGB"))
    return tensor.unsqueeze(0)
