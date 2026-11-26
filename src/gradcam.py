"""Gradient-Weighted Class Activation Mapping (Grad-CAM) Module for Day 16

Axlero AI Diagnostic System
Implements Explainable AI (XAI) for radiologic triage:
1. GradCAM class capturing activation maps and gradients on target convolutional layers.
2. Heatmap generation with ReLU activation and min-max normalization.
3. Alpha-blending of heatmaps over original chest radiographs using clinical colormaps.
4. Side-by-side diagnostic visualization generation with prediction confidence and risk tier.
5. Reusable standalone functions for FastAPI endpoint (/gradcam) and frontend dashboard.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import build_model, get_gradcam_target_layer
from src.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    denormalize_tensor,
    get_eval_transforms,
    load_image,
    preprocess_pil_image,
)
from src.utils import get_device

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class GradCAM:
    """Production Grad-CAM Engine for Convolutional Neural Networks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target convolutional layer."""
        # Ensure target layer parameters require grad for gradient backpropagation
        for p in self.target_layer.parameters():
            p.requires_grad = True

        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0]

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h_fwd, h_bwd])

    def remove_hooks(self) -> None:
        """Cleanly remove registered hooks to prevent memory leaks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate 2D normalized Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: Standardized tensor of shape (1, 3, H, W).
            target_class: Index of target class (0=NORMAL, 1=PNEUMONIA). If None, uses argmax.

        Returns:
            Tuple of (heatmap_2d [0.0, 1.0], predicted_class_idx, confidence_score).
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        # Forward pass with gradient tracking on graph
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        confidence = float(probabilities[0, target_class].item())

        # Backpropagation target
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)

        # Apply ReLU to focus exclusively on features with positive influence
        cam = F.relu(cam)

        # Interpolate to match input spatial dimensions (H, W)
        cam = F.interpolate(
            cam,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().detach().cpu().numpy()

        # Min-Max Normalization to [0.0, 1.0]
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max > c_min:
            cam_normalized = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_normalized = np.zeros_like(cam_np)

        return cam_normalized.astype(np.float32), target_class, confidence

    @staticmethod
    def overlay_heatmap(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Blend 2D normalized heatmap over RGB radiograph."""
        h, w = image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to 8-bit [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Colorize
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend
        blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
        return blended

    def visualize_side_by_side(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        target_class: Optional[int] = None,
        true_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate clinical 3-panel explainability figure and save to disk."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        pil_img = load_image(image_path)
        img_np = np.array(pil_img.resize(DEFAULT_IMAGE_SIZE))
        input_tensor = preprocess_pil_image(pil_img, DEFAULT_IMAGE_SIZE)

        heatmap, pred_idx, conf = self.generate_heatmap(input_tensor, target_class=target_class)
        overlay = self.overlay_heatmap(img_np, heatmap)

        pred_name = CLASS_NAMES[pred_idx]
        risk_tier = "HIGH RISK (URGENT REVIEW)" if pred_name == "PNEUMONIA" else "STANDARD TRIAGE"
        risk_color = "#d9534f" if pred_name == "PNEUMONIA" else "#2e7d32"

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=300)

        # Panel 1: Original Chest Radiograph
        axes[0].imshow(img_np)
        axes[0].set_title(f"A. Original Radiograph\nTrue Category: {true_label or 'Unknown'}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Raw Grad-CAM Heatmap
        im1 = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("B. Grad-CAM Activation Heatmap\n(Salient Pulmonary Regions)", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Superimposed Radiologic Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(
            f"C. Radiologic Superimposed Overlay\nPredicted: {pred_name} ({conf*100:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color=risk_color,
        )
        axes[2].axis("off")

        banner = (
            f"Axlero AI Diagnostic Explainability | Architecture: DenseNet-121 | "
            f"Prediction: {pred_name} ({conf*100:.1f}%) | Triage: {risk_tier}"
        )
        plt.suptitle(banner, fontsize=12, fontweight="bold", y=0.98, color=risk_color)
        plt.tight_layout()
        plt.savefig(out_p, bbox_inches="tight")
        plt.close()

        return {
            "image_path": str(image_path),
            "output_path": str(out_p),
            "predicted_class": pred_name,
            "predicted_idx": pred_idx,
            "confidence": round(conf, 4),
            "risk_tier": risk_tier,
            "true_label": true_label,
        }
