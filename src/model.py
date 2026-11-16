"""Deep Learning Model Architectures for Chest Radiograph Classification

Axlero AI Diagnostic System
Implements:
1. BaselineCNN: 4-block custom convolutional architecture for pediatric chest X-ray.
2. ResNet50Binary: ResNet50 transfer learning with custom classification head.
3. DenseNet121Binary: DenseNet121 transfer learning with custom classification head.
4. Model factory function (build_model) supporting frozen/fine-tuned training.
5. Target layer resolver for Grad-CAM explainability hooks.
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class BaselineCNN(nn.Module):
    """Custom 4-block Convolutional Neural Network for binary radiograph triage.

    Architecture:
    - Block 1: Conv2d(3, 32, 3) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    - Block 2: Conv2d(32, 64, 3) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    - Block 3: Conv2d(64, 128, 3) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    - Block 4: Conv2d(128, 256, 3) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    - Head: AdaptiveAvgPool2d((1, 1)) -> Flatten -> Linear(256, 128) -> ReLU -> Dropout(0.4) -> Linear(128, num_classes)
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


class ResNet50Binary(nn.Module):
    """ResNet50 Transfer Learning Architecture for Chest X-Ray Classification.

    Replaces the default 1000-class ImageNet fully connected layer with
    a regularized clinical classification head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNet121Binary(nn.Module):
    """DenseNet121 Transfer Learning Architecture for Chest X-Ray Classification.

    Leverages dense feature reuse, making it particularly effective for
    radiological feature representation with high parameter efficiency.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(
    architecture: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.4,
) -> nn.Module:
    """Factory function to build specified classification architecture.

    Args:
        architecture: One of 'baseline_cnn', 'resnet50', 'densenet121'.
        num_classes: Output classes (default 2: NORMAL, PNEUMONIA).
        pretrained: Load ImageNet pretrained weights.
        freeze_backbone: Freeze feature extractor weights.
        dropout_rate: Dropout probability in classification head.

    Returns:
        Configured PyTorch nn.Module.
    """
    arch = architecture.lower().strip()
    if arch in ["baseline", "baseline_cnn", "cnn"]:
        return BaselineCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif arch in ["resnet50", "resnet"]:
        return ResNet50Binary(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
        )
    elif arch in ["densenet121", "densenet"]:
        return DenseNet121Binary(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: 'baseline_cnn', 'resnet50', 'densenet121'."
        )


def get_gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Return the final convolutional layer of the model for Grad-CAM computation.

    Args:
        model: Instantiated neural network module.
        architecture: Architecture name ('baseline_cnn', 'resnet50', 'densenet121').

    Returns:
        Target nn.Module layer whose feature maps will be extracted.
    """
    arch = architecture.lower().strip()
    if arch in ["resnet50", "resnet"]:
        # Final bottleneck block of the 4th residual stage
        return model.backbone.layer4[-1]
    elif arch in ["densenet121", "densenet"]:
        # Final dense block of DenseNet121
        return model.backbone.features.denseblock4.denselayer16
    elif arch in ["baseline_cnn", "baseline", "cnn"]:
        # Final conv block
        return model.block4[0]
    else:
        raise ValueError(f"Unknown architecture for Grad-CAM: {architecture}")
