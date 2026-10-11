"""CNN Model Architectures for Eye-State Classification in DrishtiDrive AI.

Provides PyTorch implementations for five candidate architectures:
1. Custom Compact CNN
2. Depthwise-Separable CNN
3. MobileNetV2 (Pretrained Backbone Transfer)
4. EfficientNetB0 (Pretrained Backbone Transfer)
5. ResNet50 (Pretrained Backbone Transfer)
"""
from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torchvision.models as tv_models


class CustomCNN(nn.Module):
    """Compact baseline convolutional neural network for eye state classification."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        """Initialize CustomCNN layers."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through CustomCNN."""
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable convolution block (3x3 depthwise + 1x1 pointwise)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize depthwise and pointwise convolution layers."""
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through depthwise block."""
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class DepthwiseSeparableCNN(nn.Module):
    """Efficient lightweight depthwise-separable CNN for edge devices."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        """Initialize DepthwiseSeparableCNN layers."""
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(24, 48, stride=1),
            nn.MaxPool2d(2, 2),
            DepthwiseSeparableBlock(48, 96, stride=1),
            nn.MaxPool2d(2, 2),
            DepthwiseSeparableBlock(96, 128, stride=1),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through DepthwiseSeparableCNN."""
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.gap(x)
        return self.classifier(x)


def build_mobilenetv2(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Build MobileNetV2 with transfer learning head."""
    weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v2(weights=weights)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
    )
    return model


def build_efficientnetb0(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Build EfficientNetB0 with transfer learning head."""
    weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tv_models.efficientnet_b0(weights=weights)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
    )
    return model


def build_resnet50(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Build ResNet50 with transfer learning head."""
    weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet50(weights=weights)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
    )
    return model


def get_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Instantiate one of the five candidate models by canonical key."""
    canonical = name.lower().replace("-", "_").replace(" ", "_")
    if canonical in ("custom_cnn", "custom"):
        return CustomCNN(num_classes=num_classes, dropout=dropout)
    if canonical in ("depthwise_separable_cnn", "depthwise_cnn", "depthwise"):
        return DepthwiseSeparableCNN(num_classes=num_classes, dropout=dropout)
    if canonical in ("mobilenetv2", "mobilenet_v2"):
        return build_mobilenetv2(num_classes, pretrained, freeze_backbone)
    if canonical in ("efficientnetb0", "efficientnet_b0"):
        return build_efficientnetb0(num_classes, pretrained, freeze_backbone)
    if canonical in ("resnet50", "resnet_50"):
        return build_resnet50(num_classes, pretrained, freeze_backbone)
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return count of total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
