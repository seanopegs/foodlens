"""Model definitions: a custom CNN baseline and two ImageNet-pretrained backbones."""
from __future__ import annotations

from torch import nn
from torchvision import models

ARCHITECTURES = ("custom_cnn", "mobilenet_v2", "efficientnet_b0")
DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2, then MaxPool halves the resolution."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """Baseline trained from scratch.

    Five conv blocks take the 224 px input down to 7 px (channels 32 -> 64 -> 128 ->
    256 -> 256). Global average pooling, dropout and one fully connected layer then
    map the 256 features to the 101 classes.
    """

    def __init__(self, num_classes: int = 101, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),     # 224 -> 112
            _conv_block(32, 64),    # 112 -> 56
            _conv_block(64, 128),   # 56 -> 28
            _conv_block(128, 256),  # 28 -> 14
            _conv_block(256, 256),  # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def _freeze_all_but_last(features: nn.Sequential, trainable_blocks: int) -> None:
    """Freeze the early blocks (generic edges and textures), train the last few."""
    cut = len(features) - trainable_blocks
    for i, block in enumerate(features):
        for p in block.parameters():
            p.requires_grad = i >= cut


def build_model(arch: str, num_classes: int = 101, pretrained: bool = True,
                trainable_blocks: int = 3) -> nn.Module:
    """Build one of ``ARCHITECTURES`` with a ``num_classes``-way output layer.

    For the pretrained models the ImageNet classifier is replaced by a new linear
    layer, and only the last ``trainable_blocks`` feature blocks plus that layer
    are trained.
    """
    if arch == "custom_cnn":
        return CustomCNN(num_classes)
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown architecture {arch!r}, choose from {ARCHITECTURES}")
    if pretrained:
        _freeze_all_but_last(model.features, trainable_blocks)
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
