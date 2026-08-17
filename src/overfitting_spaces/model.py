from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


def cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Maintained torchvision ResNet18 with only the standard CIFAR stem."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
