"""ResNet models for federated learning.

Adapted for smaller input sizes (CIFAR-10, MNIST) from the standard ImageNet architecture.
"""

from typing import List, Optional, Type, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for ResNet models")


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152."""

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture adapted for CIFAR/MNIST.

    This implementation uses smaller initial convolution and no max pooling,
    making it suitable for 32x32 (CIFAR) or 28x28 (MNIST) inputs.

    Attributes:
        num_classes: Number of output classes
        in_channels: Number of input channels
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        """Initialize ResNet.

        Args:
            block: Block type (BasicBlock or Bottleneck)
            num_blocks: Number of blocks in each layer
            num_classes: Number of output classes
            in_channels: Number of input channels (3 for CIFAR, 1 for MNIST)
        """
        _check_torch()
        super().__init__()

        self.in_planes = 64
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Smaller initial convolution for CIFAR/MNIST
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet18(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """Create a ResNet-18 model.

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels

    Returns:
        ResNet-18 model
    """
    _check_torch()
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def ResNet34(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """Create a ResNet-34 model.

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels

    Returns:
        ResNet-34 model
    """
    _check_torch()
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def ResNet50(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """Create a ResNet-50 model.

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels

    Returns:
        ResNet-50 model
    """
    _check_torch()
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def get_resnet(
    variant: str = "resnet18",
    num_classes: int = 10,
    in_channels: int = 3,
) -> ResNet:
    """Get a ResNet model by name.

    Args:
        variant: Model variant ('resnet18', 'resnet34', 'resnet50')
        num_classes: Number of output classes
        in_channels: Number of input channels

    Returns:
        ResNet model

    Raises:
        ValueError: If variant is not recognized
    """
    variants = {
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
    }

    if variant.lower() not in variants:
        raise ValueError(
            f"Unknown ResNet variant: {variant}. Choose from {list(variants.keys())}"
        )

    return variants[variant.lower()](num_classes, in_channels)
