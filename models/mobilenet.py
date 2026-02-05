"""MobileNet models for federated learning.

Adapted for smaller input sizes (CIFAR-10, MNIST) from the standard ImageNet architecture.
"""

from typing import List, Optional, Callable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for MobileNet models")


class ConvBNReLU(nn.Sequential):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2."""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # Depthwise
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # Pointwise linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 adapted for CIFAR/MNIST.

    This implementation modifies the stride and removes the initial downsampling
    to work with 32x32 or 28x28 inputs.

    Attributes:
        num_classes: Number of output classes
        in_channels: Number of input channels
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        width_mult: float = 1.0,
    ) -> None:
        """Initialize MobileNetV2.

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            width_mult: Width multiplier for the network
        """
        _check_torch()
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Building inverted residual blocks
        # t: expansion factor, c: output channels, n: number of blocks, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Changed stride from 2 to 1 for small images
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # First layer - stride 1 for small images
        features: List[nn.Module] = [ConvBNReLU(in_channels, input_channel, stride=1)]

        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel

        # Last layer
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(
        self,
        input_channels: int,
        squeeze_factor: int = 4,
    ) -> None:
        super().__init__()
        squeeze_channels = input_channels // squeeze_factor
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class InvertedResidualV3(nn.Module):
    """Inverted residual block for MobileNetV3."""

    def __init__(
        self,
        inp: int,
        hidden_dim: int,
        oup: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []

        # Expand
        if hidden_dim != inp:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        # Depthwise
        layers.append(
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_dim,
            )
        )

        # Squeeze-and-excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))

        # Project
        layers.extend(
            [
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small adapted for CIFAR/MNIST.

    A lightweight variant of MobileNetV3 optimized for mobile devices.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        width_mult: float = 1.0,
    ) -> None:
        _check_torch()
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Configuration: kernel, exp_size, out, SE, NL, stride
        cfgs = [
            # k, exp, out, SE, NL, s
            [3, 16, 16, True, False, 1],  # Changed stride for small images
            [3, 72, 24, False, False, 2],
            [3, 88, 24, False, False, 1],
            [5, 96, 40, True, True, 2],
            [5, 240, 40, True, True, 1],
            [5, 240, 40, True, True, 1],
            [5, 120, 48, True, True, 1],
            [5, 144, 48, True, True, 1],
            [5, 288, 96, True, True, 2],
            [5, 576, 96, True, True, 1],
            [5, 576, 96, True, True, 1],
        ]

        input_channel = int(16 * width_mult)
        last_channel = int(576 * width_mult)

        # First layer
        features: List[nn.Module] = [ConvBNReLU(in_channels, input_channel, stride=1)]

        # Inverted residual blocks
        for k, exp, out, se, nl, s in cfgs:
            hidden_dim = int(exp * width_mult)
            output_channel = int(out * width_mult)
            features.append(
                InvertedResidualV3(
                    input_channel, hidden_dim, output_channel, k, s, se, nl
                )
            )
            input_channel = output_channel

        # Last layers
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV3Large(nn.Module):
    """MobileNetV3-Large adapted for CIFAR/MNIST.

    A larger variant of MobileNetV3 with better accuracy.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        width_mult: float = 1.0,
    ) -> None:
        _check_torch()
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Configuration for MobileNetV3-Large
        cfgs = [
            # k, exp, out, SE, NL, s
            [3, 16, 16, False, False, 1],
            [3, 64, 24, False, False, 1],  # Changed stride for small images
            [3, 72, 24, False, False, 1],
            [5, 72, 40, True, False, 2],
            [5, 120, 40, True, False, 1],
            [5, 120, 40, True, False, 1],
            [3, 240, 80, False, True, 2],
            [3, 200, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 480, 112, True, True, 1],
            [3, 672, 112, True, True, 1],
            [5, 672, 160, True, True, 2],
            [5, 960, 160, True, True, 1],
            [5, 960, 160, True, True, 1],
        ]

        input_channel = int(16 * width_mult)
        last_channel = int(960 * width_mult)

        # First layer
        features: List[nn.Module] = [ConvBNReLU(in_channels, input_channel, stride=1)]

        # Inverted residual blocks
        for k, exp, out, se, nl, s in cfgs:
            hidden_dim = int(exp * width_mult)
            output_channel = int(out * width_mult)
            features.append(
                InvertedResidualV3(
                    input_channel, hidden_dim, output_channel, k, s, se, nl
                )
            )
            input_channel = output_channel

        # Last layers
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_mobilenet(
    variant: str = "mobilenetv2",
    num_classes: int = 10,
    in_channels: int = 3,
    width_mult: float = 1.0,
) -> nn.Module:
    """Get a MobileNet model by name.

    Args:
        variant: Model variant ('mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large')
        num_classes: Number of output classes
        in_channels: Number of input channels
        width_mult: Width multiplier

    Returns:
        MobileNet model

    Raises:
        ValueError: If variant is not recognized
    """
    _check_torch()

    variants = {
        "mobilenetv2": MobileNetV2,
        "mobilenetv3_small": MobileNetV3Small,
        "mobilenetv3_large": MobileNetV3Large,
    }

    if variant.lower() not in variants:
        raise ValueError(
            f"Unknown MobileNet variant: {variant}. Choose from {list(variants.keys())}"
        )

    return variants[variant.lower()](num_classes, in_channels, width_mult)
