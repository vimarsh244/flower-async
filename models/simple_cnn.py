"""Simple CNN and MLP models for federated learning.

These models are lightweight alternatives for quick experiments.
"""

from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for neural network models")


class SimpleCNN(nn.Module):
    """Simple CNN for image classification.
    
    A lightweight CNN suitable for CIFAR-10 and MNIST datasets.
    
    Attributes:
        num_classes: Number of output classes
        in_channels: Number of input channels
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        input_size: int = 32,
    ) -> None:
        """Initialize SimpleCNN.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (3 for CIFAR, 1 for MNIST)
            input_size: Input image size (32 for CIFAR, 28 for MNIST)
        """
        _check_torch()
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate feature map size after convolutions
        feature_size = input_size // 8  # After 3 pooling layers
        self.fc_input_dim = 128 * feature_size * feature_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten and FC layers
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for classification.
    
    A basic MLP that can be used for any flattened input.
    
    Attributes:
        num_classes: Number of output classes
        input_dim: Flattened input dimension
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        input_size: int = 28,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
    ) -> None:
        """Initialize SimpleMLP.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            input_size: Input image size
            hidden_dims: Tuple of hidden layer dimensions
        """
        _check_torch()
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = input_size
        self.input_dim = in_channels * input_size * input_size
        
        # Build layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x.view(-1, self.input_dim)
        return self.network(x)


class LeNet(nn.Module):
    """LeNet-5 architecture.
    
    Classic CNN architecture, good for MNIST.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
    ) -> None:
        """Initialize LeNet.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
        """
        _check_torch()
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
