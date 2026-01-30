"""Model utilities for federated learning."""

from typing import Dict, Any, Optional

try:
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for model utilities")


def get_model(
    model_name: str,
    num_classes: int = 10,
    in_channels: int = 3,
    input_size: int = 32,
    **kwargs: Any,
) -> "nn.Module":
    """Get a model by name.

    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        in_channels: Number of input channels
        input_size: Input image size
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model

    Raises:
        ValueError: If model name is not recognized
    """
    _check_torch()

    model_name = model_name.lower()

    # ResNet variants
    if model_name.startswith("resnet"):
        from .resnet import get_resnet

        return get_resnet(model_name, num_classes, in_channels)

    # MobileNet variants
    if model_name.startswith("mobilenet"):
        from .mobilenet import get_mobilenet

        width_mult = kwargs.get("width_mult", 1.0)
        return get_mobilenet(model_name, num_classes, in_channels, width_mult)

    # Simple models
    if model_name == "simplecnn":
        from .simple_cnn import SimpleCNN

        return SimpleCNN(num_classes, in_channels, input_size)

    if model_name == "simplemlp":
        from .simple_cnn import SimpleMLP

        hidden_dims = kwargs.get("hidden_dims", (512, 256, 128))
        return SimpleMLP(num_classes, in_channels, input_size, hidden_dims)

    if model_name == "lenet":
        from .simple_cnn import LeNet

        return LeNet(num_classes, in_channels)

    raise ValueError(
        f"Unknown model: {model_name}. Available models: "
        "resnet18, resnet34, resnet50, mobilenetv2, mobilenetv3_small, "
        "mobilenetv3_large, simplecnn, simplemlp, lenet"
    )


def count_parameters(model: "nn.Module", trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    _check_torch()

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_info(model: "nn.Module") -> Dict[str, Any]:
    """Get information about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    _check_torch()

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    # Get layer info
    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_info.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "params": sum(p.numel() for p in module.parameters()),
                }
            )

    return {
        "model_name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "num_layers": len(layer_info),
        "layers": layer_info,
        "size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }


def get_model_for_dataset(
    dataset_name: str,
    model_name: str = "resnet18",
    **kwargs: Any,
) -> "nn.Module":
    """Get a model configured for a specific dataset.

    Args:
        dataset_name: Name of the dataset ('cifar10', 'mnist')
        model_name: Name of the model
        **kwargs: Additional model arguments

    Returns:
        Configured model
    """
    _check_torch()

    dataset_configs = {
        "cifar10": {"num_classes": 10, "in_channels": 3, "input_size": 32},
        "cifar100": {"num_classes": 100, "in_channels": 3, "input_size": 32},
        "mnist": {"num_classes": 10, "in_channels": 1, "input_size": 28},
        "fashion_mnist": {"num_classes": 10, "in_channels": 1, "input_size": 28},
        "emnist": {"num_classes": 62, "in_channels": 1, "input_size": 28},
    }

    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_configs:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from {list(dataset_configs.keys())}"
        )

    config = dataset_configs[dataset_name]
    config.update(kwargs)

    return get_model(model_name, **config)
