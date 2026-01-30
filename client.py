"""Federated Learning Client for Flower Async.

This module provides a configurable FL client that works with the async server.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitRes,
    EvaluateRes,
    GetParametersRes,
    Status,
    Code,
)

from models import get_model
from datasets import get_cifar10_client_data, get_mnist_client_data


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for the FL client")


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning.

    This client supports multiple datasets, models, and training configurations.

    Attributes:
        client_id: Unique identifier for this client
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        num_samples: Number of training samples
        device: Computation device (CPU/GPU)
    """

    def __init__(
        self,
        client_id: int,
        model: "nn.Module",
        train_loader: "DataLoader",
        test_loader: "DataLoader",
        num_samples: int,
        config: Dict[str, Any],
    ) -> None:
        """Initialize the client.

        Args:
            client_id: Client ID
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            num_samples: Number of training samples
            config: Training configuration
        """
        _check_torch()

        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_samples = num_samples
        self.config = config

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training config
        self.learning_rate = config.get("learning_rate", 0.01)
        self.momentum = config.get("momentum", 0.9)
        self.weight_decay = config.get("weight_decay", 0.0001)
        self.local_epochs = config.get("local_epochs", 1)

        # For gradient mode
        self.send_gradients = config.get("send_gradients", False)
        self.initial_params: Optional[List[torch.Tensor]] = None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as numpy arrays.

        Args:
            config: Configuration dictionary

        Returns:
            List of numpy arrays representing model parameters
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays.

        Args:
            parameters: List of numpy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data.

        Args:
            parameters: Initial model parameters
            config: Training configuration

        Returns:
            Tuple of (updated parameters, num samples, metrics)
        """
        # Set initial parameters
        self.set_parameters(parameters)

        # Store initial params if sending gradients
        if self.send_gradients:
            self.initial_params = [p.clone().detach() for p in self.model.parameters()]

        # Get training config from server
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("learning_rate", self.learning_rate))

        # Train
        start_time = time.time()
        train_loss, train_acc = self._train(local_epochs, lr)
        training_time = time.time() - start_time

        # Get updated parameters or gradients
        if self.send_gradients and self.initial_params is not None:
            # Compute gradients (new - old)
            updated_params = [
                (p.data - init_p).cpu().numpy()
                for p, init_p in zip(self.model.parameters(), self.initial_params)
            ]
        else:
            updated_params = self.get_parameters(config={})

        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "training_time": float(training_time),
            "client_id": self.client_id,
        }

        return updated_params, self.num_samples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local test data.

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num samples, metrics)
        """
        self.set_parameters(parameters)

        loss, accuracy = self._test()

        return float(loss), self.num_samples, {"accuracy": float(accuracy)}

    def _train(
        self,
        epochs: int,
        learning_rate: float,
    ) -> Tuple[float, float]:
        """Train the model.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def _test(self) -> Tuple[float, float]:
        """Test the model.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy


def create_client(
    client_id: int,
    num_clients: int,
    dataset_name: str = "cifar10",
    model_name: str = "resnet18",
    config: Optional[Dict[str, Any]] = None,
) -> FlowerClient:
    """Create a Flower client.

    Args:
        client_id: Client ID
        num_clients: Total number of clients
        dataset_name: Name of the dataset
        model_name: Name of the model
        config: Training configuration

    Returns:
        Configured FlowerClient
    """
    _check_torch()

    config = config or {}

    # Get dataset
    if dataset_name.lower() == "cifar10":
        train_loader, test_loader, num_samples = get_cifar10_client_data(
            client_id=client_id,
            num_clients=num_clients,
            data_dir=config.get("data_dir", "./data"),
            iid=config.get("iid", True),
            batch_size=config.get("batch_size", 32),
            alpha=config.get("alpha", 0.5),
        )
        in_channels = 3
        num_classes = 10
        input_size = 32
    elif dataset_name.lower() == "mnist":
        train_loader, test_loader, num_samples = get_mnist_client_data(
            client_id=client_id,
            num_clients=num_clients,
            data_dir=config.get("data_dir", "./data"),
            iid=config.get("iid", True),
            batch_size=config.get("batch_size", 32),
            alpha=config.get("alpha", 0.5),
        )
        in_channels = 1
        num_classes = 10
        input_size = 28
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get model
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        input_size=input_size,
    )

    return FlowerClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_samples=num_samples,
        config=config,
    )


def start_client(
    server_address: str = "[::]:8080",
    client_id: int = 0,
    num_clients: int = 10,
    dataset_name: str = "cifar10",
    model_name: str = "resnet18",
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Start a Flower client.

    Args:
        server_address: Server address
        client_id: Client ID
        num_clients: Total number of clients
        dataset_name: Dataset name
        model_name: Model name
        config: Configuration dictionary
    """
    client = create_client(
        client_id=client_id,
        num_clients=num_clients,
        dataset_name=dataset_name,
        model_name=model_name,
        config=config,
    )

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flower Async Client")
    parser.add_argument(
        "--server", type=str, default="[::]:8080", help="Server address"
    )
    parser.add_argument("--client-id", type=int, default=0, help="Client ID")
    parser.add_argument("--num-clients", type=int, default=10, help="Total clients")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--iid", action="store_true", help="Use IID partitioning")

    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "learning_rate": args.lr,
        "iid": args.iid,
    }

    start_client(
        server_address=args.server,
        client_id=args.client_id,
        num_clients=args.num_clients,
        dataset_name=args.dataset,
        model_name=args.model,
        config=config,
    )
