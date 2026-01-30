"""MNIST dataset utilities for federated learning."""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Subset, Dataset
    from torchvision import datasets, transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DataLoader = Any
    Dataset = Any


# Standard MNIST normalization values
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
MNIST_CLASSES = tuple(str(i) for i in range(10))


def get_mnist_transforms(
    train: bool = True,
    augment: bool = True,
) -> "transforms.Compose":
    """Get MNIST transforms.
    
    Args:
        train: Whether these are training transforms
        augment: Whether to apply data augmentation
        
    Returns:
        Composed transforms
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and torchvision are required for MNIST")
    
    if train and augment:
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])


def load_mnist(
    data_dir: str = "./data",
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Load MNIST dataset.
    
    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and torchvision are required for MNIST")
    
    train_transform = get_mnist_transforms(train=True)
    test_transform = get_mnist_transforms(train=False)
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )
    
    return train_dataset, test_dataset


def get_mnist_client_data(
    client_id: int,
    num_clients: int,
    data_dir: str = "./data",
    iid: bool = True,
    batch_size: int = 32,
    num_workers: int = 2,
    alpha: float = 0.5,
) -> Tuple[DataLoader, DataLoader, int]:
    """Get MNIST data for a specific client.
    
    Args:
        client_id: Client ID (0 to num_clients-1)
        num_clients: Total number of clients
        data_dir: Data directory
        iid: Whether to use IID partitioning
        batch_size: Batch size for data loaders
        num_workers: Number of data loading workers
        alpha: Dirichlet alpha for non-IID partitioning
        
    Returns:
        Tuple of (train_loader, test_loader, num_samples)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and torchvision are required for MNIST")
    
    train_dataset, test_dataset = load_mnist(data_dir)
    
    # Get indices for this client
    if iid:
        train_indices = _get_iid_indices(
            len(train_dataset), num_clients, client_id
        )
        test_indices = _get_iid_indices(
            len(test_dataset), num_clients, client_id
        )
    else:
        train_indices = _get_non_iid_indices(
            train_dataset, num_clients, client_id, alpha
        )
        test_indices = _get_iid_indices(
            len(test_dataset), num_clients, client_id
        )
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader, len(train_indices)


def _get_iid_indices(
    dataset_size: int,
    num_clients: int,
    client_id: int,
) -> List[int]:
    """Get IID partition indices for a client."""
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_size = dataset_size // num_clients
    start_idx = client_id * split_size
    end_idx = start_idx + split_size if client_id < num_clients - 1 else dataset_size
    
    return indices[start_idx:end_idx]


def _get_non_iid_indices(
    dataset: Dataset,
    num_clients: int,
    client_id: int,
    alpha: float = 0.5,
) -> List[int]:
    """Get non-IID partition indices using Dirichlet distribution."""
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    num_classes = len(np.unique(labels))
    
    np.random.seed(42)
    
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        start = 0
        for c_id in range(num_clients):
            end = start + proportions[c_id]
            client_indices[c_id].extend(class_indices[start:end].tolist())
            start = end
    
    return client_indices[client_id]


def get_mnist_info() -> Dict[str, Any]:
    """Get information about MNIST dataset.
    
    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "MNIST",
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_size": 60000,
        "test_size": 10000,
        "classes": MNIST_CLASSES,
        "mean": MNIST_MEAN,
        "std": MNIST_STD,
    }
