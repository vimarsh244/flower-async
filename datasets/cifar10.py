"""CIFAR-10 dataset utilities for federated learning."""

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


# Standard CIFAR-10 normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def get_cifar10_transforms(
    train: bool = True,
    augment: bool = True,
) -> "transforms.Compose":
    """Get CIFAR-10 transforms.
    
    Args:
        train: Whether these are training transforms
        augment: Whether to apply data augmentation
        
    Returns:
        Composed transforms
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and torchvision are required for CIFAR-10")
    
    if train and augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def load_cifar10(
    data_dir: str = "./data",
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and torchvision are required for CIFAR-10")
    
    train_transform = get_cifar10_transforms(train=True)
    test_transform = get_cifar10_transforms(train=False)
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )
    
    return train_dataset, test_dataset


def get_cifar10_client_data(
    client_id: int,
    num_clients: int,
    data_dir: str = "./data",
    iid: bool = True,
    batch_size: int = 32,
    num_workers: int = 2,
    alpha: float = 0.5,
) -> Tuple[DataLoader, DataLoader, int]:
    """Get CIFAR-10 data for a specific client.
    
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
        raise ImportError("PyTorch and torchvision are required for CIFAR-10")
    
    train_dataset, test_dataset = load_cifar10(data_dir)
    
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
    """Get IID partition indices for a client.
    
    Args:
        dataset_size: Total size of dataset
        num_clients: Number of clients
        client_id: Client ID
        
    Returns:
        List of indices for this client
    """
    indices = list(range(dataset_size))
    np.random.seed(42)  # Reproducible partitioning
    np.random.shuffle(indices)
    
    # Split evenly
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
    """Get non-IID partition indices using Dirichlet distribution.
    
    Args:
        dataset: The dataset
        num_clients: Number of clients
        client_id: Client ID
        alpha: Dirichlet concentration parameter
        
    Returns:
        List of indices for this client
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    num_classes = len(np.unique(labels))
    
    np.random.seed(42)  # Reproducible partitioning
    
    # Generate Dirichlet distribution for each class
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_indices)).astype(int)
        
        # Adjust to ensure all samples are assigned
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        # Assign indices to clients
        start = 0
        for c_id in range(num_clients):
            end = start + proportions[c_id]
            client_indices[c_id].extend(class_indices[start:end].tolist())
            start = end
    
    return client_indices[client_id]


def get_cifar10_info() -> Dict[str, Any]:
    """Get information about CIFAR-10 dataset.
    
    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "CIFAR-10",
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "train_size": 50000,
        "test_size": 10000,
        "classes": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
    }
