"""Data partitioning utilities for federated learning."""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod

try:
    from torch.utils.data import Dataset, Subset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = Any
    Subset = Any


class DataPartitioner(ABC):
    """Abstract base class for data partitioners."""
    
    @abstractmethod
    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        """Partition dataset into client subsets.
        
        Args:
            dataset: The dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of indices
        """
        pass


class IIDPartitioner(DataPartitioner):
    """IID (Independent and Identically Distributed) partitioner."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize the partitioner.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
    
    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        """Partition dataset IID among clients.
        
        Args:
            dataset: The dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of indices
        """
        np.random.seed(self.seed)
        
        indices = np.random.permutation(len(dataset))
        split_size = len(dataset) // num_clients
        
        partitions = {}
        for i in range(num_clients):
            start = i * split_size
            end = start + split_size if i < num_clients - 1 else len(dataset)
            partitions[i] = indices[start:end].tolist()
        
        return partitions


class DirichletPartitioner(DataPartitioner):
    """Non-IID partitioner using Dirichlet distribution."""
    
    def __init__(
        self,
        alpha: float = 0.5,
        seed: int = 42,
        min_samples: int = 10,
    ) -> None:
        """Initialize the partitioner.
        
        Args:
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            seed: Random seed for reproducibility
            min_samples: Minimum samples per client
        """
        self.alpha = alpha
        self.seed = seed
        self.min_samples = min_samples
    
    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        """Partition dataset non-IID using Dirichlet distribution.
        
        Args:
            dataset: The dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of indices
        """
        np.random.seed(self.seed)
        
        # Get labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            labels = np.array([y for _, y in dataset])
        
        num_classes = len(np.unique(labels))
        
        partitions: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
        
        for class_idx in range(num_classes):
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet(np.repeat(self.alpha, num_clients))
            proportions = np.array([
                p * (len(idx) >= self.min_samples)
                for p, idx in zip(proportions, [partitions[i] for i in range(num_clients)])
            ])
            proportions = proportions / proportions.sum()
            proportions = (proportions * len(class_indices)).astype(int)
            proportions[-1] = len(class_indices) - proportions[:-1].sum()
            
            start = 0
            for c_id in range(num_clients):
                end = start + proportions[c_id]
                partitions[c_id].extend(class_indices[start:end].tolist())
                start = end
        
        return partitions


class ShardPartitioner(DataPartitioner):
    """Non-IID partitioner using shards (each client gets specific classes)."""
    
    def __init__(
        self,
        shards_per_client: int = 2,
        seed: int = 42,
    ) -> None:
        """Initialize the partitioner.
        
        Args:
            shards_per_client: Number of class shards per client
            seed: Random seed for reproducibility
        """
        self.shards_per_client = shards_per_client
        self.seed = seed
    
    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        """Partition dataset using shards.
        
        Args:
            dataset: The dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of indices
        """
        np.random.seed(self.seed)
        
        # Get labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            labels = np.array([y for _, y in dataset])
        
        num_classes = len(np.unique(labels))
        
        # Sort by label
        sorted_indices = np.argsort(labels)
        
        # Create shards
        total_shards = num_clients * self.shards_per_client
        shard_size = len(dataset) // total_shards
        shards = [
            sorted_indices[i * shard_size:(i + 1) * shard_size].tolist()
            for i in range(total_shards)
        ]
        
        # Randomly assign shards to clients
        shard_indices = np.random.permutation(total_shards)
        
        partitions = {}
        for i in range(num_clients):
            client_shards = shard_indices[
                i * self.shards_per_client:(i + 1) * self.shards_per_client
            ]
            partitions[i] = []
            for shard_idx in client_shards:
                partitions[i].extend(shards[shard_idx])
        
        return partitions


def partition_data(
    dataset: Dataset,
    num_clients: int,
    strategy: str = "iid",
    **kwargs: Any,
) -> Dict[int, List[int]]:
    """Partition dataset among clients using specified strategy.
    
    Args:
        dataset: The dataset to partition
        num_clients: Number of clients
        strategy: Partitioning strategy ('iid', 'dirichlet', 'shard')
        **kwargs: Additional arguments for the partitioner
        
    Returns:
        Dictionary mapping client_id to list of indices
    """
    if strategy == "iid":
        partitioner = IIDPartitioner(seed=kwargs.get("seed", 42))
    elif strategy == "dirichlet":
        partitioner = DirichletPartitioner(
            alpha=kwargs.get("alpha", 0.5),
            seed=kwargs.get("seed", 42),
            min_samples=kwargs.get("min_samples", 10),
        )
    elif strategy == "shard":
        partitioner = ShardPartitioner(
            shards_per_client=kwargs.get("shards_per_client", 2),
            seed=kwargs.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown partitioning strategy: {strategy}")
    
    return partitioner.partition(dataset, num_clients)


def create_iid_partitions(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Create IID partitions (convenience function).
    
    Args:
        dataset: The dataset to partition
        num_clients: Number of clients
        seed: Random seed
        
    Returns:
        Dictionary mapping client_id to list of indices
    """
    return partition_data(dataset, num_clients, "iid", seed=seed)


def create_non_iid_partitions(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Create non-IID partitions using Dirichlet (convenience function).
    
    Args:
        dataset: The dataset to partition
        num_clients: Number of clients
        alpha: Dirichlet alpha parameter
        seed: Random seed
        
    Returns:
        Dictionary mapping client_id to list of indices
    """
    return partition_data(dataset, num_clients, "dirichlet", alpha=alpha, seed=seed)


def get_partition_stats(
    partitions: Dict[int, List[int]],
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Get statistics about partitions.
    
    Args:
        partitions: Dictionary mapping client_id to indices
        labels: Array of labels for the dataset
        
    Returns:
        Dictionary with partition statistics
    """
    num_classes = len(np.unique(labels))
    
    stats = {
        "num_clients": len(partitions),
        "total_samples": sum(len(idx) for idx in partitions.values()),
        "samples_per_client": {
            cid: len(idx) for cid, idx in partitions.items()
        },
        "class_distribution": {},
    }
    
    # Calculate class distribution per client
    for cid, indices in partitions.items():
        client_labels = labels[indices]
        class_counts = {}
        for c in range(num_classes):
            class_counts[c] = int(np.sum(client_labels == c))
        stats["class_distribution"][cid] = class_counts
    
    return stats
