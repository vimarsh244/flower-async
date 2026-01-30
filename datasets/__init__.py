"""Dataset utilities for federated learning."""

from .cifar10 import load_cifar10, get_cifar10_client_data
from .mnist import load_mnist, get_mnist_client_data
from .data_utils import (
    partition_data,
    create_iid_partitions,
    create_non_iid_partitions,
    DataPartitioner,
)

__all__ = [
    "load_cifar10",
    "get_cifar10_client_data",
    "load_mnist",
    "get_mnist_client_data",
    "partition_data",
    "create_iid_partitions",
    "create_non_iid_partitions",
    "DataPartitioner",
]
