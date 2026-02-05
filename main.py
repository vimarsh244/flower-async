"""Main entry point for Flower Async FL experiments.

This script uses Hydra for configuration management.

Usage:
    python main.py                           # Run with default config
    python main.py dataset=mnist             # Use MNIST dataset
    python main.py model=mobilenetv2         # Use MobileNetV2
    python main.py strategy=fedasync_gradients  # Use gradient aggregation
    python main.py training.num_rounds=50    # Override specific values
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False
    DictConfig = dict

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import flwr as fl
from flwr.common import ndarrays_to_parameters, Metrics

# Local imports
from async_server import AsyncServer
from async_client_manager import AsyncClientManager
from async_strategy import AsynchronousStrategy
from async_history import AsyncHistory
from monitoring_sync import Tracker, MetricsLogger
from models import get_model, get_model_info
from datasets import load_cifar10, load_mnist

log = logging.getLogger(__name__)


def get_initial_parameters(
    model_name: str,
    dataset_config: Dict[str, Any],
) -> fl.common.Parameters:
    """Get initial model parameters.

    Args:
        model_name: Name of the model
        dataset_config: Dataset configuration

    Returns:
        Initial parameters
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required")

    model = get_model(
        model_name=model_name,
        num_classes=dataset_config.get("num_classes", 10),
        in_channels=dataset_config.get("in_channels", 3),
        input_size=dataset_config.get("input_size", 32),
    )

    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(params)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics.

    Args:
        metrics: List of (num_samples, metrics_dict) tuples

    Returns:
        Aggregated metrics
    """
    if not metrics:
        return {}

    # Get all metric keys
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())

    aggregated = {}
    total_samples = sum(num for num, _ in metrics)

    for key in all_keys:
        values = [(num, m.get(key, 0)) for num, m in metrics if key in m]
        if values and total_samples > 0:
            weighted_sum = sum(num * val for num, val in values)
            aggregated[key] = weighted_sum / total_samples

    return aggregated


def create_strategy(
    strategy_config: Dict[str, Any],
    num_clients: int,
    total_samples: int,
    initial_parameters: fl.common.Parameters,
) -> AsynchronousStrategy:
    """Create an asynchronous strategy.

    Args:
        strategy_config: Strategy configuration
        num_clients: Number of clients
        total_samples: Total samples across clients
        initial_parameters: Initial model parameters

    Returns:
        Configured strategy
    """
    return AsynchronousStrategy(
        total_samples=total_samples,
        staleness_alpha=strategy_config.get("staleness_alpha", 0.5),
        fedasync_mixing_alpha=strategy_config.get("fedasync_mixing_alpha", 0.5),
        fedasync_a=strategy_config.get("fedasync_a", 0.5),
        num_clients=num_clients,
        async_aggregation_strategy=strategy_config.get(
            "async_aggregation_strategy", "fedasync"
        ),
        use_staleness=strategy_config.get("use_staleness", True),
        use_sample_weighing=strategy_config.get("use_sample_weighing", True),
        send_gradients=strategy_config.get("send_gradients", False),
        server_artificial_delay=strategy_config.get("server_artificial_delay", False),
    )


def run_server(cfg: Dict[str, Any]) -> None:
    """Run the FL server.

    Args:
        cfg: Configuration dictionary
    """
    # Extract configs
    experiment_cfg = cfg.get("experiment", {})
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    strategy_cfg = cfg.get("strategy", {})
    training_cfg = cfg.get("training", {})
    client_cfg = cfg.get("client", {})
    server_cfg = cfg.get("server", {})

    # Setup
    experiment_name = experiment_cfg.get("name", "async_fl")
    output_dir = experiment_cfg.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tracker
    tracker = Tracker(
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=cfg,
    )

    # Get initial parameters
    initial_params = get_initial_parameters(
        model_name=model_cfg.get("name", "resnet18"),
        dataset_config=dataset_cfg,
    )

    # Log model info
    if HAS_TORCH:
        model = get_model(
            model_name=model_cfg.get("name", "resnet18"),
            num_classes=dataset_cfg.get("num_classes", 10),
            in_channels=dataset_cfg.get("in_channels", 3),
            input_size=dataset_cfg.get("input_size", 32),
        )
        model_info = get_model_info(model)
        log.info(f"Model: {model_info['model_name']}")
        log.info(f"Parameters: {model_info['total_parameters']:,}")
        log.info(f"Size: {model_info['size_mb']:.2f} MB")

    # Create strategy
    num_clients = client_cfg.get("num_clients", 10)
    # Estimate total samples
    if dataset_cfg.get("name") == "mnist":
        total_samples = 60000
    else:
        total_samples = 50000

    async_strategy = create_strategy(
        strategy_config=strategy_cfg,
        num_clients=num_clients,
        total_samples=total_samples,
        initial_parameters=initial_params,
    )

    # Create FL strategy with FedAvg base
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=training_cfg.get("fraction_fit", 0.5),
        fraction_evaluate=training_cfg.get("fraction_evaluate", 0.5),
        min_fit_clients=training_cfg.get("min_fit_clients", 2),
        min_evaluate_clients=training_cfg.get("min_evaluate_clients", 2),
        min_available_clients=training_cfg.get("min_available_clients", 2),
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Create client manager
    client_manager = AsyncClientManager()

    # Run server
    log.info(f"Starting FL server with {num_clients} clients")
    log.info(f"Dataset: {dataset_cfg.get('name', 'cifar10')}")
    log.info(f"Model: {model_cfg.get('name', 'resnet18')}")
    log.info(f"Strategy: {strategy_cfg.get('name', 'fedasync')}")
    log.info(f"Rounds: {training_cfg.get('num_rounds', 100)}")

    # Start server
    history = fl.server.start_server(
        server_address=server_cfg.get("server_address", "[::]:8080"),
        config=fl.server.ServerConfig(
            num_rounds=training_cfg.get("num_rounds", 100),
        ),
        strategy=strategy,
        client_manager=client_manager,
    )

    # Save results
    tracker.save()

    log.info("Training complete!")
    if history.losses_distributed:
        log.info(f"Final distributed loss: {history.losses_distributed[-1][1]:.4f}")
    if history.metrics_distributed:
        log.info(f"Final metrics: {history.metrics_distributed}")


def run_experiment(cfg: DictConfig) -> None:
    """Run a complete FL experiment.

    Args:
        cfg: Hydra configuration
    """
    if HAS_HYDRA:
        # Convert OmegaConf to dict
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = dict(cfg)

    log.info("Configuration:")
    log.info(f"  Dataset: {cfg_dict.get('dataset', {}).get('name', 'cifar10')}")
    log.info(f"  Model: {cfg_dict.get('model', {}).get('name', 'resnet18')}")
    log.info(f"  Strategy: {cfg_dict.get('strategy', {}).get('name', 'fedasync')}")

    run_server(cfg_dict)


if HAS_HYDRA:

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        """Main entry point with Hydra."""
        run_experiment(cfg)

else:

    def main() -> None:
        """Main entry point without Hydra."""
        # Default configuration
        default_cfg = {
            "experiment": {
                "name": "async_fl_experiment",
                "output_dir": "./outputs",
            },
            "dataset": {
                "name": "cifar10",
                "num_classes": 10,
                "in_channels": 3,
                "input_size": 32,
            },
            "model": {
                "name": "resnet18",
            },
            "strategy": {
                "name": "fedasync",
                "staleness_alpha": 0.5,
                "fedasync_mixing_alpha": 0.5,
                "fedasync_a": 0.5,
                "use_staleness": True,
                "use_sample_weighing": True,
                "send_gradients": False,
                "async_aggregation_strategy": "fedasync",
                "server_artificial_delay": False,
            },
            "training": {
                "num_rounds": 100,
                "fraction_fit": 0.5,
                "fraction_evaluate": 0.5,
                "min_fit_clients": 2,
                "min_evaluate_clients": 2,
                "min_available_clients": 2,
            },
            "client": {
                "num_clients": 10,
            },
            "server": {
                "server_address": "[::]:8080",
            },
        }
        run_experiment(default_cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
