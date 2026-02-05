# Flower Async

**Asynchronous Federated Learning with Flower**

Implementation of asynchronous federated learning built on top of the [Flower](https://flower.ai/) framework. Compatible with **flwr 1.25.0+**.

## Features

- **Asynchronous FL Server**: Non-blocking client updates with staleness-aware aggregation
- **Multiple Aggregation Strategies**:
  - **FedAsync**: `global_new = (1-α) * global_old + α * local_new`
  - **AsyncFedED**: `global_new = global_old + η * (local_new - global_old)`
  - Staleness-based weighting (polynomial, hinge, constant)
  - Sample-based weighting
- **Multiple Datasets**: CIFAR-10, MNIST with IID and non-IID partitioning
- **Multiple Models**: ResNet18/34/50, MobileNetV2/V3, SimpleCNN, LeNet
- **Hydra Configuration**: Flexible config management
- **Monitoring**: Comprehensive tracking and visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flower-async.git
cd flower-async

# Install dependencies
pip install -r requirements.txt

# Or with conda
conda create -n flops python=3.10
conda activate flops
pip install -r requirements.txt
```

## Quick Start

### Running the Server

```python
from async_server import AsyncServer
from async_client_manager import AsyncClientManager
from async_strategy import AsynchronousStrategy
from flwr.server.strategy import FedAvg

# Create async strategy
async_strategy = AsynchronousStrategy(
    total_samples=50000,
    staleness_alpha=0.5,
    fedasync_mixing_alpha=0.5,
    fedasync_a=0.5,
    num_clients=10,
    async_aggregation_strategy="fedasync",
    use_staleness=True,
    use_sample_weighing=True,
    send_gradients=False,
    server_artificial_delay=False,
)

# Create server
server = AsyncServer(
    strategy=FedAvg(...),
    client_manager=AsyncClientManager(),
    async_strategy=async_strategy,
)
```

### Running with Hydra Config

```bash
# Default configuration (CIFAR-10 + ResNet18 + FedAsync)
python main.py

# Use MNIST dataset
python main.py dataset=mnist

# Use MobileNetV2 model
python main.py model=mobilenetv2

# Use gradient aggregation
python main.py strategy=fedasync_gradients

# Combine multiple overrides
python main.py dataset=mnist model=simplecnn training.num_rounds=50

# Run client
python client.py --client-id 0 --num-clients 10 --dataset cifar10 --model resnet18
```

## Project Structure

```
flower-async/
├── conf/                      # Hydra configuration files
│   ├── config.yaml           # Main config
│   ├── dataset/              # Dataset configs (cifar10, mnist)
│   ├── model/                # Model configs (resnet18, mobilenetv2, etc.)
│   ├── strategy/             # Strategy configs (fedasync, etc.)
│   └── server/               # Server configs
├── datasets/                  # Dataset utilities
│   ├── cifar10.py            # CIFAR-10 loader
│   ├── mnist.py              # MNIST loader
│   └── data_utils.py         # Partitioning utilities
├── models/                    # Neural network models
│   ├── resnet.py             # ResNet variants
│   ├── mobilenet.py          # MobileNet variants
│   └── simple_cnn.py         # SimpleCNN, MLP, LeNet
├── monitoring_sync/           # Monitoring and tracking
│   ├── tracker.py            # Experiment tracker
│   └── metrics.py            # Metrics logger
├── async_server.py           # Async FL server
├── async_client_manager.py   # Client manager with free/busy states
├── async_strategy.py         # Async aggregation strategies
├── async_history.py          # Extended history with timestamps
├── async_visualizer.py       # Visualization utilities
├── client.py                 # FL client implementation
├── main.py                   # Main entry point
└── requirements.txt          # Dependencies
```

## Configuration Options

### Dataset Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset.name` | Dataset name (cifar10, mnist) | cifar10 |
| `data.iid` | IID partitioning | true |
| `data.alpha` | Dirichlet alpha for non-IID | 0.5 |

### Model Configuration

| Model | Config File | Parameters |
|-------|-------------|------------|
| ResNet-18 | `model/resnet18.yaml` | ~11M |
| ResNet-34 | `model/resnet34.yaml` | ~21M |
| MobileNetV2 | `model/mobilenetv2.yaml` | ~3.5M |
| MobileNetV3-Small | `model/mobilenetv3_small.yaml` | ~2.5M |
| SimpleCNN | `model/simplecnn.yaml` | ~1M |

### Strategy Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `strategy.use_staleness` | Enable staleness weighting | true |
| `strategy.staleness_alpha` | Staleness decay parameter | 0.5 |
| `strategy.fedasync_mixing_alpha` | FedAsync mixing coefficient | 0.5 |
| `strategy.fedasync_a` | Polynomial decay parameter | 0.5 |
| `strategy.use_sample_weighing` | Enable sample weighting | true |
| `strategy.send_gradients` | Send gradients instead of models | false |

## Monitoring

Use the built-in tracker to monitor experiments:

```python
from monitoring_sync import Tracker

# Initialize tracker
tracker = Tracker(
    experiment_name="my_experiment",
    output_dir="./outputs",
    config={"model": "resnet18", "dataset": "cifar10"},
)

# Log client updates
tracker.log_client_start(client_id="client_0", round_number=1)
tracker.log_client_end(client_id="client_0", round_completed=3, metrics={"loss": 0.5})

# Log aggregation
tracker.log_aggregation(round_number=3, num_clients=5, loss=0.4, accuracy=0.85)

# Save results
tracker.save()
```

## References

This implementation is based on the following papers:

- [Asynchronous Federated Optimization (FedAsync)](https://arxiv.org/pdf/1903.03934.pdf)
- [Privacy-Preserving Asynchronous FL (PAFLM)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9022982)
- [AsyncFedED: Euclidean Distance Adaptive Aggregation](https://arxiv.org/pdf/2205.13797.pdf)
- [Federated Learning on Non-IID Data Silos](https://arxiv.org/abs/2102.02079)

## License

MIT License - see [LICENSE](LICENSE) for details.
