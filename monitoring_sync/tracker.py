"""Tracker for monitoring async federated learning experiments.

This module provides a Tracker class that records metrics, client states,
and timing information during async FL training.
"""

import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from threading import Lock
from pathlib import Path


@dataclass
class ClientUpdate:
    """Record of a single client update."""
    client_id: str
    round_started: int
    round_completed: int
    timestamp_start: float
    timestamp_end: float
    num_samples: int
    staleness: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoundMetrics:
    """Metrics for a single aggregation round."""
    round_number: int
    timestamp: float
    num_clients_aggregated: int
    global_loss: Optional[float] = None
    global_accuracy: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class Tracker:
    """Track and monitor async federated learning experiments.
    
    This class provides comprehensive tracking for async FL experiments,
    including client updates, aggregation rounds, timing, and custom metrics.
    
    Attributes:
        experiment_name: Name of the experiment
        output_dir: Directory to save tracking data
        client_updates: List of all client updates
        round_metrics: List of metrics per round
    """

    def __init__(
        self,
        experiment_name: str = "async_fl_experiment",
        output_dir: str = "./outputs",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save outputs
            config: Optional configuration dictionary to save
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.start_time = time.time()
        self.start_datetime = datetime.now().isoformat()
        
        # Thread-safe tracking
        self._lock = Lock()
        
        # Client tracking
        self.client_updates: List[ClientUpdate] = []
        self.client_states: Dict[str, str] = {}  # client_id -> "free" | "busy"
        self.client_round_map: Dict[str, int] = {}  # client_id -> current round
        
        # Round tracking
        self.round_metrics: List[RoundMetrics] = []
        self.current_round = 0
        
        # Timing
        self.timestamps: List[Tuple[float, str, Any]] = []
        
        # Custom metrics
        self.custom_metrics: Dict[str, List[Tuple[float, Any]]] = {}
        
        # Loss and accuracy history
        self.loss_history: List[Tuple[float, float]] = []  # (timestamp, loss)
        self.accuracy_history: List[Tuple[float, float]] = []  # (timestamp, acc)
        
    def log_client_start(
        self,
        client_id: str,
        round_number: int,
        num_samples: int = 0,
    ) -> None:
        """Log when a client starts training.
        
        Args:
            client_id: Unique identifier for the client
            round_number: The global round when client started
            num_samples: Number of training samples
        """
        with self._lock:
            self.client_states[client_id] = "busy"
            self.client_round_map[client_id] = round_number
            self.timestamps.append((time.time(), "client_start", {
                "client_id": client_id,
                "round": round_number,
                "num_samples": num_samples,
            }))
    
    def log_client_end(
        self,
        client_id: str,
        round_completed: int,
        metrics: Optional[Dict[str, float]] = None,
        num_samples: int = 0,
    ) -> ClientUpdate:
        """Log when a client finishes training.
        
        Args:
            client_id: Unique identifier for the client
            round_completed: The global round when client finished
            metrics: Training metrics (loss, accuracy, etc.)
            num_samples: Number of training samples
            
        Returns:
            ClientUpdate record
        """
        with self._lock:
            timestamp_end = time.time()
            round_started = self.client_round_map.get(client_id, round_completed)
            staleness = round_completed - round_started
            
            # Find the start timestamp
            timestamp_start = timestamp_end
            for ts, event, data in reversed(self.timestamps):
                if event == "client_start" and data.get("client_id") == client_id:
                    timestamp_start = ts
                    break
            
            update = ClientUpdate(
                client_id=client_id,
                round_started=round_started,
                round_completed=round_completed,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                num_samples=num_samples,
                staleness=staleness,
                metrics=metrics or {},
            )
            
            self.client_updates.append(update)
            self.client_states[client_id] = "free"
            self.timestamps.append((timestamp_end, "client_end", {
                "client_id": client_id,
                "round": round_completed,
                "staleness": staleness,
            }))
            
            return update
    
    def log_aggregation(
        self,
        round_number: int,
        num_clients: int,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log an aggregation event.
        
        Args:
            round_number: The aggregation round number
            num_clients: Number of clients aggregated
            loss: Global model loss
            accuracy: Global model accuracy
            metrics: Additional metrics
        """
        with self._lock:
            timestamp = time.time()
            
            round_metric = RoundMetrics(
                round_number=round_number,
                timestamp=timestamp,
                num_clients_aggregated=num_clients,
                global_loss=loss,
                global_accuracy=accuracy,
                metrics=metrics or {},
            )
            
            self.round_metrics.append(round_metric)
            self.current_round = round_number
            
            if loss is not None:
                self.loss_history.append((timestamp - self.start_time, loss))
            if accuracy is not None:
                self.accuracy_history.append((timestamp - self.start_time, accuracy))
            
            self.timestamps.append((timestamp, "aggregation", {
                "round": round_number,
                "num_clients": num_clients,
                "loss": loss,
                "accuracy": accuracy,
            }))
    
    def log_metric(self, name: str, value: Any) -> None:
        """Log a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        with self._lock:
            timestamp = time.time() - self.start_time
            if name not in self.custom_metrics:
                self.custom_metrics[name] = []
            self.custom_metrics[name].append((timestamp, value))
    
    def get_client_staleness_stats(self) -> Dict[str, float]:
        """Get staleness statistics across all client updates.
        
        Returns:
            Dictionary with mean, max, min staleness
        """
        if not self.client_updates:
            return {"mean": 0.0, "max": 0.0, "min": 0.0}
        
        staleness_values = [u.staleness for u in self.client_updates]
        return {
            "mean": sum(staleness_values) / len(staleness_values),
            "max": max(staleness_values),
            "min": min(staleness_values),
        }
    
    def get_training_duration(self) -> float:
        """Get total training duration in seconds."""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment.
        
        Returns:
            Dictionary with experiment summary
        """
        staleness_stats = self.get_client_staleness_stats()
        
        return {
            "experiment_name": self.experiment_name,
            "start_datetime": self.start_datetime,
            "duration_seconds": self.get_training_duration(),
            "total_rounds": self.current_round,
            "total_client_updates": len(self.client_updates),
            "staleness_stats": staleness_stats,
            "final_loss": self.loss_history[-1][1] if self.loss_history else None,
            "final_accuracy": self.accuracy_history[-1][1] if self.accuracy_history else None,
            "config": self.config,
        }
    
    def save(self, filename: Optional[str] = None) -> str:
        """Save tracking data to JSON file.
        
        Args:
            filename: Optional filename, defaults to experiment_name
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "summary": self.get_summary(),
            "client_updates": [asdict(u) for u in self.client_updates],
            "round_metrics": [asdict(r) for r in self.round_metrics],
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
            "custom_metrics": self.custom_metrics,
            "timestamps": self.timestamps,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        return str(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "Tracker":
        """Load tracking data from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Tracker instance with loaded data
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        summary = data.get("summary", {})
        tracker = cls(
            experiment_name=summary.get("experiment_name", "loaded_experiment"),
            config=summary.get("config", {}),
        )
        
        tracker.client_updates = [
            ClientUpdate(**u) for u in data.get("client_updates", [])
        ]
        tracker.round_metrics = [
            RoundMetrics(**r) for r in data.get("round_metrics", [])
        ]
        tracker.loss_history = data.get("loss_history", [])
        tracker.accuracy_history = data.get("accuracy_history", [])
        tracker.custom_metrics = data.get("custom_metrics", {})
        tracker.timestamps = data.get("timestamps", [])
        
        return tracker
