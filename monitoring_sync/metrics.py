"""Metrics logging utilities for async federated learning."""

import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import threading


class MetricType(Enum):
    """Types of metrics that can be logged."""
    LOSS = "loss"
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    STALENESS = "staleness"
    CUSTOM = "custom"


@dataclass
class MetricEntry:
    """A single metric entry with timestamp and metadata."""
    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsLogger:
    """Logger for collecting and aggregating metrics.
    
    This class provides utilities for logging various metrics during
    federated learning training, with support for aggregation and callbacks.
    
    Attributes:
        metrics: Dictionary mapping metric names to lists of entries
        callbacks: List of callback functions called on each log
    """
    
    def __init__(self, start_time: Optional[float] = None) -> None:
        """Initialize the MetricsLogger.
        
        Args:
            start_time: Optional start time for relative timestamps
        """
        self.start_time = start_time or time.time()
        self.metrics: Dict[str, List[MetricEntry]] = defaultdict(list)
        self.callbacks: List[Callable[[MetricEntry], None]] = []
        self._lock = threading.Lock()
    
    def log(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricEntry:
        """Log a metric value.
        
        Args:
            name: Name of the metric
            value: Metric value
            metric_type: Type of metric
            metadata: Optional metadata dictionary
            
        Returns:
            The created MetricEntry
        """
        entry = MetricEntry(
            name=name,
            value=value,
            timestamp=time.time() - self.start_time,
            metric_type=metric_type,
            metadata=metadata or {},
        )
        
        with self._lock:
            self.metrics[name].append(entry)
            
            # Call registered callbacks
            for callback in self.callbacks:
                try:
                    callback(entry)
                except Exception:
                    pass  # Don't let callback errors break logging
        
        return entry
    
    def log_loss(self, value: float, **metadata: Any) -> MetricEntry:
        """Log a loss value.
        
        Args:
            value: Loss value
            **metadata: Additional metadata
            
        Returns:
            The created MetricEntry
        """
        return self.log("loss", value, MetricType.LOSS, metadata)
    
    def log_accuracy(self, value: float, **metadata: Any) -> MetricEntry:
        """Log an accuracy value.
        
        Args:
            value: Accuracy value
            **metadata: Additional metadata
            
        Returns:
            The created MetricEntry
        """
        return self.log("accuracy", value, MetricType.ACCURACY, metadata)
    
    def log_latency(self, value: float, **metadata: Any) -> MetricEntry:
        """Log a latency value.
        
        Args:
            value: Latency in seconds
            **metadata: Additional metadata
            
        Returns:
            The created MetricEntry
        """
        return self.log("latency", value, MetricType.LATENCY, metadata)
    
    def log_staleness(self, value: float, **metadata: Any) -> MetricEntry:
        """Log a staleness value.
        
        Args:
            value: Staleness (rounds behind)
            **metadata: Additional metadata
            
        Returns:
            The created MetricEntry
        """
        return self.log("staleness", value, MetricType.STALENESS, metadata)
    
    def add_callback(self, callback: Callable[[MetricEntry], None]) -> None:
        """Add a callback to be called on each log.
        
        Args:
            callback: Function that takes a MetricEntry
        """
        self.callbacks.append(callback)
    
    def get_latest(self, name: str) -> Optional[MetricEntry]:
        """Get the latest entry for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Latest MetricEntry or None
        """
        entries = self.metrics.get(name, [])
        return entries[-1] if entries else None
    
    def get_values(self, name: str) -> List[float]:
        """Get all values for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of values
        """
        return [e.value for e in self.metrics.get(name, [])]
    
    def get_timestamps(self, name: str) -> List[float]:
        """Get all timestamps for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of timestamps
        """
        return [e.timestamp for e in self.metrics.get(name, [])]
    
    def get_mean(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get the mean of a metric.
        
        Args:
            name: Metric name
            last_n: Optional number of recent entries to average
            
        Returns:
            Mean value or None if no entries
        """
        values = self.get_values(name)
        if not values:
            return None
        if last_n is not None:
            values = values[-last_n:]
        return sum(values) / len(values)
    
    def get_min(self, name: str) -> Optional[float]:
        """Get the minimum value of a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Minimum value or None
        """
        values = self.get_values(name)
        return min(values) if values else None
    
    def get_max(self, name: str) -> Optional[float]:
        """Get the maximum value of a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Maximum value or None
        """
        values = self.get_values(name)
        return max(values) if values else None
    
    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert all metrics to a dictionary.
        
        Returns:
            Dictionary with metric names as keys
        """
        return {
            name: [
                {
                    "value": e.value,
                    "timestamp": e.timestamp,
                    "type": e.metric_type.value,
                    "metadata": e.metadata,
                }
                for e in entries
            ]
            for name, entries in self.metrics.items()
        }
    
    def clear(self) -> None:
        """Clear all logged metrics."""
        with self._lock:
            self.metrics.clear()
