"""Performance monitoring and metrics collection."""

import functools
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

# Constants
MAX_PARAMS_FOR_LOGGING = 10


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    execution_time: float
    memory_usage_mb: float | None = None
    function_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self):
        self.metrics: dict[str, list[PerformanceMetrics]] = defaultdict(list)
        self.enabled = True

    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        if not self.enabled:
            return

        self.metrics[metric.function_name].append(metric)

        # Log slow operations
        if metric.execution_time > 1.0:  # > 1 second
            logger.warning(
                f"Slow operation: {metric.function_name} took {metric.execution_time:.2f}s"
            )

    def get_stats(self, function_name: str) -> dict[str, float]:
        """Get statistics for a function."""
        if function_name not in self.metrics:
            return {}

        times = [m.execution_time for m in self.metrics[function_name]]
        return {
            "count": len(times),
            "total_time": sum(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all functions."""
        return {name: self.get_stats(name) for name in self.metrics}

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()

    def enable(self) -> None:
        """Enable metrics collection."""
        self.enabled = True

    def disable(self) -> None:
        """Disable metrics collection."""
        self.enabled = False


# Global metrics collector
metrics_collector = MetricsCollector()


def performance_monitor(
    track_memory: bool = False, log_params: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to monitor function performance.

    Args:
        track_memory: Whether to track memory usage (requires psutil)
        log_params: Whether to log function parameters
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            memory_before = None

            if track_memory and psutil is not None:
                try:
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    logger.warning("psutil not available for memory tracking")

            try:
                result = func(*args, **kwargs)

                execution_time = time.time() - start_time
                memory_usage = None

                if track_memory and memory_before is not None and psutil is not None:
                    try:
                        process = psutil.Process()
                        memory_after = process.memory_info().rss / 1024 / 1024  # MB
                        memory_usage = memory_after - memory_before
                    except ImportError:
                        pass

                # Prepare parameters for logging
                params = {}
                if (
                    log_params and len(args) + len(kwargs) < MAX_PARAMS_FOR_LOGGING
                ):  # Avoid huge param logs
                    params = {
                        "args_count": len(args),
                        "kwargs": list(kwargs.keys())[:5],  # First 5 kwarg names
                    }

                metric = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    function_name=f"{func.__module__}.{func.__name__}",
                    parameters=params,
                )

                metrics_collector.record_metric(metric)

                return result

            except Exception as e:
                # Still record the metric for failed operations
                execution_time = time.time() - start_time
                metric = PerformanceMetrics(
                    execution_time=execution_time,
                    function_name=f"{func.__module__}.{func.__name__}",
                    parameters={"error": str(e)[:100]},  # First 100 chars of error
                )
                metrics_collector.record_metric(metric)
                raise

        return wrapper

    return decorator


@contextmanager
def performance_context(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        metric = PerformanceMetrics(
            execution_time=execution_time, function_name=operation_name
        )
        metrics_collector.record_metric(metric)


# Convenience decorator with default settings
monitor = performance_monitor()
monitor_with_memory = performance_monitor(track_memory=True)
monitor_with_params = performance_monitor(log_params=True)
