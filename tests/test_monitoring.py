"""Tests for performance monitoring."""

import time
from unittest.mock import patch

import pytest

from app.core.monitoring import MetricsCollector
from app.core.monitoring import PerformanceMetrics
from app.core.monitoring import metrics_collector
from app.core.monitoring import performance_context
from app.core.monitoring import performance_monitor


class TestMetricsCollector:
    """Test metrics collector functionality."""

    def __init__(self):
        self.collector: MetricsCollector

    def setup_method(self):
        """Setup test fixtures."""
        self.collector = MetricsCollector()

    def test_record_metric(self):
        """Test recording metrics."""
        metric = PerformanceMetrics(execution_time=0.5, function_name="test_function")

        self.collector.record_metric(metric)

        assert len(self.collector.metrics["test_function"]) == 1
        assert self.collector.metrics["test_function"][0].execution_time == 0.5

    def test_get_stats(self):
        """Test getting statistics."""
        # Record multiple metrics
        for i in range(3):
            metric = PerformanceMetrics(
                execution_time=0.1 * (i + 1),  # 0.1, 0.2, 0.3
                function_name="test_function",
            )
            self.collector.record_metric(metric)

        stats = self.collector.get_stats("test_function")

        assert stats["count"] == 3
        assert abs(stats["total_time"] - 0.6) < 1e-10
        assert abs(stats["avg_time"] - 0.2) < 1e-10
        assert stats["min_time"] == 0.1
        assert abs(stats["max_time"] - 0.3) < 1e-10

    def test_get_stats_empty(self):
        """Test getting stats for non-existent function."""
        stats = self.collector.get_stats("nonexistent")
        assert stats == {}

    def test_clear_metrics(self):
        """Test clearing metrics."""
        metric = PerformanceMetrics(execution_time=0.5, function_name="test_function")
        self.collector.record_metric(metric)

        assert len(self.collector.metrics) == 1

        self.collector.clear_metrics()
        assert len(self.collector.metrics) == 0

    def test_enable_disable(self):
        """Test enabling/disabling metrics collection."""
        metric = PerformanceMetrics(execution_time=0.5, function_name="test_function")

        self.collector.disable()
        self.collector.record_metric(metric)
        assert len(self.collector.metrics) == 0

        self.collector.enable()
        self.collector.record_metric(metric)
        assert len(self.collector.metrics["test_function"]) == 1


class TestPerformanceMonitor:
    """Test performance monitoring decorator."""

    def test_monitor_decorator(self):
        """Test basic monitoring decorator."""
        collector = MetricsCollector()

        @performance_monitor()
        def test_function(duration=0.1):
            time.sleep(duration)
            return "result"

        # Patch the global collector
        with patch("app.core.monitoring.metrics_collector", collector):
            result = test_function(0.05)

            assert result == "result"
            function_name = f"{test_function.__module__}.{test_function.__name__}"
            stats = collector.get_stats(function_name)
            assert stats["count"] == 1
            assert stats["avg_time"] >= 0.05

    def test_monitor_decorator_with_exception(self):
        """Test monitoring decorator with exceptions."""
        collector = MetricsCollector()

        @performance_monitor()
        def failing_function():
            raise ValueError("Test error")

        with patch("app.core.monitoring.metrics_collector", collector):
            with pytest.raises(ValueError, match="Test error"):
                failing_function()

            function_name = f"{failing_function.__module__}.{failing_function.__name__}"
            stats = collector.get_stats(function_name)
            assert stats["count"] == 1

    @pytest.mark.slow
    def test_slow_operation_warning(self, caplog):
        """Test warning for slow operations."""
        collector = MetricsCollector()

        @performance_monitor()
        def slow_function():
            time.sleep(1.1)  # > 1 second
            return "done"

        with patch("app.core.monitoring.metrics_collector", collector):
            slow_function()

            assert "Slow operation" in caplog.text


class TestPerformanceContext:
    """Test performance context manager."""

    def test_context_manager(self):
        """Test performance context manager."""
        collector = MetricsCollector()

        with patch("app.core.monitoring.metrics_collector", collector):
            with performance_context("test_operation"):
                time.sleep(0.05)

            stats = collector.get_stats("test_operation")
            assert stats["count"] == 1
            assert stats["avg_time"] >= 0.05

    def test_context_manager_with_exception(self):
        """Test context manager with exceptions."""
        collector = MetricsCollector()

        with (
            patch("app.core.monitoring.metrics_collector", collector),
            pytest.raises(ValueError, match="Test error"),
        ):
            with performance_context("test_operation"):
                raise ValueError("Test error")

        stats = collector.get_stats("test_operation")
        assert stats["count"] == 1


@pytest.fixture(autouse=True)
def reset_global_collector():
    """Reset global metrics collector after each test."""
    yield
    metrics_collector.clear_metrics()
    metrics_collector.enable()
