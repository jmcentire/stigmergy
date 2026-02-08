"""Observability and metrics collection.

Provides the interface for monitoring system behavior:
- Signal processing latency (p50, p95, p99)
- Familiarity score distributions
- Consensus resolution rates
- Constraint kill/redact/pass rates
- Context lifecycle events (fork/merge/decay)
- Token budget utilization
- Agent confidence distributions
- Queue depths and backpressure events

Same interface whether backed by Prometheus, DataDog, OpenTelemetry,
or the in-memory mock. The system emits metric events; the collector
routes them to whatever backend is configured.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricEvent:
    """A single metric observation."""
    name: str
    value: float
    type: MetricType
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


class MetricsBackend(Protocol):
    """Protocol for metrics backends (Prometheus, DataDog, OTel, etc.)."""
    def emit(self, event: MetricEvent) -> None: ...
    def flush(self) -> None: ...


class MetricsCollector:
    """In-memory metrics collector with aggregation.

    Collects all metric events for analysis and test assertions.
    Provides pre-built aggregations (percentiles, rates, distributions).

    In production, swap this for an OTel-backed collector that exports
    to your monitoring stack. The emit() interface doesn't change.
    """

    def __init__(self) -> None:
        self._events: list[MetricEvent] = []
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._timers: dict[str, list[float]] = defaultdict(list)

    def emit(self, event: MetricEvent) -> None:
        """Record a metric event."""
        self._events.append(event)
        if event.type == MetricType.COUNTER:
            key = self._key(event.name, event.tags)
            self._counters[key] += event.value
        elif event.type == MetricType.GAUGE:
            key = self._key(event.name, event.tags)
            self._gauges[key] = event.value
        elif event.type == MetricType.HISTOGRAM:
            key = self._key(event.name, event.tags)
            self._histograms[key].append(event.value)
        elif event.type == MetricType.TIMER:
            key = self._key(event.name, event.tags)
            self._timers[key].append(event.value)

    # -- Convenience emitters --

    def count(self, name: str, value: float = 1.0, **tags: str) -> None:
        self.emit(MetricEvent(name=name, value=value, type=MetricType.COUNTER, tags=tags))

    def gauge(self, name: str, value: float, **tags: str) -> None:
        self.emit(MetricEvent(name=name, value=value, type=MetricType.GAUGE, tags=tags))

    def histogram(self, name: str, value: float, **tags: str) -> None:
        self.emit(MetricEvent(name=name, value=value, type=MetricType.HISTOGRAM, tags=tags))

    def timer(self, name: str, duration_ms: float, **tags: str) -> None:
        self.emit(MetricEvent(name=name, value=duration_ms, type=MetricType.TIMER, tags=tags))

    def start_timer(self) -> _TimerContext:
        """Context manager for timing operations."""
        return _TimerContext(self)

    # -- Aggregations --

    def counter_value(self, name: str, **tags: str) -> float:
        return self._counters.get(self._key(name, tags), 0.0)

    def gauge_value(self, name: str, **tags: str) -> float | None:
        return self._gauges.get(self._key(name, tags))

    def percentile(self, name: str, p: float, **tags: str) -> float | None:
        """Get the p-th percentile of a histogram or timer. p in [0, 100]."""
        key = self._key(name, tags)
        values = self._histograms.get(key) or self._timers.get(key)
        if not values:
            return None
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def mean(self, name: str, **tags: str) -> float | None:
        key = self._key(name, tags)
        values = self._histograms.get(key) or self._timers.get(key)
        if not values:
            return None
        return statistics.mean(values)

    def stdev(self, name: str, **tags: str) -> float | None:
        key = self._key(name, tags)
        values = self._histograms.get(key) or self._timers.get(key)
        if not values or len(values) < 2:
            return None
        return statistics.stdev(values)

    def rate(self, name: str, window_seconds: float = 60.0, **tags: str) -> float:
        """Events per second in the last window."""
        cutoff = time.monotonic() - window_seconds
        matching = [
            e for e in self._events
            if e.name == name and e.timestamp >= cutoff
            and all(e.tags.get(k) == v for k, v in tags.items())
        ]
        if not matching:
            return 0.0
        return len(matching) / window_seconds

    def summary(self, name: str, **tags: str) -> dict[str, Any]:
        """Full statistical summary for a histogram or timer metric."""
        key = self._key(name, tags)
        values = self._histograms.get(key) or self._timers.get(key)
        if not values:
            return {"count": 0}
        sorted_vals = sorted(values)
        return {
            "count": len(values),
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": statistics.mean(values),
            "p50": sorted_vals[len(sorted_vals) // 2],
            "p95": sorted_vals[int(len(sorted_vals) * 0.95)],
            "p99": sorted_vals[int(len(sorted_vals) * 0.99)],
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    # -- Query --

    def events(self, name: str | None = None, **tags: str) -> list[MetricEvent]:
        """Query raw events. Filter by name and/or tags."""
        results = self._events
        if name:
            results = [e for e in results if e.name == name]
        if tags:
            results = [e for e in results if all(e.tags.get(k) == v for k, v in tags.items())]
        return results

    @property
    def event_count(self) -> int:
        return len(self._events)

    def flush(self) -> None:
        """No-op for mock. Real backends would flush buffers."""
        pass

    def reset(self) -> None:
        self._events.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timers.clear()

    @staticmethod
    def _key(name: str, tags: dict[str, str]) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"


class _TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector) -> None:
        self._collector = collector
        self._start: float = 0.0
        self._name: str = ""
        self._tags: dict[str, str] = {}

    def __call__(self, name: str, **tags: str) -> _TimerContext:
        self._name = name
        self._tags = tags
        return self

    def __enter__(self) -> _TimerContext:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.monotonic() - self._start) * 1000
        self._collector.timer(self._name, elapsed_ms, **self._tags)
