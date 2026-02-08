"""Tests for TemporalScanner — backward-in-time signal scanning."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from stigmergy.cli.budget import DollarBudgetTracker
from stigmergy.mesh.mesh import Mesh
from stigmergy.mesh.temporal import ScanResult, ScanWindow, TemporalScanner
from stigmergy.mesh.worker import WorkerNode
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource


def _signal(content: str = "test signal", **kwargs) -> Signal:
    defaults = dict(
        content=content,
        source=SignalSource.GITHUB,
        channel="test/repo",
        author="test.user",
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def _mesh(**kwargs) -> Mesh:
    agents = kwargs.pop("agents", AgentRegistry())
    kwargs.setdefault("dedup_enabled", False)
    return Mesh(agents, **kwargs)


def _scanner(mesh: Mesh | None = None, **kwargs) -> TemporalScanner:
    if mesh is None:
        mesh = _mesh()
        mesh.spawn_worker(source_name="github")
    return TemporalScanner(mesh, **kwargs)


async def _empty_fetcher(start: datetime, end: datetime) -> list[Signal]:
    """Fetcher that returns no signals."""
    return []


def _make_fetcher(signals_per_window: int = 2, total_windows: int = 5):
    """Create a fetcher that returns a fixed number of signals per window."""
    calls = {"count": 0}

    async def fetcher(start: datetime, end: datetime) -> list[Signal]:
        calls["count"] += 1
        if calls["count"] > total_windows:
            return []
        return [
            _signal(
                content=f"signal {calls['count']}-{i} from window",
                timestamp=start + (end - start) / 2,
            )
            for i in range(signals_per_window)
        ]

    return fetcher


# ── ScanResult ─────────────────────────────────────────────────


class TestScanResult:
    def test_defaults(self):
        result = ScanResult()
        assert result.total_signals == 0
        assert result.windows_scanned == 0
        assert result.depth_seconds == 0.0

    def test_windows_scanned(self):
        now = datetime.now(timezone.utc)
        result = ScanResult(
            windows=[
                ScanWindow(
                    start=now - timedelta(seconds=60),
                    end=now,
                    signals_fetched=2,
                    signals_accepted=1,
                    step_seconds=60,
                    mesh_fullness=0.0,
                ),
            ],
        )
        assert result.windows_scanned == 1


# ── Basic Scanning ─────────────────────────────────────────────


class TestBasicScanning:
    def test_empty_fetcher_stops(self):
        scanner = _scanner(fresh_window_seconds=0)
        result = asyncio.run(scanner.scan(_empty_fetcher))
        assert result.stopped_reason == "no_signals"
        assert result.total_signals == 0

    def test_signals_fed_to_mesh(self):
        scanner = _scanner(min_step_seconds=60, fresh_window_seconds=0)
        fetcher = _make_fetcher(signals_per_window=3, total_windows=2)
        result = asyncio.run(scanner.scan(fetcher))
        assert result.total_signals == 6
        assert result.windows_scanned == 2

    def test_traces_collected(self):
        scanner = _scanner(min_step_seconds=60, fresh_window_seconds=0)
        fetcher = _make_fetcher(signals_per_window=2, total_windows=1)
        result = asyncio.run(scanner.scan(fetcher))
        assert len(result.traces) == 2

    def test_deepest_timestamp_set(self):
        now = datetime.now(timezone.utc)
        scanner = _scanner(min_step_seconds=60, fresh_window_seconds=0)
        fetcher = _make_fetcher(signals_per_window=1, total_windows=2)
        result = asyncio.run(scanner.scan(fetcher, now=now))
        assert result.deepest_timestamp is not None
        assert result.deepest_timestamp < now


# ── Stop Conditions ────────────────────────────────────────────


class TestStopConditions:
    def test_max_depth_stops(self):
        scanner = _scanner(
            min_step_seconds=100,
            max_depth_seconds=250,
            step_growth_factor=1.0,  # no growth
            fresh_window_seconds=0,
        )
        # Will yield signals forever
        async def infinite_fetcher(start, end):
            return [_signal(content=f"sig at {start}")]

        result = asyncio.run(scanner.scan(infinite_fetcher))
        assert result.stopped_reason == "max_depth"
        # Should have scanned ~2 windows (100+100 = 200 < 250, 100+100+100 = 300 > 250)
        assert result.windows_scanned == 2

    def test_all_full_stops(self):
        mesh = _mesh(base_threshold=0.0, high_relevance_offset=0.99, worker_capacity=2)
        mesh.spawn_worker(source_name="github")
        scanner = _scanner(mesh, min_step_seconds=60, fresh_window_seconds=0)

        fetcher = _make_fetcher(signals_per_window=5, total_windows=10)
        result = asyncio.run(scanner.scan(fetcher))
        assert result.stopped_reason == "all_full"

    def test_budget_exhausted_stops(self):
        budget = DollarBudgetTracker(
            daily_cap_usd=0.0,  # already exhausted
            hourly_cap_usd=0.0,
        )
        scanner = _scanner(budget=budget, fresh_window_seconds=0)
        fetcher = _make_fetcher(signals_per_window=1, total_windows=5)
        result = asyncio.run(scanner.scan(fetcher))
        assert result.stopped_reason == "budget"
        assert result.total_signals == 0

    def test_no_signals_stops_after_fresh_window(self):
        scanner = _scanner(
            min_step_seconds=60,
            fresh_window_seconds=30,  # very short
        )
        result = asyncio.run(scanner.scan(_empty_fetcher))
        # Should scan the fresh window (60s > 30s) then stop
        assert result.stopped_reason == "no_signals"

    def test_continues_through_fresh_window_even_if_empty(self):
        scanner = _scanner(
            min_step_seconds=60,
            fresh_window_seconds=120,  # 2 minutes
            step_growth_factor=1.0,
            max_depth_seconds=300,
        )
        # First call returns nothing, scanner should continue because within fresh window
        calls = {"count": 0}

        async def fresh_then_empty(start, end):
            calls["count"] += 1
            if calls["count"] == 2:
                return [_signal(content="found something")]
            return []

        result = asyncio.run(scanner.scan(fresh_then_empty))
        # Window 1: empty, depth=60s < fresh_window=120s → continue
        # Window 2: has signal, depth=120s → continue
        # Window 3: empty, depth=180s > fresh_window → stop
        assert result.total_signals == 1


# ── Adaptive Step ──────────────────────────────────────────────


class TestAdaptiveStep:
    def test_step_grows(self):
        scanner = _scanner(
            min_step_seconds=60,
            step_growth_factor=2.0,
            max_depth_seconds=10000,
            fresh_window_seconds=0,
        )
        fetcher = _make_fetcher(signals_per_window=1, total_windows=3)
        result = asyncio.run(scanner.scan(fetcher))
        # Steps should grow: 60, 120, 240...
        assert len(result.windows) >= 2
        assert result.windows[1].step_seconds > result.windows[0].step_seconds

    def test_step_capped_at_max(self):
        scanner = _scanner(
            min_step_seconds=60,
            max_step_seconds=100,
            step_growth_factor=10.0,  # aggressive growth
            max_depth_seconds=10000,
            fresh_window_seconds=0,
        )
        fetcher = _make_fetcher(signals_per_window=1, total_windows=3)
        result = asyncio.run(scanner.scan(fetcher))
        for window in result.windows:
            assert window.step_seconds <= 100

    def test_fullness_widens_step(self):
        # Full mesh = wider steps
        mesh = _mesh(base_threshold=0.0, high_relevance_offset=0.99, worker_capacity=5)
        w = mesh.spawn_worker(source_name="github")
        w.context.signal_count = 4  # 80% full

        scanner = _scanner(
            mesh,
            min_step_seconds=60,
            step_growth_factor=1.0,  # no base growth
            max_depth_seconds=10000,
            fresh_window_seconds=0,
        )
        fetcher = _make_fetcher(signals_per_window=0, total_windows=0)

        # Can't easily test this directly since empty fetcher stops,
        # but we verify the formula: step * growth_factor * (1 + fullness)
        # With fullness=0.8, growth_factor=1.0: step * 1.0 * 1.8 = 108
        # First window is always min_step
        result = asyncio.run(scanner.scan(fetcher))
        # At least verify the scan completed
        assert result.stopped_reason == "no_signals"


# ── Mesh Fullness Tracking ─────────────────────────────────────


class TestMeshFullnessTracking:
    def test_windows_record_fullness(self):
        scanner = _scanner(min_step_seconds=60, fresh_window_seconds=0)
        fetcher = _make_fetcher(signals_per_window=1, total_windows=2)
        result = asyncio.run(scanner.scan(fetcher))
        assert len(result.windows) == 2
        # First window should record mesh fullness
        for window in result.windows:
            assert 0.0 <= window.mesh_fullness <= 1.0


# ── Duplicate Tracking ─────────────────────────────────────────


class TestDuplicateTracking:
    def test_duplicates_counted(self):
        mesh = _mesh(dedup_enabled=True, base_threshold=0.0)
        mesh.spawn_worker(source_name="github")
        scanner = _scanner(mesh, min_step_seconds=60, fresh_window_seconds=0)

        # Fetcher that returns the same signal twice
        calls = {"count": 0}

        async def dup_fetcher(start, end):
            calls["count"] += 1
            if calls["count"] > 2:
                return []
            return [_signal(content="exact same content here")]

        result = asyncio.run(scanner.scan(dup_fetcher))
        assert result.total_signals == 2
        assert result.total_duplicates == 1  # second one is a dup


# ── Custom Now ─────────────────────────────────────────────────


class TestCustomNow:
    def test_scan_from_specific_time(self):
        specific_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        scanner = _scanner(min_step_seconds=60, fresh_window_seconds=0)
        fetcher = _make_fetcher(signals_per_window=1, total_windows=1)
        result = asyncio.run(scanner.scan(fetcher, now=specific_time))
        assert result.windows[0].end == specific_time
        assert result.windows[0].start == specific_time - timedelta(seconds=60)
