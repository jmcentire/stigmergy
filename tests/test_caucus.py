"""Tests for Caucus — mesh warm-start seed phase."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.mesh.caucus import (
    Caucus,
    CaucusConfig,
    CaucusResult,
    RunStats,
    build_activation_matrix,
    compute_caucus_size,
    execute_caucus,
    negotiate_assignments,
    project_volume,
    select_diverse_seeds,
    suggest_worker_count,
)
from stigmergy.mesh.mesh import Mesh
from stigmergy.mesh.state_store import (
    load_mesh_state,
    restore_run_stats,
    save_mesh_state,
)
from stigmergy.mesh.worker import WorkerNode
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource


# ── Helpers (same pattern as test_mesh.py) ────────────────────


def _ctx(*, signal_count: int = 0, capacity: int = 200, **kwargs) -> Context:
    ctx = Context(capacity=capacity, **kwargs)
    ctx.signal_count = signal_count
    return ctx


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


def _worker(*, signal_count: int = 0, capacity: int = 200, source_name=None) -> WorkerNode:
    ctx = _ctx(signal_count=signal_count, capacity=capacity)
    return WorkerNode(ctx, source_name=source_name)


# ── TestProjectVolume ─────────────────────────────────────────


class TestProjectVolume:
    def test_no_stats_returns_zero(self):
        assert project_volume(None, 100.0) == 0

    def test_proportional_projection(self):
        stats = RunStats(signals=100, duration_seconds=3600.0, workers=3,
                         timestamp_iso="2025-01-01T00:00:00+00:00")
        # 2x elapsed → 2x signals
        assert project_volume(stats, 7200.0) == 200

    def test_same_duration_returns_same(self):
        stats = RunStats(signals=50, duration_seconds=1800.0, workers=2,
                         timestamp_iso="2025-01-01T00:00:00+00:00")
        assert project_volume(stats, 1800.0) == 50

    def test_zero_duration_returns_zero(self):
        stats = RunStats(signals=100, duration_seconds=0.0, workers=3,
                         timestamp_iso="2025-01-01T00:00:00+00:00")
        assert project_volume(stats, 100.0) == 0

    def test_zero_elapsed_returns_zero(self):
        stats = RunStats(signals=100, duration_seconds=3600.0, workers=3,
                         timestamp_iso="2025-01-01T00:00:00+00:00")
        assert project_volume(stats, 0.0) == 0


# ── TestComputeCaucusSize ─────────────────────────────────────


class TestComputeCaucusSize:
    def test_min_floor(self):
        # Even with low projection, floor is min_caucus
        assert compute_caucus_size(10) >= CaucusConfig().min_caucus

    def test_max_ceiling(self):
        # Very large projection capped at max_caucus
        assert compute_caucus_size(10000) <= CaucusConfig().max_caucus

    def test_fraction_range(self):
        cfg = CaucusConfig(min_caucus=1, max_caucus=100, fraction=0.10)
        # 200 signals → 10% = 20
        assert compute_caucus_size(200, cfg) == 20

    def test_zero_projected_returns_min(self):
        cfg = CaucusConfig(min_caucus=5)
        assert compute_caucus_size(0, cfg) == 5


# ── TestBuildActivationMatrix ─────────────────────────────────


class TestBuildActivationMatrix:
    def test_shape(self):
        workers = [_worker() for _ in range(3)]
        signals = [_signal(content=f"signal {i}") for i in range(5)]
        matrix = build_activation_matrix(signals, workers)
        assert matrix.shape == (3, 5)

    def test_value_range(self):
        workers = [_worker() for _ in range(2)]
        signals = [_signal() for _ in range(3)]
        matrix = build_activation_matrix(signals, workers)
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)

    def test_differentiated_workers(self):
        """Workers with different terms produce different activation profiles."""
        w1 = _worker()
        w1.context.terms = {"pricing", "revenue", "billing", "subscription"}
        w1.context.term_bloom.add_many(w1.context.terms)
        w1.context.signal_count = 5

        w2 = _worker()
        w2.context.terms = {"deploy", "kubernetes", "docker", "infrastructure"}
        w2.context.term_bloom.add_many(w2.context.terms)
        w2.context.signal_count = 5

        sig_pricing = _signal(content="pricing revenue billing model subscription tier")
        sig_deploy = _signal(content="deploy kubernetes docker infrastructure cluster")

        matrix = build_activation_matrix([sig_pricing, sig_deploy], [w1, w2])
        # w1 should score higher on pricing signal
        assert matrix[0, 0] > matrix[1, 0]
        # w2 should score higher on deploy signal
        assert matrix[1, 1] > matrix[0, 1]

    def test_empty_workers(self):
        """Empty workers all produce identical (zero-ish) scores."""
        workers = [_worker() for _ in range(3)]
        signals = [_signal()]
        matrix = build_activation_matrix(signals, workers)
        # All workers should score similarly (all empty)
        assert np.allclose(matrix[:, 0], matrix[0, 0])


# ── TestSelectDiverseSeeds ────────────────────────────────────


class TestSelectDiverseSeeds:
    def test_selects_requested_count(self):
        signals = [_signal(content=f"unique topic {i} content") for i in range(10)]
        indices = select_diverse_seeds(signals, 3)
        assert len(indices) == 3

    def test_avoids_duplicates(self):
        """Similar signals should not both be selected."""
        signals = [
            _signal(content="booking sync webhook PMS external properties"),
            _signal(content="booking sync failing PMS external webhook"),
            _signal(content="pricing revenue billing subscription model"),
            _signal(content="deploy kubernetes docker infrastructure cluster"),
        ]
        indices = select_diverse_seeds(signals, 3)
        assert len(indices) == 3
        # Signals 0 and 1 are nearly identical — at most one should be selected
        assert not (0 in indices and 1 in indices)

    def test_count_exceeds_signals(self):
        signals = [_signal(content="test") for _ in range(2)]
        indices = select_diverse_seeds(signals, 5)
        assert len(indices) == 2

    def test_empty_signals(self):
        assert select_diverse_seeds([], 3) == []

    def test_zero_count(self):
        signals = [_signal(content="test")]
        assert select_diverse_seeds(signals, 0) == []


# ── TestNegotiateAssignments ──────────────────────────────────


class TestNegotiateAssignments:
    def test_all_assigned(self):
        """Every signal gets assigned to some worker."""
        matrix = np.array([[0.5, 0.3], [0.2, 0.7]])
        workers = [_worker() for _ in range(2)]
        assignments = negotiate_assignments(matrix, workers)
        assert len(assignments) == 2
        assert set(assignments.keys()) == {0, 1}

    def test_highest_wins(self):
        matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        workers = [_worker() for _ in range(2)]
        assignments = negotiate_assignments(matrix, workers)
        assert assignments[0] == 0  # sig 0 → worker 0
        assert assignments[1] == 1  # sig 1 → worker 1

    def test_tiebreak_spreads_then_rolling_avg(self):
        """When scores tie, signals spread across workers first."""
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        w1 = _worker()
        w1.rolling_avg_familiarity = 0.3
        w2 = _worker()
        w2.rolling_avg_familiarity = 0.8
        assignments = negotiate_assignments(matrix, [w1, w2])
        # With 2 signals and 2 workers, each should get one
        assert set(assignments.values()) == {0, 1}

    def test_tiebreak_rolling_avg_after_balance(self):
        """After balancing, rolling_avg_familiarity breaks remaining ties."""
        # 3 signals, 2 workers: first 2 spread, 3rd goes to higher rolling_avg
        matrix = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        w1 = _worker()
        w1.rolling_avg_familiarity = 0.3
        w2 = _worker()
        w2.rolling_avg_familiarity = 0.8
        assignments = negotiate_assignments(matrix, [w1, w2])
        # Both workers should get at least 1 signal
        counts = {0: 0, 1: 0}
        for w_idx in assignments.values():
            counts[w_idx] += 1
        assert counts[0] >= 1
        assert counts[1] >= 1

    def test_distribution_not_all_one_worker(self):
        """With differentiated scores, signals spread across workers."""
        matrix = np.array([
            [0.9, 0.1, 0.5],
            [0.1, 0.9, 0.3],
            [0.3, 0.3, 0.9],
        ])
        workers = [_worker() for _ in range(3)]
        assignments = negotiate_assignments(matrix, workers)
        assigned_workers = set(assignments.values())
        assert len(assigned_workers) == 3  # all 3 workers got signals


# ── TestSuggestWorkerCount ────────────────────────────────────


class TestSuggestWorkerCount:
    def test_normal_no_change(self):
        """With balanced load, no scaling suggestion."""
        matrix = np.random.rand(3, 10)
        suggested, merges, fork = suggest_worker_count(
            matrix, projected_volume=100, current_count=3,
            worker_capacity=200,
        )
        assert suggested is None
        assert merges == []
        assert fork is None

    def test_underload_suggests_merge(self):
        """Very few signals per worker suggests merge."""
        # 2 workers, only 5 projected signals → < 10/worker underload threshold
        # Make workers have very similar activation profiles for merge
        row = np.random.rand(10)
        matrix = np.vstack([row, row + 0.01])  # nearly identical
        suggested, merges, fork = suggest_worker_count(
            matrix, projected_volume=5, current_count=2,
            worker_capacity=200,
        )
        assert len(merges) > 0
        assert suggested is not None
        assert suggested < 2

    def test_overload_suggests_fork(self):
        """Too many signals per worker suggests fork."""
        matrix = np.random.rand(2, 10)
        # projected_volume=300, capacity=200, overload_factor=0.5 → 150/worker > 100
        suggested, merges, fork = suggest_worker_count(
            matrix, projected_volume=300, current_count=2,
            worker_capacity=200,
        )
        assert fork is not None
        assert suggested is not None
        assert suggested > 2


# ── TestExecuteCaucus ─────────────────────────────────────────


class TestExecuteCaucus:
    def test_terms_seeded(self):
        """Workers get terms from assigned signals."""
        w = _worker()
        sig = _signal(content="pricing revenue billing model")
        matrix = np.array([[0.5]])
        execute_caucus([sig], {0: 0}, [w], matrix)
        assert len(w.context.terms) > 0
        assert "pricing" in w.context.terms

    def test_all_assigned_workers_warm(self):
        """All workers that received assignments have signal_count == 1 (warm)."""
        workers = [_worker() for _ in range(3)]
        signals = [
            _signal(content="pricing billing revenue"),
            _signal(content="deploy kubernetes docker"),
            _signal(content="testing coverage quality"),
        ]
        # Assign one signal per worker
        assignments = {0: 0, 1: 1, 2: 2}
        matrix = np.ones((3, 3)) * 0.5
        execute_caucus(signals, assignments, workers, matrix)
        for w in workers:
            # Each worker is minimally warm (signal_count = 1),
            # not incremented per signal (that's ingest's job)
            assert w.context.signal_count == 1

    def test_position_dirty(self):
        """Workers with seeded terms have dirty position caches."""
        w = _worker()
        # Force clean position cache
        _ = w.position  # triggers position computation
        w._position_dirty = False

        sig = _signal(content="pricing revenue billing")
        matrix = np.array([[0.5]])
        execute_caucus([sig], {0: 0}, [w], matrix)
        assert w._position_dirty is True

    def test_no_source_author_double_count(self):
        """Caucus does NOT seed source/author counts (ingest handles them)."""
        w = _worker()
        sig = _signal(content="test content here", source="slack", author="alice")
        matrix = np.array([[0.5]])
        execute_caucus([sig], {0: 0}, [w], matrix)
        # Source/author counts stay at 0 — ingest will set them correctly
        assert w.context.source_counts.get("slack", 0) == 0
        assert w.context.author_counts.get("alice", 0) == 0

    def test_multiple_signals_same_worker_signal_count_stays_one(self):
        """Multiple signals seeded to one worker still sets signal_count = 1."""
        w = _worker()
        signals = [_signal(content=f"content {i}") for i in range(5)]
        assignments = {i: 0 for i in range(5)}
        matrix = np.ones((1, 5)) * 0.5
        execute_caucus(signals, assignments, [w], matrix)
        assert w.context.signal_count == 1  # warm, not inflated


# ── TestCaucusOrchestrator ────────────────────────────────────


class TestCaucusOrchestrator:
    def test_no_stats_fallback(self):
        """Without previous stats, caucus still runs (uses worker-scaled min)."""
        mesh = _mesh()
        mesh.spawn_worker(source_name="test1")
        mesh.spawn_worker(source_name="test2")
        signals = [_signal(content=f"signal {i} content here") for i in range(5)]

        caucus = Caucus(mesh, prev_stats=None, elapsed_seconds=0.0)
        size = caucus.compute_size(len(signals))
        # Should scale with worker count: 2 workers * 2 min_per_worker = 4
        assert size >= 2 * CaucusConfig().min_per_worker

        result = caucus.run(signals[:size])
        assert isinstance(result, CaucusResult)
        assert result.caucus_size == size

    def test_empty_mesh(self):
        """Caucus with no workers returns empty result."""
        mesh = _mesh()
        caucus = Caucus(mesh)
        result = caucus.run([_signal()])
        assert result.caucus_size == 0
        assert result.assignments == {}

    def test_end_to_end_differentiation(self):
        """After caucus with pre-seeded workers, signals distribute based on affinity."""
        mesh = _mesh()
        w1 = mesh.spawn_worker(source_name="test1")
        w2 = mesh.spawn_worker(source_name="test2")

        # Pre-seed workers with distinct terms so familiarity differentiates
        w1.context.terms = {"pricing", "revenue", "billing"}
        w1.context.term_bloom.add_many(w1.context.terms)
        w1.context.signal_count = 3

        w2.context.terms = {"deploy", "kubernetes", "docker"}
        w2.context.term_bloom.add_many(w2.context.terms)
        w2.context.signal_count = 3

        # Signals with distinct vocabularies matching each worker
        signals = [
            _signal(content="pricing revenue billing subscription model tier"),
            _signal(content="deploy kubernetes docker infrastructure cluster node"),
            _signal(content="pricing cost revenue margin discount percentage"),
            _signal(content="deploy rollback canary release pipeline cicd"),
        ]

        stats = RunStats(signals=40, duration_seconds=60.0, workers=2,
                         timestamp_iso="2025-01-01T00:00:00+00:00")
        caucus = Caucus(mesh, prev_stats=stats, elapsed_seconds=60.0)
        size = caucus.compute_size(len(signals))
        result = caucus.run(signals[:size])

        # Both workers should be warm (signal_count = 1 from caucus seeding)
        assert w1.context.signal_count >= 1
        assert w2.context.signal_count >= 1

        # Signals should distribute across workers
        assigned_workers = set(result.assignments.values())
        assert len(assigned_workers) == 2
        assert len(result.assignments) == size

    def test_no_signals_returns_empty(self):
        """Caucus with no signals returns empty result."""
        mesh = _mesh()
        mesh.spawn_worker()
        caucus = Caucus(mesh)
        result = caucus.run([])
        assert result.caucus_size == 0

    def test_size_scales_with_worker_count(self):
        """Caucus size scales so each worker gets min_per_worker signals."""
        mesh = _mesh()
        for i in range(5):
            mesh.spawn_worker(source_name=f"test{i}")
        signals = [_signal(content=f"signal {i}") for i in range(20)]

        cfg = CaucusConfig(min_per_worker=2)
        caucus = Caucus(mesh, config=cfg)
        size = caucus.compute_size(len(signals))
        # 5 workers * 2 min_per_worker = 10
        assert size >= 10

    def test_two_phase_clusters_by_topic(self):
        """Cold-start two-phase negotiation clusters related signals together."""
        mesh = _mesh()
        w1 = mesh.spawn_worker(source_name="test1")
        w2 = mesh.spawn_worker(source_name="test2")
        w3 = mesh.spawn_worker(source_name="test3")

        # 9 signals: 3 topics × 3 signals each
        pricing_signals = [
            _signal(content="pricing revenue billing subscription model tier"),
            _signal(content="pricing discount margin percentage cost"),
            _signal(content="revenue seasonal multiplier pricing engine"),
        ]
        deploy_signals = [
            _signal(content="deploy kubernetes docker infrastructure cluster"),
            _signal(content="deploy rollback canary release pipeline"),
            _signal(content="kubernetes helm chart deployment node"),
        ]
        booking_signals = [
            _signal(content="booking reservation checkin checkout guest"),
            _signal(content="booking calendar availability property sync"),
            _signal(content="reservation confirmation guest notification"),
        ]
        signals = pricing_signals + deploy_signals + booking_signals

        caucus = Caucus(mesh)
        result = caucus.run(signals)

        # All 3 workers should be warm
        for w in [w1, w2, w3]:
            assert w.context.signal_count > 0

        # Signals should cluster: related signals assigned to same worker
        # Check that not all signals go to one worker
        worker_counts = {}
        for w_idx in result.assignments.values():
            worker_counts[w_idx] = worker_counts.get(w_idx, 0) + 1
        assert len(worker_counts) >= 2, "Signals should spread across workers"
        # No single worker should get everything
        assert max(worker_counts.values()) < len(signals)

    def test_single_worker_still_works(self):
        """Caucus with only one worker assigns everything to it."""
        mesh = _mesh()
        w = mesh.spawn_worker()
        signals = [_signal(content=f"signal {i}") for i in range(3)]
        caucus = Caucus(mesh)
        result = caucus.run(signals)
        # All assignments should point to worker 0
        assert all(v == 0 for v in result.assignments.values())
        # Worker is warm (signal_count = 1), not inflated to 3
        assert w.context.signal_count == 1


# ── TestRunStatsRoundTrip ─────────────────────────────────────


class TestRunStatsRoundTrip:
    def test_save_load_cycle(self, tmp_path):
        """RunStats survives save/load round trip."""
        stats = RunStats(
            signals=42, duration_seconds=123.4, workers=3,
            timestamp_iso="2025-06-15T12:00:00+00:00",
        )
        path = tmp_path / "mesh_state.json"
        save_mesh_state(workers=[], agents=[], run_stats=stats, path=path)

        loaded = load_mesh_state(path=path)
        restored = restore_run_stats(loaded)
        assert restored is not None
        assert restored.signals == 42
        assert restored.duration_seconds == 123.4
        assert restored.workers == 3
        assert restored.timestamp_iso == "2025-06-15T12:00:00+00:00"

    def test_missing_returns_none(self):
        """Missing run stats returns None."""
        restored = restore_run_stats({})
        assert restored is None

    def test_empty_state_returns_none(self):
        """Empty saved state returns None."""
        restored = restore_run_stats({"workers": {}, "agents": {}})
        assert restored is None
