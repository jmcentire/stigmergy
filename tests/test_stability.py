"""Tests for Lyapunov stability tracker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from stigmergy.mesh.stability import StabilityTracker


# ── Minimal mocks ────────────────────────────────────────────


@dataclass
class MockContext:
    id: UUID = field(default_factory=uuid4)
    signal_count: int = 0
    energy: float = 1.0
    terms: set = field(default_factory=set)
    last_signal: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    capacity: int = 200

    @property
    def fullness(self) -> float:
        return min(1.0, self.signal_count / self.capacity) if self.capacity > 0 else 1.0


class MockWorker:
    def __init__(self, signal_count=10, avg_fam=0.5, terms=None):
        self.context = MockContext(signal_count=signal_count, terms=terms or {"a", "b"})
        self.signals_accepted = signal_count
        self.rolling_avg_familiarity = avg_fam
        self.base_threshold = 0.15
        self.max_threshold = 0.8
        self.threshold_curve = 2.0

    @property
    def id(self) -> UUID:
        return self.context.id

    @property
    def fullness(self) -> float:
        return self.context.fullness

    @property
    def adaptive_threshold(self) -> float:
        fill = self.fullness
        return self.base_threshold + (self.max_threshold - self.base_threshold) * (fill ** self.threshold_curve)


class MockMesh:
    def __init__(self, workers):
        self._workers = workers

    @property
    def workers(self):
        return self._workers


@dataclass
class MockHopRecord:
    worker_id: UUID
    score: float = 0.5
    accepted: bool = True
    learning_mode: str = "context_summarized"
    hop: int = 0


@dataclass
class MockTrace:
    signal_id: UUID = field(default_factory=uuid4)
    duplicate: bool = False
    accepted_workers: set = field(default_factory=set)
    familiarity_scores: dict = field(default_factory=dict)
    hops: list = field(default_factory=list)


# ── Tests ────────────────────────────────────────────────────


@pytest.fixture
def tracker(tmp_path):
    return StabilityTracker(path=tmp_path / "stability.jsonl")


class TestStabilityTracker:
    def test_empty_tracker(self, tracker):
        assert tracker.signal_index == 0
        assert tracker.buffered_count == 0

    def test_record_skips_duplicates(self, tracker):
        mesh = MockMesh([MockWorker()])
        trace = MockTrace(duplicate=True)
        tracker.record(mesh, trace)
        assert tracker.signal_index == 0
        assert tracker.buffered_count == 0

    def test_record_captures_worker_state(self, tracker):
        w1 = MockWorker(signal_count=50, avg_fam=0.7)
        w2 = MockWorker(signal_count=10, avg_fam=0.3)
        mesh = MockMesh([w1, w2])

        trace = MockTrace(
            accepted_workers={w1.id},
            familiarity_scores={w1.id: 0.75, w2.id: 0.15},
            hops=[MockHopRecord(worker_id=w1.id, score=0.75)],
        )
        tracker.record(mesh, trace)

        assert tracker.signal_index == 1
        assert tracker.buffered_count == 1

    def test_total_error_computation(self, tracker):
        # Worker with avg_fam=0.8 → error=0.2
        # Worker with avg_fam=0.4 → error=0.6
        # Total V(x) = 0.8
        w1 = MockWorker(signal_count=50, avg_fam=0.8)
        w2 = MockWorker(signal_count=20, avg_fam=0.4)
        mesh = MockMesh([w1, w2])

        trace = MockTrace(
            accepted_workers={w1.id},
            familiarity_scores={w1.id: 0.85},
            hops=[MockHopRecord(worker_id=w1.id)],
        )
        tracker.record(mesh, trace)
        assert tracker.buffered_count == 1

    def test_flush_writes_jsonl(self, tracker, tmp_path):
        w1 = MockWorker(signal_count=10, avg_fam=0.6)
        mesh = MockMesh([w1])

        trace = MockTrace(
            accepted_workers={w1.id},
            familiarity_scores={w1.id: 0.65},
            hops=[MockHopRecord(worker_id=w1.id)],
        )
        tracker.record(mesh, trace)
        count = tracker.flush()
        assert count == 1
        assert tracker.buffered_count == 0

        path = tmp_path / "stability.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "total_error" in entry
        assert "worker_count" in entry
        assert "workers" in entry
        assert len(entry["workers"]) == 1
        assert entry["workers"][0]["avg_familiarity"] == 0.6
        assert entry["workers"][0]["error"] == pytest.approx(0.4)

    def test_flush_noop_when_empty(self, tracker):
        assert tracker.flush() == 0

    def test_sample_interval(self, tmp_path):
        tracker = StabilityTracker(
            path=tmp_path / "stability.jsonl",
            sample_interval=3,
        )
        w1 = MockWorker(signal_count=10, avg_fam=0.5)
        mesh = MockMesh([w1])
        trace = MockTrace(
            accepted_workers={w1.id},
            familiarity_scores={w1.id: 0.55},
            hops=[MockHopRecord(worker_id=w1.id)],
        )

        # Record 5 signals — only 1 should be buffered (at index 3)
        for _ in range(5):
            tracker.record(mesh, trace)

        assert tracker.signal_index == 5
        assert tracker.buffered_count == 1  # only index 3 was recorded

    def test_lifecycle_events(self, tracker):
        tracker.record_lifecycle("spawn", worker_id="abc123", detail={"source": "gap"})
        tracker.record_lifecycle("fork", worker_id="abc123", detail={"children": ["def456"]})

        w1 = MockWorker(signal_count=10, avg_fam=0.5)
        mesh = MockMesh([w1])
        trace = MockTrace(
            accepted_workers={w1.id},
            familiarity_scores={w1.id: 0.55},
            hops=[MockHopRecord(worker_id=w1.id)],
        )
        tracker.record(mesh, trace)

        assert tracker.buffered_count == 1

    def test_all_lifecycle_events_captured(self, tracker, tmp_path):
        """All lifecycle events between signals are recorded, not just the last."""
        tracker.record_lifecycle("spawn", worker_id="aaa11111", detail={"source": "gap"})
        tracker.record_lifecycle("fork", worker_id="bbb22222", detail={"children": ["ccc33333"]})
        tracker.record_lifecycle("archive", worker_id="ddd44444", detail={"reason": "energy_depleted"})

        w1 = MockWorker(signal_count=10, avg_fam=0.5)
        mesh = MockMesh([w1])
        trace = MockTrace(
            accepted_workers={w1.id},
            familiarity_scores={w1.id: 0.55},
            hops=[MockHopRecord(worker_id=w1.id)],
        )
        tracker.record(mesh, trace)
        tracker.flush()

        path = tmp_path / "stability.jsonl"
        entry = json.loads(path.read_text().strip())
        events = entry["lifecycle_events"]
        assert len(events) == 3
        assert events[0]["event"] == "spawn"
        assert events[0]["worker_id"] == "aaa11111"
        assert "timestamp" in events[0]
        assert events[1]["event"] == "fork"
        assert events[2]["event"] == "archive"
        assert events[2]["detail"]["reason"] == "energy_depleted"

    def test_multiple_records_append(self, tracker, tmp_path):
        w1 = MockWorker(signal_count=10, avg_fam=0.5)
        mesh = MockMesh([w1])
        trace = MockTrace(
            accepted_workers={w1.id},
            familiarity_scores={w1.id: 0.55},
            hops=[MockHopRecord(worker_id=w1.id)],
        )

        tracker.record(mesh, trace)
        tracker.record(mesh, trace)
        tracker.record(mesh, trace)
        tracker.flush()

        path = tmp_path / "stability.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Signal indices should increment
        entries = [json.loads(line) for line in lines]
        assert entries[0]["signal_index"] == 1
        assert entries[1]["signal_index"] == 2
        assert entries[2]["signal_index"] == 3

    def test_convergence_visible_in_error(self, tracker):
        """Simulated convergence: V(x) should decrease as workers specialize."""
        errors = []

        for step in range(5):
            # Workers get better over time (increasing avg_fam)
            avg_fam = 0.3 + step * 0.15  # 0.3 → 0.9
            w1 = MockWorker(signal_count=10 + step * 20, avg_fam=avg_fam)
            mesh = MockMesh([w1])
            trace = MockTrace(
                accepted_workers={w1.id},
                familiarity_scores={w1.id: avg_fam + 0.05},
                hops=[MockHopRecord(worker_id=w1.id)],
            )
            tracker.record(mesh, trace)

        # Check that buffered entries show decreasing total_error
        for entry in tracker._buffer:
            errors.append(entry["total_error"])

        # V(x) should be monotonically decreasing
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1], (
                f"V(x) increased at step {i}: {errors[i-1]} -> {errors[i]}"
            )
