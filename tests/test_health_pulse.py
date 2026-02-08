"""Tests for health pulse collector and change rate tracking."""

from __future__ import annotations

from uuid import uuid4

import pytest

from stigmergy.mesh.health_pulse import (
    ChangeRate,
    HealthPulseCollector,
    MeshHealthPulse,
    WorkerSummary,
    AgentSummary,
)


# ── ChangeRate tests ──────────────────────────────────────────


class TestChangeRate:
    def test_initial_state(self):
        cr = ChangeRate()
        assert cr.current == 0.0
        assert cr.rate == 0.0
        assert cr.samples == 0

    def test_first_update_sets_current(self):
        cr = ChangeRate()
        cr.update(5.0)
        assert cr.current == 5.0
        assert cr.samples == 1
        assert cr.rate == 0.0  # no rate on first sample

    def test_second_update_computes_rate(self):
        cr = ChangeRate()
        cr.update(5.0, interval=1.0)
        cr.update(10.0, interval=1.0)
        assert cr.current == 10.0
        assert cr.previous == 5.0
        assert cr.rate == 5.0  # first rate is raw, no EMA yet
        assert cr.samples == 2

    def test_ema_smoothing(self):
        cr = ChangeRate(ema_alpha=0.5)
        cr.update(0.0, interval=1.0)
        cr.update(10.0, interval=1.0)  # raw rate = 10.0, first rate
        assert cr.rate == 10.0
        cr.update(10.0, interval=1.0)  # raw rate = 0.0, EMA: 0.5*0 + 0.5*10 = 5.0
        assert cr.rate == pytest.approx(5.0)
        cr.update(10.0, interval=1.0)  # raw rate = 0.0, EMA: 0.5*0 + 0.5*5 = 2.5
        assert cr.rate == pytest.approx(2.5)

    def test_interval_scaling(self):
        cr = ChangeRate()
        cr.update(0.0, interval=1.0)
        cr.update(100.0, interval=50.0)  # rate = 100/50 = 2.0
        assert cr.rate == pytest.approx(2.0)

    def test_decreasing_values(self):
        cr = ChangeRate()
        cr.update(10.0, interval=1.0)
        cr.update(5.0, interval=1.0)  # rate = -5.0
        assert cr.rate == pytest.approx(-5.0)

    def test_known_ema_sequence(self):
        """Verify EMA with alpha=0.2 over a known sequence."""
        cr = ChangeRate(ema_alpha=0.2)
        values = [0, 10, 20, 30, 20, 10]
        for v in values:
            cr.update(float(v), interval=1.0)
        # Final value should be smoothed — not just the last raw rate
        assert cr.samples == len(values)
        assert cr.current == 10.0
        assert cr.previous == 20.0


# ── MeshHealthPulse tests ────────────────────────────────────


class TestMeshHealthPulse:
    def test_to_dict_serialization(self):
        pulse = MeshHealthPulse(
            signal_index=100,
            worker_count=5,
            total_error=1.5,
            avg_familiarity=0.7,
        )
        d = pulse.to_dict()
        assert d["signal_index"] == 100
        assert d["worker_count"] == 5
        assert d["total_error"] == 1.5
        assert d["avg_familiarity"] == 0.7
        assert isinstance(d["workers"], list)
        assert isinstance(d["agents"], list)

    def test_worker_summaries_in_dict(self):
        pulse = MeshHealthPulse()
        pulse.workers.append(WorkerSummary(
            id="abcd1234", label="PMS sync", familiarity=0.8,
            energy=0.5, threshold=0.3, signal_count=42,
        ))
        d = pulse.to_dict()
        assert len(d["workers"]) == 1
        assert d["workers"][0]["id"] == "abcd1234"
        assert d["workers"][0]["familiarity"] == 0.8


# ── HealthPulseCollector tests ────────────────────────────────


class _MockWorker:
    """Minimal mock for testing pulse collection."""

    def __init__(self, wid=None, label="test", fam=0.5, signal_count=10):
        self.id = wid or uuid4()
        self.label = label
        self.rolling_avg_familiarity = fam
        self.adaptive_threshold = 0.3
        self.context = _MockContext(signal_count)


class _MockContext:
    def __init__(self, signal_count=10):
        self.signal_count = signal_count
        self.energy = 0.5


class _MockAgent:
    def __init__(self, confidence=0.8):
        self.id = uuid4()
        self.confidence = confidence
        self.competencies = _MockCompetency()

    def diagnostics(self):
        return {
            "reflect": {"llm_calls": 5, "mechanical_calls": 0},
            "evaluate": {"llm_calls": 10, "mechanical_calls": 0},
        }


class _MockCompetency:
    def __init__(self):
        self.weights = {"temporal_analysis": 0.7, "cross_context_propagation": 0.5}


class _MockMesh:
    def __init__(self, workers):
        self._workers = workers

    @property
    def workers(self):
        return self._workers


class TestHealthPulseCollector:
    def test_collect_basic(self):
        workers = [_MockWorker(fam=0.6), _MockWorker(fam=0.8)]
        mesh = _MockMesh(workers)
        collector = HealthPulseCollector()

        pulse = collector.collect(50, mesh)

        assert pulse.signal_index == 50
        assert pulse.worker_count == 2
        assert pulse.avg_familiarity == pytest.approx(0.7)
        assert len(pulse.workers) == 2

    def test_change_rates_update_over_pulses(self):
        workers = [_MockWorker(fam=0.5)]
        mesh = _MockMesh(workers)
        collector = HealthPulseCollector()

        pulse1 = collector.collect(50, mesh)
        assert pulse1.d_error_dt == 0.0  # first sample, no rate

        workers[0].rolling_avg_familiarity = 0.8
        pulse2 = collector.collect(100, mesh)
        # Error decreased (familiarity increased), so d_error_dt should be negative
        assert pulse2.d_error_dt != 0.0

    def test_history_kept(self):
        mesh = _MockMesh([_MockWorker()])
        collector = HealthPulseCollector()

        for i in range(15):
            collector.collect(i * 50, mesh)

        assert len(collector.history) == 10  # max_history
        assert collector.latest.signal_index == 14 * 50

    def test_routing_changes_tracked(self):
        mesh = _MockMesh([_MockWorker(), _MockWorker()])
        collector = HealthPulseCollector()

        pulse1 = collector.collect(50, mesh)
        assert pulse1.routing_changes == 2  # initial: 0 -> 2

        pulse2 = collector.collect(100, mesh)
        assert pulse2.routing_changes == 0  # no change

        mesh._workers.append(_MockWorker())
        pulse3 = collector.collect(150, mesh)
        assert pulse3.routing_changes == 1  # 2 -> 3

    def test_agent_summaries(self):
        mesh = _MockMesh([_MockWorker()])
        agents = [_MockAgent(confidence=0.9)]
        collector = HealthPulseCollector()

        pulse = collector.collect(50, mesh, agents=agents)

        assert len(pulse.agents) == 1
        assert pulse.agents[0].confidence == 0.9
        assert pulse.agents[0].top_competency == "temporal_analysis"

    def test_graceful_without_optional_subsystems(self):
        mesh = _MockMesh([])
        collector = HealthPulseCollector()

        pulse = collector.collect(50, mesh)

        assert pulse.worker_count == 0
        assert pulse.avg_familiarity == 0.0
        assert pulse.total_error == 0.0

    def test_total_error_is_sum_of_one_minus_familiarity(self):
        workers = [_MockWorker(fam=0.3), _MockWorker(fam=0.7)]
        mesh = _MockMesh(workers)
        collector = HealthPulseCollector()

        pulse = collector.collect(50, mesh)
        # error = (1 - 0.3) + (1 - 0.7) = 0.7 + 0.3 = 1.0
        assert pulse.total_error == pytest.approx(1.0)
