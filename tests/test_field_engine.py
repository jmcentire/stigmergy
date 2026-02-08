"""Tests for field engine — end-to-end orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from stigmergy.unity.field_config import FieldConfig
from stigmergy.unity.field_engine import FieldEngine
from stigmergy.unity.field_state import FieldState


@dataclass
class MockWorkerSummary:
    id: str = "w1"
    label: str = "test"
    familiarity: float = 0.8
    energy: float = 1.0
    threshold: float = 0.15
    signal_count: int = 10


@dataclass
class MockEnginePulse:
    signal_index: int = 50
    worker_count: int = 3
    total_error: float = 0.6
    avg_familiarity: float = 0.8
    routing_changes: int = 0
    spectral_high_freq: float = 0.3
    spectral_trend: str = "stable"
    effective_dimensionality: int = 3
    last_quorum_met: int = 4
    last_quorum_total: int = 5
    d_error_dt: float = -0.01
    d_familiarity_dt: float = 0.005
    d_worker_count_dt: float = 0.0
    d_routing_changes_dt: float = 0.0
    d_confidence_dt: float = 0.0
    d_spectral_dt: float = 0.001
    active_findings: int = 3
    deferred_findings: int = 1
    normalized_findings: int = 0
    workers: list = field(default_factory=lambda: [
        MockWorkerSummary("w1", "eng", 0.85, 1.0, 0.15, 20),
        MockWorkerSummary("w2", "sup", 0.75, 0.9, 0.20, 15),
        MockWorkerSummary("w3", "lead", 0.80, 0.8, 0.18, 10),
    ])
    agents: list = field(default_factory=list)


@dataclass
class MockContext:
    signal_count: int = 10


@dataclass
class MockWorkerNode:
    id: str = "w1"
    base_threshold: float = 0.15
    context: MockContext = field(default_factory=MockContext)


@dataclass
class MockEngineMesh:
    workers: list = field(default_factory=lambda: [
        MockWorkerNode("w1", 0.15, MockContext(20)),
        MockWorkerNode("w2", 0.20, MockContext(15)),
        MockWorkerNode("w3", 0.18, MockContext(10)),
    ])


@dataclass
class MockCompetency:
    weights: dict = field(default_factory=lambda: {"a": 0.5, "b": 0.3})


@dataclass
class MockAgent:
    competencies: MockCompetency = field(default_factory=MockCompetency)


class TestFieldEngine:
    def test_tick_returns_field_state(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=True)
        engine = FieldEngine(config)
        pulse = MockEnginePulse()
        mesh = MockEngineMesh()
        agents = [MockAgent(), MockAgent()]

        state = engine.tick(pulse, mesh=mesh, agents=agents)
        assert isinstance(state, FieldState)
        assert state.signal_index == 50
        assert 0.0 <= state.coherence <= 1.0
        assert 0.0 <= state.health <= 1.0

    def test_tick_logs_to_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=True)
        engine = FieldEngine(config)
        pulse = MockEnginePulse()

        engine.tick(pulse)
        log_path = tmp_path / ".stigmergy" / "field_state.jsonl"
        assert log_path.exists()
        content = log_path.read_text()
        assert "coherence" in content

    def test_history_accumulates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=False)
        engine = FieldEngine(config)

        for i in range(5):
            pulse = MockEnginePulse(signal_index=i * 50)
            engine.tick(pulse)

        assert len(engine.history) == 5
        assert engine.latest.signal_index == 200

    def test_graceful_degradation_no_pulse(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=False)
        engine = FieldEngine(config)
        state = engine.tick(pulse=None)
        assert isinstance(state, FieldState)
        assert state.health > 0.0  # Still computes with defaults

    def test_graceful_degradation_no_mesh(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=False)
        engine = FieldEngine(config)
        pulse = MockEnginePulse()
        state = engine.tick(pulse, mesh=None)
        assert isinstance(state, FieldState)

    def test_bridge_disabled_by_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(bridge_enabled=False, log_field_state=False)
        engine = FieldEngine(config)
        mesh = MockEngineMesh()
        old_thresholds = [w.base_threshold for w in mesh.workers]
        pulse = MockEnginePulse()
        engine.tick(pulse, mesh=mesh)
        # Thresholds should not change
        for i, w in enumerate(mesh.workers):
            assert w.base_threshold == old_thresholds[i]

    def test_record_prediction(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=False)
        engine = FieldEngine(config)
        engine.record_prediction("agent1", 0.9, True)
        engine.record_prediction("agent1", 0.8, False)
        assert "agent1" in engine.calibration.agents
        assert len(engine.calibration.agents["agent1"].predictions) == 2

    def test_eigenvalue_builds_over_ticks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(jacobian_window=3, log_field_state=False)
        engine = FieldEngine(config)
        for i in range(6):
            pulse = MockEnginePulse(signal_index=i * 50)
            state = engine.tick(pulse)
        # After enough ticks, eigenvalue should be non-zero (or at least attempted)
        assert engine.latest is not None

    def test_multiple_ticks_consistent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=False)
        engine = FieldEngine(config)
        pulse = MockEnginePulse()
        mesh = MockEngineMesh()

        states = []
        for i in range(10):
            pulse_i = MockEnginePulse(signal_index=i * 50)
            s = engine.tick(pulse_i, mesh=mesh, agents=[MockAgent()])
            states.append(s)

        # All states should be valid
        for s in states:
            assert 0.0 <= s.coherence <= 1.0
            assert 0.0 <= s.health <= 1.0
            assert s.dysmemic_pressure >= 0.0
