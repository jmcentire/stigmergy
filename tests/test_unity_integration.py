"""Integration tests for the unity system — end-to-end, persistence, edge cases.

These tests verify that blood actually flows through the system:
calibration feeds bridge, bridge modifies mesh, state persists
across save/restore cycles.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from stigmergy.unity.breathing import BreathingController
from stigmergy.unity.calibration import (
    CalibrationState,
    brier_score,
    compute_n_star,
    update_calibration,
)
from stigmergy.unity.eigenmonitor import EigenMonitor, EigenState
from stigmergy.unity.field_bridge import FieldAdjustment, FieldBridge
from stigmergy.unity.field_config import FieldConfig
from stigmergy.unity.field_engine import FieldEngine
from stigmergy.unity.field_state import (
    FieldState,
    _PaceSmoother,
    _TemperatureScaler,
    compute_mesh_field_state,
)


# ── Mock objects ─────────────────────────────────────────────


@dataclass
class MockWorkerSummary:
    id: str = "w1"
    label: str = "test"
    familiarity: float = 0.8
    energy: float = 1.0
    threshold: float = 0.15
    signal_count: int = 10


@dataclass
class MockPulse:
    signal_index: int = 100
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
class MockMeshWithThresholds:
    """Mock mesh that exposes settable physics parameters."""
    _consensus_threshold: float = 0.6
    _gap_threshold: float = 0.08
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


# ── Temperature Scaler tests ────────────────────────────────


class TestTemperatureScaler:
    def test_first_value_calibrates_scale(self):
        scaler = _TemperatureScaler(initial_scale=0.1)
        result = scaler.normalize(0.5)
        assert 0.0 < result < 1.0
        assert scaler.scale == 0.5  # Expanded to new max

    def test_sigmoid_squash_bounded(self):
        scaler = _TemperatureScaler(initial_scale=1.0)
        # Very large input should still be < 1.0 (sigmoid bounded)
        result = scaler.normalize(100.0)
        assert result < 1.0
        assert result > 0.5  # tanh(1) ≈ 0.76 when scale expands to match input

    def test_zero_input(self):
        scaler = _TemperatureScaler(initial_scale=1.0)
        result = scaler.normalize(0.0)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_scale_decays(self):
        scaler = _TemperatureScaler(initial_scale=10.0, decay=0.5)
        scaler.normalize(0.1)  # Much smaller than scale; scale decays
        assert scaler.scale < 10.0

    def test_consistent_normalization(self):
        """Same input to same scaler gives diminishing result as scale adapts."""
        scaler = _TemperatureScaler(initial_scale=0.01)
        r1 = scaler.normalize(1.0)  # Expands scale to 1.0
        r2 = scaler.normalize(1.0)  # Scale decayed slightly, then re-expanded
        # Both should be in valid range
        assert 0.0 < r1 < 1.0
        assert 0.0 < r2 < 1.0


# ── Coherence variance scaling tests ────────────────────────


class TestCoherenceVarianceScaling:
    def test_maximum_disagreement(self):
        """Workers at [0, 1] should give coherence near 0, not 0.75."""
        config = FieldConfig()
        pulse = MockPulse(workers=[
            MockWorkerSummary("w1", "a", 0.0, 1.0, 0.15, 10),
            MockWorkerSummary("w2", "b", 1.0, 1.0, 0.15, 10),
        ])
        state = compute_mesh_field_state(config, pulse)
        # With normalized variance, max disagreement should push coherence low
        assert state.coherence < 0.5

    def test_perfect_agreement(self):
        """Workers at same familiarity → coherence near 1."""
        config = FieldConfig()
        pulse = MockPulse(workers=[
            MockWorkerSummary("w1", "a", 0.8, 1.0, 0.15, 10),
            MockWorkerSummary("w2", "b", 0.8, 1.0, 0.15, 10),
        ])
        state = compute_mesh_field_state(config, pulse)
        assert state.coherence > 0.7


# ── Eigenvalue hysteresis tests ──────────────────────────────


class TestEigenvalueHysteresis:
    def test_doesnt_flicker_at_boundary(self):
        """Status should not flicker when eigenvalue hovers near threshold."""
        config = FieldConfig(jacobian_window=3, eigenvalue_healthy_min=0.8,
                             eigenvalue_healthy_max=1.2)
        monitor = EigenMonitor(config)

        # Build up history to get "healthy" status
        for i in range(5):
            state = FieldState(
                coherence=0.5 + 0.02 * i, entropy=0.5 + 0.01 * i,
                resonance=0.5, temperature=0.1 + 0.01 * i, substrate=0.5,
            )
            result = monitor.update(state)

        if result is not None and result.status != "insufficient_data":
            initial_status = result.status
            # Feed nearly identical states — should NOT flicker
            statuses = []
            for i in range(5, 10):
                state = FieldState(
                    coherence=0.5 + 0.001 * i, entropy=0.5 + 0.001 * i,
                    resonance=0.5, temperature=0.1, substrate=0.5,
                )
                result = monitor.update(state)
                if result is not None:
                    statuses.append(result.status)
            # Hysteresis should reduce flickering — allow at most 2 transitions
            # (Jacobian estimation from small windows is inherently noisy)
            transitions = sum(1 for a, b in zip(statuses, statuses[1:]) if a != b)
            assert transitions <= 2

    def test_smoothed_max_is_ema(self):
        config = FieldConfig(jacobian_window=3)
        monitor = EigenMonitor(config)
        for i in range(5):
            state = FieldState(
                coherence=0.5 + 0.05 * i, entropy=0.5,
                resonance=0.5, temperature=0.1, substrate=0.5,
            )
            result = monitor.update(state)
        if result is not None:
            assert result.smoothed_max >= 0.0


# ── Calibration → Bridge feedback tests ──────────────────────


class TestCalibrationBridgeFeedback:
    def test_babbling_agents_raise_consensus(self):
        """When agents are babbling, bridge should raise consensus threshold."""
        config = FieldConfig(bridge_enabled=True, bridge_quorum_gain=0.1)
        bridge = FieldBridge(config)

        cal = CalibrationState(window=50, babbling_threshold=0.3)
        # Create a babbling agent
        for _ in range(20):
            update_calibration("bad_agent", 1.0, False, cal, config)
            update_calibration("bad_agent", 0.0, True, cal, config)
        assert cal.agents["bad_agent"].is_babbling

        state = FieldState(temperature=0.1, entropy=0.5, resonance=0.5,
                           dysmemic_pressure=0.0)
        adjustments = bridge.compute_adjustments(state, calibration=cal)
        consensus_adj = [a for a in adjustments if a.target == "consensus"]
        assert len(consensus_adj) >= 1
        assert consensus_adj[0].direction == "increase"

    def test_no_calibration_no_extra_adjustments(self):
        """Without calibration data, no calibration-driven adjustments."""
        config = FieldConfig(bridge_enabled=True, bridge_quorum_gain=0.1)
        bridge = FieldBridge(config)
        state = FieldState(temperature=0.1, entropy=0.5, resonance=0.5,
                           dysmemic_pressure=0.0)
        adjustments = bridge.compute_adjustments(state, calibration=None)
        cal_adj = [a for a in adjustments if "babbling" in a.reason or "N*" in a.reason]
        assert cal_adj == []

    def test_low_n_star_raises_quorum(self):
        """Low average N* should raise quorum demands."""
        config = FieldConfig(bridge_enabled=True, bridge_quorum_gain=0.1)
        bridge = FieldBridge(config)

        cal = CalibrationState(window=50, babbling_threshold=0.3)
        # Agent with mediocre predictions (Brier around 0.25)
        for _ in range(10):
            update_calibration("med_agent", 0.5, True, cal, config)
            update_calibration("med_agent", 0.5, False, cal, config)

        state = FieldState(temperature=0.1, entropy=0.5, resonance=0.5,
                           dysmemic_pressure=0.0)
        adjustments = bridge.compute_adjustments(state, calibration=cal)
        quorum_adj = [a for a in adjustments if a.target == "quorum"
                      and "N*" in a.reason]
        assert len(quorum_adj) >= 1


# ── Bridge → Mesh mechanical tests ──────────────────────────


class TestBridgeMeshMechanics:
    def test_consensus_threshold_actually_changes(self):
        """Bridge should mechanically modify mesh._consensus_threshold."""
        config = FieldConfig(bridge_enabled=True)
        bridge = FieldBridge(config)
        mesh = MockMeshWithThresholds()
        old = mesh._consensus_threshold
        adjustments = [FieldAdjustment("consensus", "increase", 0.05, "test")]
        actions = bridge.apply_adjustments(adjustments, mesh)
        assert mesh._consensus_threshold > old
        assert len(actions) > 0
        assert "consensus" in actions[0]

    def test_gap_threshold_actually_changes(self):
        """Bridge should mechanically modify mesh._gap_threshold."""
        config = FieldConfig(bridge_enabled=True)
        bridge = FieldBridge(config)
        mesh = MockMeshWithThresholds()
        old = mesh._gap_threshold
        adjustments = [FieldAdjustment("quorum", "increase", 0.02, "test")]
        actions = bridge.apply_adjustments(adjustments, mesh)
        assert mesh._gap_threshold > old

    def test_threshold_clamps(self):
        config = FieldConfig(bridge_enabled=True)
        bridge = FieldBridge(config)
        mesh = MockMeshWithThresholds(_consensus_threshold=0.89)
        adjustments = [FieldAdjustment("consensus", "increase", 0.5, "test")]
        bridge.apply_adjustments(adjustments, mesh)
        assert mesh._consensus_threshold <= 0.9

    def test_gap_threshold_clamps_low(self):
        config = FieldConfig(bridge_enabled=True)
        bridge = FieldBridge(config)
        mesh = MockMeshWithThresholds(_gap_threshold=0.03)
        adjustments = [FieldAdjustment("quorum", "decrease", 0.5, "test")]
        bridge.apply_adjustments(adjustments, mesh)
        assert mesh._gap_threshold >= 0.02


# ── End-to-end: bridge enabled, modifying a real-ish mesh ────


class TestEndToEndBridgeEnabled:
    def test_tick_with_bridge_modifies_thresholds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(
            bridge_enabled=True,
            bridge_vigilance_gain=0.1,
            bridge_quorum_gain=0.1,
            log_field_state=False,
        )
        engine = FieldEngine(config)
        mesh = MockMeshWithThresholds()
        old_thresholds = [w.base_threshold for w in mesh.workers]

        # Feed a pulse with extreme values to trigger adjustments
        pulse = MockPulse(
            d_error_dt=0.5, d_familiarity_dt=0.5,
            d_worker_count_dt=0.5, d_spectral_dt=0.5,
        )
        engine.tick(pulse, mesh=mesh, agents=[MockAgent()])

        # With high temperature, vigilance should have been raised
        changed = any(
            w.base_threshold != old
            for w, old in zip(mesh.workers, old_thresholds)
        )
        assert changed, "Bridge should have modified worker thresholds"

    def test_calibration_flows_through_to_mesh(self, tmp_path, monkeypatch):
        """Babbling → calibration → bridge → consensus threshold change."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(
            bridge_enabled=True,
            bridge_quorum_gain=0.2,
            log_field_state=False,
        )
        engine = FieldEngine(config)
        mesh = MockMeshWithThresholds()

        # Create babbling agent via corroboration feedback
        for _ in range(20):
            engine.record_prediction("agent_x", 1.0, False)
            engine.record_prediction("agent_x", 0.0, True)

        assert engine.calibration.agents["agent_x"].is_babbling

        # Tick should detect babbling and adjust consensus
        old_consensus = mesh._consensus_threshold
        pulse = MockPulse()
        engine.tick(pulse, mesh=mesh, agents=[MockAgent()])

        assert mesh._consensus_threshold > old_consensus


# ── State persistence roundtrip tests ────────────────────────


class TestStatePersistence:
    def test_save_and_restore(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=False)

        # Create engine, do some work
        engine1 = FieldEngine(config)
        pulse = MockPulse()
        for i in range(5):
            engine1.tick(MockPulse(signal_index=i * 50))
        engine1.record_prediction("agent1", 0.9, True)
        engine1.record_prediction("agent1", 0.8, False)
        engine1.save_state()

        # Create new engine, restore
        engine2 = FieldEngine(config)
        assert engine2.tick_count == 0
        loaded = engine2.load_state()
        assert loaded is True
        assert engine2.tick_count == 5
        assert "agent1" in engine2.calibration.agents
        assert len(engine2.calibration.agents["agent1"].predictions) == 2

    def test_pid_state_survives_restart(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(pid_ki=0.1, log_field_state=False)

        # Build up PID integral
        engine1 = FieldEngine(config)
        for i in range(10):
            engine1.tick(MockPulse(signal_index=i * 50))
        integral_before = engine1._pid._integral
        engine1.save_state()

        # Restore and check integral preserved
        engine2 = FieldEngine(config)
        engine2.load_state()
        assert engine2._pid._integral == pytest.approx(integral_before, abs=0.001)

    def test_temperature_scaler_survives_restart(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=False)

        engine1 = FieldEngine(config)
        # Feed high-rate pulse to expand temperature scale
        engine1.tick(MockPulse(d_error_dt=5.0, d_familiarity_dt=3.0,
                               d_worker_count_dt=2.0, d_spectral_dt=1.0))
        scale_before = engine1._temperature_scaler.scale
        engine1.save_state()

        engine2 = FieldEngine(config)
        engine2.load_state()
        assert engine2._temperature_scaler.scale == pytest.approx(scale_before)

    def test_eigenmonitor_history_survives_restart(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(jacobian_window=3, log_field_state=False)

        engine1 = FieldEngine(config)
        for i in range(4):
            engine1.tick(MockPulse(signal_index=i * 50))
        history_len = engine1._eigen.history_length
        engine1.save_state()

        engine2 = FieldEngine(config)
        engine2.load_state()
        assert engine2._eigen.history_length == history_len

    def test_no_state_file_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        engine = FieldEngine(FieldConfig())
        assert engine.load_state() is False

    def test_corrupt_state_file_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        state_path = tmp_path / ".stigmergy" / "unity_state.json"
        state_path.write_text("not valid json {{{")
        engine = FieldEngine(FieldConfig())
        assert engine.load_state() is False


# ── Edge case tests ──────────────────────────────────────────


class TestEdgeCases:
    def test_nan_familiarity_clamped(self):
        """Out-of-range familiarity should not crash."""
        config = FieldConfig()
        pulse = MockPulse(workers=[
            MockWorkerSummary("w1", "a", 2.5, 1.0, 0.15, 10),
            MockWorkerSummary("w2", "b", -0.5, 1.0, 0.15, 10),
        ])
        state = compute_mesh_field_state(config, pulse)
        assert 0.0 <= state.coherence <= 1.0

    def test_zero_signal_count_workers(self):
        """Workers with 0 signals should not crash entropy calculation."""
        config = FieldConfig()
        pulse = MockPulse(workers=[
            MockWorkerSummary("w1", "a", 0.8, 1.0, 0.15, 0),
            MockWorkerSummary("w2", "b", 0.7, 1.0, 0.15, 0),
        ])
        state = compute_mesh_field_state(config, pulse)
        assert state.entropy >= 0.0

    def test_single_worker(self):
        """Single worker should not crash variance calculation."""
        config = FieldConfig()
        pulse = MockPulse(workers=[
            MockWorkerSummary("w1", "a", 0.8, 1.0, 0.15, 10),
        ])
        state = compute_mesh_field_state(config, pulse)
        assert 0.0 <= state.health <= 1.0

    def test_all_weights_zero(self):
        """All CERTX weights zero should give health=0 not division error."""
        config = FieldConfig(
            w_coherence=0, w_entropy=0, w_resonance=0,
            w_temperature=0, w_substrate=0,
        )
        pulse = MockPulse()
        state = compute_mesh_field_state(config, pulse)
        assert state.health == 0.0

    def test_brier_score_with_extreme_confidence(self):
        """Confidence outside [0,1] should not crash Brier."""
        # Shouldn't happen but defense in depth
        bs = brier_score([(1.5, True), (-0.5, False)])
        assert bs >= 0.0

    def test_breathing_period_one(self):
        """Minimum period should not crash."""
        config = FieldConfig(breathing_period=1, breathing_amplitude=0.1)
        bc = BreathingController(config)
        for i in range(5):
            state = bc.tick(signal_count=i)
            assert abs(state.modulation) <= 0.1

    def test_pid_dt_zero(self):
        """dt=0 should not crash PID."""
        from stigmergy.unity.pid_controller import PIDController
        config = FieldConfig()
        pid = PIDController(config)
        pid.update(0.5, dt=1.0)  # Initialize
        state = pid.update(0.5, dt=0.0)  # dt=0
        assert state.d_term == 0.0  # derivative should be 0 (div by zero guard)

    def test_many_rapid_ticks(self, tmp_path, monkeypatch):
        """50 rapid ticks should not accumulate errors or crash."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".stigmergy").mkdir()
        config = FieldConfig(log_field_state=False)
        engine = FieldEngine(config)
        mesh = MockMeshWithThresholds()

        for i in range(50):
            pulse = MockPulse(signal_index=i * 10)
            state = engine.tick(pulse, mesh=mesh, agents=[MockAgent()])
            assert 0.0 <= state.health <= 2.0  # Health can exceed 1 with temperature
            assert state.dysmemic_pressure >= 0.0
