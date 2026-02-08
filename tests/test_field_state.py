"""Tests for CERTX field state computation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from stigmergy.unity.field_config import FieldConfig
from stigmergy.unity.field_state import (
    FieldState,
    _PaceSmoother,
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
        MockWorkerSummary("w2", "support", 0.75, 0.9, 0.20, 15),
        MockWorkerSummary("w3", "leadership", 0.80, 0.8, 0.18, 10),
    ])
    agents: list = field(default_factory=list)


@dataclass
class MockActionCorrelation:
    topic: str = "test"
    ratio: float = 0.5
    exploration_ratio: float = 0.4


@dataclass
class MockPhaseLockAlert:
    framing_similarity: float = 0.3
    risk_topics: list = field(default_factory=list)


@dataclass
class MockCompressionTracker:
    current_index: float = 0.2


# ── FieldState tests ─────────────────────────────────────────


class TestFieldState:
    def test_default_values(self):
        fs = FieldState()
        assert fs.coherence == 0.5
        assert fs.entropy == 0.5
        assert fs.resonance == 0.5
        assert fs.temperature == 0.0
        assert fs.substrate == 0.5
        assert fs.level == "mesh"

    def test_to_dict_roundtrip(self):
        fs = FieldState(coherence=0.72, entropy=0.58, health=0.65)
        d = fs.to_dict()
        assert d["coherence"] == 0.72
        assert d["entropy"] == 0.58
        assert d["health"] == 0.65
        assert "raw" in d

    def test_certx_summary(self):
        fs = FieldState(
            coherence=0.72, entropy=0.58, resonance=0.81,
            temperature=0.15, substrate=0.66, health=0.74,
            dysmemic_pressure=0.06, stability_eigenvalue=0.95,
        )
        s = fs.certx_summary()
        assert "C=0.72" in s
        assert "E=0.58" in s
        assert "health=0.74" in s
        assert "dysmemic=0.06" in s

    def test_dysmemic_calculation(self):
        """Dysmemic pressure = max(0, C - X)."""
        fs = FieldState(coherence=0.9, substrate=0.3, dysmemic_pressure=0.6)
        assert fs.dysmemic_pressure == pytest.approx(0.6)


# ── CERTX Computation tests ─────────────────────────────────


class TestComputeMeshFieldState:
    def test_basic_computation(self):
        config = FieldConfig()
        pulse = MockPulse()
        state = compute_mesh_field_state(config, pulse, signal_index=100)
        assert 0.0 <= state.coherence <= 1.0
        assert 0.0 <= state.entropy <= 1.0
        assert 0.0 <= state.resonance <= 1.0
        assert state.temperature >= 0.0
        assert 0.0 <= state.substrate <= 1.0
        assert state.signal_index == 100
        assert state.level == "mesh"

    def test_health_weighted_combination(self):
        config = FieldConfig(
            w_coherence=1.0, w_entropy=0.0,
            w_resonance=0.0, w_temperature=0.0, w_substrate=0.0,
        )
        pulse = MockPulse()
        state = compute_mesh_field_state(config, pulse)
        # Health should be driven entirely by coherence
        assert state.health == pytest.approx(state.coherence, abs=0.01)

    def test_dysmemic_pressure_computed(self):
        config = FieldConfig()
        pulse = MockPulse()
        state = compute_mesh_field_state(config, pulse)
        expected = max(0.0, state.coherence - state.substrate)
        assert state.dysmemic_pressure == pytest.approx(expected)

    def test_graceful_degradation_no_pulse(self):
        """Missing pulse produces moderate defaults."""
        config = FieldConfig()
        state = compute_mesh_field_state(config, pulse=None)
        assert state.coherence == 0.5
        assert state.temperature == 0.0
        assert state.health > 0.0  # Still computes

    def test_graceful_degradation_empty_workers(self):
        """Pulse with no workers still works."""
        config = FieldConfig()
        pulse = MockPulse(workers=[], worker_count=0, avg_familiarity=0.0)
        state = compute_mesh_field_state(config, pulse)
        assert 0.0 <= state.health <= 1.0

    def test_graceful_degradation_no_subsystems(self):
        """No optional subsystems produces moderate defaults."""
        config = FieldConfig()
        pulse = MockPulse()
        state = compute_mesh_field_state(
            config, pulse,
            spectral=None,
            compression_tracker=None,
            action_correlations=None,
            phase_lock_alerts=None,
        )
        assert state.health > 0.0

    def test_lyapunov_dv_dt_from_pulse(self):
        config = FieldConfig()
        pulse = MockPulse(d_error_dt=-0.05)
        state = compute_mesh_field_state(config, pulse)
        assert state.lyapunov_dv_dt == pytest.approx(-0.05)

    def test_substrate_with_compression(self):
        config = FieldConfig()
        pulse = MockPulse()
        tracker = MockCompressionTracker(current_index=0.8)
        state = compute_mesh_field_state(
            config, pulse, compression_tracker=tracker,
        )
        # High compression → low substrate
        assert state.substrate < 0.5

    def test_substrate_with_phase_lock(self):
        config = FieldConfig()
        pulse = MockPulse()
        alerts = [MockPhaseLockAlert(framing_similarity=0.9)]
        state = compute_mesh_field_state(
            config, pulse, phase_lock_alerts=alerts,
        )
        # High phase-lock → low substrate
        assert state.substrate < 0.5

    def test_resonance_with_action_correlation(self):
        config = FieldConfig()
        pulse = MockPulse()
        # Low action-correlation → lower resonance
        low_ac = [MockActionCorrelation(ratio=0.05)]
        state = compute_mesh_field_state(
            config, pulse, action_correlations=low_ac, diversity_index=0.5,
        )
        assert state.resonance < 0.8

    def test_raw_values_stored(self):
        config = FieldConfig()
        pulse = MockPulse()
        state = compute_mesh_field_state(config, pulse)
        assert "raw_coherence" in state.raw
        assert "raw_entropy" in state.raw
        assert "raw_temperature" in state.raw


# ── Pace Layer Smoother tests ────────────────────────────────


class TestPaceSmoother:
    def test_first_value_passthrough(self):
        config = FieldConfig()
        smoother = _PaceSmoother(config)
        assert smoother.smooth("coherence", 0.8) == 0.8

    def test_ema_smoothing(self):
        config = FieldConfig(pace_coherence=0.5)
        smoother = _PaceSmoother(config)
        smoother.smooth("coherence", 1.0)  # first value: 1.0
        result = smoother.smooth("coherence", 0.0)  # EMA: 0.5*0 + 0.5*1 = 0.5
        assert result == pytest.approx(0.5)

    def test_different_rates_per_dimension(self):
        config = FieldConfig(pace_temperature=0.5, pace_substrate=0.01)
        smoother = _PaceSmoother(config)
        # Temperature responds fast
        smoother.smooth("temperature", 0.0)
        t = smoother.smooth("temperature", 1.0)  # 0.5*1 + 0.5*0 = 0.5
        # Substrate responds slowly
        smoother.smooth("substrate", 0.0)
        x = smoother.smooth("substrate", 1.0)  # 0.01*1 + 0.99*0 = 0.01
        assert t > x  # Temperature responds faster

    def test_smoothing_in_computation(self):
        config = FieldConfig(pace_coherence=0.5)
        smoother = _PaceSmoother(config)
        pulse = MockPulse()
        # First computation: raw pass-through
        s1 = compute_mesh_field_state(config, pulse, smoother=smoother)
        # Second with different pulse: smoothed
        pulse2 = MockPulse(workers=[
            MockWorkerSummary("w1", "a", 0.1, 1.0, 0.15, 10),
            MockWorkerSummary("w2", "b", 0.1, 0.9, 0.20, 15),
        ])
        s2 = compute_mesh_field_state(config, pulse2, smoother=smoother)
        # Coherence should be smoothed (not raw jump)
        assert s2.coherence != s2.raw["raw_coherence"]
