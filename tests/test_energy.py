"""Tests for energy model.

Verify exponential math, threshold triggers, and status transitions.
"""

from __future__ import annotations

import math

import pytest

from stigmergy.core.energy import ContextStatus, EnergyState, compute_energy


class TestExponentialDecay:
    def test_no_decay_at_t0(self):
        state = compute_energy(base_energy=1.0, decay_rate=0.001, elapsed_seconds=0.0)
        assert state.energy == pytest.approx(1.0)
        assert state.status == ContextStatus.ACTIVE

    def test_half_life(self):
        """Verify energy is halved at the correct time."""
        decay_rate = 0.001
        half_life = math.log(2) / decay_rate  # ~693 seconds
        state = compute_energy(base_energy=1.0, decay_rate=decay_rate, elapsed_seconds=half_life)
        assert state.energy == pytest.approx(0.5, abs=0.01)

    def test_approaches_zero(self):
        state = compute_energy(base_energy=1.0, decay_rate=0.001, elapsed_seconds=10000)
        assert state.energy < 0.001

    def test_high_energy_stays_active(self):
        state = compute_energy(base_energy=5.0, decay_rate=0.001, elapsed_seconds=100)
        assert state.status == ContextStatus.ACTIVE

    def test_preserves_scale(self):
        """Higher base energy decays proportionally."""
        state_low = compute_energy(base_energy=1.0, decay_rate=0.001, elapsed_seconds=500)
        state_high = compute_energy(base_energy=2.0, decay_rate=0.001, elapsed_seconds=500)
        assert state_high.energy == pytest.approx(state_low.energy * 2, abs=0.01)


class TestStatusThresholds:
    def test_active_above_decay_threshold(self):
        state = compute_energy(
            base_energy=1.0, decay_rate=0.001, elapsed_seconds=0,
            decay_threshold=0.3, archive_threshold=0.05,
        )
        assert state.status == ContextStatus.ACTIVE

    def test_decaying_below_decay_threshold(self):
        # energy = 1.0 * e^(-0.001 * 1300) ≈ 0.272
        state = compute_energy(
            base_energy=1.0, decay_rate=0.001, elapsed_seconds=1300,
            decay_threshold=0.3, archive_threshold=0.05,
        )
        assert state.status == ContextStatus.DECAYING

    def test_archived_below_archive_threshold(self):
        # energy = 1.0 * e^(-0.001 * 3500) ≈ 0.030
        state = compute_energy(
            base_energy=1.0, decay_rate=0.001, elapsed_seconds=3500,
            decay_threshold=0.3, archive_threshold=0.05,
        )
        assert state.status == ContextStatus.ARCHIVED

    def test_custom_thresholds(self):
        state = compute_energy(
            base_energy=1.0, decay_rate=0.01, elapsed_seconds=100,
            decay_threshold=0.5, archive_threshold=0.3,
        )
        # energy = e^(-1) ≈ 0.368
        assert state.status == ContextStatus.DECAYING


class TestEdgeCases:
    def test_zero_base_energy(self):
        state = compute_energy(base_energy=0.0, decay_rate=0.001, elapsed_seconds=0)
        assert state.energy == 0.0
        assert state.status == ContextStatus.ARCHIVED

    def test_zero_decay_rate(self):
        """No decay → energy stays constant."""
        state = compute_energy(base_energy=1.0, decay_rate=0.0, elapsed_seconds=999999)
        assert state.energy == pytest.approx(1.0)
        assert state.status == ContextStatus.ACTIVE

    def test_very_high_decay_rate(self):
        state = compute_energy(base_energy=1.0, decay_rate=1.0, elapsed_seconds=10)
        assert state.energy < 0.001
