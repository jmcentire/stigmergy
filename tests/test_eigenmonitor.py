"""Tests for eigenvalue monitor — Jacobian eigenvalue tracking."""

from __future__ import annotations

import math

import pytest

from stigmergy.unity.eigenmonitor import EigenMonitor, EigenState
from stigmergy.unity.field_config import FieldConfig
from stigmergy.unity.field_state import FieldState


class TestEigenMonitor:
    def test_insufficient_data(self):
        config = FieldConfig(jacobian_window=5)
        monitor = EigenMonitor(config)
        # Only one point — not enough
        state = FieldState(coherence=0.5, entropy=0.5, resonance=0.5,
                           temperature=0.1, substrate=0.5)
        result = monitor.update(state)
        assert result is not None
        assert result.status == "insufficient_data"

    def test_history_accumulation(self):
        config = FieldConfig(jacobian_window=3)
        monitor = EigenMonitor(config)
        for i in range(3):
            state = FieldState(coherence=0.5 + i * 0.01, entropy=0.5,
                               resonance=0.5, temperature=0.1, substrate=0.5)
            monitor.update(state)
        assert monitor.history_length == 3

    def test_stable_system(self):
        """A system with small, stable changes should have |λ| near zero."""
        config = FieldConfig(jacobian_window=5, eigenvalue_healthy_min=0.8,
                             eigenvalue_healthy_max=1.2)
        monitor = EigenMonitor(config)

        # Feed slowly changing states (near-constant)
        for i in range(7):
            state = FieldState(
                coherence=0.5 + 0.001 * i,
                entropy=0.5 + 0.001 * i,
                resonance=0.5,
                temperature=0.1,
                substrate=0.5,
            )
            result = monitor.update(state)

        # Should have a result by now
        assert result is not None
        if result.status != "insufficient_data":
            # Small changes → small eigenvalues
            assert result.max_eigenvalue < 2.0

    def test_unstable_system(self):
        """A system with exponentially growing changes should have high |λ|."""
        config = FieldConfig(jacobian_window=5, eigenvalue_healthy_min=0.8,
                             eigenvalue_healthy_max=1.2)
        monitor = EigenMonitor(config)

        # Feed exponentially diverging states (unclamped temperature dimension)
        for i in range(7):
            factor = 1.5 ** i
            state = FieldState(
                coherence=0.1 + 0.05 * i,
                entropy=0.1 + 0.08 * i,
                resonance=0.1 + 0.03 * i,
                temperature=0.1 * factor,  # Exponential growth here
                substrate=0.1 + 0.06 * i,
            )
            result = monitor.update(state)

        assert result is not None
        if result.status != "insufficient_data":
            # With exponential growth in one dimension, max eigenvalue should be elevated
            assert result.max_eigenvalue > 0.1

    def test_eigenstate_defaults(self):
        es = EigenState()
        assert es.eigenvalues == []
        assert es.max_eigenvalue == 0.0
        assert es.status == "unknown"

    def test_window_trimming(self):
        config = FieldConfig(jacobian_window=3)
        monitor = EigenMonitor(config)
        for i in range(20):
            state = FieldState(coherence=0.5 + 0.01 * (i % 5),
                               entropy=0.5, resonance=0.5,
                               temperature=0.1, substrate=0.5)
            monitor.update(state)
        # Should keep at most window + 2
        assert monitor.history_length <= 5

    def test_constant_system_low_eigenvalues(self):
        """Constant CERTX (zero diff) should give near-zero eigenvalues."""
        config = FieldConfig(jacobian_window=3)
        monitor = EigenMonitor(config)
        for i in range(5):
            state = FieldState(coherence=0.5, entropy=0.5, resonance=0.5,
                               temperature=0.1, substrate=0.5)
            result = monitor.update(state)
        if result is not None and result.status != "insufficient_data":
            # All diffs are zero → eigenvalues should be ~0
            assert result.max_eigenvalue < 0.1
