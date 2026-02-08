"""Tests for breathing controller — accumulate/integrate oscillation."""

from __future__ import annotations

import math

import pytest

from stigmergy.unity.breathing import BreathingController, BreathingState
from stigmergy.unity.field_config import FieldConfig


class TestBreathingController:
    def test_initial_tick(self):
        config = FieldConfig(breathing_period=100, breathing_amplitude=0.1, accumulate_ratio=0.6)
        bc = BreathingController(config)
        state = bc.tick(signal_count=0)
        assert state.phase == "accumulate"
        assert state.cycle_position == 0.0
        assert state.modulation == pytest.approx(0.0, abs=0.001)  # sin(0) = 0

    def test_accumulate_phase_negative_modulation(self):
        """Accumulate phase should produce negative modulation (lower thresholds)."""
        config = FieldConfig(breathing_period=100, breathing_amplitude=0.1, accumulate_ratio=0.6)
        bc = BreathingController(config)
        # Midpoint of accumulate phase: position=0.3, which is 50% through accumulate
        state = bc.tick(signal_count=30)
        assert state.phase == "accumulate"
        assert state.modulation < 0  # Negative = lower thresholds
        assert state.modulation >= -0.1  # Bounded by amplitude

    def test_integrate_phase_positive_modulation(self):
        """Integrate phase should produce positive modulation (raise thresholds)."""
        config = FieldConfig(breathing_period=100, breathing_amplitude=0.1, accumulate_ratio=0.6)
        bc = BreathingController(config)
        # Midpoint of integrate phase: position=0.8, which is 50% through integrate
        state = bc.tick(signal_count=80)
        assert state.phase == "integrate"
        assert state.modulation > 0  # Positive = raise thresholds
        assert state.modulation <= 0.1  # Bounded by amplitude

    def test_cycle_period(self):
        """After one full period, should be back at start."""
        config = FieldConfig(breathing_period=100, breathing_amplitude=0.1, accumulate_ratio=0.6)
        bc = BreathingController(config)
        state_0 = bc.tick(signal_count=0)
        state_100 = bc.tick(signal_count=100)
        assert state_0.cycle_position == state_100.cycle_position
        assert state_0.phase == state_100.phase

    def test_asymmetric_split(self):
        """60/40 split: accumulate for 60%, integrate for 40%."""
        config = FieldConfig(breathing_period=100, breathing_amplitude=0.1, accumulate_ratio=0.6)
        bc = BreathingController(config)
        # At position 59 (last of accumulate)
        state_59 = bc.tick(signal_count=59)
        assert state_59.phase == "accumulate"
        # At position 60 (first of integrate)
        state_60 = bc.tick(signal_count=60)
        assert state_60.phase == "integrate"

    def test_sinusoidal_shape(self):
        """Modulation should follow sinusoidal envelope."""
        config = FieldConfig(breathing_period=200, breathing_amplitude=0.2, accumulate_ratio=0.5)
        bc = BreathingController(config)
        # Peak of accumulate at 25% of period (50% through accumulate)
        state = bc.tick(signal_count=50)
        expected = -0.2 * math.sin(math.pi * 0.5)  # -0.2
        assert state.modulation == pytest.approx(expected, abs=0.01)

    def test_amplitude_zero_disables(self):
        """amplitude=0 produces zero modulation always."""
        config = FieldConfig(breathing_period=100, breathing_amplitude=0.0, accumulate_ratio=0.6)
        bc = BreathingController(config)
        for i in range(100):
            state = bc.tick(signal_count=i)
            assert state.modulation == pytest.approx(0.0)

    def test_smooth_transition(self):
        """No discrete jump at phase boundary."""
        config = FieldConfig(breathing_period=1000, breathing_amplitude=0.1, accumulate_ratio=0.6)
        bc = BreathingController(config)
        # Right before transition
        s1 = bc.tick(signal_count=599)
        # Right after transition
        s2 = bc.tick(signal_count=600)
        # Both should be near zero at boundaries (sin(pi*~1) and sin(pi*0))
        assert abs(s1.modulation) < 0.01
        assert abs(s2.modulation) < 0.01

    def test_internal_counter(self):
        """tick() without argument increments internal counter."""
        config = FieldConfig(breathing_period=100, breathing_amplitude=0.1)
        bc = BreathingController(config)
        bc.tick()
        bc.tick()
        assert bc.signal_count == 2
