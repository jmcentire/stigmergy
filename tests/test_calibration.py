"""Tests for calibration tracker — Brier scores, N*, babbling detection."""

from __future__ import annotations

import pytest

from stigmergy.unity.calibration import (
    AgentCalibration,
    CalibrationState,
    brier_score,
    compute_n_star,
    update_calibration,
    apply_session_decay,
)
from stigmergy.unity.field_config import FieldConfig


class TestBrierScore:
    def test_perfect_predictions(self):
        """BS=0 for perfect predictions."""
        predictions = [
            (1.0, True),   # Confident yes, was yes
            (0.0, False),  # Confident no, was no
            (0.9, True),   # High confidence, was yes
        ]
        assert brier_score(predictions) == pytest.approx(0.003333, abs=0.01)

    def test_always_half(self):
        """BS=0.25 for always guessing 0.5."""
        predictions = [(0.5, True), (0.5, False), (0.5, True), (0.5, False)]
        assert brier_score(predictions) == pytest.approx(0.25)

    def test_always_wrong(self):
        """BS=1.0 for maximally wrong predictions."""
        predictions = [
            (1.0, False),  # Confident yes, was no
            (0.0, True),   # Confident no, was yes
        ]
        assert brier_score(predictions) == pytest.approx(1.0)

    def test_empty(self):
        assert brier_score([]) == 0.0

    def test_babbling_range(self):
        """Babbling predictions should score > 0.33."""
        # Random-ish predictions
        predictions = [
            (0.9, False), (0.1, True),
            (0.8, False), (0.2, True),
        ]
        bs = brier_score(predictions)
        assert bs > 0.33


class TestComputeNStar:
    def test_perfect_gives_high_n(self):
        """Perfect calibration → N* >> 2."""
        n = compute_n_star(0.01)
        assert n > 5.0

    def test_random_gives_low_n(self):
        """Random guessing (BS=0.25) → N* ≈ 1 (barely above minimum)."""
        n = compute_n_star(0.25)
        assert n == pytest.approx(1.0)  # Formula: max(1, -0.5 + 0.5*sqrt(9)) = 1.0

    def test_babbling_gives_one(self):
        """Babbling (BS > 0.5) → N* ≈ 1."""
        n = compute_n_star(0.5)
        assert n < 2.0

    def test_minimum_one(self):
        """N* never goes below 1."""
        n = compute_n_star(100.0)
        assert n >= 1.0

    def test_zero_brier_clamped(self):
        """BS=0 is clamped to 0.01 to avoid division by zero."""
        n = compute_n_star(0.0)
        assert n > 1.0


class TestUpdateCalibration:
    def test_new_agent(self):
        state = CalibrationState(window=50, babbling_threshold=0.3)
        config = FieldConfig()
        result = update_calibration("agent1", 0.8, True, state, config)
        assert "agent1" in state.agents
        assert len(result.predictions) == 1
        assert result.brier_score >= 0.0

    def test_window_trimming(self):
        state = CalibrationState(window=5, babbling_threshold=0.3)
        config = FieldConfig()
        for i in range(10):
            update_calibration("a", 0.5, i % 2 == 0, state, config)
        assert len(state.agents["a"].predictions) == 5

    def test_babbling_detection(self):
        state = CalibrationState(window=50, babbling_threshold=0.3)
        config = FieldConfig()
        # Maximally wrong predictions → babbling
        for _ in range(20):
            update_calibration("a", 1.0, False, state, config)
            update_calibration("a", 0.0, True, state, config)
        assert state.agents["a"].is_babbling is True

    def test_good_calibration_not_babbling(self):
        state = CalibrationState(window=50, babbling_threshold=0.3)
        config = FieldConfig()
        # Good predictions
        for _ in range(20):
            update_calibration("a", 0.9, True, state, config)
            update_calibration("a", 0.1, False, state, config)
        assert state.agents["a"].is_babbling is False
        assert state.agents["a"].brier_score < 0.1


class TestSessionDecay:
    def test_decay_reduces_predictions(self):
        state = CalibrationState(window=50, babbling_threshold=0.3, decay=0.5)
        config = FieldConfig()
        for _ in range(10):
            update_calibration("a", 0.5, True, state, config)
        assert len(state.agents["a"].predictions) == 10
        apply_session_decay(state)
        assert len(state.agents["a"].predictions) < 10

    def test_decay_one_preserves_all(self):
        state = CalibrationState(window=50, babbling_threshold=0.3, decay=1.0)
        config = FieldConfig()
        for _ in range(10):
            update_calibration("a", 0.5, True, state, config)
        apply_session_decay(state)
        # decay=1.0 keeps all (int(10*1.0) = 10)
        assert len(state.agents["a"].predictions) == 10
