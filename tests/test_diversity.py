"""Tests for diversity guard — requisite variety and HOT reserve."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from stigmergy.unity.diversity import (
    DiversityState,
    _simpsons_index,
    _shannon_normalized,
    compute_diversity,
)
from stigmergy.unity.field_config import FieldConfig


@dataclass
class MockContext:
    signal_count: int = 10


@dataclass
class MockWorker:
    id: str = "w1"
    context: MockContext = None

    def __post_init__(self):
        if self.context is None:
            self.context = MockContext()


@dataclass
class MockCompetency:
    weights: dict = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {"a": 0.5, "b": 0.3, "c": 0.2}


@dataclass
class MockAgent:
    competencies: MockCompetency = None

    def __post_init__(self):
        if self.competencies is None:
            self.competencies = MockCompetency()


class TestSimpsonsIndex:
    def test_monoculture(self):
        """Single item = 0 diversity."""
        assert _simpsons_index([1.0]) == pytest.approx(0.0)

    def test_monoculture_with_zeros(self):
        """All signals in one worker."""
        assert _simpsons_index([100.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_uniform_distribution(self):
        """Equal distribution approaches 1 - 1/N."""
        result = _simpsons_index([10.0, 10.0, 10.0, 10.0])
        expected = 1.0 - 4 * (0.25 ** 2)  # = 0.75
        assert result == pytest.approx(expected)

    def test_empty(self):
        assert _simpsons_index([]) == 0.0

    def test_all_zeros(self):
        assert _simpsons_index([0.0, 0.0]) == 0.0


class TestShannonNormalized:
    def test_uniform(self):
        """Uniform distribution = normalized entropy 1.0."""
        result = _shannon_normalized([10.0, 10.0, 10.0, 10.0])
        assert result == pytest.approx(1.0)

    def test_concentrated(self):
        """All in one bin = entropy 0."""
        result = _shannon_normalized([100.0, 0.0, 0.0])
        # Only one non-zero → effectively one category
        assert result < 0.5

    def test_single_item(self):
        assert _shannon_normalized([10.0]) == 0.0


class TestComputeDiversity:
    def test_basic_computation(self):
        workers = [
            MockWorker("w1", MockContext(20)),
            MockWorker("w2", MockContext(15)),
            MockWorker("w3", MockContext(10)),
        ]
        agents = [MockAgent(), MockAgent()]
        config = FieldConfig()
        state = compute_diversity(workers, agents, config)
        assert state.worker_diversity > 0.0
        assert state.agent_diversity > 0.0
        assert isinstance(state.requisite_variety_met, bool)

    def test_hot_reserve_above_floor(self):
        workers = [
            MockWorker("w1", MockContext(10)),
            MockWorker("w2", MockContext(10)),
            MockWorker("w3", MockContext(10)),
        ]
        config = FieldConfig(min_diversity_index=0.3)
        state = compute_diversity(workers, None, config)
        # Uniform 3-worker distribution: Simpson's = 1 - 3*(1/3)^2 = 0.667
        assert state.hot_reserve > 0.0
        assert state.hot_reserve == pytest.approx(state.worker_diversity - 0.3)

    def test_hot_reserve_below_floor(self):
        workers = [MockWorker("w1", MockContext(100))]
        config = FieldConfig(min_diversity_index=0.3)
        state = compute_diversity(workers, None, config)
        assert state.hot_reserve == 0.0

    def test_over_specialization_detected(self):
        workers = [
            MockWorker("w1", MockContext(95)),
            MockWorker("w2", MockContext(3)),
            MockWorker("w3", MockContext(2)),
        ]
        config = FieldConfig(max_specialization=0.9)
        state = compute_diversity(workers, None, config)
        assert len(state.over_specialized) > 0

    def test_no_over_specialization(self):
        workers = [
            MockWorker("w1", MockContext(10)),
            MockWorker("w2", MockContext(10)),
        ]
        config = FieldConfig(max_specialization=0.9)
        state = compute_diversity(workers, None, config)
        assert len(state.over_specialized) == 0

    def test_empty_workers(self):
        config = FieldConfig()
        state = compute_diversity([], None, config)
        assert state.worker_diversity == 0.0
        assert state.hot_reserve == 0.0

    def test_none_workers(self):
        config = FieldConfig()
        state = compute_diversity(None, None, config)
        assert state.worker_diversity == 0.0

    def test_requisite_variety(self):
        """V_C >= V_E when controller variety is sufficient."""
        workers = [
            MockWorker("w1", MockContext(10)),
            MockWorker("w2", MockContext(10)),
            MockWorker("w3", MockContext(10)),
        ]
        agents = [MockAgent(), MockAgent()]
        config = FieldConfig()
        state = compute_diversity(workers, agents, config)
        assert state.controller_variety == pytest.approx(
            state.worker_diversity * state.agent_diversity
        )
