"""Tests for field bridge — mechanism design, org state -> mesh physics."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from stigmergy.unity.field_bridge import FieldAdjustment, FieldBridge
from stigmergy.unity.field_config import FieldConfig
from stigmergy.unity.field_state import FieldState


@dataclass
class MockWorkerForBridge:
    id: str = "w1"
    base_threshold: float = 0.15


@dataclass
class MockMesh:
    workers: list = field(default_factory=lambda: [
        MockWorkerForBridge("w1", 0.15),
        MockWorkerForBridge("w2", 0.20),
    ])


class TestFieldBridge:
    def test_bridge_disabled_returns_empty(self):
        config = FieldConfig(bridge_enabled=False)
        bridge = FieldBridge(config)
        state = FieldState(temperature=1.0, entropy=0.1, dysmemic_pressure=0.5)
        adjustments = bridge.compute_adjustments(state)
        assert adjustments == []

    def test_bridge_disabled_no_apply(self):
        config = FieldConfig(bridge_enabled=False)
        bridge = FieldBridge(config)
        mesh = MockMesh()
        actions = bridge.apply_adjustments(
            [FieldAdjustment("vigilance", "increase", 0.1, "test")],
            mesh,
        )
        assert actions == []

    def test_high_temperature_raises_vigilance(self):
        config = FieldConfig(bridge_enabled=True, bridge_vigilance_gain=0.1)
        bridge = FieldBridge(config)
        state = FieldState(temperature=0.8, entropy=0.5, resonance=0.5,
                           dysmemic_pressure=0.0)
        adjustments = bridge.compute_adjustments(state)
        vigilance_adj = [a for a in adjustments if a.target == "vigilance"
                         and a.direction == "increase"
                         and "temperature" in a.reason]
        assert len(vigilance_adj) >= 1

    def test_low_entropy_lowers_vigilance(self):
        config = FieldConfig(bridge_enabled=True, bridge_vigilance_gain=0.1)
        bridge = FieldBridge(config)
        state = FieldState(temperature=0.1, entropy=0.1, resonance=0.5,
                           dysmemic_pressure=0.0)
        adjustments = bridge.compute_adjustments(state)
        vig_decrease = [a for a in adjustments if a.target == "vigilance"
                        and a.direction == "decrease"
                        and "entropy" in a.reason]
        assert len(vig_decrease) >= 1

    def test_high_dysmemic_raises_quorum(self):
        config = FieldConfig(bridge_enabled=True, bridge_quorum_gain=0.1)
        bridge = FieldBridge(config)
        state = FieldState(temperature=0.1, entropy=0.5, resonance=0.5,
                           dysmemic_pressure=0.5)
        adjustments = bridge.compute_adjustments(state)
        quorum_adj = [a for a in adjustments if a.target == "quorum"
                      and a.direction == "increase"]
        assert len(quorum_adj) >= 1

    def test_low_resonance_lowers_vigilance(self):
        config = FieldConfig(bridge_enabled=True, bridge_vigilance_gain=0.1)
        bridge = FieldBridge(config)
        state = FieldState(temperature=0.1, entropy=0.5, resonance=0.1,
                           dysmemic_pressure=0.0)
        adjustments = bridge.compute_adjustments(state)
        vig_decrease = [a for a in adjustments if a.target == "vigilance"
                        and a.direction == "decrease"
                        and "resonance" in a.reason]
        assert len(vig_decrease) >= 1

    def test_pid_output_adjustment(self):
        config = FieldConfig(bridge_enabled=True, bridge_vigilance_gain=0.1)
        bridge = FieldBridge(config)
        state = FieldState(temperature=0.1, entropy=0.5, resonance=0.5,
                           dysmemic_pressure=0.0)
        adjustments = bridge.compute_adjustments(state, pid_output=0.5)
        pid_adj = [a for a in adjustments if "PID" in a.reason]
        assert len(pid_adj) >= 1
        assert pid_adj[0].direction == "increase"

    def test_breathing_modulation(self):
        config = FieldConfig(bridge_enabled=True, bridge_vigilance_gain=0.1)
        bridge = FieldBridge(config)
        state = FieldState(temperature=0.1, entropy=0.5, resonance=0.5,
                           dysmemic_pressure=0.0)
        adjustments = bridge.compute_adjustments(
            state, breathing_modulation=-0.05)
        breath_adj = [a for a in adjustments if "breathing" in a.reason]
        assert len(breath_adj) >= 1
        assert breath_adj[0].direction == "decrease"

    def test_all_gains_zero_no_adjustments(self):
        config = FieldConfig(
            bridge_enabled=True,
            bridge_vigilance_gain=0.0,
            bridge_learning_gain=0.0,
            bridge_quorum_gain=0.0,
        )
        bridge = FieldBridge(config)
        state = FieldState(temperature=1.0, entropy=0.0, resonance=0.0,
                           dysmemic_pressure=1.0)
        adjustments = bridge.compute_adjustments(state, pid_output=1.0)
        assert adjustments == []

    def test_apply_adjustments_modifies_thresholds(self):
        config = FieldConfig(bridge_enabled=True, bridge_vigilance_gain=0.1)
        bridge = FieldBridge(config)
        mesh = MockMesh()
        old_thresholds = [w.base_threshold for w in mesh.workers]
        adjustments = [FieldAdjustment("vigilance", "increase", 0.05, "test")]
        actions = bridge.apply_adjustments(adjustments, mesh)
        assert len(actions) > 0
        for i, w in enumerate(mesh.workers):
            assert w.base_threshold > old_thresholds[i]

    def test_apply_adjustments_clamps_thresholds(self):
        config = FieldConfig(bridge_enabled=True)
        bridge = FieldBridge(config)
        mesh = MockMesh()
        mesh.workers[0].base_threshold = 0.94
        adjustments = [FieldAdjustment("vigilance", "increase", 0.5, "test")]
        bridge.apply_adjustments(adjustments, mesh)
        assert mesh.workers[0].base_threshold <= 0.95
