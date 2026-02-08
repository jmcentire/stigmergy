"""FieldBridge — org state -> mesh physics adjustments.

Mechanism design: not commands — physics shifts. All gains are config
parameters. All gains at zero = no adjustment. bridge_enabled=False =
entire component disabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from stigmergy.unity.field_config import FieldConfig

if TYPE_CHECKING:
    from stigmergy.unity.calibration import CalibrationState

logger = logging.getLogger(__name__)


@dataclass
class FieldAdjustment:
    """A single mesh physics adjustment."""

    target: str           # "vigilance", "learning_rate", "quorum", "consensus"
    direction: str        # "increase" or "decrease"
    magnitude: float      # [0, 1] strength of adjustment
    reason: str           # Why this adjustment


class FieldBridge:
    """Translate organizational state into mesh incentive adjustments.

    Reads FieldState + controller outputs, produces adjustments to
    worker vigilance, learning rates, and quorum thresholds.
    """

    def __init__(self, config: FieldConfig) -> None:
        self._enabled = config.bridge_enabled
        self._vigilance_gain = config.bridge_vigilance_gain
        self._learning_gain = config.bridge_learning_gain
        self._quorum_gain = config.bridge_quorum_gain

    def compute_adjustments(
        self,
        mesh_state: Any,
        pid_output: float = 0.0,
        breathing_modulation: float = 0.0,
        calibration: CalibrationState | None = None,
    ) -> list[FieldAdjustment]:
        """Compute physics adjustments from field state.

        Args:
            mesh_state: FieldState with CERTX dimensions.
            pid_output: PID controller output [-1, 1].
            breathing_modulation: Breathing phase modulation.

        Returns:
            List of FieldAdjustment objects. Empty if bridge disabled.
        """
        if not self._enabled:
            return []

        adjustments: list[FieldAdjustment] = []

        # High temperature (volatile env) → raise vigilance (be more selective)
        if hasattr(mesh_state, "temperature") and mesh_state.temperature > 0.5:
            mag = min(1.0, mesh_state.temperature - 0.5) * self._vigilance_gain
            if mag > 0.001:
                adjustments.append(FieldAdjustment(
                    target="vigilance",
                    direction="increase",
                    magnitude=mag,
                    reason=f"high temperature ({mesh_state.temperature:.2f})",
                ))

        # Low entropy (monoculture) → lower vigilance (accept more diversity)
        if hasattr(mesh_state, "entropy") and mesh_state.entropy < 0.3:
            mag = (0.3 - mesh_state.entropy) * self._vigilance_gain
            if mag > 0.001:
                adjustments.append(FieldAdjustment(
                    target="vigilance",
                    direction="decrease",
                    magnitude=mag,
                    reason=f"low entropy ({mesh_state.entropy:.2f})",
                ))

        # High dysmemic pressure → raise quorum (demand more evidence)
        if hasattr(mesh_state, "dysmemic_pressure") and mesh_state.dysmemic_pressure > 0.2:
            mag = mesh_state.dysmemic_pressure * self._quorum_gain
            if mag > 0.001:
                adjustments.append(FieldAdjustment(
                    target="quorum",
                    direction="increase",
                    magnitude=mag,
                    reason=f"dysmemic pressure ({mesh_state.dysmemic_pressure:.2f})",
                ))

        # Low resonance → lower vigilance (re-couple to environment)
        if hasattr(mesh_state, "resonance") and mesh_state.resonance < 0.3:
            mag = (0.3 - mesh_state.resonance) * self._vigilance_gain
            if mag > 0.001:
                adjustments.append(FieldAdjustment(
                    target="vigilance",
                    direction="decrease",
                    magnitude=mag,
                    reason=f"low resonance ({mesh_state.resonance:.2f})",
                ))

        # PID output → vigilance adjustment
        if abs(pid_output) > 0.01 and self._vigilance_gain > 0:
            direction = "increase" if pid_output > 0 else "decrease"
            mag = abs(pid_output) * self._vigilance_gain
            adjustments.append(FieldAdjustment(
                target="vigilance",
                direction=direction,
                magnitude=mag,
                reason=f"PID output ({pid_output:.3f})",
            ))

        # Breathing modulation → vigilance (additive)
        if abs(breathing_modulation) > 0.001 and self._vigilance_gain > 0:
            direction = "increase" if breathing_modulation > 0 else "decrease"
            mag = abs(breathing_modulation)
            adjustments.append(FieldAdjustment(
                target="vigilance",
                direction=direction,
                magnitude=mag,
                reason=f"breathing ({breathing_modulation:+.3f})",
            ))

        # Calibration feedback: babbling agents → tighten quorum,
        # low average N* → raise consensus threshold (demand more agreement)
        if calibration is not None and calibration.agents and self._quorum_gain > 0:
            babbling_count = sum(1 for ac in calibration.agents.values() if ac.is_babbling)
            total_agents = len(calibration.agents)
            if babbling_count > 0 and total_agents > 0:
                babbling_fraction = babbling_count / total_agents
                mag = babbling_fraction * self._quorum_gain
                if mag > 0.001:
                    adjustments.append(FieldAdjustment(
                        target="consensus",
                        direction="increase",
                        magnitude=mag,
                        reason=f"babbling agents ({babbling_count}/{total_agents})",
                    ))

            # Average N* across agents — low N* means poor signal-to-noise
            n_stars = [ac.n_star for ac in calibration.agents.values()
                       if len(ac.predictions) >= 5]  # Only agents with enough data
            if n_stars:
                avg_n_star = sum(n_stars) / len(n_stars)
                if avg_n_star < 3.0:  # Below useful partition count
                    mag = (3.0 - avg_n_star) / 3.0 * self._quorum_gain
                    if mag > 0.001:
                        adjustments.append(FieldAdjustment(
                            target="quorum",
                            direction="increase",
                            magnitude=mag,
                            reason=f"low avg N* ({avg_n_star:.1f})",
                        ))

        return adjustments

    def apply_adjustments(
        self,
        adjustments: list[FieldAdjustment],
        mesh: Any,
    ) -> list[str]:
        """Apply adjustments to mesh physics.

        Mechanically modifies:
        - Worker base_threshold (vigilance)
        - Mesh _consensus_threshold (quorum/consensus)
        - Mesh _gap_threshold (worker spawning sensitivity)

        Returns log of actions taken.
        """
        if not self._enabled or not adjustments:
            return []

        actions: list[str] = []
        workers = list(mesh.workers) if mesh is not None and hasattr(mesh, "workers") else []

        for adj in adjustments:
            if adj.target == "vigilance" and workers:
                delta = adj.magnitude if adj.direction == "increase" else -adj.magnitude
                for w in workers:
                    if hasattr(w, "base_threshold"):
                        old = w.base_threshold
                        w.base_threshold = max(0.05, min(0.95, old + delta))
                actions.append(f"vigilance {adj.direction} by {adj.magnitude:.4f}: {adj.reason}")

            elif adj.target == "quorum" and mesh is not None:
                # Adjust gap threshold — controls how eagerly new workers spawn
                delta = adj.magnitude if adj.direction == "increase" else -adj.magnitude
                if hasattr(mesh, "_gap_threshold"):
                    old = mesh._gap_threshold
                    mesh._gap_threshold = max(0.02, min(0.25, old + delta))
                    actions.append(f"gap_threshold {old:.4f}→{mesh._gap_threshold:.4f}: {adj.reason}")

            elif adj.target == "consensus" and mesh is not None:
                # Adjust consensus threshold — controls agreement requirement
                delta = adj.magnitude if adj.direction == "increase" else -adj.magnitude
                if hasattr(mesh, "_consensus_threshold"):
                    old = mesh._consensus_threshold
                    mesh._consensus_threshold = max(0.2, min(0.9, old + delta))
                    actions.append(f"consensus {old:.4f}→{mesh._consensus_threshold:.4f}: {adj.reason}")

            elif adj.target == "learning_rate":
                actions.append(f"learning {adj.direction} by {adj.magnitude:.4f}: {adj.reason}")

        if actions:
            logger.info("Field bridge applied: %s", "; ".join(actions))

        return actions
