"""CalibrationState, Brier scores, N* estimation, babbling detection.

Proper scoring rules for agent prediction quality. Called after
corroboration resolves each finding.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from stigmergy.unity.field_config import FieldConfig


@dataclass
class AgentCalibration:
    """Calibration state for a single agent."""

    predictions: list[tuple[float, bool]] = field(default_factory=list)
    brier_score: float = 0.0
    n_star: float = 1.0
    is_babbling: bool = False


@dataclass
class CalibrationState:
    """Calibration tracking across all agents."""

    agents: dict[str, AgentCalibration] = field(default_factory=dict)
    window: int = 50
    babbling_threshold: float = 0.3
    decay: float = 0.95


def brier_score(predictions: list[tuple[float, bool]]) -> float:
    """Compute Brier score from (forecast, outcome) pairs.

    BS = (1/N) * sum((forecast_i - outcome_i)^2)
    BS=0: perfect. BS=0.25: always guessing 0.5. BS>0.33: babbling.
    """
    if not predictions:
        return 0.0
    total = sum(
        (forecast - (1.0 if outcome else 0.0)) ** 2
        for forecast, outcome in predictions
    )
    return total / len(predictions)


def compute_n_star(bs: float) -> float:
    """Crawford-Sobel partition count from calibration quality.

    Perfect (BS~0): N*->10+. Random (BS~0.25): N*~2. Babbling (BS>0.33): N*~1.
    """
    bs_clamped = max(bs, 0.01)
    return max(1.0, -0.5 + 0.5 * math.sqrt(1 + 2 / bs_clamped))


def update_calibration(
    agent_id: str,
    confidence: float,
    was_corroborated: bool,
    state: CalibrationState,
    config: FieldConfig,
) -> AgentCalibration:
    """Record a (confidence, outcome) pair and recompute Brier score.

    Called after corroboration resolves each finding.
    """
    if agent_id not in state.agents:
        state.agents[agent_id] = AgentCalibration()

    agent_cal = state.agents[agent_id]

    # Record prediction
    agent_cal.predictions.append((confidence, was_corroborated))

    # Trim to window
    if len(agent_cal.predictions) > state.window:
        agent_cal.predictions = agent_cal.predictions[-state.window:]

    # Recompute Brier score
    agent_cal.brier_score = brier_score(agent_cal.predictions)

    # N* estimation
    agent_cal.n_star = compute_n_star(agent_cal.brier_score)

    # Babbling detection
    agent_cal.is_babbling = agent_cal.brier_score > state.babbling_threshold

    return agent_cal


def apply_session_decay(state: CalibrationState) -> None:
    """Apply per-session decay to all predictions."""
    for agent_cal in state.agents.values():
        # Decay by removing oldest predictions probabilistically
        if agent_cal.predictions and state.decay < 1.0:
            keep_count = max(1, int(len(agent_cal.predictions) * state.decay))
            agent_cal.predictions = agent_cal.predictions[-keep_count:]
            agent_cal.brier_score = brier_score(agent_cal.predictions)
            agent_cal.n_star = compute_n_star(agent_cal.brier_score)
            agent_cal.is_babbling = agent_cal.brier_score > state.babbling_threshold
