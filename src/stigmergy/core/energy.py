"""Energy model for context lifecycle.

energy(t) = base_energy * e^(-lambda * elapsed_seconds)

When energy drops below decay_threshold, context enters DECAYING.
When energy drops below archive_threshold, context archives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class ContextStatus:
    """Well-known context lifecycle states.

    Agents may observe additional states or propose transitions
    beyond these three as the system evolves.
    """

    ACTIVE = "active"
    DECAYING = "decaying"
    ARCHIVED = "archived"


@dataclass
class EnergyState:
    energy: float
    status: str


def compute_energy(
    base_energy: float,
    decay_rate: float,
    elapsed_seconds: float,
    decay_threshold: float = 0.3,
    archive_threshold: float = 0.05,
) -> EnergyState:
    """Compute current energy and lifecycle status.

    Pure function. No side effects.

    Args:
        base_energy: Energy at time of last signal.
        decay_rate: Lambda (exponential decay rate).
        elapsed_seconds: Time since last signal.
        decay_threshold: Below this, context is DECAYING.
        archive_threshold: Below this, context is ARCHIVED.

    Returns:
        EnergyState with current energy and status.
    """
    energy = base_energy * math.exp(-decay_rate * elapsed_seconds)

    if energy < archive_threshold:
        status = ContextStatus.ARCHIVED
    elif energy < decay_threshold:
        status = ContextStatus.DECAYING
    else:
        status = ContextStatus.ACTIVE

    return EnergyState(energy=energy, status=status)
