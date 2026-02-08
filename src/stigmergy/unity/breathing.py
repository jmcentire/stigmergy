"""BreathingController — accumulate/integrate oscillation.

The mesh breathes. Accumulate phase lowers thresholds (accept more).
Integrate phase raises thresholds (consolidate). Sinusoidal —
no discrete switching, smooth transitions. amplitude=0 disables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from stigmergy.unity.field_config import FieldConfig


@dataclass
class BreathingState:
    """Output of a breathing tick."""

    phase: str = "accumulate"   # "accumulate" or "integrate"
    modulation: float = 0.0     # Threshold adjustment [-amplitude, +amplitude]
    cycle_position: float = 0.0  # [0, 1) position in current cycle


class BreathingController:
    """Sinusoidal accumulate/integrate oscillation.

    Accumulate (first portion of cycle): Lower thresholds, accept more signals.
    Integrate (rest of cycle): Raise thresholds, consolidate.
    """

    def __init__(self, config: FieldConfig) -> None:
        self._period = max(1, config.breathing_period)
        self._amplitude = config.breathing_amplitude
        self._accumulate_ratio = max(0.01, min(0.99, config.accumulate_ratio))
        self._signal_count: int = 0

    def tick(self, signal_count: int | None = None) -> BreathingState:
        """Advance the breathing cycle.

        Args:
            signal_count: Override internal counter. If None, uses internal.
        """
        if signal_count is not None:
            self._signal_count = signal_count
        else:
            self._signal_count += 1

        cycle_position = (self._signal_count % self._period) / self._period

        if cycle_position < self._accumulate_ratio:
            phase = "accumulate"
            # Progress within accumulate phase: [0, 1]
            phase_progress = cycle_position / self._accumulate_ratio
            # Lower thresholds (negative modulation)
            modulation = -self._amplitude * math.sin(math.pi * phase_progress)
        else:
            phase = "integrate"
            # Progress within integrate phase: [0, 1]
            phase_progress = (cycle_position - self._accumulate_ratio) / (1.0 - self._accumulate_ratio)
            # Raise thresholds (positive modulation)
            modulation = self._amplitude * math.sin(math.pi * phase_progress)

        return BreathingState(
            phase=phase,
            modulation=modulation,
            cycle_position=cycle_position,
        )

    @property
    def signal_count(self) -> int:
        return self._signal_count
