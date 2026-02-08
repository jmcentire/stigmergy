"""EigenMonitor — Jacobian eigenvalue tracking.

The "heartbeat" diagnostic. Estimates the Jacobian of the CERTX
state space from recent history, computes eigenvalues.

|lambda| < 0.8: rigid fossil (over-damped)
|lambda| in [0.8, 1.2]: healthy breathing
|lambda| > 1.2: unstable (diverging)

Includes EMA smoothing on max eigenvalue and hysteresis bands
to prevent status flickering on noisy data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stigmergy.unity.field_config import FieldConfig
    from stigmergy.unity.field_state import FieldState

logger = logging.getLogger(__name__)

# Hysteresis margin: must cross threshold +/- margin to change status.
# Prevents flickering when eigenvalue hovers near a boundary.
_HYSTERESIS_MARGIN = 0.05

# EMA alpha for smoothing max eigenvalue across ticks
_EIGEN_EMA_ALPHA = 0.3


@dataclass
class EigenState:
    """Result of eigenvalue monitoring."""

    eigenvalues: list[float] = field(default_factory=list)
    max_eigenvalue: float = 0.0
    min_eigenvalue: float = 0.0
    smoothed_max: float = 0.0
    status: str = "unknown"  # "rigid", "healthy", "unstable", "insufficient_data"


class EigenMonitor:
    """Jacobian eigenvalue monitoring from CERTX state history.

    Accumulates CERTX vectors; when enough history exists, estimates
    the Jacobian and computes eigenvalues. EMA smooths the result;
    hysteresis bands prevent status flickering.
    """

    def __init__(self, config: FieldConfig) -> None:
        self._window = max(2, config.jacobian_window)
        self._healthy_min = config.eigenvalue_healthy_min
        self._healthy_max = config.eigenvalue_healthy_max
        self._history: list[list[float]] = []
        self._smoothed_max: float = 0.0
        self._prev_status: str = "insufficient_data"
        self._initialized: bool = False

    def update(self, state: FieldState) -> EigenState | None:
        """Update with new CERTX state. Returns EigenState or None if insufficient data."""
        certx = [
            state.coherence,
            state.entropy,
            state.resonance,
            state.temperature,
            state.substrate,
        ]
        self._history.append(certx)

        # Keep window + 1 for diff computation
        max_keep = self._window + 2
        if len(self._history) > max_keep:
            self._history = self._history[-max_keep:]

        # Need at least window + 1 points to compute Jacobian
        if len(self._history) < self._window + 1:
            return EigenState(status="insufficient_data")

        try:
            import numpy as np

            # Build state matrix and difference matrix
            states = np.array(self._history[-(self._window + 1):])
            X = states[:-1].T    # 5 x window
            dX = np.diff(states, axis=0).T  # 5 x window

            # Check for degenerate cases (all-zero diffs or near-singular X)
            if np.allclose(dX, 0.0):
                # Constant system — eigenvalues are zero
                self._update_smoothed(0.0)
                return EigenState(
                    eigenvalues=[0.0] * 5,
                    max_eigenvalue=0.0,
                    min_eigenvalue=0.0,
                    smoothed_max=self._smoothed_max,
                    status=self._classify(self._smoothed_max),
                )

            # Estimate Jacobian: dX = J @ X => J = dX @ pinv(X)
            X_pinv = np.linalg.pinv(X, rcond=1e-6)
            J = dX @ X_pinv

            # Check for NaN/inf in Jacobian
            if not np.all(np.isfinite(J)):
                logger.debug("Jacobian contains NaN/inf, skipping")
                return EigenState(
                    smoothed_max=self._smoothed_max,
                    status=self._prev_status,
                )

            # Compute eigenvalues
            eigvals = np.abs(np.linalg.eigvals(J))
            eigvals_list = sorted(eigvals.tolist(), reverse=True)

            # Filter out NaN/inf eigenvalues
            eigvals_list = [e for e in eigvals_list if e == e and e != float("inf")]
            if not eigvals_list:
                return EigenState(
                    smoothed_max=self._smoothed_max,
                    status=self._prev_status,
                )

            max_eig = eigvals_list[0]
            min_eig = eigvals_list[-1]

            # EMA smooth the max eigenvalue
            self._update_smoothed(max_eig)
            status = self._classify(self._smoothed_max)

            return EigenState(
                eigenvalues=eigvals_list,
                max_eigenvalue=max_eig,
                min_eigenvalue=min_eig,
                smoothed_max=self._smoothed_max,
                status=status,
            )

        except ImportError:
            logger.debug("numpy not available for eigenvalue monitoring")
            return EigenState(status="insufficient_data")
        except Exception as exc:
            logger.debug("Eigenvalue computation failed: %s", exc)
            return EigenState(
                smoothed_max=self._smoothed_max,
                status=self._prev_status,
            )

    def _update_smoothed(self, raw_max: float) -> None:
        """EMA smooth the max eigenvalue."""
        if not self._initialized:
            self._smoothed_max = raw_max
            self._initialized = True
        else:
            self._smoothed_max = (
                _EIGEN_EMA_ALPHA * raw_max
                + (1 - _EIGEN_EMA_ALPHA) * self._smoothed_max
            )

    def _classify(self, smoothed: float) -> str:
        """Classify status with hysteresis to prevent flickering.

        To transition OUT of a status, the smoothed eigenvalue must
        cross the boundary plus a margin. To stay, it just needs to
        be within the band.
        """
        prev = self._prev_status
        margin = _HYSTERESIS_MARGIN

        if prev == "healthy":
            # Must exceed boundary + margin to leave healthy
            if smoothed > self._healthy_max + margin:
                status = "unstable"
            elif smoothed < self._healthy_min - margin:
                status = "rigid"
            else:
                status = "healthy"
        elif prev == "unstable":
            # Must drop below boundary - margin to leave unstable
            if smoothed < self._healthy_max - margin:
                status = "healthy" if smoothed >= self._healthy_min else "rigid"
            else:
                status = "unstable"
        elif prev == "rigid":
            # Must rise above boundary + margin to leave rigid
            if smoothed > self._healthy_min + margin:
                status = "healthy" if smoothed <= self._healthy_max else "unstable"
            else:
                status = "rigid"
        else:
            # First classification — no hysteresis
            if smoothed > self._healthy_max:
                status = "unstable"
            elif smoothed < self._healthy_min:
                status = "rigid"
            else:
                status = "healthy"

        self._prev_status = status
        return status

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def smoothed_max(self) -> float:
        return self._smoothed_max
