"""PIDController — stability envelope.

Replaces proportional-only fullness->threshold with full P+I+D
control on V(x). Ki=0, Kd=0 degrades to current proportional-only
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from stigmergy.unity.field_config import FieldConfig


@dataclass
class PIDState:
    """Output of a PID update step."""

    output: float = 0.0
    error: float = 0.0
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0


class PIDController:
    """PID controller for mesh stability envelope.

    P: Current distance from target V(x). Acts immediately.
    I: Accumulated error. Eliminates steady-state offset.
    D: Rate of error change. Prevents overshoot on transient spikes.
    """

    def __init__(self, config: FieldConfig) -> None:
        self._kp = config.pid_kp
        self._ki = config.pid_ki
        self._kd = config.pid_kd
        self._integral_max = config.pid_integral_max
        self._setpoint = config.pid_setpoint
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._initialized: bool = False

    def update(self, measurement: float, dt: float = 1.0) -> PIDState:
        """Compute PID output from current measurement.

        Args:
            measurement: Current value (e.g., normalized V(x) or health).
            dt: Time step (signals between updates).

        Returns:
            PIDState with output clamped to [-1, 1].
        """
        error = self._setpoint - measurement

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral = max(-self._integral_max, min(self._integral_max, self._integral))

        # Derivative (zero on first call to avoid spike)
        if not self._initialized:
            derivative = 0.0
            self._initialized = True
        else:
            derivative = (error - self._prev_error) / dt if dt > 0 else 0.0

        self._prev_error = error

        p_term = self._kp * error
        i_term = self._ki * self._integral
        d_term = self._kd * derivative
        output = p_term + i_term + d_term

        # Clamp output
        output = max(-1.0, min(1.0, output))

        return PIDState(
            output=output,
            error=error,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
        )

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False
