"""Tests for PID controller — stability envelope."""

from __future__ import annotations

import pytest

from stigmergy.unity.field_config import FieldConfig
from stigmergy.unity.pid_controller import PIDController, PIDState


class TestPIDController:
    def test_initial_state(self):
        config = FieldConfig()
        pid = PIDController(config)
        state = pid.update(0.7)  # Exactly at setpoint
        assert state.error == pytest.approx(0.0)
        assert state.output == pytest.approx(0.0, abs=0.01)

    def test_proportional_term(self):
        config = FieldConfig(pid_kp=1.0, pid_ki=0.0, pid_kd=0.0, pid_setpoint=0.7)
        pid = PIDController(config)
        state = pid.update(0.5)  # Below setpoint
        assert state.error == pytest.approx(0.2)
        assert state.p_term == pytest.approx(0.2)
        assert state.i_term == pytest.approx(0.0)  # ki=0
        assert state.d_term == pytest.approx(0.0)  # first call, no derivative

    def test_integral_accumulation(self):
        config = FieldConfig(pid_kp=0.0, pid_ki=1.0, pid_kd=0.0, pid_setpoint=0.7)
        pid = PIDController(config)
        pid.update(0.5, dt=1.0)  # error=0.2, integral=0.2
        state = pid.update(0.5, dt=1.0)  # error=0.2, integral=0.4
        assert state.i_term == pytest.approx(0.4)

    def test_anti_windup(self):
        config = FieldConfig(pid_kp=0.0, pid_ki=1.0, pid_kd=0.0,
                             pid_setpoint=0.7, pid_integral_max=0.5)
        pid = PIDController(config)
        # Keep adding positive error — integral should clamp at 0.5
        for _ in range(20):
            state = pid.update(0.0, dt=1.0)  # error=0.7 each time
        assert state.i_term == pytest.approx(0.5)

    def test_derivative_term(self):
        config = FieldConfig(pid_kp=0.0, pid_ki=0.0, pid_kd=1.0, pid_setpoint=0.7)
        pid = PIDController(config)
        pid.update(0.5, dt=1.0)  # error=0.2, no derivative (first call)
        state = pid.update(0.3, dt=1.0)  # error=0.4, derivative=(0.4-0.2)/1=0.2
        assert state.d_term == pytest.approx(0.2)

    def test_no_derivative_spike_on_first_call(self):
        config = FieldConfig(pid_kp=0.0, pid_ki=0.0, pid_kd=1.0, pid_setpoint=0.7)
        pid = PIDController(config)
        state = pid.update(0.0, dt=1.0)  # Large error but first call
        assert state.d_term == 0.0

    def test_output_clamped(self):
        config = FieldConfig(pid_kp=10.0, pid_ki=0.0, pid_kd=0.0, pid_setpoint=0.7)
        pid = PIDController(config)
        state = pid.update(0.0)  # error=0.7, P=7.0 -> clamped to 1.0
        assert state.output == pytest.approx(1.0)
        state = pid.update(1.5)  # error=-0.8, P=-8.0 -> clamped to -1.0
        assert state.output == pytest.approx(-1.0)

    def test_convergence_to_setpoint(self):
        """Step response should converge with reasonable gains."""
        config = FieldConfig(pid_kp=0.5, pid_ki=0.1, pid_kd=0.05, pid_setpoint=0.7)
        pid = PIDController(config)
        measurement = 0.3  # Start far from setpoint
        for _ in range(50):
            state = pid.update(measurement, dt=1.0)
            measurement += state.output * 0.1  # Simple plant response
        # Should be close to setpoint after convergence
        assert abs(measurement - 0.7) < 0.1

    def test_ki_zero_kd_zero_is_proportional_only(self):
        """Backward compatibility: Ki=0, Kd=0 degrades to proportional."""
        config = FieldConfig(pid_kp=0.3, pid_ki=0.0, pid_kd=0.0, pid_setpoint=0.7)
        pid = PIDController(config)
        state = pid.update(0.5)
        assert state.i_term == 0.0
        assert state.d_term == 0.0
        assert state.output == pytest.approx(0.3 * 0.2)

    def test_reset(self):
        config = FieldConfig(pid_kp=0.3, pid_ki=0.1, pid_kd=0.1, pid_setpoint=0.7)
        pid = PIDController(config)
        pid.update(0.5)
        pid.update(0.5)
        pid.reset()
        state = pid.update(0.5)
        # After reset, should behave like first call (no derivative)
        assert state.d_term == 0.0

    def test_dt_scaling(self):
        config = FieldConfig(pid_kp=0.0, pid_ki=1.0, pid_kd=0.0, pid_setpoint=0.7)
        pid = PIDController(config)
        pid.update(0.5, dt=2.0)  # error=0.2, integral=0.2*2=0.4
        state = pid.update(0.5, dt=2.0)  # integral += 0.2*2=0.4, total=0.8
        assert state.i_term == pytest.approx(0.8)
