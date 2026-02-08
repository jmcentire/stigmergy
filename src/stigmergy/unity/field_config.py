"""FieldConfig — all knobs (the constitution).

System 5 identity. Every number is a knob. Zero disables the component.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FieldConfig:
    """Constitutional parameters for the unity field system."""

    # --- CERTX dimension weights ---
    w_coherence: float = 1.0       # C: cross-evaluator agreement
    w_entropy: float = 1.0         # E: information diversity
    w_resonance: float = 1.0       # R: mesh-environment coupling
    w_temperature: float = 1.0     # T: rate of state change
    w_substrate: float = 1.0       # X: grounding in evidence

    # --- PID gains ---
    pid_kp: float = 0.3            # Proportional
    pid_ki: float = 0.05           # Integral (accumulated error)
    pid_kd: float = 0.1            # Derivative (rate dampening)
    pid_integral_max: float = 2.0  # Anti-windup clamp
    pid_setpoint: float = 0.7      # Target: V(x)/V_max normalized

    # --- Breathing rhythm ---
    breathing_period: int = 200    # Signals per full cycle
    breathing_amplitude: float = 0.1  # Max modulation of thresholds
    accumulate_ratio: float = 0.6  # Fraction of cycle in accumulate phase

    # --- Diversity guard ---
    min_diversity_index: float = 0.3  # Floor — below this is fragile
    max_specialization: float = 0.9   # Cap per-worker signal share

    # --- Calibration ---
    calibration_window: int = 50   # Findings kept for Brier
    babbling_threshold: float = 0.3  # Brier above this = babbling
    calibration_decay: float = 0.95  # Per-session decay

    # --- Pace layering (EMA alpha per dimension) ---
    pace_coherence: float = 0.1    # Fast — changes quickly
    pace_entropy: float = 0.05     # Medium
    pace_resonance: float = 0.03   # Slow — structural
    pace_temperature: float = 0.2  # Fastest — instantaneous rate
    pace_substrate: float = 0.02   # Slowest — deep evidence

    # --- Field bridge gains ---
    bridge_enabled: bool = False   # Opt-in: mechanical adjustment
    bridge_vigilance_gain: float = 0.1
    bridge_learning_gain: float = 0.05
    bridge_quorum_gain: float = 0.05

    # --- Eigenvalue monitoring ---
    eigenvalue_healthy_min: float = 0.8
    eigenvalue_healthy_max: float = 1.2
    jacobian_window: int = 5       # Pulses for Jacobian estimation

    # --- Meta ---
    enabled: bool = False          # Master switch
    log_field_state: bool = True   # Emit to .stigmergy/field_state.jsonl
