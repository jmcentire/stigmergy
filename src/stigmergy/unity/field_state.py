"""FieldState + compute_mesh_field_state() — the CERTX state vector.

Same structure at mesh level and org level (VSM recursion). Pure
computation — reads existing subsystem data, no LLM calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from stigmergy.unity.field_config import FieldConfig


@dataclass
class FieldState:
    """CERTX state vector — describes mesh or org health."""

    timestamp: str = ""
    signal_index: int = 0
    level: str = "mesh"  # "mesh" or "org"

    # CERTX dimensions
    coherence: float = 0.5     # C: [0,1]
    entropy: float = 0.5       # E: [0,1] normalized Shannon
    resonance: float = 0.5     # R: [0,1]
    temperature: float = 0.0   # T: [0,+inf) magnitude of change-rate vector
    substrate: float = 0.5     # X: [0,1]

    # Derived
    health: float = 0.0               # Weighted CERTX combination
    lyapunov_dv_dt: float = 0.0       # dV/dt from health pulse
    stability_eigenvalue: float = 0.0  # From eigenmonitor (0 until enough history)
    dysmemic_pressure: float = 0.0    # max(0, C - X) — collective delusion risk

    # Components (for transparency)
    raw: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "signal_index": self.signal_index,
            "level": self.level,
            "coherence": round(self.coherence, 4),
            "entropy": round(self.entropy, 4),
            "resonance": round(self.resonance, 4),
            "temperature": round(self.temperature, 4),
            "substrate": round(self.substrate, 4),
            "health": round(self.health, 4),
            "lyapunov_dv_dt": round(self.lyapunov_dv_dt, 6),
            "stability_eigenvalue": round(self.stability_eigenvalue, 4),
            "dysmemic_pressure": round(self.dysmemic_pressure, 4),
            "raw": {k: round(v, 4) for k, v in self.raw.items()},
        }

    def certx_summary(self) -> str:
        """Compact CERTX summary for meta-modeler prompt injection (~50 tokens)."""
        return (
            f"CERTX: C={self.coherence:.2f} E={self.entropy:.2f} "
            f"R={self.resonance:.2f} T={self.temperature:.2f} "
            f"X={self.substrate:.2f} | health={self.health:.2f} | "
            f"dysmemic={self.dysmemic_pressure:.2f} | "
            f"\u03bb={self.stability_eigenvalue:.2f}"
        )


class _PaceSmoother:
    """EMA smoother per CERTX dimension — pace layering."""

    def __init__(self, config: FieldConfig) -> None:
        self._alphas = {
            "coherence": config.pace_coherence,
            "entropy": config.pace_entropy,
            "resonance": config.pace_resonance,
            "temperature": config.pace_temperature,
            "substrate": config.pace_substrate,
        }
        self._values: dict[str, float | None] = {k: None for k in self._alphas}

    def smooth(self, dimension: str, raw_value: float) -> float:
        alpha = self._alphas.get(dimension, 0.1)
        prev = self._values.get(dimension)
        if prev is None:
            self._values[dimension] = raw_value
            return raw_value
        smoothed = alpha * raw_value + (1 - alpha) * prev
        self._values[dimension] = smoothed
        return smoothed


def _safe_mean(values: list[float], default: float = 0.5) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _compute_coherence(
    pulse: Any,
    spectral: Any | None,
) -> float:
    """C: agreement between independent evaluators."""
    components: list[float] = []

    # Worker agreement: 1 - normalized variance of familiarities
    # Max variance for [0,1] values is 0.25 (half at 0, half at 1).
    # Dividing by 0.25 maps variance to [0,1], giving coherence full range.
    if pulse is not None and pulse.workers:
        fams = [w.familiarity for w in pulse.workers]
        if len(fams) >= 2:
            mean_fam = sum(fams) / len(fams)
            var_fam = sum((f - mean_fam) ** 2 for f in fams) / len(fams)
            normalized_var = min(1.0, var_fam / 0.25)  # 0.25 = max variance for [0,1]
            components.append(max(0.0, 1.0 - normalized_var))

    # Quorum consensus ratio
    if pulse is not None and pulse.last_quorum_total > 0:
        components.append(pulse.last_quorum_met / pulse.last_quorum_total)

    # Spectral balance: 1 - |high_freq_ratio - 0.5| * 2
    if spectral is not None:
        hf = getattr(spectral, "high_freq_ratio", None)
        if hf is None and pulse is not None:
            hf = pulse.spectral_high_freq
        if hf is not None:
            components.append(max(0.0, min(1.0, 1.0 - abs(hf - 0.5) * 2)))

    return _safe_mean(components)


def _compute_entropy(
    pulse: Any,
    agents: list | None,
) -> float:
    """E: information diversity (normalized Shannon)."""
    # Source entropy from worker signal counts
    if pulse is not None and pulse.workers:
        counts = [w.signal_count for w in pulse.workers if w.signal_count > 0]
        if len(counts) >= 2:
            total = sum(counts)
            probs = [c / total for c in counts]
            h = -sum(p * math.log2(p) for p in probs if p > 0)
            h_max = math.log2(len(counts))
            if h_max > 0:
                return max(0.0, min(1.0, h / h_max))

    # Fallback: agent competency diversity
    if agents:
        weights_lists = []
        for agent in agents:
            if hasattr(agent, "competencies") and agent.competencies and agent.competencies.weights:
                weights_lists.extend(agent.competencies.weights.values())
        if len(weights_lists) >= 2:
            total = sum(weights_lists)
            if total > 0:
                probs = [w / total for w in weights_lists]
                h = -sum(p * math.log2(p) for p in probs if p > 0)
                h_max = math.log2(len(probs))
                if h_max > 0:
                    return max(0.0, min(1.0, h / h_max))

    return 0.5  # moderate default


def _compute_resonance(
    pulse: Any,
    action_correlations: list | None,
    diversity_index: float,
) -> float:
    """R: mesh-environment coupling quality."""
    components: list[float] = []

    # Workers match signals
    if pulse is not None:
        components.append(max(0.0, min(1.0, pulse.avg_familiarity)))

    # Discussion-to-action coupling
    if action_correlations:
        low_action = sum(1 for ac in action_correlations if getattr(ac, "ratio", 0.5) < 0.1)
        total = len(action_correlations)
        if total > 0:
            components.append(1.0 - low_action / total)

    # Requisite variety
    components.append(max(0.0, min(1.0, diversity_index)))

    return _safe_mean(components)


class _TemperatureScaler:
    """Adaptive normalizer for temperature magnitude.

    Tracks running max of raw magnitudes and uses it to normalize
    future values to [0,1]. New maxima expand the scale; the scale
    decays slowly so the system doesn't get stuck on ancient spikes.
    """

    def __init__(self, initial_scale: float = 0.1, decay: float = 0.995) -> None:
        self._scale = max(0.001, initial_scale)
        self._decay = decay

    def normalize(self, raw_magnitude: float) -> float:
        # Decay scale toward a minimum so system stays responsive
        self._scale = max(0.001, self._scale * self._decay)
        # Expand scale if we see a new max
        if raw_magnitude > self._scale:
            self._scale = raw_magnitude
        # Normalize and squash through sigmoid-like mapping
        # 2/(1+e^(-2x))-1 maps [0,inf)→[0,1) with x=1 mapping to ~0.76
        normalized = raw_magnitude / self._scale if self._scale > 0 else 0.0
        return 2.0 / (1.0 + math.exp(-2.0 * normalized)) - 1.0

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = max(0.001, value)


# Module-level default scaler; replaced by engine's persistent one
_default_temperature_scaler = _TemperatureScaler()


def _compute_temperature(pulse: Any, scaler: _TemperatureScaler | None = None) -> float:
    """T: rate of state change (magnitude of change-rate vector)."""
    if pulse is None:
        return 0.0

    rates = [
        getattr(pulse, "d_error_dt", 0.0),
        getattr(pulse, "d_familiarity_dt", 0.0),
        getattr(pulse, "d_worker_count_dt", 0.0),
        getattr(pulse, "d_spectral_dt", 0.0),
    ]
    magnitude = math.sqrt(sum(r * r for r in rates))
    if scaler is not None:
        return scaler.normalize(magnitude)
    return _default_temperature_scaler.normalize(magnitude)


def _compute_substrate(
    compression_tracker: Any | None,
    action_correlations: list | None,
    phase_lock_alerts: list | None,
) -> float:
    """X: grounding in concrete evidence."""
    components: list[float] = []

    # Direct language (not hedged): 1 - compression_index
    if compression_tracker is not None:
        ci = getattr(compression_tracker, "current_index", None)
        if ci is None:
            ci = getattr(compression_tracker, "compression_index", None)
        if ci is not None:
            components.append(max(0.0, min(1.0, 1.0 - ci)))

    # Exploration ratio
    if action_correlations:
        exploration_ratios = [
            getattr(ac, "exploration_ratio", 0.5)
            for ac in action_correlations
        ]
        if exploration_ratios:
            components.append(_safe_mean(exploration_ratios))

    # Not phase-locked: 1 - max(framing_similarity)
    if phase_lock_alerts:
        max_sim = max(
            getattr(alert, "framing_similarity", 0.0)
            for alert in phase_lock_alerts
        )
        components.append(max(0.0, min(1.0, 1.0 - max_sim)))

    return _safe_mean(components)


def compute_mesh_field_state(
    config: FieldConfig,
    pulse: Any,
    signal_index: int = 0,
    agents: list | None = None,
    stability_tracker: Any | None = None,
    spectral: Any | None = None,
    compression_tracker: Any | None = None,
    action_correlations: list | None = None,
    phase_lock_alerts: list | None = None,
    diversity_index: float = 0.5,
    smoother: _PaceSmoother | None = None,
    temperature_scaler: _TemperatureScaler | None = None,
) -> FieldState:
    """Compute the CERTX state vector from existing subsystem data.

    All inputs are optional — missing subsystems produce moderate defaults.
    Zero new LLM calls.
    """
    # Raw CERTX computation
    raw_c = _compute_coherence(pulse, spectral)
    raw_e = _compute_entropy(pulse, agents)
    raw_r = _compute_resonance(pulse, action_correlations, diversity_index)
    raw_t = _compute_temperature(pulse, scaler=temperature_scaler)
    raw_x = _compute_substrate(compression_tracker, action_correlations, phase_lock_alerts)

    # Pace-layer EMA smoothing
    if smoother is not None:
        c = smoother.smooth("coherence", raw_c)
        e = smoother.smooth("entropy", raw_e)
        r = smoother.smooth("resonance", raw_r)
        t = smoother.smooth("temperature", raw_t)
        x = smoother.smooth("substrate", raw_x)
    else:
        c, e, r, t, x = raw_c, raw_e, raw_r, raw_t, raw_x

    # Weighted health
    weights = [config.w_coherence, config.w_entropy, config.w_resonance,
               config.w_temperature, config.w_substrate]
    dims = [c, e, r, t, x]
    total_weight = sum(weights)
    health = sum(w * d for w, d in zip(weights, dims)) / total_weight if total_weight > 0 else 0.0

    # Lyapunov dV/dt
    dv_dt = getattr(pulse, "d_error_dt", 0.0) if pulse is not None else 0.0

    # Dysmemic pressure: everyone agrees but agreement isn't grounded
    dysmemic = max(0.0, c - x)

    state = FieldState(
        timestamp=datetime.now(timezone.utc).isoformat(),
        signal_index=signal_index,
        level="mesh",
        coherence=c,
        entropy=e,
        resonance=r,
        temperature=t,
        substrate=x,
        health=health,
        lyapunov_dv_dt=dv_dt,
        stability_eigenvalue=0.0,  # Set by eigenmonitor later
        dysmemic_pressure=dysmemic,
        raw={
            "raw_coherence": raw_c,
            "raw_entropy": raw_e,
            "raw_resonance": raw_r,
            "raw_temperature": raw_t,
            "raw_substrate": raw_x,
        },
    )
    return state
