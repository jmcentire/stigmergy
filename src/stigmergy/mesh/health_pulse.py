"""Health pulse: periodic mesh vitals snapshot.

Pure mechanical computation — no LLM calls. Assembles a snapshot of
mesh state every checkpoint interval: worker counts, error rates,
familiarity trends, spectral state, corroboration outcomes.

The pulse feeds the meta-modeler (when enabled) and provides the raw
data for self-assessment. Without the meta-modeler, pulses still
accumulate in history for diagnostic access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class ChangeRate:
    """Tracks a metric's value and its EMA-smoothed rate of change.

    The rate captures velocity — is V(x) decreasing quickly or slowly?
    Is worker count stable or oscillating? EMA smoothing prevents
    single-sample spikes from dominating.
    """

    current: float = 0.0
    previous: float = 0.0
    rate: float = 0.0
    ema_alpha: float = 0.2
    samples: int = 0

    def update(self, value: float, interval: float = 1.0) -> None:
        """Update with new value. interval = time/signals between samples."""
        self.previous = self.current
        self.current = value
        if self.samples > 0 and interval > 0:
            raw_rate = (value - self.previous) / interval
            if self.samples == 1:
                self.rate = raw_rate
            else:
                self.rate = self.ema_alpha * raw_rate + (1 - self.ema_alpha) * self.rate
        self.samples += 1


@dataclass
class WorkerSummary:
    """Per-worker vitals for the health pulse."""

    id: str
    label: str
    familiarity: float
    energy: float
    threshold: float
    signal_count: int


@dataclass
class AgentSummary:
    """Per-agent vitals for the health pulse."""

    id: str
    confidence: float
    llm_calls: int
    mechanical_calls: int
    top_competency: str


@dataclass
class MeshHealthPulse:
    """Periodic snapshot of mesh state — the system's vital signs.

    Collected at checkpoint intervals (every 50 signals). Each field
    is a mechanical measurement, not an interpretation. The meta-modeler
    adds interpretation.
    """

    signal_index: int = 0
    timestamp: str = ""

    # Aggregate
    worker_count: int = 0
    total_error: float = 0.0  # V(x) — aggregate familiarity error
    avg_familiarity: float = 0.0
    routing_changes: int = 0  # workers spawned/pruned since last pulse

    # Per-worker summaries
    workers: list[WorkerSummary] = field(default_factory=list)

    # Per-agent summaries
    agents: list[AgentSummary] = field(default_factory=list)

    # Spectral state
    spectral_high_freq: float = 0.0
    spectral_trend: str = ""  # "stable", "increasing", "decreasing"
    effective_dimensionality: int = 0

    # Corroboration
    last_quorum_met: int = 0
    last_quorum_total: int = 0

    # Change rates (EMA-smoothed)
    d_error_dt: float = 0.0
    d_familiarity_dt: float = 0.0
    d_worker_count_dt: float = 0.0
    d_routing_changes_dt: float = 0.0
    d_confidence_dt: float = 0.0
    d_spectral_dt: float = 0.0

    # Finding lifecycle
    active_findings: int = 0
    deferred_findings: int = 0
    normalized_findings: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for prompt injection and storage."""
        return {
            "signal_index": self.signal_index,
            "timestamp": self.timestamp,
            "worker_count": self.worker_count,
            "total_error": round(self.total_error, 4),
            "avg_familiarity": round(self.avg_familiarity, 4),
            "routing_changes": self.routing_changes,
            "spectral_high_freq": round(self.spectral_high_freq, 4),
            "spectral_trend": self.spectral_trend,
            "effective_dimensionality": self.effective_dimensionality,
            "d_error_dt": round(self.d_error_dt, 6),
            "d_familiarity_dt": round(self.d_familiarity_dt, 6),
            "d_worker_count_dt": round(self.d_worker_count_dt, 6),
            "d_routing_changes_dt": round(self.d_routing_changes_dt, 6),
            "d_confidence_dt": round(self.d_confidence_dt, 6),
            "d_spectral_dt": round(self.d_spectral_dt, 6),
            "last_quorum_met": self.last_quorum_met,
            "last_quorum_total": self.last_quorum_total,
            "active_findings": self.active_findings,
            "deferred_findings": self.deferred_findings,
            "normalized_findings": self.normalized_findings,
            "workers": [
                {
                    "id": w.id, "label": w.label,
                    "familiarity": round(w.familiarity, 4),
                    "energy": round(w.energy, 4),
                    "threshold": round(w.threshold, 4),
                    "signal_count": w.signal_count,
                }
                for w in self.workers
            ],
            "agents": [
                {
                    "id": a.id, "confidence": round(a.confidence, 4),
                    "llm_calls": a.llm_calls,
                    "mechanical_calls": a.mechanical_calls,
                    "top_competency": a.top_competency,
                }
                for a in self.agents
            ],
        }


class HealthPulseCollector:
    """Assembles health pulses from mesh subsystems.

    Pure mechanical computation — reads state from workers, agents,
    stability tracker, spectral state, and finding registries. Maintains
    EMA-smoothed change rates for trend detection.

    Fires at checkpoint intervals (every 50 signals), same cadence as
    the CadenceMonitor.
    """

    def __init__(self) -> None:
        self._rates: dict[str, ChangeRate] = {
            "error": ChangeRate(),
            "familiarity": ChangeRate(),
            "worker_count": ChangeRate(),
            "routing_changes": ChangeRate(),
            "confidence": ChangeRate(),
            "spectral": ChangeRate(),
        }
        self._history: list[MeshHealthPulse] = []
        self._max_history = 10
        self._prev_worker_count: int = 0

    def collect(
        self,
        signal_index: int,
        mesh: Any,
        stability_tracker: Any | None = None,
        spectral: Any | None = None,
        agents: list | None = None,
        finding_registry: Any | None = None,
        compression_tracker: Any | None = None,
    ) -> MeshHealthPulse:
        """Assemble a health pulse from current mesh state.

        All parameters are optional to support partial mesh configurations.
        Missing subsystems produce zero values — graceful degradation.
        """
        pulse = MeshHealthPulse(
            signal_index=signal_index,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Worker state
        workers = list(mesh.workers) if mesh is not None else []
        pulse.worker_count = len(workers)

        # Routing changes since last pulse
        routing_changes = abs(pulse.worker_count - self._prev_worker_count)
        pulse.routing_changes = routing_changes
        self._prev_worker_count = pulse.worker_count

        # Per-worker summaries
        total_familiarity = 0.0
        for w in workers:
            fam = w.rolling_avg_familiarity
            total_familiarity += fam
            pulse.workers.append(WorkerSummary(
                id=str(w.id)[:8],
                label=w.label[:30] if hasattr(w, "label") else "",
                familiarity=fam,
                energy=getattr(w.context, "energy", 0.0),
                threshold=w.adaptive_threshold,
                signal_count=w.context.signal_count,
            ))

        pulse.avg_familiarity = total_familiarity / max(len(workers), 1)

        # Total error — V(x): sum of (1 - familiarity) across workers
        # Lower is better — mesh is fitting signals well
        pulse.total_error = sum(1.0 - w.familiarity for w in pulse.workers)

        # Agent summaries
        if agents:
            total_confidence = 0.0
            for agent in agents:
                diags = agent.diagnostics()
                top_comp = ""
                if agent.competencies.weights:
                    top_comp = max(agent.competencies.weights, key=agent.competencies.weights.get)
                pulse.agents.append(AgentSummary(
                    id=str(agent.id)[:8],
                    confidence=agent.confidence,
                    llm_calls=diags["reflect"]["llm_calls"],
                    mechanical_calls=diags["reflect"]["mechanical_calls"],
                    top_competency=top_comp,
                ))
                total_confidence += agent.confidence
            avg_confidence = total_confidence / max(len(agents), 1)
        else:
            avg_confidence = 0.0

        # Spectral state from stability tracker
        if stability_tracker is not None:
            try:
                v_history = stability_tracker.v_history
                if v_history:
                    pulse.spectral_high_freq = abs(v_history[-1] - v_history[-2]) if len(v_history) >= 2 else 0.0
                    if len(v_history) >= 3:
                        recent = v_history[-3:]
                        if recent[-1] > recent[0]:
                            pulse.spectral_trend = "increasing"
                        elif recent[-1] < recent[0]:
                            pulse.spectral_trend = "decreasing"
                        else:
                            pulse.spectral_trend = "stable"
                    else:
                        pulse.spectral_trend = "stable"
            except (AttributeError, IndexError):
                pulse.spectral_trend = "unknown"

        pulse.effective_dimensionality = len(workers)

        # Finding registry state
        if finding_registry is not None:
            try:
                summary = finding_registry.summary()
                pulse.active_findings = summary.get("active", 0)
                pulse.deferred_findings = summary.get("deferred", 0)
                pulse.normalized_findings = summary.get("normalized", 0)
            except (AttributeError, TypeError):
                pass

        # Update change rates
        interval = 50.0  # checkpoint interval in signals
        self._rates["error"].update(pulse.total_error, interval)
        self._rates["familiarity"].update(pulse.avg_familiarity, interval)
        self._rates["worker_count"].update(float(pulse.worker_count), interval)
        self._rates["routing_changes"].update(float(routing_changes), interval)
        self._rates["confidence"].update(avg_confidence, interval)
        self._rates["spectral"].update(pulse.spectral_high_freq, interval)

        # Copy rates into pulse
        pulse.d_error_dt = self._rates["error"].rate
        pulse.d_familiarity_dt = self._rates["familiarity"].rate
        pulse.d_worker_count_dt = self._rates["worker_count"].rate
        pulse.d_routing_changes_dt = self._rates["routing_changes"].rate
        pulse.d_confidence_dt = self._rates["confidence"].rate
        pulse.d_spectral_dt = self._rates["spectral"].rate

        # Keep history
        self._history.append(pulse)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.debug(
            "Health pulse %d: %d workers, error=%.4f, d_error=%.6f, fam=%.4f",
            signal_index, pulse.worker_count, pulse.total_error,
            pulse.d_error_dt, pulse.avg_familiarity,
        )

        return pulse

    @property
    def history(self) -> list[MeshHealthPulse]:
        """Last N health pulses for trend analysis."""
        return list(self._history)

    @property
    def latest(self) -> MeshHealthPulse | None:
        """Most recent pulse, or None if no pulses collected."""
        return self._history[-1] if self._history else None
