"""DiversityState + compute_diversity() — requisite variety, HOT reserve.

Ashby's Law: V_C >= V_E (controller variety must match environmental variety).
Carlson-Doyle HOT: distance from fragility floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from stigmergy.unity.field_config import FieldConfig


@dataclass
class DiversityState:
    """Diversity measurement results."""

    worker_diversity: float = 0.0      # Simpson's on worker signal distribution
    agent_diversity: float = 0.0       # Simpson's on agent competency weights
    source_diversity: float = 0.0      # Normalized Shannon on source distribution
    requisite_variety_met: bool = True  # V_C >= V_E
    controller_variety: float = 0.0    # V_C = worker_div * agent_div
    environmental_variety: float = 0.0  # V_E = source_div
    hot_reserve: float = 0.0          # Distance from fragility floor
    over_specialized: list[str] = field(default_factory=list)  # Worker IDs


def _simpsons_index(proportions: list[float]) -> float:
    """Simpson's Diversity Index: 1 - sum(p_i^2).

    0 = monoculture (one item dominates), approaches 1 for uniform.
    """
    if not proportions:
        return 0.0
    total = sum(proportions)
    if total <= 0:
        return 0.0
    probs = [p / total for p in proportions]
    return 1.0 - sum(p * p for p in probs)


def _shannon_normalized(counts: list[float]) -> float:
    """Normalized Shannon entropy: H / log2(N). Returns [0, 1]."""
    if len(counts) < 2:
        return 0.0
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    if len(probs) < 2:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs)
    h_max = math.log2(len(counts))
    return h / h_max if h_max > 0 else 0.0


def compute_diversity(
    workers: list | None,
    agents: list | None,
    config: FieldConfig,
) -> DiversityState:
    """Compute diversity metrics for requisite variety and HOT reserve.

    Args:
        workers: WorkerNode list (must have .context.signal_count, .id).
        agents: Agent list (must have .competencies.weights).
        config: FieldConfig for thresholds.
    """
    state = DiversityState()

    # Worker signal distribution (Simpson's)
    if workers:
        signal_counts = []
        total_signals = 0
        for w in workers:
            sc = getattr(getattr(w, "context", None), "signal_count", 0) if hasattr(w, "context") else 0
            signal_counts.append(float(sc))
            total_signals += sc

        if total_signals > 0:
            state.worker_diversity = _simpsons_index(signal_counts)

            # Over-specialization detection
            for i, w in enumerate(workers):
                share = signal_counts[i] / total_signals
                if share > config.max_specialization:
                    wid = str(getattr(w, "id", f"worker_{i}"))[:8]
                    state.over_specialized.append(wid)

    # Agent competency diversity (Simpson's on all weights)
    if agents:
        all_weights: list[float] = []
        for agent in agents:
            comp = getattr(agent, "competencies", None)
            if comp is not None and hasattr(comp, "weights") and comp.weights:
                all_weights.extend(comp.weights.values())
        if all_weights:
            state.agent_diversity = _simpsons_index(all_weights)

    # Source diversity: Shannon on worker signal counts (proxy for source variety)
    if workers:
        source_counts = [
            float(getattr(getattr(w, "context", None), "signal_count", 0))
            for w in workers
            if hasattr(w, "context")
        ]
        source_counts = [c for c in source_counts if c > 0]
        state.source_diversity = _shannon_normalized(source_counts)

    # Requisite Variety: V_C >= V_E
    state.controller_variety = state.worker_diversity * state.agent_diversity
    state.environmental_variety = state.source_diversity
    state.requisite_variety_met = state.controller_variety >= state.environmental_variety

    # HOT Reserve: distance from fragility floor
    state.hot_reserve = max(0.0, state.worker_diversity - config.min_diversity_index)

    return state
