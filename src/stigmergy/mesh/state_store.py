"""Mesh state persistence — warm restart without losing self-organization.

The mesh self-organizes through signal processing: workers specialize,
agents develop competencies, compression trackers accumulate trends.
This state is the system's value — losing it on restart means cold-starting
back to the uniform initial state.

Analogy: taking dice out of a cube container. Getting them packed back
in requires intentional placement. The d4 goes next to the d20.

This store serializes the organized state to .stigmergy/mesh_state.json
on shutdown and restores it on startup. Three categories:

1. Worker routing stats: signals_received, signals_accepted,
   rolling_avg_familiarity — how workers learned to route.
2. Agent competency weights: evolved from config seeds through
   processing — what agents got good at.
3. Compression tracker: ND trend memory — EMA values, initial
   baselines, LD/SE history per channel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

MESH_STATE_PATH = Path(".stigmergy") / "mesh_state.json"


def save_mesh_state(
    *,
    workers: list,
    agents: list,
    compression_tracker: Any | None = None,
    run_stats: Any | None = None,
    path: Path = MESH_STATE_PATH,
) -> None:
    """Serialize mesh self-organized state to disk.

    Args:
        workers: list of WorkerNode objects
        agents: list of Agent objects
        compression_tracker: CompressionTracker or None
        path: output path
    """
    state: dict[str, Any] = {}

    # Worker routing state — the accumulated routing intelligence
    state["workers"] = {}
    for w in workers:
        state["workers"][str(w.id)] = {
            "signals_received": w.signals_received,
            "signals_accepted": w.signals_accepted,
            "signals_forwarded": w.signals_forwarded,
            "rolling_avg_familiarity": round(w.rolling_avg_familiarity, 6),
            "source_name": w.source_name,
        }

    # Agent competency weights — evolved through processing
    state["agents"] = {}
    for agent in agents:
        state["agents"][str(agent.id)] = {
            "competencies": {k: round(v, 6) for k, v in agent.competencies.weights.items()},
            "confidence": round(agent.confidence, 6),
        }

    # Compression tracker — ND trend memory
    if compression_tracker is not None:
        ct_state: dict[str, Any] = {}
        for key in compression_tracker._ema:
            ct_state[key] = {
                "ema": round(compression_tracker._ema[key], 6),
                "count": compression_tracker._counts[key],
                "peak": round(compression_tracker._peaks[key], 6),
                "prev_ema": round(compression_tracker._prev_ema[key], 6),
                "initial": round(compression_tracker._initial.get(key, 0.0), 6),
                "ld": round(compression_tracker._ld.get(key, 0.0), 6),
                "se": round(compression_tracker._se.get(key, 0.0), 6),
            }
        state["compression_tracker"] = ct_state

    # Run stats — for caucus phase volume projection
    if run_stats is not None:
        state["last_run_stats"] = {
            "signals": run_stats.signals,
            "duration_seconds": run_stats.duration_seconds,
            "workers": run_stats.workers,
            "timestamp_iso": run_stats.timestamp_iso,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.info("Saved mesh state: %d workers, %d agents, %d compression keys",
                len(state["workers"]), len(state["agents"]),
                len(state.get("compression_tracker", {})))


def load_mesh_state(path: Path = MESH_STATE_PATH) -> dict[str, Any]:
    """Load mesh state from disk. Returns empty dict if no state exists."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load mesh state: %s", exc)
        return {}


def restore_worker_state(worker, saved: dict[str, Any]) -> None:
    """Restore routing stats to a worker from saved state.

    Only restores stats that help routing — not signal_count (fullness)
    which is intentionally reset per session (see bootstrap_mesh design
    decision comment).
    """
    worker_state = saved.get("workers", {}).get(str(worker.id))
    if worker_state is None:
        return

    # Routing intelligence: what this worker learned about its role
    worker.signals_received = worker_state.get("signals_received", 0)
    worker.signals_accepted = worker_state.get("signals_accepted", 0)
    worker.signals_forwarded = worker_state.get("signals_forwarded", 0)
    worker.rolling_avg_familiarity = worker_state.get("rolling_avg_familiarity", 0.0)


def restore_agent_state(agent, saved: dict[str, Any]) -> None:
    """Restore competency weights to an agent from saved state."""
    agent_state = saved.get("agents", {}).get(str(agent.id))
    if agent_state is None:
        return

    saved_weights = agent_state.get("competencies", {})
    if saved_weights:
        # Merge with current weights — saved values override seeds,
        # but new competencies from config are preserved
        agent.competencies.weights.update(saved_weights)

    saved_confidence = agent_state.get("confidence")
    if saved_confidence is not None:
        agent.confidence = saved_confidence


def restore_compression_tracker(tracker, saved: dict[str, Any]) -> None:
    """Restore compression tracker state from saved data.

    This is the ND trend memory — EMA values, initial baselines,
    sample counts. Without this, the "Born Caged" detection and
    trend analysis restart from zero every session.
    """
    ct_state = saved.get("compression_tracker", {})
    if not ct_state:
        return

    for key, data in ct_state.items():
        tracker._ema[key] = data.get("ema", 0.0)
        tracker._counts[key] = data.get("count", 0)
        tracker._peaks[key] = data.get("peak", 0.0)
        tracker._prev_ema[key] = data.get("prev_ema", 0.0)
        tracker._initial[key] = data.get("initial", 0.0)
        tracker._ld[key] = data.get("ld", 0.0)
        tracker._se[key] = data.get("se", 0.0)

    logger.info("Restored compression tracker: %d channels", len(ct_state))


def restore_run_stats(saved: dict[str, Any]):
    """Restore RunStats from saved state. Returns None if not present."""
    from stigmergy.mesh.caucus import RunStats

    data = saved.get("last_run_stats")
    if data is None:
        return None
    return RunStats(
        signals=data.get("signals", 0),
        duration_seconds=data.get("duration_seconds", 0.0),
        workers=data.get("workers", 0),
        timestamp_iso=data.get("timestamp_iso", ""),
    )
