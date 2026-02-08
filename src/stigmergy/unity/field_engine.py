"""FieldEngine — orchestrator, composes all components into a single tick().

Fires at checkpoint intervals (every 50 signals), same cadence as
health pulse and cadence monitor. Zero new LLM calls.

Persists state across sessions to .stigmergy/unity_state.json so
PID integral, pace smoothing, calibration, and eigenvalue history
survive restarts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from stigmergy.unity.breathing import BreathingController
from stigmergy.unity.calibration import CalibrationState, update_calibration
from stigmergy.unity.diversity import compute_diversity
from stigmergy.unity.eigenmonitor import EigenMonitor
from stigmergy.unity.field_bridge import FieldBridge
from stigmergy.unity.field_config import FieldConfig
from stigmergy.unity.field_state import (
    FieldState,
    _PaceSmoother,
    _TemperatureScaler,
    compute_mesh_field_state,
)
from stigmergy.unity.pid_controller import PIDController

logger = logging.getLogger(__name__)

UNITY_STATE_PATH = Path(".stigmergy") / "unity_state.json"


class FieldEngine:
    """Unity system orchestrator — composes all components.

    tick() runs the full field equation pipeline:
    1. Compute diversity
    2. Advance breathing
    3. Compute CERTX mesh state
    4. PID update on health
    5. Eigenvalue monitoring
    6. Field bridge: state -> adjustments -> mesh physics
    7. Record history, log to field_state.jsonl
    """

    def __init__(self, config: FieldConfig) -> None:
        self._config = config
        self._pid = PIDController(config)
        self._breathing = BreathingController(config)
        self._eigen = EigenMonitor(config)
        self._bridge = FieldBridge(config)
        self._calibration = CalibrationState(
            window=config.calibration_window,
            babbling_threshold=config.babbling_threshold,
            decay=config.calibration_decay,
        )
        self._smoother = _PaceSmoother(config)
        self._temperature_scaler = _TemperatureScaler()
        self._history: list[FieldState] = []
        self._max_history = 50
        self._log_path = Path(".stigmergy") / "field_state.jsonl"
        self._tick_count: int = 0

    def tick(
        self,
        pulse: Any,
        mesh: Any = None,
        agents: list | None = None,
        stability_tracker: Any | None = None,
        spectral: Any | None = None,
        compression_tracker: Any | None = None,
        action_correlations: list | None = None,
        phase_lock_alerts: list | None = None,
    ) -> FieldState:
        """Run one full field equation cycle.

        All inputs are optional — graceful degradation with moderate defaults.
        """
        self._tick_count += 1
        signal_index = getattr(pulse, "signal_index", 0) if pulse is not None else 0

        # 1. Compute diversity
        workers = list(mesh.workers) if mesh is not None and hasattr(mesh, "workers") else []
        diversity = compute_diversity(workers, agents, self._config)
        diversity_index = diversity.worker_diversity

        # 2. Advance breathing
        breathing = self._breathing.tick(signal_count=signal_index)

        # 3. Compute CERTX mesh state
        field_state = compute_mesh_field_state(
            self._config,
            pulse,
            signal_index=signal_index,
            agents=agents,
            stability_tracker=stability_tracker,
            spectral=spectral,
            compression_tracker=compression_tracker,
            action_correlations=action_correlations,
            phase_lock_alerts=phase_lock_alerts,
            diversity_index=diversity_index,
            smoother=self._smoother,
            temperature_scaler=self._temperature_scaler,
        )

        # 4. PID update on health
        pid_state = self._pid.update(field_state.health, dt=1.0)

        # 5. Eigenvalue monitoring
        eigen_state = self._eigen.update(field_state)
        if eigen_state is not None:
            # Use smoothed max for stability — less noisy than raw eigenvalue
            field_state.stability_eigenvalue = eigen_state.smoothed_max

        # 6. Field bridge: state -> adjustments -> mesh physics
        adjustments = self._bridge.compute_adjustments(
            field_state,
            pid_output=pid_state.output,
            breathing_modulation=breathing.modulation,
            calibration=self._calibration,
        )
        if adjustments and mesh is not None:
            self._bridge.apply_adjustments(adjustments, mesh)

        # 7. Record history
        self._history.append(field_state)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Log to file
        if self._config.log_field_state:
            self._log_field_state(field_state, pid_state, breathing, diversity, eigen_state)

        logger.info(
            "Field tick %d: %s | PID=%.3f | breath=%s(%.3f) | eigen=%s",
            signal_index, field_state.certx_summary(),
            pid_state.output, breathing.phase, breathing.modulation,
            eigen_state.status if eigen_state else "n/a",
        )

        return field_state

    def record_prediction(
        self,
        agent_id: str,
        confidence: float,
        was_corroborated: bool,
    ) -> None:
        """Feed calibration tracker after corroboration."""
        update_calibration(
            agent_id, confidence, was_corroborated,
            self._calibration, self._config,
        )

    # ── Persistence ────────────────────────────────────────────

    def save_state(self, path: Path = UNITY_STATE_PATH) -> None:
        """Serialize engine state for warm restart.

        Saves: PID integral/error, pace smoother EMA values,
        temperature scaler, calibration predictions, eigenmonitor
        history, tick count.
        """
        state: dict[str, Any] = {}

        # PID controller state
        state["pid"] = {
            "integral": self._pid._integral,
            "prev_error": self._pid._prev_error,
            "initialized": self._pid._initialized,
        }

        # Pace smoother EMA values
        state["smoother"] = {
            k: v for k, v in self._smoother._values.items() if v is not None
        }

        # Temperature scaler
        state["temperature_scaler"] = {
            "scale": self._temperature_scaler.scale,
        }

        # Calibration state — all agent predictions
        cal_state: dict[str, Any] = {}
        for agent_id, agent_cal in self._calibration.agents.items():
            cal_state[agent_id] = {
                "predictions": [
                    [round(f, 4), o] for f, o in agent_cal.predictions
                ],
                "brier_score": round(agent_cal.brier_score, 6),
                "n_star": round(agent_cal.n_star, 4),
                "is_babbling": agent_cal.is_babbling,
            }
        state["calibration"] = cal_state

        # Eigenmonitor CERTX history
        state["eigenmonitor"] = {
            "history": [
                [round(v, 6) for v in vec] for vec in self._eigen._history
            ],
            "smoothed_max": round(self._eigen._smoothed_max, 6),
            "prev_status": self._eigen._prev_status,
            "initialized": self._eigen._initialized,
        }

        # Tick count
        state["tick_count"] = self._tick_count

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(
            "Saved unity state: %d ticks, %d calibrated agents, %d eigen history",
            self._tick_count, len(cal_state), len(self._eigen._history),
        )

    def load_state(self, path: Path = UNITY_STATE_PATH) -> bool:
        """Restore engine state from previous session.

        Returns True if state was loaded, False if no state exists.
        """
        if not path.exists():
            return False

        try:
            with open(path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load unity state: %s", exc)
            return False

        # PID controller
        pid_state = state.get("pid", {})
        if pid_state:
            self._pid._integral = pid_state.get("integral", 0.0)
            self._pid._prev_error = pid_state.get("prev_error", 0.0)
            self._pid._initialized = pid_state.get("initialized", False)

        # Pace smoother
        smoother_state = state.get("smoother", {})
        for dim, val in smoother_state.items():
            if dim in self._smoother._values:
                self._smoother._values[dim] = val

        # Temperature scaler
        temp_state = state.get("temperature_scaler", {})
        if "scale" in temp_state:
            self._temperature_scaler.scale = temp_state["scale"]

        # Calibration
        from stigmergy.unity.calibration import AgentCalibration
        cal_state = state.get("calibration", {})
        for agent_id, data in cal_state.items():
            agent_cal = AgentCalibration(
                predictions=[(f, o) for f, o in data.get("predictions", [])],
                brier_score=data.get("brier_score", 0.0),
                n_star=data.get("n_star", 1.0),
                is_babbling=data.get("is_babbling", False),
            )
            self._calibration.agents[agent_id] = agent_cal

        # Eigenmonitor
        eigen_state = state.get("eigenmonitor", {})
        if eigen_state:
            self._eigen._history = eigen_state.get("history", [])
            self._eigen._smoothed_max = eigen_state.get("smoothed_max", 0.0)
            self._eigen._prev_status = eigen_state.get("prev_status", "insufficient_data")
            self._eigen._initialized = eigen_state.get("initialized", False)

        # Tick count
        self._tick_count = state.get("tick_count", 0)

        logger.info(
            "Restored unity state: %d ticks, %d calibrated agents",
            self._tick_count, len(cal_state),
        )
        return True

    # ── Properties ─────────────────────────────────────────────

    @property
    def history(self) -> list[FieldState]:
        return list(self._history)

    @property
    def latest(self) -> FieldState | None:
        return self._history[-1] if self._history else None

    @property
    def calibration(self) -> CalibrationState:
        return self._calibration

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def _log_field_state(
        self,
        field_state: FieldState,
        pid_state: Any,
        breathing: Any,
        diversity: Any,
        eigen_state: Any,
    ) -> None:
        """Append field state to JSONL log."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            record = field_state.to_dict()
            record["pid_output"] = round(pid_state.output, 4)
            record["breathing_phase"] = breathing.phase
            record["breathing_modulation"] = round(breathing.modulation, 4)
            record["diversity_worker"] = round(diversity.worker_diversity, 4)
            record["diversity_agent"] = round(diversity.agent_diversity, 4)
            record["requisite_variety_met"] = diversity.requisite_variety_met
            record["hot_reserve"] = round(diversity.hot_reserve, 4)
            if eigen_state is not None:
                record["eigen_status"] = eigen_state.status
                record["eigen_max"] = round(eigen_state.max_eigenvalue, 4)
                record["eigen_smoothed"] = round(eigen_state.smoothed_max, 4)
            # Calibration summary
            if self._calibration.agents:
                babbling = sum(1 for a in self._calibration.agents.values() if a.is_babbling)
                record["calibration_agents"] = len(self._calibration.agents)
                record["calibration_babbling"] = babbling
            record["tick_count"] = self._tick_count
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("Failed to log field state: %s", exc)
