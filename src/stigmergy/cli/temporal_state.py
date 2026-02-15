"""Temporal state management: --since flag parsing and state.yaml persistence.

Three-layer architecture:
  1. Pure frozen types (SincePolicy, SinceSpec, RunState)
  2. Pure functions (parse_since_argument, resolve_effective_since)
  3. Async I/O shell (load_state, save_state)

All datetime values are UTC-aware (tzinfo=datetime.UTC).
State file writes are atomic (tempfile + os.replace).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# ── Pure Types ───────────────────────────────────────────────────────


class SincePolicy(Enum):
    """The three semantic states of the --since CLI flag.

    ALL: full reprocessing (--since all).
    STORED: use stored last-run timestamp (--since omitted).
    EXPLICIT: user-provided ISO 8601 timestamp.
    """

    ALL = "ALL"
    STORED = "STORED"
    EXPLICIT = "EXPLICIT"


class SinceSpec(BaseModel):
    """Parsed --since CLI argument.

    Combines the policy enum with an optional explicit timestamp.
    timestamp is non-None only when policy is EXPLICIT.
    """

    model_config = ConfigDict(frozen=True)

    policy: SincePolicy
    timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def _check_timestamp_policy_consistency(self) -> "SinceSpec":
        if self.policy == SincePolicy.EXPLICIT and self.timestamp is None:
            raise ValueError(
                "timestamp must be provided when policy is EXPLICIT"
            )
        if self.policy != SincePolicy.EXPLICIT and self.timestamp is not None:
            raise ValueError(
                "timestamp must be None when policy is not EXPLICIT"
            )
        return self


class RunState(BaseModel):
    """Persisted state in .stigmergy/state.yaml.

    Contains a schema version (always 1) and the last successful run timestamp.
    """

    model_config = ConfigDict(frozen=True)

    version: int = Field(default=1)
    last_run_timestamp: datetime

    @model_validator(mode="after")
    def _validate(self) -> "RunState":
        if self.version != 1:
            raise ValueError(
                f"version must be 1, got {self.version}"
            )
        if self.last_run_timestamp.tzinfo is None:
            raise ValueError(
                "last_run_timestamp must be UTC-aware"
            )
        if self.last_run_timestamp.utcoffset() != timedelta(0):
            raise ValueError(
                "last_run_timestamp must have UTC offset (0)"
            )
        return self


# ── Pure Functions ───────────────────────────────────────────────────


def parse_since_argument(raw: Optional[str]) -> SinceSpec:
    """Convert raw CLI --since argument string into a typed SinceSpec.

    None input (flag omitted) -> STORED policy.
    'all' (case-insensitive) -> ALL policy.
    Any other string -> parse as ISO 8601, EXPLICIT policy with UTC datetime.

    Pure function, no I/O.
    """
    if raw is None:
        return SinceSpec(policy=SincePolicy.STORED)

    if not raw:
        raise ValueError(
            "--since value must not be an empty string. "
            "Use 'all' for full reprocessing or provide an ISO 8601 datetime."
        )

    if raw.lower() == "all":
        return SinceSpec(policy=SincePolicy.ALL)

    # Parse as ISO 8601
    try:
        dt = datetime.fromisoformat(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot parse '{raw}' as ISO 8601 datetime. "
            "Expected format: YYYY-MM-DDTHH:MM:SS[+HH:MM] or 'all'."
        ) from exc

    # Normalize to UTC
    if dt.tzinfo is None:
        # Naive datetime — assume UTC
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC
        dt = dt.astimezone(timezone.utc)

    return SinceSpec(policy=SincePolicy.EXPLICIT, timestamp=dt)


def resolve_effective_since(
    spec: SinceSpec,
    stored: Optional[datetime],
) -> Optional[datetime]:
    """Combine SinceSpec with stored last-run timestamp to get effective cutoff.

    Returns None to mean 'process all signals' (no cutoff), or a UTC-aware
    datetime cutoff.

    Pure function, no I/O.
    """
    if spec.policy == SincePolicy.ALL:
        return None

    if spec.policy == SincePolicy.STORED:
        return stored  # None if no stored state (first run -> process all)

    # EXPLICIT
    return spec.timestamp


# ── Async I/O Shell ──────────────────────────────────────────────────

_STATE_FILENAME = "state.yaml"
_STATE_DIR_NAME = ".stigmergy"


def _state_file_path(state_dir: Path) -> Path:
    return state_dir / _STATE_DIR_NAME / _STATE_FILENAME


def _load_state_sync(state_dir: Path) -> Optional[RunState]:
    """Synchronous implementation of load_state."""
    path = _state_file_path(state_dir)

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        logger.warning("Cannot read state file at %s: %s", path, exc)
        return None
    except OSError as exc:
        logger.warning(
            "Unexpected error reading state file at %s: %s", path, exc
        )
        return None

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("Corrupt state file at %s, ignoring: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning(
            "Invalid state file at %s, ignoring: expected dict, got %s",
            path,
            type(data).__name__,
        )
        return None

    # Parse timestamp string back to datetime
    ts_raw = data.get("last_run_timestamp")
    if ts_raw is not None and isinstance(ts_raw, str):
        try:
            dt = datetime.fromisoformat(ts_raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            data["last_run_timestamp"] = dt
        except (ValueError, TypeError):
            pass  # Let Pydantic validation catch it
    elif ts_raw is not None and isinstance(ts_raw, datetime):
        # yaml.safe_load may parse timestamps as datetime objects
        if ts_raw.tzinfo is None:
            data["last_run_timestamp"] = ts_raw.replace(tzinfo=timezone.utc)
        else:
            data["last_run_timestamp"] = ts_raw.astimezone(timezone.utc)

    try:
        return RunState(**data)
    except Exception as exc:
        logger.warning(
            "Invalid state file at %s, ignoring: %s", path, exc
        )
        return None


def _save_state_sync(state_dir: Path, timestamp: datetime) -> None:
    """Synchronous implementation of save_state."""
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError(
            f"timestamp must be UTC-aware (tzinfo=datetime.UTC), "
            f"got: {timestamp.tzinfo}"
        )

    stigmergy_dir = state_dir / _STATE_DIR_NAME
    try:
        stigmergy_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Cannot create state directory at {stigmergy_dir}: {exc}"
        ) from exc

    state_path = stigmergy_dir / _STATE_FILENAME

    data = {
        "version": 1,
        "last_run_timestamp": timestamp.isoformat(),
    }

    # Atomic write: tempfile in same directory, then os.replace
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(stigmergy_dir), suffix=".tmp", prefix="state_"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None  # os.fdopen takes ownership
            yaml.safe_dump(data, f, default_flow_style=False)

        os.replace(tmp_path, state_path)
        tmp_path = None  # replaced successfully
    except OSError as exc:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise OSError(
            f"Failed to write state file at {state_path}: {exc}"
        ) from exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


async def load_state(state_dir: Path) -> Optional[RunState]:
    """Read and parse .stigmergy/state.yaml.

    Returns RunState on success, None if missing/corrupt.
    Graceful degradation: logs warnings, never raises for I/O issues.
    """
    return await asyncio.to_thread(_load_state_sync, state_dir)


async def save_state(state_dir: Path, timestamp: datetime) -> None:
    """Atomically write RunState to .stigmergy/state.yaml.

    Creates .stigmergy/ directory if needed. Atomic via tempfile + os.replace.
    """
    await asyncio.to_thread(_save_state_sync, state_dir, timestamp)
