"""Tests for temporal state management (--since flag, state.yaml persistence)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from stigmergy.cli.temporal_state import (
    RunState,
    SincePolicy,
    SinceSpec,
    load_state,
    parse_since_argument,
    resolve_effective_since,
    save_state,
)


# ── SincePolicy / SinceSpec ─────────────────────────────────────────


class TestSinceSpec:
    def test_stored_policy(self):
        spec = SinceSpec(policy=SincePolicy.STORED)
        assert spec.policy == SincePolicy.STORED
        assert spec.timestamp is None

    def test_all_policy(self):
        spec = SinceSpec(policy=SincePolicy.ALL)
        assert spec.policy == SincePolicy.ALL
        assert spec.timestamp is None

    def test_explicit_policy(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spec = SinceSpec(policy=SincePolicy.EXPLICIT, timestamp=ts)
        assert spec.policy == SincePolicy.EXPLICIT
        assert spec.timestamp == ts

    def test_explicit_requires_timestamp(self):
        with pytest.raises(ValueError, match="timestamp must be provided"):
            SinceSpec(policy=SincePolicy.EXPLICIT)

    def test_stored_rejects_timestamp(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="timestamp must be None"):
            SinceSpec(policy=SincePolicy.STORED, timestamp=ts)

    def test_all_rejects_timestamp(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="timestamp must be None"):
            SinceSpec(policy=SincePolicy.ALL, timestamp=ts)

    def test_frozen(self):
        spec = SinceSpec(policy=SincePolicy.STORED)
        with pytest.raises(Exception):
            spec.policy = SincePolicy.ALL


# ── RunState ─────────────────────────────────────────────────────────


class TestRunState:
    def test_valid(self):
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        state = RunState(last_run_timestamp=ts)
        assert state.version == 1
        assert state.last_run_timestamp == ts

    def test_version_must_be_1(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="version must be 1"):
            RunState(version=2, last_run_timestamp=ts)

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="UTC-aware"):
            RunState(last_run_timestamp=datetime(2026, 1, 1))

    def test_non_utc_offset_rejected(self):
        eastern = timezone(timedelta(hours=-5))
        with pytest.raises(ValueError, match="UTC offset"):
            RunState(
                last_run_timestamp=datetime(2026, 1, 1, tzinfo=eastern)
            )

    def test_frozen(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        state = RunState(last_run_timestamp=ts)
        with pytest.raises(Exception):
            state.version = 2


# ── parse_since_argument ─────────────────────────────────────────────


class TestParseSinceArgument:
    def test_none_returns_stored(self):
        spec = parse_since_argument(None)
        assert spec.policy == SincePolicy.STORED
        assert spec.timestamp is None

    def test_all_lowercase(self):
        spec = parse_since_argument("all")
        assert spec.policy == SincePolicy.ALL
        assert spec.timestamp is None

    def test_all_uppercase(self):
        spec = parse_since_argument("ALL")
        assert spec.policy == SincePolicy.ALL

    def test_all_mixed_case(self):
        spec = parse_since_argument("All")
        assert spec.policy == SincePolicy.ALL

    def test_iso8601_utc(self):
        spec = parse_since_argument("2026-01-15T12:00:00+00:00")
        assert spec.policy == SincePolicy.EXPLICIT
        assert spec.timestamp == datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

    def test_iso8601_naive_assumed_utc(self):
        spec = parse_since_argument("2026-01-15T12:00:00")
        assert spec.policy == SincePolicy.EXPLICIT
        assert spec.timestamp.tzinfo == timezone.utc
        assert spec.timestamp == datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

    def test_iso8601_with_offset_converted_to_utc(self):
        spec = parse_since_argument("2026-01-15T17:00:00+05:00")
        assert spec.policy == SincePolicy.EXPLICIT
        assert spec.timestamp == datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

    def test_date_only(self):
        spec = parse_since_argument("2026-01-15")
        assert spec.policy == SincePolicy.EXPLICIT
        assert spec.timestamp.year == 2026
        assert spec.timestamp.month == 1
        assert spec.timestamp.day == 15

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be an empty string"):
            parse_since_argument("")

    def test_invalid_iso_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_since_argument("not-a-date")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_since_argument("yesterday")


# ── resolve_effective_since ──────────────────────────────────────────


class TestResolveEffectiveSince:
    def test_all_returns_none(self):
        spec = SinceSpec(policy=SincePolicy.ALL)
        result = resolve_effective_since(spec, datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert result is None

    def test_stored_with_stored_ts(self):
        stored = datetime(2026, 1, 10, tzinfo=timezone.utc)
        spec = SinceSpec(policy=SincePolicy.STORED)
        result = resolve_effective_since(spec, stored)
        assert result == stored

    def test_stored_without_stored_ts(self):
        spec = SinceSpec(policy=SincePolicy.STORED)
        result = resolve_effective_since(spec, None)
        assert result is None

    def test_explicit(self):
        ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
        spec = SinceSpec(policy=SincePolicy.EXPLICIT, timestamp=ts)
        result = resolve_effective_since(spec, datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert result == ts

    def test_explicit_ignores_stored(self):
        explicit_ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
        stored_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spec = SinceSpec(policy=SincePolicy.EXPLICIT, timestamp=explicit_ts)
        result = resolve_effective_since(spec, stored_ts)
        assert result == explicit_ts


# ── load_state / save_state ──────────────────────────────────────────


class TestLoadState:
    async def test_missing_file_returns_none(self, tmp_path):
        result = await load_state(tmp_path)
        assert result is None

    async def test_valid_state(self, tmp_path):
        state_dir = tmp_path / ".stigmergy"
        state_dir.mkdir()
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        data = {
            "version": 1,
            "last_run_timestamp": ts.isoformat(),
        }
        (state_dir / "state.yaml").write_text(
            yaml.safe_dump(data), encoding="utf-8"
        )
        result = await load_state(tmp_path)
        assert result is not None
        assert result.version == 1
        assert result.last_run_timestamp == ts

    async def test_corrupt_yaml_returns_none(self, tmp_path):
        state_dir = tmp_path / ".stigmergy"
        state_dir.mkdir()
        (state_dir / "state.yaml").write_text(
            "{{{{invalid yaml", encoding="utf-8"
        )
        result = await load_state(tmp_path)
        assert result is None

    async def test_invalid_schema_returns_none(self, tmp_path):
        state_dir = tmp_path / ".stigmergy"
        state_dir.mkdir()
        data = {"version": 99, "last_run_timestamp": "2026-01-01T00:00:00+00:00"}
        (state_dir / "state.yaml").write_text(
            yaml.safe_dump(data), encoding="utf-8"
        )
        result = await load_state(tmp_path)
        assert result is None

    async def test_missing_timestamp_returns_none(self, tmp_path):
        state_dir = tmp_path / ".stigmergy"
        state_dir.mkdir()
        data = {"version": 1}
        (state_dir / "state.yaml").write_text(
            yaml.safe_dump(data), encoding="utf-8"
        )
        result = await load_state(tmp_path)
        assert result is None

    async def test_non_dict_returns_none(self, tmp_path):
        state_dir = tmp_path / ".stigmergy"
        state_dir.mkdir()
        (state_dir / "state.yaml").write_text(
            "just a string\n", encoding="utf-8"
        )
        result = await load_state(tmp_path)
        assert result is None


class TestSaveState:
    async def test_creates_directory_and_file(self, tmp_path):
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        await save_state(tmp_path, ts)

        state_path = tmp_path / ".stigmergy" / "state.yaml"
        assert state_path.exists()

        data = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["last_run_timestamp"] == ts.isoformat()

    async def test_overwrites_existing(self, tmp_path):
        ts1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2026, 6, 15, tzinfo=timezone.utc)

        await save_state(tmp_path, ts1)
        await save_state(tmp_path, ts2)

        state_path = tmp_path / ".stigmergy" / "state.yaml"
        data = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        assert data["last_run_timestamp"] == ts2.isoformat()

    async def test_naive_timestamp_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="UTC-aware"):
            await save_state(tmp_path, datetime(2026, 1, 1))

    async def test_non_utc_timestamp_rejected(self, tmp_path):
        eastern = timezone(timedelta(hours=-5))
        with pytest.raises(ValueError, match="UTC-aware"):
            await save_state(tmp_path, datetime(2026, 1, 1, tzinfo=eastern))


class TestRoundTrip:
    async def test_save_then_load(self, tmp_path):
        ts = datetime(2026, 3, 14, 15, 9, 26, tzinfo=timezone.utc)
        await save_state(tmp_path, ts)
        state = await load_state(tmp_path)
        assert state is not None
        assert state.last_run_timestamp == ts
        assert state.version == 1

    async def test_multiple_saves_load_latest(self, tmp_path):
        ts1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        await save_state(tmp_path, ts1)
        await save_state(tmp_path, ts2)

        state = await load_state(tmp_path)
        assert state is not None
        assert state.last_run_timestamp == ts2
