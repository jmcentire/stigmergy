"""Resume / backfill-window regression tests.

Guards the fix for the incident where a stale checkpoint (months old) made
every automated run re-fetch the whole backlog, time out before draining it,
and never converge — so findings froze for ~4 months while the run still
"succeeded". The core rule lives in `_resolve_backfill_since`.
"""

from datetime import datetime, timedelta, timezone

import pytest

from stigmergy.cli import run_cmd
from stigmergy.cli.run_cmd import _resolve_backfill_since

NOW = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)


def _iso(days_ago):
    return (NOW - timedelta(days=days_ago)).isoformat()


def _checkpoint(days_ago, processed=1486, total=29991, spend=9.5):
    return {
        "timestamp": _iso(days_ago),
        "signals_processed": processed,
        "total_signals": total,
        "spend_usd": spend,
    }


@pytest.fixture(autouse=True)
def _fixed_window(monkeypatch):
    # Pin the clamp ceiling so assertions don't depend on the env override.
    monkeypatch.setattr(run_cmd, "MAX_BACKFILL_DAYS", 7)


def test_stale_checkpoint_is_clamped_to_floor():
    # The actual incident: checkpoint stuck ~175 days back. Automated path
    # (since_days=None) must clamp to now-7d so the run can converge.
    state = {"checkpoint": _checkpoint(175)}
    since, notes = _resolve_backfill_since(state, since_days=None, now=NOW)
    assert since == NOW - timedelta(days=7)
    assert any("Clamping backfill window" in n for n in notes)
    assert any("Resuming from checkpoint" in n for n in notes)


def test_recent_checkpoint_resumes_without_clamp():
    # A checkpoint inside the window must be honored verbatim (true resume).
    state = {"checkpoint": _checkpoint(3)}
    since, notes = _resolve_backfill_since(state, since_days=None, now=NOW)
    assert since == datetime.fromisoformat(_iso(3))
    assert any("Resuming from checkpoint" in n for n in notes)
    assert not any("Clamping" in n for n in notes)


def test_no_checkpoint_recent_last_run_is_incremental():
    state = {"last_run": _iso(1)}
    since, notes = _resolve_backfill_since(state, since_days=None, now=NOW)
    assert since == datetime.fromisoformat(_iso(1))
    assert notes == []


def test_no_checkpoint_stale_last_run_is_clamped():
    state = {"last_run": _iso(40)}
    since, notes = _resolve_backfill_since(state, since_days=None, now=NOW)
    assert since == NOW - timedelta(days=7)
    assert any("Clamping" in n for n in notes)


def test_empty_state_defaults_to_30_days():
    since, notes = _resolve_backfill_since({}, since_days=None, now=NOW)
    # 30d default is older than the 7d floor, so it clamps.
    assert since == NOW - timedelta(days=7)


def test_explicit_since_bypasses_clamp_for_catch_up():
    # Manual catch-up: --since 90 must reach 90 days back even though that is
    # far older than the automated clamp floor.
    state = {"checkpoint": _checkpoint(175)}
    since, notes = _resolve_backfill_since(state, since_days=90, now=NOW)
    assert since == NOW - timedelta(days=90)
    assert not any("Clamping" in n for n in notes)


def test_explicit_since_resumes_from_newer_checkpoint():
    # --since 14 but a checkpoint 3 days back exists -> resume from checkpoint
    # (newer) to avoid re-processing.
    state = {"checkpoint": _checkpoint(3)}
    since, notes = _resolve_backfill_since(state, since_days=14, now=NOW)
    assert since == datetime.fromisoformat(_iso(3))
    assert any("Resuming from checkpoint" in n for n in notes)


def test_convergence_walk_does_not_regress_below_floor():
    # Simulate successive automated runs: a stale checkpoint is clamped to the
    # floor; once the checkpoint advances inside the window it is honored.
    # Across runs the window must never reach back past the floor.
    for cp_days in (175, 120, 30, 7, 5, 1):
        since, _ = _resolve_backfill_since(
            {"checkpoint": _checkpoint(cp_days)}, since_days=None, now=NOW
        )
        assert since >= NOW - timedelta(days=7)
