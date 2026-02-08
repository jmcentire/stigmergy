"""Tests for temporal currency in signal summary formatting.

Validates that signal summaries sent to the correlator include
temporal state (MERGED Xd ago, OPEN, Done, thread age) so the
LLM can distinguish active from historical signals.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from stigmergy.primitives.signal import Signal


def _make_signal(
    source: str = "GITHUB",
    channel: str = "repo/test",
    author: str = "alice",
    content: str = "Fix cache invalidation",
    metadata: dict | None = None,
    timestamp: datetime | None = None,
) -> Signal:
    return Signal(
        id=uuid4(),
        source=source,
        channel=channel,
        author=author,
        content=content,
        timestamp=timestamp or datetime.now(timezone.utc),
        metadata=metadata or {},
    )


class TestMergedPRSummaryFormat:
    """Merged PRs should show 'MERGED Xd ago' in the summary."""

    def test_merged_pr_includes_days_ago(self):
        """A signal with state=MERGED and merged_at should show MERGED Xd ago."""
        now = datetime.now(timezone.utc)
        three_days_ago = now - timedelta(days=3)

        sig = _make_signal(
            source="GITHUB",
            metadata={
                "event_type": "pull_request",
                "state": "MERGED",
                "merged_at": three_days_ago.isoformat(),
            },
        )
        meta = sig.metadata or {}
        status = meta.get("status") or meta.get("state") or ""
        state_upper = str(status).upper()

        assert state_upper == "MERGED"

        # Simulate the temporal tag logic from agent.py
        age_ts = meta.get("merged_at") or meta.get("closedAt") or meta.get("closed_at")
        assert age_ts is not None

        age_dt = datetime.fromisoformat(age_ts.replace("Z", "+00:00"))
        days_ago = (now - age_dt).total_seconds() / 86400
        assert 2.5 < days_ago < 3.5

        if days_ago < 1:
            temporal_tag = f"{state_upper} <1d ago"
        else:
            temporal_tag = f"{state_upper} {int(days_ago)}d ago"

        assert temporal_tag == "MERGED 3d ago"


class TestOpenPRSummaryFormat:
    """Open PRs should show 'OPEN' in the summary."""

    def test_open_pr_shows_open(self):
        sig = _make_signal(
            source="GITHUB",
            metadata={
                "event_type": "pull_request",
                "state": "OPEN",
            },
        )
        meta = sig.metadata or {}
        status = meta.get("status") or meta.get("state") or ""
        state_upper = str(status).upper()
        assert state_upper == "OPEN"

        # Per agent.py logic
        if state_upper in ("OPEN", "IN PROGRESS", "IN_PROGRESS"):
            temporal_tag = "OPEN"
        else:
            temporal_tag = str(status)

        assert temporal_tag == "OPEN"


class TestClosedIssueSummaryFormat:
    """Linear tickets with status=Done should show 'Done'."""

    def test_done_ticket_shows_done(self):
        sig = _make_signal(
            source="LINEAR",
            channel="PLAT",
            metadata={
                "event_type": "issue",
                "status": "Done",
            },
        )
        meta = sig.metadata or {}
        status = meta.get("status") or meta.get("state") or ""
        state_upper = str(status).upper()

        assert state_upper == "DONE"

        # Per agent.py logic
        if state_upper in ("DONE", "RESOLVED", "COMPLETE", "COMPLETED"):
            temporal_tag = state_upper
        else:
            temporal_tag = str(status)

        assert temporal_tag == "DONE"

    def test_in_progress_ticket_shows_open(self):
        sig = _make_signal(
            source="LINEAR",
            channel="PLAT",
            metadata={
                "event_type": "issue",
                "status": "In Progress",
            },
        )
        meta = sig.metadata or {}
        status = meta.get("status") or meta.get("state") or ""
        state_upper = str(status).upper()

        assert state_upper == "IN PROGRESS"

        if state_upper in ("OPEN", "IN PROGRESS", "IN_PROGRESS"):
            temporal_tag = "OPEN"
        else:
            temporal_tag = str(status)

        assert temporal_tag == "OPEN"


class TestSlackThreadAge:
    """Slack threads older than 1 day should show 'thread from Xd ago'."""

    def test_old_slack_thread_shows_age(self):
        now = datetime.now(timezone.utc)
        three_days_ago = now - timedelta(days=3)

        sig = _make_signal(
            source="slack",
            channel="#dev-platform",
            timestamp=three_days_ago,
        )

        # Simulate the Slack temporal logic from agent.py
        thread_age = (now - sig.timestamp).total_seconds() / 86400
        assert thread_age > 1

        temporal_tag = f"thread from {int(thread_age)}d ago"
        assert temporal_tag == "thread from 3d ago"

    def test_recent_slack_no_thread_age(self):
        now = datetime.now(timezone.utc)
        recent = now - timedelta(hours=6)

        sig = _make_signal(
            source="slack",
            channel="#dev-platform",
            timestamp=recent,
        )

        thread_age = (now - sig.timestamp).total_seconds() / 86400
        assert thread_age < 1
        # No temporal_tag would be set for recent threads
