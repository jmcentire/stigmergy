"""Mock GitHub adapter. Returns signals shaped like real GitHub events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Callable

from stigmergy.primitives.signal import Signal, SignalSource

_MOCK_EVENTS = [
    {
        "content": "PR #487: Fix cache invalidation race condition. Changes src/cache/invalidation.ts and adds retry logic with exponential backoff.",
        "channel": "acme-org/backend",
        "author": "alice.chen",
        "metadata": {
            "event_type": "pull_request",
            "action": "opened",
            "pr_number": 487,
            "state": "OPEN",
            "files_changed": ["src/cache/invalidation.ts", "src/cache/retry.ts"],
            "additions": 142,
            "deletions": 23,
            "review_decision": "",
            "is_draft": False,
            "branch": "fix/cache-race",
        },
    },
    {
        "content": "Looks like this duplicates the fix already done in PR #480. We should close one of these.",
        "channel": "acme-org/backend",
        "author": "bob.martinez",
        "metadata": {
            "event_type": "pr_comment",
            "pr_number": 487,
            "parent_url": "https://github.com/acme-org/backend/pull/487",
        },
    },
    {
        "content": "Review (CHANGES_REQUESTED): The retry backoff should cap at 30s, not grow indefinitely. Also missing unit test for the race condition.",
        "channel": "acme-org/backend",
        "author": "carol.park",
        "metadata": {
            "event_type": "review",
            "review_state": "CHANGES_REQUESTED",
            "pr_number": 487,
            "parent_url": "https://github.com/acme-org/backend/pull/487",
        },
    },
    {
        "content": "PR #488: Add backpressure handling with configurable queue depth and rate limiting.",
        "channel": "acme-org/sync-service",
        "author": "dave.kim",
        "metadata": {
            "event_type": "pull_request",
            "action": "opened",
            "pr_number": 488,
            "state": "OPEN",
            "files_changed": ["src/adapters/ingestion.ts", "src/queue/backpressure.ts"],
            "additions": 298,
            "deletions": 15,
            "review_decision": "",
            "is_draft": False,
            "branch": "feature/backpressure",
        },
    },
    {
        "content": "Merged PR #485: Pricing engine v3 dynamic pricing model with 15% accuracy improvement.",
        "channel": "acme-org/pricing",
        "author": "bob.martinez",
        "metadata": {
            "event_type": "pull_request",
            "action": "closed",
            "merged": True,
            "pr_number": 485,
            "state": "MERGED",
            "files_changed": ["src/engine/v3/model.ts", "src/engine/v3/features.ts"],
            "merged_at": "2025-01-15T10:30:00Z",
            "merged_by": "carol.park",
            "review_decision": "APPROVED",
            "is_draft": False,
            "branch": "feature/pricing-v3",
        },
    },
    {
        "content": "CI failure: test-suite on main",
        "channel": "acme-org/backend",
        "author": "github-actions",
        "metadata": {
            "event_type": "check_run",
            "conclusion": "failure",
            "run_name": "test-suite",
            "branch": "main",
            "url": "https://github.com/acme-org/backend/actions/runs/2341",
        },
    },
    {
        "content": "CI timed_out: integration-tests on feature/sync-v2",
        "channel": "acme-org/sync-service",
        "author": "github-actions",
        "metadata": {
            "event_type": "check_run",
            "conclusion": "timed_out",
            "run_name": "integration-tests",
            "branch": "feature/sync-v2",
            "url": "https://github.com/acme-org/sync-service/actions/runs/891",
        },
    },
]


class MockGitHubAdapter:
    """Generates realistic GitHub-shaped signals for testing."""

    def __init__(self) -> None:
        self._callback: Callable[[Signal], None] | None = None
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def subscribe(self, callback: Callable[[Signal], None]) -> None:
        self._callback = callback

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]:
        base_time = since
        for i, event in enumerate(_MOCK_EVENTS):
            yield Signal(
                content=event["content"],
                source=SignalSource.GITHUB,
                channel=event["channel"],
                author=event["author"],
                timestamp=base_time + timedelta(hours=i * 3),
                metadata=event["metadata"],
            )

    async def emit_all(self) -> list[Signal]:
        signals = []
        async for signal in self.backfill(datetime.now(timezone.utc) - timedelta(days=1)):
            signals.append(signal)
            if self._callback:
                self._callback(signal)
        return signals
