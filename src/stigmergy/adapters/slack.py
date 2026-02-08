"""Mock Slack adapter. Returns signals shaped like real Slack messages."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Callable

from stigmergy.primitives.signal import Signal, SignalSource

# Realistic mock Slack messages for testing signal flow
_MOCK_MESSAGES = [
    {
        "content": "Hey team, the booking sync is failing for external PMS properties again. Getting 500 errors on the webhook callback.",
        "channel": "#engineering",
        "author": "bob.martinez",
        "metadata": {"thread_ts": "1706000000.000001", "reaction_count": 3},
    },
    {
        "content": "Just pushed a fix for the cache invalidation bug. PR #487 is ready for review.",
        "channel": "#engineering",
        "author": "alice.chen",
        "metadata": {"thread_ts": "1706001000.000002", "has_link": True},
    },
    {
        "content": "Customer at the lakefront property is reporting double bookings again. This is the third time this month.",
        "channel": "#support",
        "author": "eve.santos",
        "metadata": {"thread_ts": "1706002000.000003", "urgency": "high"},
    },
    {
        "content": "Q1 revenue projections look solid. We should talk about expanding to the coastal market.",
        "channel": "#leadership",
        "author": "carol.park",
        "metadata": {"thread_ts": "1706003000.000004"},
    },
    {
        "content": "Anyone want to grab lunch? Thinking tacos.",
        "channel": "#general",
        "author": "frank.reyes",
        "metadata": {"thread_ts": "1706004000.000005"},
    },
    {
        "content": "The pricing engine v3 is showing a 15% improvement in dynamic pricing accuracy compared to v2.",
        "channel": "#data-science",
        "author": "bob.martinez",
        "metadata": {"thread_ts": "1706005000.000006", "has_attachment": True},
    },
    {
        "content": "Deploy to production scheduled for 3pm EST. Pricing service and sync service going out together.",
        "channel": "#deploys",
        "author": "grace.liu",
        "metadata": {"thread_ts": "1706006000.000007", "deploy_id": "deploy-2024-01-23"},
    },
    {
        "content": "The external integration is dropping events during high load. We need to add backpressure handling.",
        "channel": "#engineering",
        "author": "dave.kim",
        "metadata": {"thread_ts": "1706007000.000008", "mentions": ["alice.chen"]},
    },
]


class MockSlackAdapter:
    """Generates realistic Slack-shaped signals for testing."""

    def __init__(self) -> None:
        self._callback: Callable[[Signal], None] | None = None
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def subscribe(self, callback: Callable[[Signal], None]) -> None:
        self._callback = callback

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]:
        base_time = since
        for i, msg in enumerate(_MOCK_MESSAGES):
            yield Signal(
                content=msg["content"],
                source=SignalSource.SLACK,
                channel=msg["channel"],
                author=msg["author"],
                timestamp=base_time + timedelta(hours=i),
                metadata=msg["metadata"],
            )

    async def emit_all(self) -> list[Signal]:
        """Emit all mock messages as signals (for testing)."""
        signals = []
        async for signal in self.backfill(datetime.now(timezone.utc) - timedelta(days=1)):
            signals.append(signal)
            if self._callback:
                self._callback(signal)
        return signals
