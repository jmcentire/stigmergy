"""Mock Linear adapter. Returns signals shaped like real Linear tickets."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Callable

from stigmergy.primitives.signal import Signal, SignalSource

_MOCK_TICKETS = [
    {
        "content": "Booking sync fails for external PMS properties with 500 error on webhook callback. Affects 12 properties.",
        "channel": "ENG-1234",
        "author": "bob.martinez",
        "metadata": {
            "identifier": "ENG-1234",
            "title": "PMS webhook sync failure",
            "status": "In Progress",
            "priority": 1,
            "labels": ["bug", "sync", "integrations"],
            "assignee": "alice.chen",
            "team": "Engineering",
            "estimate": 5,
            "cycle": "Sprint 12",
            "cycle_start": "2025-01-06",
            "cycle_end": "2025-01-20",
            "relations": [
                {"type": "related", "issue": "ENG-1236", "title": "Cache race condition"},
            ],
        },
    },
    {
        "content": "This is already fixed in PR #480 on the backend repo. The webhook handler was missing the retry header.",
        "channel": "ENG-1234",
        "author": "alice.chen",
        "metadata": {
            "event_type": "linear_comment",
            "identifier": "ENG-1234",
            "team": "Engineering",
        },
    },
    {
        "content": "Implement backpressure handling for event ingestion during high load periods.",
        "channel": "ENG-1235",
        "author": "dave.kim",
        "metadata": {
            "identifier": "ENG-1235",
            "title": "Event backpressure handling",
            "status": "Todo",
            "priority": 2,
            "labels": ["enhancement", "sync", "performance"],
            "assignee": "alice.chen",
            "team": "Engineering",
            "estimate": 8,
            "project": "Sync Re-platform",
            "parent": "ENG-1200",
            "parent_title": "Sync service re-architecture",
        },
    },
    {
        "content": "Cache invalidation race condition causing stale data for 30-60 seconds after booking.",
        "channel": "ENG-1236",
        "author": "alice.chen",
        "metadata": {
            "identifier": "ENG-1236",
            "title": "Cache race condition",
            "status": "Done",
            "priority": 1,
            "labels": ["bug", "availability", "cache"],
            "assignee": "alice.chen",
            "team": "Engineering",
            "estimate": 3,
            "due_date": "2025-01-15",
            "children": [
                {"id": "ENG-1236-A", "title": "Add cache TTL config", "status": "Done"},
                {"id": "ENG-1236-B", "title": "Write invalidation tests", "status": "Done"},
            ],
        },
    },
    {
        "content": "Pricing engine v3 should support seasonal multipliers configurable per-property.",
        "channel": "ENG-1237",
        "author": "carol.park",
        "metadata": {
            "identifier": "ENG-1237",
            "title": "Per-property seasonal pricing multipliers",
            "status": "Backlog",
            "priority": 3,
            "labels": ["enhancement", "pricing"],
            "assignee": "bob.martinez",
            "team": "Engineering",
            "project": "Pricing Re-platform",
        },
    },
    {
        "content": "Should we model seasonal multipliers as a separate table or embed in the property config? Separate table is more flexible.",
        "channel": "ENG-1237",
        "author": "bob.martinez",
        "metadata": {
            "event_type": "linear_comment",
            "identifier": "ENG-1237",
            "team": "Engineering",
        },
    },
]


class MockLinearAdapter:
    """Generates realistic Linear-shaped signals for testing."""

    def __init__(self) -> None:
        self._callback: Callable[[Signal], None] | None = None
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def subscribe(self, callback: Callable[[Signal], None]) -> None:
        self._callback = callback

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]:
        base_time = since
        for i, ticket in enumerate(_MOCK_TICKETS):
            yield Signal(
                content=ticket["content"],
                source=SignalSource.LINEAR,
                channel=ticket["channel"],
                author=ticket["author"],
                timestamp=base_time + timedelta(hours=i * 2),
                metadata=ticket["metadata"],
            )

    async def emit_all(self) -> list[Signal]:
        signals = []
        async for signal in self.backfill(datetime.now(timezone.utc) - timedelta(days=1)):
            signals.append(signal)
            if self._callback:
                self._callback(signal)
        return signals
