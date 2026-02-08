"""Source adapter protocol.

Each adapter normalizes source data into Signals and computes embeddings.
When real MCPs are connected, the adapter code shouldn't change —
only the transport layer swaps.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Callable, Protocol

from stigmergy.primitives.signal import Signal


class SourceAdapter(Protocol):
    """Protocol for all source adapters (real or mock)."""

    async def connect(self) -> None: ...

    async def subscribe(self, callback: Callable[[Signal], None]) -> None: ...

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]: ...
