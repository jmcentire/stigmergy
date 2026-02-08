"""Ingestion queue: the only centralized component.

A FIFO buffer with priority weighting. It does not route.
It holds signals until agents claim them.
"""

from __future__ import annotations

import asyncio

from stigmergy.primitives.signal import Signal


class IngestionQueue:
    """Priority queue for incoming signals. Lower priority number = higher priority."""

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[float, int, Signal]] = asyncio.PriorityQueue()
        self._counter = 0  # tiebreaker for equal priorities

    async def put(self, signal: Signal, priority: float = 0.5) -> None:
        """Enqueue a signal. priority in [0,1], lower = more urgent."""
        self._counter += 1
        await self._queue.put((-priority, self._counter, signal))

    async def get(self) -> Signal:
        """Dequeue the highest-priority signal. Blocks if empty."""
        _, _, signal = await self._queue.get()
        return signal

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
