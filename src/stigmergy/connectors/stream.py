"""Mock message stream connector (Kafka/Redis Streams interface).

Implements the producer/consumer pattern the ingestion pipeline uses.
Same interface whether backed by Kafka, Redis Streams, or NATS.

Features:
- Topic-based pub/sub
- Consumer groups (multiple consumers, each gets unique messages)
- Message acknowledgment
- Dead letter queue for failed processing
- Backpressure (configurable max queue depth)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from uuid import uuid4


@dataclass
class StreamMessage:
    """A message on the stream."""
    id: str = field(default_factory=lambda: str(uuid4()))
    topic: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)
    attempts: int = 0


class StreamProducer(Protocol):
    """Protocol for stream producers."""
    async def send(self, topic: str, payload: dict[str, Any]) -> str: ...


class StreamConsumer(Protocol):
    """Protocol for stream consumers."""
    async def receive(self, topic: str, timeout: float = 5.0) -> StreamMessage | None: ...
    async def ack(self, message_id: str) -> None: ...
    async def nack(self, message_id: str) -> None: ...


class MockStream:
    """In-memory message stream with consumer groups and DLQ.

    Tracks all operations for observability and test assertions.
    """

    def __init__(self, max_queue_depth: int = 10000, max_retries: int = 3) -> None:
        self._queues: dict[str, asyncio.Queue[StreamMessage]] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_queue_depth)
        )
        self._max_retries = max_retries
        self._pending: dict[str, StreamMessage] = {}  # message_id -> message (awaiting ack)
        self._dlq: list[StreamMessage] = []
        self._stats: dict[str, _TopicStats] = defaultdict(_TopicStats)
        self._subscribers: dict[str, list[Callable[[StreamMessage], Any]]] = defaultdict(list)

    async def send(self, topic: str, payload: dict[str, Any]) -> str:
        """Produce a message to a topic. Returns message ID."""
        msg = StreamMessage(topic=topic, payload=payload)
        try:
            self._queues[topic].put_nowait(msg)
        except asyncio.QueueFull:
            self._stats[topic].dropped += 1
            raise BackpressureError(f"Topic '{topic}' queue full ({self._queues[topic].maxsize})")
        self._stats[topic].produced += 1
        # Notify subscribers
        for callback in self._subscribers.get(topic, []):
            callback(msg)
        return msg.id

    async def receive(self, topic: str, timeout: float = 5.0) -> StreamMessage | None:
        """Consume a message from a topic. Returns None on timeout."""
        try:
            msg = await asyncio.wait_for(self._queues[topic].get(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None
        msg.attempts += 1
        self._pending[msg.id] = msg
        self._stats[topic].consumed += 1
        return msg

    async def ack(self, message_id: str) -> None:
        """Acknowledge successful processing of a message."""
        msg = self._pending.pop(message_id, None)
        if msg:
            self._stats[msg.topic].acked += 1

    async def nack(self, message_id: str) -> None:
        """Negative acknowledge — message failed processing."""
        msg = self._pending.pop(message_id, None)
        if msg is None:
            return
        self._stats[msg.topic].nacked += 1
        if msg.attempts >= self._max_retries:
            self._dlq.append(msg)
            self._stats[msg.topic].dead_lettered += 1
        else:
            # Re-enqueue for retry
            try:
                self._queues[msg.topic].put_nowait(msg)
            except asyncio.QueueFull:
                self._dlq.append(msg)

    async def subscribe(self, topic: str, callback: Callable[[StreamMessage], Any]) -> None:
        """Register a callback for new messages on a topic."""
        self._subscribers[topic].append(callback)

    @property
    def dead_letter_queue(self) -> list[StreamMessage]:
        return list(self._dlq)

    def stats(self, topic: str | None = None) -> dict[str, Any]:
        """Get stats for a topic or all topics."""
        if topic:
            s = self._stats[topic]
            return {
                "produced": s.produced,
                "consumed": s.consumed,
                "acked": s.acked,
                "nacked": s.nacked,
                "dropped": s.dropped,
                "dead_lettered": s.dead_lettered,
                "pending": sum(1 for m in self._pending.values() if m.topic == topic),
                "queue_depth": self._queues[topic].qsize(),
            }
        return {t: self.stats(t) for t in self._stats}

    def reset(self) -> None:
        self._queues.clear()
        self._pending.clear()
        self._dlq.clear()
        self._stats.clear()
        self._subscribers.clear()


class _TopicStats:
    def __init__(self) -> None:
        self.produced = 0
        self.consumed = 0
        self.acked = 0
        self.nacked = 0
        self.dropped = 0
        self.dead_lettered = 0


class BackpressureError(Exception):
    """Raised when a topic queue is full."""
    pass
