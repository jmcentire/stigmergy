"""Fixed-size ring buffer for bounded signal history per context.

Contexts don't need to remember every signal forever. The ring buffer
keeps the N most recent signals, automatically evicting the oldest.
Used for temporal analysis and active inquiry windowing.

Performance:
- append: O(1) amortized
- read: O(1) by index, O(k) for last k items
- Memory: O(capacity)
"""

from __future__ import annotations

from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Fixed-capacity circular buffer. O(1) append, O(1) index access."""

    __slots__ = ("_buffer", "_capacity", "_head", "_count")

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer: list[T | None] = [None] * capacity
        self._capacity = capacity
        self._head = 0  # next write position
        self._count = 0

    def append(self, item: T) -> T | None:
        """Append item. Returns evicted item if buffer was full, else None."""
        evicted = self._buffer[self._head] if self._count == self._capacity else None
        self._buffer[self._head] = item
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1
        return evicted

    def __getitem__(self, index: int) -> T:
        """Get item by index (0 = oldest, -1 = newest)."""
        if index < 0:
            index = self._count + index
        if index < 0 or index >= self._count:
            raise IndexError(f"index {index} out of range for buffer of size {self._count}")
        start = (self._head - self._count) % self._capacity
        actual = (start + index) % self._capacity
        return self._buffer[actual]  # type: ignore

    def last(self, n: int = 1) -> list[T]:
        """Get the last n items (most recent first)."""
        n = min(n, self._count)
        result: list[T] = []
        for i in range(n):
            pos = (self._head - 1 - i) % self._capacity
            result.append(self._buffer[pos])  # type: ignore
        return result

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def full(self) -> bool:
        return self._count == self._capacity

    def __iter__(self) -> Iterator[T]:
        """Iterate oldest to newest."""
        start = (self._head - self._count) % self._capacity
        for i in range(self._count):
            yield self._buffer[(start + i) % self._capacity]  # type: ignore

    def clear(self) -> None:
        self._buffer = [None] * self._capacity
        self._head = 0
        self._count = 0
