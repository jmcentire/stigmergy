"""Counting Bloom Filter for fast set membership and overlap estimation.

Used in familiarity scoring for O(k) term overlap instead of O(n) set
intersection. Counting variant supports removal (contexts lose terms
when signals decay).

Double-hashing technique (Kirsch & Mitzenmacher): only two base hashes
needed to simulate k hash functions. No external hash library required.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable

import numpy as np


def _fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit hash. Fast, good distribution."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _murmur_mix(data: bytes) -> int:
    """Murmur3-style finalizer for second hash. Independent from FNV."""
    h = 0
    for i in range(0, len(data) - 3, 4):
        k = struct.unpack_from("<I", data, i)[0]
        k = (k * 0xCC9E2D51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1B873593) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    # Handle remaining bytes
    remaining = data[len(data) - (len(data) % 4):]
    if remaining:
        k = 0
        for i, b in enumerate(remaining):
            k |= b << (8 * i)
        k = (k * 0xCC9E2D51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1B873593) & 0xFFFFFFFF
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


class CountingBloomFilter:
    """Counting Bloom filter with 8-bit counters.

    Supports add, remove, membership testing, and overlap estimation
    between two filters (for fast Jaccard-like similarity).

    Memory: ~size bytes (uint8 counters).
    False positive rate: configurable at construction.
    """

    __slots__ = ("_size", "_num_hashes", "_counters", "_count")

    def __init__(self, capacity: int = 10000, fp_rate: float = 0.01, num_hashes: int | None = None):
        if capacity <= 0:
            capacity = 1
        self._size = self._optimal_size(capacity, fp_rate)
        self._num_hashes = num_hashes or self._optimal_hashes(self._size, capacity)
        self._counters = np.zeros(self._size, dtype=np.uint8)
        self._count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Optimal bit array size for n items at false positive rate p."""
        m = -int(n * math.log(p) / (math.log(2) ** 2))
        return max(m, 64)  # floor at 64

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        """Optimal number of hash functions."""
        k = max(1, int((m / max(n, 1)) * math.log(2)))
        return min(k, 20)  # cap at 20

    def _indices(self, item: str) -> list[int]:
        """Generate k indices using double hashing."""
        data = item.encode("utf-8")
        h1 = _fnv1a_64(data)
        h2 = _murmur_mix(data)
        return [(h1 + i * h2) % self._size for i in range(self._num_hashes)]

    def add(self, item: str) -> None:
        """Add an item. Saturates at 255 (no overflow)."""
        for idx in self._indices(item):
            if self._counters[idx] < 255:
                self._counters[idx] += 1
        self._count += 1

    def add_many(self, items: Iterable[str]) -> None:
        """Batch add for efficiency."""
        for item in items:
            self.add(item)

    def remove(self, item: str) -> None:
        """Remove an item. Only remove if all counters are positive."""
        indices = self._indices(item)
        if all(self._counters[idx] > 0 for idx in indices):
            for idx in indices:
                self._counters[idx] -= 1
            self._count = max(0, self._count - 1)

    def __contains__(self, item: str) -> bool:
        """Probabilistic membership test. No false negatives."""
        return all(self._counters[idx] > 0 for idx in self._indices(item))

    def estimate_overlap(self, other: CountingBloomFilter) -> float:
        """Estimate overlap ratio between two filters.

        Uses minimum of counters at each position, normalized.
        Returns value in [0, 1] — proportion of shared items.
        """
        if self._size != other._size or self._num_hashes != other._num_hashes:
            raise ValueError("Filters must have same parameters for overlap estimation")
        if self._count == 0 and other._count == 0:
            return 0.0

        # Inner product of min-count vectors, normalized
        min_counts = np.minimum(self._counters, other._counters)
        max_counts = np.maximum(self._counters, other._counters)
        sum_max = max_counts.sum()
        if sum_max == 0:
            return 0.0
        return float(min_counts.sum() / sum_max)

    def estimate_jaccard(self, other: CountingBloomFilter) -> float:
        """Estimate Jaccard similarity between the sets represented by these filters.

        J(A,B) = |A ∩ B| / |A ∪ B|
        Approximated via min-count / max-count of the counter vectors.
        """
        return self.estimate_overlap(other)

    @property
    def count(self) -> int:
        """Number of items added (approximate, counts insertions not unique items)."""
        return self._count

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._counters.fill(0)
        self._count = 0
