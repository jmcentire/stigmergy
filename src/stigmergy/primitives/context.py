"""Context: a knowledge domain backed by a vector store.

Contexts are not predefined. They emerge from signal clustering and
dissolve when signals stop arriving. They have topology (neighbors),
energy (aliveness), and business weight (organizational importance).

Uses CountingBloomFilter for O(k) term overlap estimation and
RingBuffer for bounded recent signal history.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

from stigmergy.services.vector_store import InMemoryVectorStore, VectorStore
from stigmergy.structures.bloom import CountingBloomFilter
from stigmergy.structures.ring_buffer import RingBuffer


class Context:
    def __init__(
        self,
        *,
        id: UUID | None = None,
        relevance_decay: float = 0.001,
        relevance_threshold: float = 0.3,
        business_weight: float = 1.0,
        parent_id: UUID | None = None,
        vector_store: VectorStore | None = None,
        bloom_capacity: int = 10000,
        history_capacity: int = 1000,
        capacity: int = 200,
    ) -> None:
        self.id = id or uuid4()
        self.vector_store: VectorStore = vector_store or InMemoryVectorStore()
        self.active_agents: set[UUID] = set()
        self.relevance_decay = relevance_decay  # lambda
        self.relevance_threshold = relevance_threshold
        self.last_signal: datetime = datetime.now(timezone.utc)
        self.energy: float = 1.0
        self.parent_id = parent_id
        self.neighbors: set[UUID] = set()
        self.business_weight = business_weight
        self.capacity = capacity

        # Accumulated terms — both set (exact) and bloom filter (fast overlap)
        self.terms: set[str] = set()
        self.term_bloom = CountingBloomFilter(capacity=bloom_capacity)

        # Recent signal history (bounded)
        self.signal_history: RingBuffer[str] = RingBuffer(history_capacity)

        # Track signal sources and authors for affinity learning
        self.source_counts: dict[str, int] = {}
        self.author_counts: dict[str, int] = {}
        self.signal_count: int = 0

    @property
    def fullness(self) -> float:
        """How full this context is. signal_count / capacity, clamped [0, 1]."""
        if self.capacity <= 0:
            return 1.0
        return min(1.0, self.signal_count / self.capacity)

    def compute_energy(self, now: datetime | None = None) -> float:
        """Recompute energy based on exponential decay since last signal."""
        now = now or datetime.now(timezone.utc)
        elapsed = (now - self.last_signal).total_seconds()
        self.energy = self.energy * math.exp(-self.relevance_decay * elapsed)
        return self.energy

    async def ingest_signal(self, signal_id: str, embeddings: dict[str, list[float]],
                            terms: set[str], source: str, author: str,
                            score: float, metadata: dict,
                            signal_timestamp: datetime | None = None) -> None:
        """Ingest a classified signal into this context."""
        await self.vector_store.add(signal_id, embeddings, metadata)
        self.terms.update(terms)
        self.term_bloom.add_many(terms)
        self.signal_history.append(signal_id)
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        self.author_counts[author] = self.author_counts.get(author, 0) + 1
        self.signal_count += 1
        # Use the later of wall-clock and signal timestamp so that during
        # backfill, energy decay properly reflects signal-time gaps.
        now = datetime.now(timezone.utc)
        self.last_signal = max(now, signal_timestamp) if signal_timestamp else now
        self.energy += score

    def source_affinity(self, source: str) -> float:
        """How often signals from this source land in this context. [0, 1]."""
        if self.signal_count == 0:
            return 0.0
        return self.source_counts.get(source, 0) / self.signal_count

    def author_affinity(self, author: str) -> float:
        """How often this author's signals land in this context. [0, 1]."""
        if self.signal_count == 0:
            return 0.0
        return self.author_counts.get(author, 0) / self.signal_count

    def fast_term_overlap(self, other_bloom: CountingBloomFilter) -> float:
        """Estimate term overlap using bloom filters. O(size) but vectorized."""
        return self.term_bloom.estimate_jaccard(other_bloom)
