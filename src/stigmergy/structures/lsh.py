"""Locality-Sensitive Hashing for approximate nearest-neighbor search.

Used to pre-filter which contexts to evaluate for familiarity scoring.
Instead of computing cosine similarity against ALL contexts, LSH finds
candidate contexts in O(1) average case.

Uses random hyperplane hashing (SimHash variant):
- Each hash function is a random hyperplane through the origin.
- A vector's hash bit is 1 if it's on the positive side, 0 otherwise.
- Similar vectors land in the same bucket with high probability.

Multiple hash tables increase recall at the cost of memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stigmergy.services.vector_store import cosine_similarity


@dataclass
class _LSHEntry:
    id: str
    vector: np.ndarray


class LSHIndex:
    """Multi-table LSH index for cosine similarity.

    Performance:
    - add(): O(num_tables * num_hyperplanes) — hash computation
    - query(): O(num_tables * avg_bucket_size) — candidate retrieval + rerank
    - Memory: O(n * num_tables) for bucket storage
    """

    __slots__ = ("_dims", "_num_tables", "_num_hyperplanes", "_planes", "_tables", "_vectors")

    def __init__(self, dimensions: int, num_tables: int = 12, num_hyperplanes: int = 10,
                 seed: int = 42):
        """
        Args:
            dimensions: Vector dimensionality.
            num_tables: Number of hash tables (more = higher recall, more memory).
            num_hyperplanes: Bits per hash (more = higher precision, lower recall per table).
            seed: Random seed for reproducible hyperplanes.
        """
        self._dims = dimensions
        self._num_tables = num_tables
        self._num_hyperplanes = num_hyperplanes

        # Generate random hyperplanes for each table
        rng = np.random.RandomState(seed)
        self._planes: list[np.ndarray] = [
            rng.randn(num_hyperplanes, dimensions).astype(np.float32)
            for _ in range(num_tables)
        ]

        # Hash tables: table_index -> hash_key -> list of entry IDs
        self._tables: list[dict[int, list[str]]] = [
            {} for _ in range(num_tables)
        ]

        # ID -> vector for reranking
        self._vectors: dict[str, np.ndarray] = {}

    def _hash(self, vector: np.ndarray, table_idx: int) -> int:
        """Compute hash for a vector in a specific table.

        Projects vector onto random hyperplanes and converts to integer hash.
        """
        projections = self._planes[table_idx] @ vector
        bits = (projections > 0).astype(np.uint32)
        # Pack bits into integer
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    def add(self, id: str, vector: list[float] | np.ndarray) -> None:
        """Add a vector to all hash tables."""
        vec = np.asarray(vector, dtype=np.float32)
        self._vectors[id] = vec

        for t in range(self._num_tables):
            h = self._hash(vec, t)
            bucket = self._tables[t].setdefault(h, [])
            bucket.append(id)

    def remove(self, id: str) -> None:
        """Remove a vector from the index."""
        vec = self._vectors.pop(id, None)
        if vec is None:
            return
        for t in range(self._num_tables):
            h = self._hash(vec, t)
            bucket = self._tables[t].get(h)
            if bucket:
                try:
                    bucket.remove(id)
                except ValueError:
                    pass

    def query(self, vector: list[float] | np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Find approximate nearest neighbors.

        Returns list of (id, cosine_similarity) sorted by similarity, descending.
        """
        vec = np.asarray(vector, dtype=np.float32)

        # Collect candidates from all tables (set for dedup)
        candidates: set[str] = set()
        for t in range(self._num_tables):
            h = self._hash(vec, t)
            bucket = self._tables[t].get(h, [])
            candidates.update(bucket)

        if not candidates:
            return []

        # Rerank candidates by exact cosine similarity
        scored: list[tuple[str, float]] = []
        for cid in candidates:
            cvec = self._vectors.get(cid)
            if cvec is not None:
                sim = cosine_similarity(vec, cvec)
                scored.append((cid, float(sim)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def query_ids(self, vector: list[float] | np.ndarray, top_k: int = 10) -> list[str]:
        """Convenience: return just IDs of nearest neighbors."""
        return [id for id, _ in self.query(vector, top_k)]

    def update(self, id: str, vector: list[float] | np.ndarray) -> None:
        """Update a vector in place (remove + re-add)."""
        self.remove(id)
        self.add(id, vector)

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, id: str) -> bool:
        return id in self._vectors
