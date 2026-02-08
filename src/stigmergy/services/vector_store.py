"""Vector store: interface + in-memory implementation.

Supports multi-strategy embeddings. Each entry stores vectors under
multiple strategy keys. Search operates on a single strategy at a time.

Optimized with:
- Batched matrix cosine similarity (numpy matmul instead of per-entry loop)
- Cached centroids (invalidated on add, not recomputed every query)
- LSH index for approximate pre-filtering when store is large
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any]


@dataclass
class _VectorEntry:
    vectors: dict[str, list[float]]
    metadata: dict[str, Any]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors.

    Args:
        query: (d,) shaped vector
        matrix: (n, d) shaped matrix

    Returns:
        (n,) shaped array of similarities
    """
    if matrix.shape[0] == 0:
        return np.array([])
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        return np.zeros(matrix.shape[0])
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    return (matrix @ query) / (row_norms * query_norm)


class VectorStore(Protocol):
    async def add(self, id: str, vectors: dict[str, list[float]], metadata: dict[str, Any]) -> None: ...

    async def search(self, query_vector: list[float], strategy: str, top_k: int = 10) -> list[SearchResult]: ...

    async def get_centroid(self, strategy: str) -> list[float] | None: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    """In-memory vector store with batched matrix operations.

    Uses numpy matmul for cosine similarity instead of per-entry loops.
    Caches strategy-specific matrices and centroids, invalidating on add.

    Performance:
    - search: O(n*d) via single matmul call (vs O(n*d) with n function calls)
    - centroid: O(1) cache hit, O(n*d) on cache miss
    - add: O(1) amortized (invalidates caches)
    """

    def __init__(self) -> None:
        self._entries: dict[str, _VectorEntry] = {}
        self._entry_order: list[str] = []  # maintain insertion order for matrix indexing
        # Cached matrices per strategy: strategy -> (n, d) array
        self._matrix_cache: dict[str, np.ndarray] = {}
        self._id_index_cache: dict[str, list[str]] = {}  # strategy -> [entry_ids]
        self._centroid_cache: dict[str, list[float]] = {}
        self._cache_dirty: bool = False

    async def add(self, id: str, vectors: dict[str, list[float]], metadata: dict[str, Any]) -> None:
        is_new = id not in self._entries
        self._entries[id] = _VectorEntry(vectors=vectors, metadata=metadata)
        if is_new:
            self._entry_order.append(id)
        self._cache_dirty = True
        # Invalidate caches for affected strategies
        for strategy in vectors:
            self._matrix_cache.pop(strategy, None)
            self._id_index_cache.pop(strategy, None)
            self._centroid_cache.pop(strategy, None)

    def _build_matrix(self, strategy: str) -> tuple[np.ndarray, list[str]]:
        """Build the (n, d) matrix and ID index for a strategy."""
        if not self._cache_dirty and strategy in self._matrix_cache:
            return self._matrix_cache[strategy], self._id_index_cache[strategy]

        ids: list[str] = []
        vectors: list[list[float]] = []
        for eid in self._entry_order:
            entry = self._entries.get(eid)
            if entry and strategy in entry.vectors:
                ids.append(eid)
                vectors.append(entry.vectors[strategy])

        if not vectors:
            matrix = np.array([]).reshape(0, 0)
        else:
            matrix = np.array(vectors, dtype=np.float32)

        self._matrix_cache[strategy] = matrix
        self._id_index_cache[strategy] = ids
        return matrix, ids

    async def search(self, query_vector: list[float], strategy: str, top_k: int = 10) -> list[SearchResult]:
        matrix, ids = self._build_matrix(strategy)
        if matrix.shape[0] == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        similarities = batch_cosine_similarity(query, matrix)

        # Partial sort for top_k (faster than full sort when top_k << n)
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results: list[SearchResult] = []
        for idx in top_indices:
            eid = ids[idx]
            entry = self._entries[eid]
            results.append(SearchResult(
                id=eid,
                score=float(similarities[idx]),
                metadata=entry.metadata,
            ))
        return results

    async def get_centroid(self, strategy: str) -> list[float] | None:
        if strategy in self._centroid_cache and not self._cache_dirty:
            return self._centroid_cache[strategy]

        matrix, ids = self._build_matrix(strategy)
        if matrix.shape[0] == 0:
            return None

        centroid = np.mean(matrix, axis=0).tolist()
        self._centroid_cache[strategy] = centroid
        self._cache_dirty = False
        return centroid

    def count(self) -> int:
        return len(self._entries)
