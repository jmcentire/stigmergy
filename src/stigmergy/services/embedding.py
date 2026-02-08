"""Embedding service: interface + deterministic stub.

Every LLM embedding call goes through this interface. Stub returns
deterministic pseudo-random vectors based on content hash so that
familiarity scoring is reproducible in tests.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


EMBEDDING_DIMENSIONS = 384

STRATEGIES = [
    "semantic",
    "technical",
    "social",
    "temporal",
    "strategic",
]


class EmbeddingService(Protocol):
    async def embed(self, content: str, strategy: str) -> list[float]: ...

    async def embed_all(self, content: str) -> dict[str, list[float]]: ...


class StubEmbeddingService:
    """Deterministic stub: same (content, strategy) always produces the same vector."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS, strategies: list[str] | None = None):
        self._dimensions = dimensions
        self._strategies = strategies or STRATEGIES
        self._cache: dict[tuple[str, str], list[float]] = {}

    async def embed(self, content: str, strategy: str) -> list[float]:
        key = (content, strategy)
        if key not in self._cache:
            seed = hash(key) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dimensions)
            vec = vec / np.linalg.norm(vec)
            self._cache[key] = vec.tolist()
        return self._cache[key]

    async def embed_all(self, content: str) -> dict[str, list[float]]:
        return {s: await self.embed(content, s) for s in self._strategies}
