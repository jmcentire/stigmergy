"""Embedding service: interface + deterministic stub + strategy registry.

Every LLM embedding call goes through this interface. Stub returns
deterministic pseudo-random vectors based on content hash so that
familiarity scoring is reproducible in tests.

The EmbeddingStrategyRegistry provides a multi-strategy embedding system
where different embedding strategies (semantic, technical, social, etc.)
can be registered and run concurrently on signals.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from stigmergy.primitives.signal import Signal

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = 384

STRATEGIES = [
    "semantic",
    "technical",
    "social",
    "temporal",
    "strategic",
]

_STRATEGY_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


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


# ── Multi-Strategy Embedding System ──────────────────────────────


@runtime_checkable
class EmbeddingStrategy(Protocol):
    """Protocol for an embedding strategy.

    All strategies must implement async embed(signal) -> numpy ndarray.
    """

    async def embed(self, signal: Signal) -> np.ndarray: ...


@dataclass(frozen=True)
class StrategyFailure:
    """Details about a strategy that failed during embed_all."""

    strategy_name: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class StrategyInfo:
    """Information about a registered strategy."""

    name: str
    strategy_type: str
    is_available: bool


@dataclass
class EmbedAllResult:
    """Result of running all strategies on a signal."""

    embeddings: dict[str, np.ndarray]
    failures: list[StrategyFailure]


def project_to_dim(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Truncate or pad a vector to target_dim, then L2-normalize.

    Pure function. Returns a new array.
    """
    if vector.ndim != 1:
        raise ValueError(
            f"Expected 1-D array, got {vector.ndim}-D array with shape {vector.shape}."
        )
    if vector.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"Expected float32 or float64 dtype, got {vector.dtype}."
        )
    if not isinstance(target_dim, int) or target_dim < 1 or target_dim > 4096:
        raise ValueError(
            "target_dim must be an integer between 1 and 4096 inclusive."
        )
    if vector.shape[0] == 0:
        raise ValueError("Input vector must have at least one element.")

    # Truncate or pad
    if vector.shape[0] >= target_dim:
        result = vector[:target_dim].copy()
    else:
        result = np.zeros(target_dim, dtype=vector.dtype)
        result[: vector.shape[0]] = vector

    # L2-normalize (skip for zero vectors)
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm

    return result


def embeddings_to_ndarray(
    embedding: list[float],
    dtype: str = "float32",
) -> np.ndarray:
    """Convert list[float] to numpy ndarray."""
    if not embedding:
        raise ValueError("Embedding list must not be empty.")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64', got '{dtype}'.")
    arr = np.array(embedding, dtype=np.dtype(dtype))
    if not np.all(np.isfinite(arr)):
        raise ValueError("Embedding contains non-finite values (NaN or Inf).")
    return arr


def ndarray_to_embedding(array: np.ndarray) -> list[float]:
    """Convert numpy ndarray to list[float]."""
    if array.ndim != 1:
        raise ValueError(f"Expected 1-D array, got {array.ndim}-D array.")
    if array.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"Expected float32 or float64 dtype, got {array.dtype}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Array contains non-finite values (NaN or Inf).")
    return array.tolist()


class EmbeddingStrategyRegistry:
    """Registry for multiple embedding strategies.

    All strategies must produce vectors of the same fixed dimensionality.
    Strategies are run concurrently via asyncio.gather.
    """

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        if not isinstance(dimensions, int) or dimensions < 1 or dimensions > 4096:
            raise ValueError(
                "dimensions must be an integer between 1 and 4096 inclusive."
            )
        self.dimensions = dimensions
        self._strategies: dict[str, EmbeddingStrategy] = {}

    def register(self, name: str, strategy: EmbeddingStrategy) -> None:
        """Register a named embedding strategy."""
        if not _STRATEGY_NAME_PATTERN.match(name):
            raise ValueError(
                "Strategy name must start with a lowercase letter and contain "
                "only lowercase alphanumeric characters and underscores (1-64 chars)."
            )
        if not isinstance(strategy, EmbeddingStrategy):
            raise TypeError(
                "Strategy does not satisfy the EmbeddingStrategy protocol. "
                "Must have an 'embed(self, signal: Signal) -> np.ndarray' method."
            )
        if not inspect.iscoroutinefunction(strategy.embed):
            raise TypeError(
                "Strategy.embed must be an async coroutine function "
                "(defined with 'async def')."
            )
        if name in self._strategies:
            raise ValueError(
                f"A strategy with name '{name}' is already registered. "
                "Use unregister() first to replace it."
            )
        self._strategies[name] = strategy

    def unregister(self, name: str) -> None:
        """Remove a previously registered strategy."""
        if name not in self._strategies:
            raise KeyError(f"No strategy registered with name '{name}'.")
        del self._strategies[name]

    def has_strategy(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return name in self._strategies

    def list_strategies(self) -> list[StrategyInfo]:
        """Return info about all registered strategies, sorted by name."""
        return sorted(
            [
                StrategyInfo(
                    name=name,
                    strategy_type=type(s).__qualname__,
                    is_available=True,
                )
                for name, s in self._strategies.items()
            ],
            key=lambda s: s.name,
        )

    async def embed(
        self, signal: Signal, strategy: str = "semantic"
    ) -> np.ndarray:
        """Run a single strategy on a signal."""
        if strategy not in self._strategies:
            available = ", ".join(sorted(self._strategies.keys()))
            raise KeyError(
                f"No strategy registered with name '{strategy}'. "
                f"Available: {available}."
            )
        try:
            result = await self._strategies[strategy].embed(signal)
        except Exception as e:
            raise RuntimeError(f"Strategy '{strategy}' failed: {e}") from e

        result = np.asarray(result)
        if result.shape != (self.dimensions,):
            # Auto-project to target dimensions
            result = project_to_dim(
                result.astype(np.float64), self.dimensions
            )
        return result

    async def embed_all(self, signal: Signal) -> EmbedAllResult:
        """Run all strategies concurrently."""
        if not self._strategies:
            raise RuntimeError(
                "No strategies registered. Register at least one strategy "
                "before calling embed_all()."
            )

        names = list(self._strategies.keys())
        tasks = [self._strategies[n].embed(signal) for n in names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        embeddings: dict[str, np.ndarray] = {}
        failures: list[StrategyFailure] = []

        for name, result in zip(names, results):
            if isinstance(result, Exception):
                failures.append(StrategyFailure(
                    strategy_name=name,
                    error_type=type(result).__name__,
                    error_message=str(result),
                ))
                logger.warning("Strategy '%s' failed: %s", name, result)
                continue

            arr = np.asarray(result)
            if arr.ndim != 1:
                failures.append(StrategyFailure(
                    strategy_name=name,
                    error_type="ValueError",
                    error_message=f"Expected 1-D array, got {arr.ndim}-D",
                ))
                continue

            # Project to target dimensions if needed
            if arr.shape[0] != self.dimensions:
                try:
                    arr = project_to_dim(arr.astype(np.float64), self.dimensions)
                except Exception as e:
                    failures.append(StrategyFailure(
                        strategy_name=name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    ))
                    continue

            embeddings[name] = arr

        return EmbedAllResult(embeddings=embeddings, failures=failures)

    async def populate_signal_embeddings(self, signal: Signal) -> Signal:
        """Run all strategies and return a new Signal with embeddings populated."""
        result = await self.embed_all(signal)
        embedding_dict: dict[str, list[float]] = {}
        for name, arr in result.embeddings.items():
            embedding_dict[name] = ndarray_to_embedding(arr)
        return signal.model_copy(update={"embeddings": embedding_dict})
