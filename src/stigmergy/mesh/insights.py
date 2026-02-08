"""Spectral observation — agent-driven cross-signal pattern detection.

Workers ARE basis vectors in classification space. A signal's spectrum is
its familiarity score vector across all workers. Agents observe cross-signal
patterns through their competency lens and decide what's worth reporting.

No hard-coded insight types. Agents reason; code provides tools and memory.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from stigmergy.primitives.signal import Signal
from stigmergy.structures.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from stigmergy.graph.graph import CommunicationGraph
    from stigmergy.mesh.mesh import MeshTrace
    from stigmergy.mesh.worker import WorkerNode


# ── Data structures ──────────────────────────────────────────


@dataclass
class Insight:
    """A single observation an agent considers worth reporting.

    The type is an agent-chosen string — NOT an enum. Agents decide
    what categories exist based on their competencies and role.
    """

    agent_id: UUID
    type: str
    summary: str
    confidence: float  # [0, 1]
    signal_ids: list[UUID] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedSignal:
    """A signal paired with its mesh trace and spectral decomposition."""

    signal: Signal
    trace: MeshTrace
    spectrum: list[float]  # familiarity scores across workers (ordered)
    worker_ids: list[UUID]  # worker ordering for spectrum alignment
    cached_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Spectral math ────────────────────────────────────────────


def spectral_cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two spectrum vectors.

    Returns 0.0 if either vector is zero-length or all-zero.
    Handles mismatched lengths by truncating to shorter.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0

    dot = sum(a[i] * b[i] for i in range(n))
    mag_a = math.sqrt(sum(x * x for x in a[:n]))
    mag_b = math.sqrt(sum(x * x for x in b[:n]))

    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0

    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


# ── Signal Cache ─────────────────────────────────────────────


class SignalCache:
    """In-memory ring buffer of recent signals with spectral decomposition.

    The cache is the system's short-term memory. Agents query it.
    It doesn't interpret — it stores and retrieves.
    """

    def __init__(self, capacity: int = 500) -> None:
        self._buffer: RingBuffer[CachedSignal] = RingBuffer(capacity)

    def add(
        self,
        signal: Signal,
        trace: MeshTrace,
        spectrum: list[float],
        worker_ids: list[UUID],
    ) -> None:
        """Add a signal with its spectral decomposition to the cache."""
        self._buffer.append(
            CachedSignal(
                signal=signal,
                trace=trace,
                spectrum=spectrum,
                worker_ids=worker_ids,
            )
        )

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._buffer.capacity

    def recent(self, n: int = 50) -> list[CachedSignal]:
        """Get the last n cached signals (most recent first)."""
        return self._buffer.last(n)

    def spectral_neighbors(
        self,
        spectrum: list[float],
        threshold: float = 0.7,
        limit: int = 20,
    ) -> list[tuple[CachedSignal, float]]:
        """Find cached signals with similar spectra (cosine similarity)."""
        results: list[tuple[CachedSignal, float]] = []
        for entry in self._buffer.last(len(self._buffer)):
            sim = spectral_cosine(spectrum, entry.spectrum)
            if sim >= threshold:
                results.append((entry, sim))
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def by_author(self, author: str, limit: int = 20) -> list[CachedSignal]:
        """Recent signals from a specific author (most recent first)."""
        results: list[CachedSignal] = []
        for entry in self._buffer.last(len(self._buffer)):
            if entry.signal.author == author:
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def in_window(
        self,
        hours: float = 24.0,
        reference_time: datetime | None = None,
    ) -> list[CachedSignal]:
        """Signals within a time window (most recent first).

        reference_time: the "now" for the window. During backfill, this
        should be the current signal's timestamp (not wall clock), so the
        correlator sees signals within N hours of the signal being processed.
        BUG FIX: previously used datetime.now(), which meant backfilled signals
        with past timestamps were never visible to the correlator.
        """
        ref = reference_time or datetime.now(timezone.utc)
        cutoff = ref - timedelta(hours=hours)
        results: list[CachedSignal] = []
        for entry in self._buffer.last(len(self._buffer)):
            if cutoff <= entry.signal.timestamp <= ref:
                results.append(entry)
        return results


# ── Observation Context ──────────────────────────────────────


class ObservationContext:
    """Read-only view of everything an agent can observe about a signal.

    Provides generic primitives — facts, not interpretations.
    Agents use these to reason about cross-signal patterns.
    """

    def __init__(
        self,
        signal: Signal,
        trace: MeshTrace,
        spectrum: list[float],
        cache: SignalCache,
        workers: list[WorkerNode],
        graph: CommunicationGraph | None = None,
    ) -> None:
        self.signal = signal
        self.trace = trace
        self.spectrum = spectrum
        self._cache = cache
        self._workers = {w.id: w for w in workers}
        self._worker_list = workers
        self._graph = graph

    def spectral_neighbors(
        self, threshold: float = 0.7, limit: int = 20
    ) -> list[tuple[CachedSignal, float]]:
        """Signals with similar worker-excitation patterns."""
        return self._cache.spectral_neighbors(self.spectrum, threshold, limit)

    def same_author_recent(self, limit: int = 20) -> list[CachedSignal]:
        """Recent signals from this signal's author."""
        return self._cache.by_author(self.signal.author, limit)

    def cross_source_similar(
        self, threshold: float = 0.5, limit: int = 20
    ) -> list[tuple[CachedSignal, float]]:
        """Spectrally similar signals from DIFFERENT sources."""
        neighbors = self._cache.spectral_neighbors(self.spectrum, threshold, limit * 2)
        results: list[tuple[CachedSignal, float]] = []
        for cached, sim in neighbors:
            if cached.signal.source != self.signal.source:
                results.append((cached, sim))
                if len(results) >= limit:
                    break
        return results

    def worker_scores(self) -> dict[UUID, float]:
        """This signal's familiarity score at each worker."""
        return dict(self.trace.familiarity_scores)

    def score_variance(self) -> float:
        """How differently workers rated this signal (stdev of scores)."""
        scores = list(self.trace.familiarity_scores.values())
        if len(scores) < 2:
            return 0.0
        return statistics.stdev(scores)

    def temporal_cluster(self, window_hours: float = 24.0) -> list[CachedSignal]:
        """Signals within a time window relative to this signal's timestamp.

        Uses the current signal's timestamp as reference (not wall clock)
        so backfilled signals with past timestamps are visible.
        """
        return self._cache.in_window(window_hours, reference_time=self.signal.timestamp)

    def worker_state(self, worker_id: UUID) -> WorkerNode | None:
        """Access a specific worker's accumulated state."""
        return self._workers.get(worker_id)

    def metadata(self, key: str, default: Any = None) -> Any:
        """Signal metadata access (status, files_changed, labels, etc.)."""
        return self.signal.metadata.get(key, default)

    @property
    def workers(self) -> list[WorkerNode]:
        """All workers in the mesh."""
        return self._worker_list

    @property
    def accepted_workers(self) -> set[UUID]:
        """Workers that accepted this signal."""
        return self.trace.accepted_workers

    @property
    def cache_size(self) -> int:
        """How many signals are in the cache."""
        return len(self._cache)

    # ── Communication graph queries ─────────────────────────────

    def bridge_distance(self, author_a: str, author_b: str) -> float:
        """Topological distance between two people in the communication graph.

        Returns float('inf') if no graph is attached or people are disconnected.
        """
        if self._graph is None:
            return float("inf")
        from stigmergy.graph.metrics import bridge_distance as _bridge_distance
        return _bridge_distance(self._graph, author_a, author_b)

    def author_constraint(self, author: str) -> float:
        """Burt's constraint for a signal author — high = embedded in clique.

        Returns 0.0 if no graph is attached.
        """
        if self._graph is None:
            return 0.0
        from stigmergy.graph.metrics import burt_constraint
        return burt_constraint(self._graph, author)
