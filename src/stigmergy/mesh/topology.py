"""Topology: worker position model and spatial coordination.

Workers have positions in signal space derived from their bloom filters.
Position distance drives entry selection, peer recentering, and gap detection.

Workers are basis vectors in the spectral model — if they're all identical,
decomposition is meaningless. This module provides the primitives that let
competitive routing differentiate workers into distinct signal-space regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np

from stigmergy.structures.bloom import CountingBloomFilter


@dataclass(frozen=True)
class WorkerPosition:
    """A worker's position in signal space.

    Derived from the worker's bloom filter — a compressed fingerprint of
    everything the worker knows about. Two workers with similar positions
    know about similar topics.
    """

    worker_id: UUID
    bloom_digest: np.ndarray  # shape (bins,), float64
    top_terms: tuple[str, ...]  # top N most frequent terms
    signal_count: int
    fullness: float


def bloom_digest(bloom: CountingBloomFilter, bins: int = 128) -> np.ndarray:
    """Downsample a bloom filter's counters to a fixed-size digest.

    Groups consecutive counters and sums each group, producing a compact
    fingerprint suitable for fast cosine distance. A 96K-counter bloom
    becomes a 128-float vector.

    Returns a float64 array of shape (bins,).
    """
    counters = bloom._counters.astype(np.float64)
    size = len(counters)
    if size <= bins:
        # Bloom is smaller than target bins — pad with zeros
        result = np.zeros(bins, dtype=np.float64)
        result[:size] = counters
        return result

    # Split into `bins` groups and sum each
    # Handle non-evenly-divisible sizes by using array_split
    groups = np.array_split(counters, bins)
    return np.array([g.sum() for g in groups], dtype=np.float64)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def position_distance(a: WorkerPosition, b: WorkerPosition) -> float:
    """Distance between two worker positions in signal space.

    Returns 0.0 for identical positions, up to 1.0 for completely different.
    Based on cosine distance of bloom digests.
    """
    return 1.0 - _cosine_similarity(a.bloom_digest, b.bloom_digest)


def signal_position_distance(
    signal_bloom: CountingBloomFilter,
    worker_pos: WorkerPosition,
    bins: int = 128,
) -> float:
    """Distance from a signal to a worker position.

    Downsamples the signal's bloom to the same bin count as the worker
    position, then computes cosine distance.
    """
    sig_digest = bloom_digest(signal_bloom, bins=bins)
    return 1.0 - _cosine_similarity(sig_digest, worker_pos.bloom_digest)


def select_peers(
    worker_pos: WorkerPosition,
    all_positions: list[WorkerPosition],
    *,
    target: int = 6,
    min_peers: int = 4,
    max_peers: int = 8,
) -> list[UUID]:
    """Select peers that balance local routing with long-range exploration.

    Strategy:
    1. Closest 2 workers (local routing — efficient forwarding)
    2. Farthest 2 workers (long-range routing — exploration)
    3. Fill remaining slots with workers that maximize diversity
       (max min-distance to already-selected peers)

    Returns list of worker UUIDs to use as neighbors.
    """
    # Filter out self
    candidates = [p for p in all_positions if p.worker_id != worker_pos.worker_id]
    if not candidates:
        return []

    # Compute distances to all candidates
    distances = [(p, position_distance(worker_pos, p)) for p in candidates]
    distances.sort(key=lambda x: x[1])

    actual_target = min(target, len(candidates))
    actual_target = max(min(actual_target, max_peers), min(min(min_peers, len(candidates)), len(candidates)))

    selected: list[WorkerPosition] = []

    # Step 1: closest 2 (or fewer if not enough candidates)
    close_count = min(2, len(distances))
    for i in range(close_count):
        selected.append(distances[i][0])

    # Step 2: farthest 2 (that aren't already selected)
    remaining = [(p, d) for p, d in distances if p not in selected]
    remaining.sort(key=lambda x: x[1], reverse=True)
    far_count = min(2, len(remaining))
    for i in range(far_count):
        selected.append(remaining[i][0])

    # Step 3: fill remaining with diversity-maximizing picks
    remaining = [(p, d) for p, d in distances if p not in selected]
    while len(selected) < actual_target and remaining:
        # For each remaining candidate, compute min distance to any already-selected peer
        best_candidate = None
        best_min_dist = -1.0
        for p, _ in remaining:
            min_dist = min(position_distance(p, s) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = p
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining = [(p, d) for p, d in remaining if p is not best_candidate]
        else:
            break

    return [p.worker_id for p in selected]


def detect_gap(
    scores: dict[UUID, float],
    threshold: float = 0.08,
) -> bool:
    """Detect if a signal falls in uncovered territory.

    Returns True when ALL visited workers scored below the gap threshold —
    no worker specializes in this signal's domain.
    """
    if not scores:
        return True
    return all(score < threshold for score in scores.values())
