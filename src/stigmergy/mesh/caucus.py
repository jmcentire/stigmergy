"""Caucus phase: seed all workers before normal BFS routing begins.

After warm restart, signal_count resets to 0 for all workers (by design).
This creates a degenerate case: the first signal makes one worker "warm"
(signal_count=1) and that worker wins entry selection for every subsequent
signal via nearest-warm-worker. Other workers starve. The mesh collapses
to a single-pipe classifier — no spectral differentiation.

The caucus fixes this by buffering the first N signals, scoring all M
workers against all N signals (reusing familiarity()), assigning each
signal to its highest-affinity worker, and seeding worker state (terms,
bloom, counts) so that normal BFS routing sees multiple differentiated
warm workers from the start.

Key property: NO double-counting. Caucus seeds terms/bloom/counts only.
Signals are NOT added to the dedup index during seeding. When they later
go through mesh.ingest(), dedup passes (no prior fingerprint) and they
route normally — but now through a mesh with multiple warm workers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from stigmergy.core.familiarity import FamiliarityWeights, familiarity
from stigmergy.mesh.topology import _cosine_similarity, bloom_digest

if TYPE_CHECKING:
    from stigmergy.mesh.mesh import Mesh
    from stigmergy.mesh.worker import WorkerNode
    from stigmergy.primitives.signal import Signal

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────


@dataclass
class RunStats:
    """Statistics from a previous run, used to project volume."""

    signals: int
    duration_seconds: float
    workers: int
    timestamp_iso: str


@dataclass
class CaucusConfig:
    """Tuning knobs for the caucus phase."""

    min_caucus: int = 3  # Floor: always buffer at least 3
    min_per_worker: int = 2  # Each worker needs at least 2 signals for differentiation
    max_caucus: int = 50  # Ceiling: never buffer more than 50
    fraction: float = 0.10  # Buffer 10% of projected volume
    underload_factor: float = 10.0  # < 10 signals/worker → suggest merge
    overload_factor: float = 0.5  # > 50% projected fullness → suggest fork


@dataclass
class CaucusResult:
    """Output of the caucus phase."""

    assignments: dict[int, int]  # signal_idx → worker_idx
    activation_matrix: np.ndarray  # (M, N) shape
    suggested_worker_count: int | None  # None = no change
    merge_pairs: list[tuple[int, int]]  # worker pairs to merge
    fork_worker_idx: int | None  # worker to fork
    caucus_size: int
    projected_volume: int


# ── Pure functions ─────────────────────────────────────────────


def project_volume(prev_stats: RunStats | None, elapsed_seconds: float) -> int:
    """Project how many signals this run will process.

    Uses ratio projection: if the last run processed S signals in D seconds,
    and E seconds have elapsed since then, we expect roughly S * (E / D).

    Returns 0 if no previous stats or zero duration.
    """
    if prev_stats is None:
        return 0
    if prev_stats.duration_seconds <= 0:
        return 0
    if elapsed_seconds <= 0:
        return 0
    ratio = elapsed_seconds / prev_stats.duration_seconds
    return max(0, int(prev_stats.signals * ratio))


def compute_caucus_size(projected: int, config: CaucusConfig | None = None) -> int:
    """Compute how many signals to buffer for the caucus phase.

    Bounded by min_caucus and max_caucus from config.
    """
    cfg = config or CaucusConfig()
    if projected <= 0:
        return cfg.min_caucus
    raw = int(projected * cfg.fraction)
    return max(cfg.min_caucus, min(raw, cfg.max_caucus))


def build_activation_matrix(
    signals: list[Signal],
    workers: list[WorkerNode],
    weights: FamiliarityWeights | None = None,
) -> np.ndarray:
    """Build an M×N activation matrix: each worker scores each signal.

    M = len(workers), N = len(signals).
    Uses familiarity() — the same ART match function used in routing.
    """
    w = weights or FamiliarityWeights()
    m = len(workers)
    n = len(signals)
    matrix = np.zeros((m, n), dtype=np.float64)

    for j, sig in enumerate(signals):
        for i, worker in enumerate(workers):
            matrix[i, j] = familiarity(sig, worker.context, weights=w)

    return matrix


def negotiate_assignments(
    matrix: np.ndarray,
    workers: list[WorkerNode],
    config: CaucusConfig | None = None,
) -> dict[int, int]:
    """Assign each signal to the worker with highest affinity.

    Tiebreak strategy (critical for cold start):
    1. Highest activation score wins outright
    2. On tie: prefer worker with fewest assignments so far (spread signals)
    3. On further tie: highest rolling_avg_familiarity (historical specialization)

    The "fewest assignments" tiebreak is what prevents starvation: when all
    workers are empty and score identically, signals distribute round-robin
    instead of collapsing to one worker.

    Returns dict mapping signal_idx → worker_idx.
    """
    _config = config or CaucusConfig()
    m, n = matrix.shape
    assignments: dict[int, int] = {}
    # Track how many signals each worker has been assigned (for balancing)
    assignment_counts: list[int] = [0] * m

    for j in range(n):
        col = matrix[:, j]
        max_score = col.max()
        # Find all workers with the max score (ties)
        candidates = [i for i in range(m) if col[i] == max_score]
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # Tiebreak 1: fewest assignments so far (spread signals)
            min_assigned = min(assignment_counts[i] for i in candidates)
            balanced = [i for i in candidates if assignment_counts[i] == min_assigned]
            if len(balanced) == 1:
                winner = balanced[0]
            else:
                # Tiebreak 2: highest rolling_avg_familiarity
                winner = max(
                    balanced,
                    key=lambda i: workers[i].rolling_avg_familiarity,
                )
        assignments[j] = winner
        assignment_counts[winner] += 1

    return assignments


def suggest_worker_count(
    matrix: np.ndarray,
    projected_volume: int,
    current_count: int,
    worker_capacity: int,
    config: CaucusConfig | None = None,
) -> tuple[int | None, list[tuple[int, int]], int | None]:
    """Suggest scaling: merge underloaded workers, fork overloaded ones.

    Returns (suggested_count, merge_pairs, fork_idx).
    All None/empty if no change needed.
    """
    cfg = config or CaucusConfig()
    m = matrix.shape[0]

    if m < 2:
        return None, [], None

    merge_pairs: list[tuple[int, int]] = []
    fork_idx: int | None = None

    # Check for underload: if projected signals per worker is very low
    if projected_volume > 0:
        per_worker = projected_volume / current_count
        if per_worker < cfg.underload_factor and m > 1:
            # Find most similar worker pair via cosine similarity of activation rows
            best_sim = -1.0
            best_pair: tuple[int, int] | None = None
            for i in range(m):
                for j in range(i + 1, m):
                    sim = _cosine_similarity(matrix[i], matrix[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)
            if best_pair is not None and best_sim > 0.8:
                merge_pairs.append(best_pair)

    # Check for overload: if projected signals push a worker past capacity threshold
    if projected_volume > 0 and worker_capacity > 0:
        projected_per_worker = projected_volume / max(current_count, 1)
        if projected_per_worker > worker_capacity * cfg.overload_factor:
            # Find worker with most diverse activation (highest kurtosis/spread)
            variances = [matrix[i].var() for i in range(m)]
            fork_idx = int(np.argmax(variances))

    suggested: int | None = None
    if merge_pairs:
        suggested = current_count - len(merge_pairs)
    elif fork_idx is not None:
        suggested = current_count + 1

    return suggested, merge_pairs, fork_idx


def _term_jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two term sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_diverse_seeds(signals: list[Signal], count: int) -> list[int]:
    """Select `count` maximally diverse signal indices from the list.

    Uses greedy farthest-first traversal on term Jaccard distance.
    Ensures phase 1 seeds create maximum vocabulary differentiation
    regardless of chronological ordering.

    Returns list of signal indices into the original list.
    """
    n = len(signals)
    if count >= n:
        return list(range(n))
    if count <= 0:
        return []

    # Precompute term sets
    term_sets = [sig.terms for sig in signals]

    # Start with signal 0
    selected = [0]

    for _ in range(count - 1):
        best_idx = -1
        best_min_dist = -1.0
        for i in range(n):
            if i in selected:
                continue
            # Min distance to any already-selected signal
            min_dist = min(
                1.0 - _term_jaccard(term_sets[i], term_sets[j])
                for j in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx >= 0:
            selected.append(best_idx)

    return selected


def execute_caucus(
    signals: list[Signal],
    assignments: dict[int, int],
    workers: list[WorkerNode],
    matrix: np.ndarray,
) -> None:
    """Seed workers with term/bloom data from assigned signals.

    This is the single side-effect boundary in the caucus module.
    Seeds ONLY terms and bloom filters for position differentiation.
    Does NOT increment signal_count per signal, source_counts, or
    author_counts — those are handled by mesh.ingest() when signals
    are processed normally. This prevents double-counting.

    After seeding, each assigned worker gets signal_count set to 1
    (if it was 0) — just enough for _select_entry_worker to consider
    it "warm". The actual signal_count comes from ingest.

    Does NOT add signals to dedup index. Does NOT call context.ingest_signal()
    (which would add to vector store).
    """
    assigned_worker_indices: set[int] = set()

    for sig_idx, worker_idx in assignments.items():
        if sig_idx >= len(signals) or worker_idx >= len(workers):
            continue

        sig = signals[sig_idx]
        worker = workers[worker_idx]
        ctx = worker.context

        # Seed terms only (exact set + bloom filter)
        # Terms are set-based — adding existing terms is a no-op.
        # Bloom add_many is idempotent for membership queries.
        terms = sig.terms
        ctx.terms.update(terms)
        ctx.term_bloom.add_many(terms)

        # Invalidate position cache so position reflects new terms
        worker._position_dirty = True

        # Update rolling average familiarity from the activation matrix
        score = float(matrix[worker_idx, sig_idx])
        worker._update_familiarity_ema(score)

        assigned_worker_indices.add(worker_idx)

    # Make all assigned workers minimally warm (signal_count = 1).
    # This is the minimum needed for _select_entry_worker to include
    # them in the warm_workers list. Actual signal_count accumulates
    # during mesh.ingest() without double-counting.
    for w_idx in assigned_worker_indices:
        w = workers[w_idx]
        if w.context.signal_count == 0:
            w.context.signal_count = 1


# ── Orchestrator ───────────────────────────────────────────────


class Caucus:
    """Orchestrates the caucus phase between state restoration and normal routing.

    Usage:
        caucus = Caucus(mesh, weights, config, prev_stats, elapsed)
        size = caucus.compute_size(len(all_signals))
        if size > 0:
            result = caucus.run(signals[:size])
    """

    def __init__(
        self,
        mesh: Mesh,
        weights: FamiliarityWeights | None = None,
        config: CaucusConfig | None = None,
        prev_stats: RunStats | None = None,
        elapsed_seconds: float = 0.0,
    ) -> None:
        self._mesh = mesh
        self._weights = weights or FamiliarityWeights()
        self._config = config or CaucusConfig()
        self._prev_stats = prev_stats
        self._elapsed = elapsed_seconds

    def compute_size(self, actual_signal_count: int) -> int:
        """Compute caucus size, capped by actual available signals.

        Ensures each worker gets at least min_per_worker signals so that
        bloom filter differentiation is meaningful. Without this, workers
        seeded with 1 signal can't compete for position distance against
        workers that accumulate terms during normal routing.
        """
        projected = project_volume(self._prev_stats, self._elapsed)
        # If we have actual signal count, use the larger of projected vs actual
        # (projection may underestimate for first run after long gap)
        effective = max(projected, actual_signal_count)
        size = compute_caucus_size(effective, self._config)
        # Scale with worker count: each worker needs enough signals to
        # build a distinct vocabulary. Without this, the gravity well
        # effect dominates — the first worker to accept gains terms,
        # attracts more signals, gains more terms.
        worker_floor = self._mesh.worker_count * self._config.min_per_worker
        size = max(size, worker_floor)
        return min(size, actual_signal_count)

    def _are_workers_cold(self, workers: list[WorkerNode]) -> bool:
        """Check if all workers lack terms (cold start)."""
        return all(len(w.context.terms) == 0 for w in workers)

    def run(self, signals: list[Signal]) -> CaucusResult:
        """Execute the full caucus phase.

        Two-phase negotiation for cold starts:

        When all workers are empty, the activation matrix is uniform — every
        worker scores identically on every signal. Round-robin distributes
        signals evenly but RANDOMLY, so each worker gets a cross-section of
        all topics. All workers become generalists with similar bloom profiles.
        Position distance can't differentiate them, and the gravity well forms.

        Phase 1 (seed): Round-robin assigns 1 signal per worker and immediately
        seeds terms. Now workers have distinct vocabularies from different signals.

        Phase 2 (cluster): Rebuilds the activation matrix with differentiated
        workers. Signals now score differently against each worker, so
        negotiate_assignments clusters related signals on the same worker.
        Topic-based specialization emerges.

        When workers already have terms (warm restart with context restoration),
        single-phase assignment works fine since the matrix already differentiates.
        """
        workers = self._mesh.workers
        if not workers or not signals:
            return CaucusResult(
                assignments={},
                activation_matrix=np.zeros((0, 0)),
                suggested_worker_count=None,
                merge_pairs=[],
                fork_worker_idx=None,
                caucus_size=0,
                projected_volume=0,
            )

        projected = project_volume(self._prev_stats, self._elapsed)
        m = len(workers)

        if self._are_workers_cold(workers) and len(signals) > m:
            # Phase 1: pick M maximally diverse signals and assign one per worker.
            # Greedy farthest-first ensures each worker gets a distinct vocabulary
            # regardless of chronological ordering (e.g., if signals 1 and 2 are
            # both about booking sync, only one gets selected for seeding).
            seed_indices = select_diverse_seeds(signals, m)
            phase1_assignments = {
                seed_indices[i]: i for i in range(len(seed_indices))
            }
            phase1_seeded = set(seed_indices)
            phase1_matrix = build_activation_matrix(signals, workers, self._weights)
            execute_caucus(signals, phase1_assignments, workers, phase1_matrix)

            # Phase 2: rebuild matrix with now-differentiated workers
            matrix = build_activation_matrix(signals, workers, self._weights)
            assignments = negotiate_assignments(matrix, workers, self._config)

            # Phase 1 already seeded the diverse seeds. Only seed the remaining
            # signals from phase 2 to avoid double-counting terms.
            phase2_assignments = {
                k: v for k, v in assignments.items() if k not in phase1_seeded
            }
            execute_caucus(signals, phase2_assignments, workers, matrix)
        else:
            # Warm workers or few signals — single-phase works fine
            matrix = build_activation_matrix(signals, workers, self._weights)
            assignments = negotiate_assignments(matrix, workers, self._config)
            execute_caucus(signals, assignments, workers, matrix)

        suggested, merges, fork = suggest_worker_count(
            matrix,
            projected_volume=max(projected, len(signals)),
            current_count=len(workers),
            worker_capacity=self._mesh._worker_capacity,
            config=self._config,
        )

        # Log distribution
        worker_signal_counts: dict[int, int] = {}
        for w_idx in assignments.values():
            worker_signal_counts[w_idx] = worker_signal_counts.get(w_idx, 0) + 1
        for w_idx, count in sorted(worker_signal_counts.items()):
            w = workers[w_idx]
            logger.info(
                "  Caucus: worker %s seeded with %d signals (%d terms)",
                str(w.id)[:8],
                count,
                len(w.context.terms),
            )

        return CaucusResult(
            assignments=assignments,
            activation_matrix=matrix,
            suggested_worker_count=suggested,
            merge_pairs=merges,
            fork_worker_idx=fork,
            caucus_size=len(signals),
            projected_volume=projected,
        )
