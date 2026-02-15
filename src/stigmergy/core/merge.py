"""Context merge: detection and execution.

Merges contexts that have converged — sharing enough signal overlap and
assessment agreement that maintaining separate contexts adds no information.

Detection: Jaccard overlap > 0.7 AND assessment agreement > 0.8 (strict).
Execution: union vector stores, deduplicate agents, max energy, update topology.

All functions are pure — no side effects. The caller (mesh.py) applies
topology mutations atomically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ── Merge Detection ───────────────────────────────────────────────

class AssessmentSummary(BaseModel):
    """Assessment state for a single signal in a context."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    familiarity_score: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class MergeCandidate:
    """A pair of contexts that should be merged."""

    context_id_a: str
    context_id_b: str
    overlap: float
    agreement: float
    combined_score: float


@dataclass(frozen=True)
class MergeDetectionResult:
    """All merge candidates found in this cycle."""

    candidates: tuple[MergeCandidate, ...] = ()


def _compute_overlap(
    signals_a: frozenset[str],
    signals_b: frozenset[str],
) -> float:
    """Jaccard similarity between two signal ID sets."""
    if not signals_a and not signals_b:
        return 0.0
    intersection = signals_a & signals_b
    union = signals_a | signals_b
    return len(intersection) / len(union)


def _compute_agreement(
    assessments_a: dict[str, AssessmentSummary],
    assessments_b: dict[str, AssessmentSummary],
) -> float:
    """Assessment agreement on shared signals.

    Agreement per signal:
      (1.0 if accepted matches else 0.0) * (1.0 - abs(fam_a - fam_b))
    Overall: mean over shared signals.
    """
    shared_ids = set(assessments_a.keys()) & set(assessments_b.keys())
    if not shared_ids:
        return 0.0

    total = 0.0
    for sid in shared_ids:
        a = assessments_a[sid]
        b = assessments_b[sid]
        decision_match = 1.0 if a.accepted == b.accepted else 0.0
        score_similarity = 1.0 - abs(a.familiarity_score - b.familiarity_score)
        total += decision_match * score_similarity

    return total / len(shared_ids)


def detect_merge_candidates(
    signal_windows: dict[str, frozenset[str]],
    assessments: dict[str, dict[str, AssessmentSummary]],
    neighbors: dict[str, frozenset[str]],
    overlap_threshold: float = 0.7,
    agreement_threshold: float = 0.8,
) -> MergeDetectionResult:
    """Detect merge candidates among neighboring contexts.

    Pure function. Only checks neighbor pairs (not all-pairs).
    Each pair checked once (canonical ordering).
    """
    if not (0.0 <= overlap_threshold <= 1.0):
        raise ValueError(
            f"overlap_threshold must be between 0.0 and 1.0 inclusive, got {overlap_threshold}"
        )
    if not (0.0 <= agreement_threshold <= 1.0):
        raise ValueError(
            f"agreement_threshold must be between 0.0 and 1.0 inclusive, got {agreement_threshold}"
        )

    # Validate familiarity scores
    for ctx_id, ctx_assessments in assessments.items():
        for sig_id, summary in ctx_assessments.items():
            if not (0.0 <= summary.familiarity_score <= 1.0):
                raise ValueError(
                    f"familiarity_score must be between 0.0 and 1.0 inclusive "
                    f"for signal {sig_id} in context {ctx_id}"
                )

    # Track checked pairs to avoid duplicates
    checked: set[tuple[str, str]] = set()
    candidates: list[MergeCandidate] = []

    for ctx_id, neighbor_set in neighbors.items():
        for neighbor_id in neighbor_set:
            # Canonical ordering: a <= b
            pair = (min(ctx_id, neighbor_id), max(ctx_id, neighbor_id))
            if pair in checked:
                continue
            checked.add(pair)

            a_id, b_id = pair
            signals_a = signal_windows.get(a_id, frozenset())
            signals_b = signal_windows.get(b_id, frozenset())

            overlap = _compute_overlap(signals_a, signals_b)
            if overlap <= overlap_threshold:
                continue

            assessments_a = assessments.get(a_id, {})
            assessments_b = assessments.get(b_id, {})
            agreement = _compute_agreement(assessments_a, assessments_b)
            if agreement <= agreement_threshold:
                continue

            candidates.append(MergeCandidate(
                context_id_a=a_id,
                context_id_b=b_id,
                overlap=overlap,
                agreement=agreement,
                combined_score=overlap * agreement,
            ))

    # Sort by combined_score descending, break ties by IDs
    candidates.sort(key=lambda c: (-c.combined_score, c.context_id_a, c.context_id_b))

    return MergeDetectionResult(candidates=tuple(candidates))


# ── Merge Execution ───────────────────────────────────────────────

class MergeReason(StrEnum):
    SIGNAL_OVERLAP = "SIGNAL_OVERLAP"
    DECAY_HANDOFF = "DECAY_HANDOFF"


class TopologyAction(StrEnum):
    ADD_NEIGHBOR = "add_neighbor"
    REMOVE_NEIGHBOR = "remove_neighbor"


@dataclass(frozen=True)
class TopologyUpdate:
    """Atomic topology mutation command."""

    context_id: str
    action: TopologyAction
    target_id: str


class MergeError(Exception):
    """Error during merge execution."""

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.context = context


def resolve_surviving_identity(
    context_a_id: str,
    context_b_id: str,
    energy_a: float,
    energy_b: float,
) -> str:
    """Determine which context survives a merge.

    Higher energy wins. On tie, lexicographically smaller ID wins.
    """
    if context_a_id == context_b_id:
        raise MergeError(
            "Cannot resolve surviving identity for the same context.",
            context_id=context_a_id,
        )
    if energy_a > energy_b:
        return context_a_id
    elif energy_b > energy_a:
        return context_b_id
    else:
        return min(context_a_id, context_b_id)


def compute_topology_updates(
    surviving_id: str,
    removed_id: str,
    surviving_neighbor_ids: set[str],
    removed_neighbor_ids: set[str],
) -> list[TopologyUpdate]:
    """Compute topology update commands for post-merge cleanup.

    Pure function. Returns commands the caller must apply atomically.
    """
    if surviving_id == removed_id:
        raise MergeError(
            "Surviving and removed context IDs must be distinct.",
            context_id=surviving_id,
        )

    updates: list[TopologyUpdate] = []

    # Remove the removed context from the surviving context's neighbor list
    updates.append(TopologyUpdate(
        context_id=surviving_id,
        action=TopologyAction.REMOVE_NEIGHBOR,
        target_id=removed_id,
    ))

    # For each neighbor of the removed context (excluding the survivor):
    for neighbor_id in removed_neighbor_ids:
        if neighbor_id == surviving_id:
            continue

        # Remove the removed context from this neighbor's list
        updates.append(TopologyUpdate(
            context_id=neighbor_id,
            action=TopologyAction.REMOVE_NEIGHBOR,
            target_id=removed_id,
        ))

        # If this neighbor wasn't already connected to the survivor, connect them
        if neighbor_id not in surviving_neighbor_ids:
            updates.append(TopologyUpdate(
                context_id=neighbor_id,
                action=TopologyAction.ADD_NEIGHBOR,
                target_id=surviving_id,
            ))
            updates.append(TopologyUpdate(
                context_id=surviving_id,
                action=TopologyAction.ADD_NEIGHBOR,
                target_id=neighbor_id,
            ))

    return updates
