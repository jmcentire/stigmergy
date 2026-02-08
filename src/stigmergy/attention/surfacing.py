"""Surfacing decisions — the loss function that determines what to show.

surface_value(finding, person) = importance * P(doesn't_know)
                                - annoyance * P(already_knows)

Surface when positive. Rank by margin.

Key asymmetry: very important things surface even at moderate P(knows),
because the cost of not-knowing exceeds annoyance. Trivial things only
surface near zero P(knows).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from stigmergy.attention.model import AttentionModel


@dataclass
class SurfacingDecision:
    """Result of the surfacing loss function for one finding+person pair."""

    finding_hash: str
    finding_summary: str
    finding_type: str
    person_id: str
    importance: float  # S(f) score
    p_knows: float  # P(already_knows)
    surface_value: float  # importance * P(!knows) - annoyance * P(knows)
    margin: float  # same as surface_value (alias for clarity)
    contributing_signals: int  # how many of the finding's signals the person saw
    total_signals: int  # total signals contributing to the finding
    exposure_summary: str  # human-readable summary of how they were exposed


def compute_surface_value(
    importance: float,
    p_knows: float,
    importance_weight: float = 1.0,
    annoyance_weight: float = 0.3,
) -> float:
    """The core surfacing loss function.

    Returns:
        positive = surface (benefit outweighs annoyance)
        negative = suppress (annoyance outweighs benefit)
    """
    p_doesnt_know = 1.0 - p_knows
    benefit = importance_weight * importance * p_doesnt_know
    cost = annoyance_weight * p_knows
    return benefit - cost


def rank_for_person(
    findings: list[dict],
    person_id: str,
    model: AttentionModel,
    now: datetime | None = None,
) -> list[SurfacingDecision]:
    """Rank findings for a specific person by surfacing value.

    Args:
        findings: list of dicts with keys:
            - finding_hash: str
            - summary: str
            - type: str
            - sf_score: float (importance from scoring.py)
            - terms: frozenset[str] (finding topic terms)
            - signal_ids: list[str] (contributing signal IDs)
        person_id: canonical person ID
        model: the AttentionModel with accumulated exposures
        now: current time (for temporal decay)

    Returns:
        List of SurfacingDecision sorted by margin (highest first).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    params = model.params
    decisions: list[SurfacingDecision] = []

    person_state = model.get_state(person_id)

    for finding in findings:
        finding_hash = finding.get("finding_hash", "")
        summary = finding.get("summary", "")
        finding_type = finding.get("type", "")
        importance = finding.get("sf_score", 0.0)
        terms = finding.get("terms", frozenset())
        signal_ids = set(finding.get("signal_ids", []))

        p_knows = model.p_knows(person_id, terms, now)
        # C_redundant adapts to the person's false-positive rate —
        # not a static 0.3 but a function of how often they already knew.
        effective_annoyance = model.effective_annoyance_weight(person_id)
        sv = compute_surface_value(
            importance, p_knows,
            params.importance_weight,
            effective_annoyance,
        )

        # Count how many contributing signals the person was exposed to
        contributing = 0
        total = len(signal_ids)
        exposure_parts: list[str] = []

        if person_state is not None:
            seen_signal_ids = {e.signal_id for e in person_state.exposures}
            contributing = len(signal_ids & seen_signal_ids)

            # Build human-readable exposure summary
            if contributing > 0:
                # Find the most recent relevant exposure
                relevant = [
                    e for e in person_state.exposures
                    if e.signal_id in signal_ids
                ]
                if relevant:
                    latest = max(relevant, key=lambda e: e.timestamp)
                    age_days = (now - latest.timestamp).total_seconds() / 86400
                    source_label = f"{latest.source}/{latest.channel}"
                    if age_days < 1:
                        age_str = "today"
                    elif age_days < 2:
                        age_str = "yesterday"
                    else:
                        age_str = f"{age_days:.0f} days ago"
                    exposure_parts.append(
                        f"You saw {contributing}/{total} contributing signals "
                        f"({latest.exposure_type} in {source_label}, {age_str})"
                    )

        if not exposure_parts:
            exposure_summary = f"No direct exposure to {total} contributing signals"
        else:
            exposure_summary = "; ".join(exposure_parts)

        decisions.append(SurfacingDecision(
            finding_hash=finding_hash,
            finding_summary=summary,
            finding_type=finding_type,
            person_id=person_id,
            importance=importance,
            p_knows=p_knows,
            surface_value=sv,
            margin=sv,
            contributing_signals=contributing,
            total_signals=total,
            exposure_summary=exposure_summary,
        ))

    # Sort by margin (highest surface value first)
    decisions.sort(key=lambda d: d.margin, reverse=True)
    return decisions
