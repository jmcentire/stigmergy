"""Re-emission engine — event-driven retriggering of suppressed findings.

A finding that was previously surfaced and then suppressed (either because
P(knows) was too high, or importance too low, or it decayed) can be
re-emitted when context changes:

Triggers:
  (a) New signal strengthens coupling — new evidence makes a deferred/active
      finding more important than it was when last evaluated.
  (b) Related finding compounds risk — two findings about overlapping
      actors/channels/terms are individually below threshold but together
      represent a compound risk worth surfacing.
  (c) Decay stage advances — a finding moving from deferred → normalized
      deserves one last explicit call-out before it becomes organizational
      blindness.

This module is evaluated after the main surfacing pass. It produces
ReemissionEvent objects that the output layer renders distinctly from
regular findings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ReemissionEvent:
    """A finding that should be re-surfaced due to context change."""

    finding_hash: str
    summary: str
    finding_type: str
    trigger: str  # "new_evidence", "compound_risk", "decay_escalation"
    trigger_detail: str  # human-readable reason for re-emission
    original_score: float  # S(f) when first surfaced
    current_score: float  # S(f) now (may have increased)
    decay_stage: str  # current decay stage
    times_surfaced: int
    days_tracked: float
    actors: list[str] = field(default_factory=list)
    related_findings: list[str] = field(default_factory=list)  # hashes of compounding findings


def check_new_evidence(
    findings: list,
    signal_terms: frozenset[str],
    signal_source: str,
    signal_channel: str,
    score_increase_threshold: float = 0.1,
) -> list[ReemissionEvent]:
    """Check if new signals strengthen existing deferred/normalized findings.

    A finding is retriggered when:
      - It's in "deferred" or "normalized" stage
      - New signal terms overlap with the finding's summary terms (>= 2 matches)
      - The finding's S(f) score is above the increase threshold

    Args:
        findings: List of Finding objects from FindingRegistry
        signal_terms: Terms extracted from new signals this run
        signal_source: Source of the new signals (e.g. "github", "slack")
        signal_channel: Channel of the new signals
        score_increase_threshold: Minimum S(f) score to consider for re-emission
    """
    events: list[ReemissionEvent] = []

    for finding in findings:
        if finding.decay_stage not in ("deferred", "normalized"):
            continue
        if finding.max_score < score_increase_threshold:
            continue

        # Check term overlap between signal and finding summary
        summary_terms: set[str] = set()
        for word in finding.summary.lower().split():
            word = word.strip(".,!?;:()[]{}\"'*`<>#~")
            if len(word) >= 4:
                summary_terms.add(word)

        overlap = summary_terms & set(signal_terms)
        if len(overlap) < 2:
            continue

        events.append(ReemissionEvent(
            finding_hash=finding.finding_hash,
            summary=finding.summary,
            finding_type=finding.type,
            trigger="new_evidence",
            trigger_detail=(
                f"New {signal_source}/{signal_channel} signals match "
                f"{len(overlap)} terms: {', '.join(sorted(overlap)[:5])}"
            ),
            original_score=finding.max_score,
            current_score=finding.max_score,  # updated externally if re-scored
            decay_stage=finding.decay_stage,
            times_surfaced=finding.times_surfaced,
            days_tracked=finding.days_since_first,
            actors=finding.actors[:],
        ))

    return events


def check_compound_risk(
    findings: list,
    actor_threshold: int = 1,
    term_threshold: int = 2,
) -> list[ReemissionEvent]:
    """Check if multiple findings about overlapping actors/terms compound risk.

    Two findings compound when:
      - They share at least `actor_threshold` actors
      - OR they share at least `term_threshold` summary terms
      - At least one is in "deferred" or "normalized" stage
      - Together their scores suggest combined risk

    Returns one ReemissionEvent per compound cluster (not per pair).
    """
    if len(findings) < 2:
        return []

    # Build term sets for each finding
    finding_terms: dict[str, set[str]] = {}
    finding_actors: dict[str, set[str]] = {}
    for f in findings:
        terms: set[str] = set()
        for word in f.summary.lower().split():
            word = word.strip(".,!?;:()[]{}\"'*`<>#~")
            if len(word) >= 4:
                terms.add(word)
        finding_terms[f.finding_hash] = terms
        finding_actors[f.finding_hash] = {a.lower().strip() for a in f.actors}

    events: list[ReemissionEvent] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, f1 in enumerate(findings):
        for f2 in findings[i + 1:]:
            pair_key = tuple(sorted([f1.finding_hash, f2.finding_hash]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # At least one must be decaying
            if f1.decay_stage not in ("deferred", "normalized") and \
               f2.decay_stage not in ("deferred", "normalized"):
                continue

            actor_overlap = finding_actors[f1.finding_hash] & finding_actors[f2.finding_hash]
            term_overlap = finding_terms[f1.finding_hash] & finding_terms[f2.finding_hash]

            if len(actor_overlap) >= actor_threshold or len(term_overlap) >= term_threshold:
                # Compound risk detected
                compound_score = f1.max_score + f2.max_score
                # Choose the higher-score finding as the primary
                primary, secondary = (f1, f2) if f1.max_score >= f2.max_score else (f2, f1)

                detail_parts = []
                if actor_overlap:
                    detail_parts.append(f"shared actors: {', '.join(sorted(actor_overlap)[:3])}")
                if term_overlap:
                    detail_parts.append(f"shared terms: {', '.join(sorted(term_overlap)[:5])}")

                events.append(ReemissionEvent(
                    finding_hash=primary.finding_hash,
                    summary=primary.summary,
                    finding_type=primary.type,
                    trigger="compound_risk",
                    trigger_detail=(
                        f"Compounds with: \"{secondary.summary[:60]}\" — "
                        + ", ".join(detail_parts)
                    ),
                    original_score=primary.max_score,
                    current_score=compound_score,
                    decay_stage=primary.decay_stage,
                    times_surfaced=primary.times_surfaced,
                    days_tracked=primary.days_since_first,
                    actors=list(set(primary.actors + secondary.actors)),
                    related_findings=[secondary.finding_hash],
                ))

    return events


def check_decay_escalation(
    findings: list,
    min_score: float = 0.2,
) -> list[ReemissionEvent]:
    """Check for findings approaching or at Normalized Deviance.

    Produces a re-emission event for findings that:
      - Are in "deferred" stage with 2+ surfacings (approaching normalized)
      - OR are newly "normalized" (just crossed the threshold)
      - Have a meaningful S(f) score (above min_score)

    This is the system's last-chance alarm: if no one acts, this finding
    will become invisible — organizational blindness.
    """
    events: list[ReemissionEvent] = []

    for finding in findings:
        if finding.max_score < min_score:
            continue

        if finding.decay_stage == "normalized":
            events.append(ReemissionEvent(
                finding_hash=finding.finding_hash,
                summary=finding.summary,
                finding_type=finding.type,
                trigger="decay_escalation",
                trigger_detail=(
                    f"NORMALIZED DEVIANCE: surfaced {finding.times_surfaced}x "
                    f"over {finding.days_since_first:.0f} days without action. "
                    f"S(f)={finding.max_score:.3f}. Risk of organizational blindness."
                ),
                original_score=finding.max_score,
                current_score=finding.max_score,
                decay_stage=finding.decay_stage,
                times_surfaced=finding.times_surfaced,
                days_tracked=finding.days_since_first,
                actors=finding.actors[:],
            ))
        elif finding.decay_stage == "deferred" and finding.times_surfaced >= 2:
            # Approaching normalized — early warning
            events.append(ReemissionEvent(
                finding_hash=finding.finding_hash,
                summary=finding.summary,
                finding_type=finding.type,
                trigger="decay_escalation",
                trigger_detail=(
                    f"Approaching normalization: surfaced {finding.times_surfaced}x "
                    f"over {finding.days_since_first:.0f} days. "
                    f"Action needed to prevent organizational blindness."
                ),
                original_score=finding.max_score,
                current_score=finding.max_score,
                decay_stage=finding.decay_stage,
                times_surfaced=finding.times_surfaced,
                days_tracked=finding.days_since_first,
                actors=finding.actors[:],
            ))

    return events


def evaluate_reemissions(
    findings: list,
    signal_terms: frozenset[str] | None = None,
    signal_source: str = "",
    signal_channel: str = "",
) -> list[ReemissionEvent]:
    """Run all re-emission checks and return deduplicated events.

    This is the main entry point. Runs all three trigger checks and
    deduplicates by finding_hash (keeping the highest-priority trigger).

    Priority: decay_escalation > compound_risk > new_evidence
    """
    all_events: list[ReemissionEvent] = []

    # Check all triggers
    if signal_terms:
        all_events.extend(check_new_evidence(
            findings, signal_terms, signal_source, signal_channel,
        ))
    all_events.extend(check_compound_risk(findings))
    all_events.extend(check_decay_escalation(findings))

    # Deduplicate: keep highest-priority trigger per finding
    trigger_priority = {"decay_escalation": 3, "compound_risk": 2, "new_evidence": 1}
    best: dict[str, ReemissionEvent] = {}
    for event in all_events:
        existing = best.get(event.finding_hash)
        if existing is None:
            best[event.finding_hash] = event
        elif trigger_priority.get(event.trigger, 0) > trigger_priority.get(existing.trigger, 0):
            best[event.finding_hash] = event

    # Sort: decay_escalation first, then compound_risk, then new_evidence
    result = sorted(
        best.values(),
        key=lambda e: (-trigger_priority.get(e.trigger, 0), -e.current_score),
    )
    return result
