"""Phase-lock detection (Paper Section 5.2).

"Topic-variance over time (is the framing of a known risk converging
across teams?)" — monitoring whether independent risks are synchronizing.

Tracks term-set variance across independent findings over time. When
variance decreases (framing converges) while underlying risks remain
unresolved, flags phase-lock.

Pure computation — no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from stigmergy.primitives.signal import extract_terms


@dataclass
class PhaseLockAlert:
    """Alert for converging risk framing (phase-lock)."""

    risk_topics: list[str]            # the converging risk topics
    framing_similarity: float         # Jaccard similarity between finding term sets
    finding_count: int                # number of findings in the cluster
    finding_hashes: list[str]         # related finding hashes
    action_correlation_avg: float     # average action-correlation for these topics
    severity: str                     # "warning" or "critical"


# ── Jaccard similarity ─────────────────────────────────────


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity between two term sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


# ── Core detection ─────────────────────────────────────────


def detect_phase_lock(
    findings: list[dict],
    *,
    similarity_threshold: float = 0.4,
    min_findings: int = 2,
    action_correlations: dict[str, float] | None = None,
) -> list[PhaseLockAlert]:
    """Detect phase-lock: converging risk framing across findings.

    Args:
        findings: list of dicts with keys: finding_hash, summary, type,
                  sf_score, actors, channels.
        similarity_threshold: minimum Jaccard similarity to cluster.
        min_findings: minimum findings in a cluster to trigger alert.
        action_correlations: optional mapping of term -> action-correlation
                            ratio from compute_action_correlations().

    Returns:
        List of PhaseLockAlert objects, sorted by severity.
    """
    if len(findings) < min_findings:
        return []

    action_correlations = action_correlations or {}

    # Extract term sets from each finding
    finding_terms: list[tuple[dict, frozenset[str]]] = []
    for f in findings:
        terms = extract_terms(f.get("summary", ""))
        if terms:
            finding_terms.append((f, frozenset(terms)))

    if len(finding_terms) < min_findings:
        return []

    # Build clusters of similar findings
    clusters: list[list[tuple[dict, frozenset[str]]]] = []
    used: set[int] = set()

    for i, (f_i, terms_i) in enumerate(finding_terms):
        if i in used:
            continue

        cluster = [(f_i, terms_i)]
        used.add(i)

        for j, (f_j, terms_j) in enumerate(finding_terms):
            if j in used:
                continue
            sim = _jaccard(terms_i, terms_j)
            if sim >= similarity_threshold:
                cluster.append((f_j, terms_j))
                used.add(j)

        if len(cluster) >= min_findings:
            clusters.append(cluster)

    # Generate alerts from clusters
    alerts: list[PhaseLockAlert] = []

    for cluster in clusters:
        # Compute average pairwise similarity
        sims = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                sims.append(_jaccard(cluster[i][1], cluster[j][1]))
        avg_sim = sum(sims) / len(sims) if sims else 0.0

        # Collect finding info
        hashes = [f.get("finding_hash", "") for f, _ in cluster]
        all_terms: set[str] = set()
        for _, terms in cluster:
            all_terms.update(terms)

        # Top terms as risk topics
        risk_topics = sorted(all_terms)[:5]

        # Action correlation for these terms
        ac_values = [
            action_correlations[t]
            for t in all_terms
            if t in action_correlations
        ]
        ac_avg = sum(ac_values) / len(ac_values) if ac_values else 0.5

        # Severity: high similarity + low action-correlation = critical
        if avg_sim > 0.6 and ac_avg < 0.2:
            severity = "critical"
        elif avg_sim > 0.5 or ac_avg < 0.1:
            severity = "warning"
        else:
            severity = "warning"

        alerts.append(PhaseLockAlert(
            risk_topics=risk_topics,
            framing_similarity=round(avg_sim, 4),
            finding_count=len(cluster),
            finding_hashes=hashes,
            action_correlation_avg=round(ac_avg, 4),
            severity=severity,
        ))

    # Sort: critical first, then by similarity descending
    severity_order = {"critical": 0, "warning": 1}
    alerts.sort(key=lambda a: (severity_order.get(a.severity, 2), -a.framing_similarity))
    return alerts
