"""Portfolio risk — compound scoring across related findings.

Individual findings may each fall below the surfacing threshold, but when
multiple findings share actors, channels, or topic terms, their collective
risk compounds. A cluster of four "medium" findings about the same service
is often more actionable than a single "high" finding in isolation.

The portfolio model:
  1. Clusters findings by shared actors, channels, and terms
  2. Computes a compound score for each cluster
  3. Escalates clusters that exceed threshold even if no individual finding does

Compound score:
    cluster_score = max(individual_scores) + sum(other_scores) * decay_factor

Where decay_factor (0.3 default) prevents runaway scores while still
rewarding convergent evidence. A cluster of 4 findings at 0.3 each:
    0.3 + (0.3 + 0.3 + 0.3) * 0.3 = 0.3 + 0.27 = 0.57

This surfaces when any individual 0.3 would not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RiskCluster:
    """A cluster of related findings with compound risk."""

    cluster_id: str  # derived from primary finding hash
    findings: list[dict] = field(default_factory=list)  # finding dicts
    shared_actors: list[str] = field(default_factory=list)
    shared_terms: list[str] = field(default_factory=list)
    shared_channels: list[str] = field(default_factory=list)
    compound_score: float = 0.0
    max_individual_score: float = 0.0
    escalated: bool = False  # True if cluster exceeds threshold but no individual does


def _extract_terms(text: str) -> set[str]:
    """Extract meaningful terms from text for overlap checking."""
    terms: set[str] = set()
    for word in text.lower().split():
        word = word.strip(".,!?;:()[]{}\"'*`<>#~")
        if len(word) >= 4:
            terms.add(word)
    return terms


def _similarity(f1: dict, f2: dict) -> tuple[set[str], set[str], set[str]]:
    """Compute shared actors, terms, and channels between two findings."""
    actors1 = {a.lower().strip() for a in f1.get("actors", [])}
    actors2 = {a.lower().strip() for a in f2.get("actors", [])}
    shared_actors = actors1 & actors2

    terms1 = _extract_terms(f1.get("summary", ""))
    terms2 = _extract_terms(f2.get("summary", ""))
    shared_terms = terms1 & terms2

    channels1 = set(f1.get("channels", []))
    channels2 = set(f2.get("channels", []))
    shared_channels = channels1 & channels2

    return shared_actors, shared_terms, shared_channels


def cluster_findings(
    findings: list[dict],
    actor_threshold: int = 1,
    term_threshold: int = 2,
) -> list[RiskCluster]:
    """Cluster findings by shared actors, terms, or channels.

    Uses single-linkage clustering: if finding A relates to B and B relates
    to C, all three end up in the same cluster.

    Args:
        findings: List of finding dicts with keys:
            - finding_hash: str
            - summary: str
            - sf_score: float
            - actors: list[str]
            - channels: list[str] (optional)
        actor_threshold: Min shared actors to consider related
        term_threshold: Min shared terms to consider related
    """
    if len(findings) < 2:
        return []

    n = len(findings)
    # Union-find for single-linkage clustering
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build adjacency
    for i in range(n):
        for j in range(i + 1, n):
            shared_actors, shared_terms, shared_channels = _similarity(
                findings[i], findings[j],
            )
            if (len(shared_actors) >= actor_threshold or
                    len(shared_terms) >= term_threshold or
                    len(shared_channels) >= 1):
                union(i, j)

    # Group by cluster root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    clusters: list[RiskCluster] = []
    for root, indices in groups.items():
        if len(indices) < 2:
            continue  # Single findings are not clusters

        cluster_findings_list = [findings[i] for i in indices]

        # Compute shared attributes across all findings in the cluster
        all_actors: set[str] = set()
        all_terms: set[str] = set()
        all_channels: set[str] = set()
        for i in indices:
            for j in indices:
                if i < j:
                    sa, st, sc = _similarity(findings[i], findings[j])
                    all_actors.update(sa)
                    all_terms.update(st)
                    all_channels.update(sc)

        cluster = RiskCluster(
            cluster_id=findings[root].get("finding_hash", str(root)),
            findings=cluster_findings_list,
            shared_actors=sorted(all_actors),
            shared_terms=sorted(all_terms),
            shared_channels=sorted(all_channels),
        )
        clusters.append(cluster)

    return clusters


def score_clusters(
    clusters: list[RiskCluster],
    decay_factor: float = 0.3,
    escalation_threshold: float = 0.4,
) -> list[RiskCluster]:
    """Compute compound scores for clusters and mark escalated ones.

    Compound score = max(scores) + sum(other_scores) * decay_factor

    A cluster is "escalated" if its compound score exceeds the threshold
    but no individual finding's score does.

    Args:
        clusters: List of RiskCluster objects (findings already grouped)
        decay_factor: Weight for non-primary findings (default 0.3)
        escalation_threshold: Score above which a cluster is surfaced
    """
    for cluster in clusters:
        scores = sorted(
            [f.get("sf_score", 0.0) for f in cluster.findings],
            reverse=True,
        )
        if not scores:
            continue

        max_score = scores[0]
        other_sum = sum(scores[1:])
        cluster.compound_score = max_score + other_sum * decay_factor
        cluster.max_individual_score = max_score

        # Escalate if compound exceeds threshold but max individual doesn't
        cluster.escalated = (
            cluster.compound_score >= escalation_threshold
            and max_score < escalation_threshold
        )

    return clusters


def build_portfolio(
    findings: list[dict],
    actor_threshold: int = 1,
    term_threshold: int = 2,
    decay_factor: float = 0.3,
    escalation_threshold: float = 0.4,
) -> list[RiskCluster]:
    """Full pipeline: cluster → score → filter.

    Returns only clusters with 2+ findings, sorted by compound score.
    """
    clusters = cluster_findings(findings, actor_threshold, term_threshold)
    clusters = score_clusters(clusters, decay_factor, escalation_threshold)
    clusters.sort(key=lambda c: c.compound_score, reverse=True)
    return clusters
