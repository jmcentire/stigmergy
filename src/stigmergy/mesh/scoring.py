"""S(f) scoring function — Equation 3 from the paper.

S(f) = d_bridge(f) × c_entity(f) × r_coupling(f) × γ(f)

Four multiplicative components, each a necessary condition:
- d_bridge: topological bridge distance — how far outside known subspace K
- c_entity: entity resolution confidence — can we trust the identities?
- r_coupling: risk coupling — proximity to control surfaces
- γ: communicability — can the finding be transmitted with useful precision?

Multiplicative form implements logical conjunction: every component must be
nonzero for the finding to warrant attention. A perfectly communicated
finding about a resolved dependency is worthless.

The components are topological, not sentiment-based. The only way to
change the score is to change the underlying structure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stigmergy.graph.graph import CommunicationGraph
    from stigmergy.mesh.insights import Insight

logger = logging.getLogger(__name__)


@dataclass
class ScoringComponents:
    """Individual S(f) components for a finding."""

    d_bridge: float  # [0, 1] normalized bridge distance
    c_entity: float  # [0, 1] entity resolution confidence
    r_coupling: float  # [0, 1] risk coupling
    gamma: float  # [0, 1] communicability
    score: float  # product of all four

    def to_dict(self) -> dict[str, float]:
        return {
            "d_bridge": round(self.d_bridge, 4),
            "c_entity": round(self.c_entity, 4),
            "r_coupling": round(self.r_coupling, 4),
            "gamma": round(self.gamma, 4),
            "score": round(self.score, 4),
        }


def _sigmoid(x: float) -> float:
    """Standard sigmoid σ(x) = 1 / (1 + e^(-x))."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_bridge_distance(
    insight: Insight,
    graph: CommunicationGraph | None,
) -> float:
    """d_bridge: structural hole distance via Burt's effective size.

    High d_bridge = finding connects actors who span structural holes
    (non-redundant, diverse contact networks). This is more nuanced than
    raw BFS hop count — two people 2 hops apart in a dense clique score
    low, while 2 hops across a structural hole scores high.

    Uses Burt's effective size: high effective size means the actor has
    many non-redundant contacts (bridges multiple groups). When both actors
    have high effective size AND are distant in BFS, the finding connects
    truly remote organizational regions.

    Normalized to [0, 1] via sigmoid.
    """
    if graph is None:
        return 0.5  # Unknown — moderate prior

    actors = insight.details.get("actors", [])
    if len(actors) < 2:
        return 0.5  # Single actor — no bridge to measure

    from stigmergy.graph.metrics import bridge_distance, effective_size

    # Compute BFS distance between actor pairs
    distances: list[float] = []
    for i, a in enumerate(actors):
        for b in actors[i + 1:]:
            d = bridge_distance(graph, a, b)
            if d != float("inf"):
                distances.append(d)

    if not distances:
        # Actors disconnected in graph — maximum bridge distance
        return 0.95

    # Compute effective size for each actor (Burt's structural holes)
    eff_sizes: list[float] = []
    for actor in actors:
        es = effective_size(graph, actor)
        eff_sizes.append(es)

    max_dist = max(distances)
    # Average effective size across actors — higher means they each
    # bridge more structural holes
    avg_eff_size = sum(eff_sizes) / len(eff_sizes) if eff_sizes else 1.0

    # Combine: BFS distance weighted by structural diversity.
    # Effective size > 1 amplifies distance (actors bridge holes).
    # Effective size ≤ 1 dampens distance (actors in same clique).
    # Log scale on effective size to prevent extreme amplification.
    import math
    structural_weight = math.log1p(avg_eff_size)  # log(1 + eff_size)
    combined = max_dist * structural_weight

    # Normalize: sigmoid centered at 2, scaled so:
    #   combined ≈ 1 → ~0.27, combined ≈ 2 → ~0.5, combined ≈ 4 → ~0.88
    return _sigmoid(combined - 2)


def compute_entity_confidence(
    insight: Insight,
    graph: CommunicationGraph | None,
) -> float:
    """c_entity: entity resolution confidence.

    How confident are we that the actors in the insight are correctly
    resolved to real organizational entities?

    Based on: whether actors are in the identity resolver, whether
    they have multiple confirmed aliases, whether they appear in
    the communication graph with established edges.
    """
    actors = insight.details.get("actors", [])
    if not actors:
        return 0.3  # No actors identified — low confidence

    if graph is None:
        return 0.5  # No graph — moderate prior

    resolved_count = 0
    for actor in actors:
        canonical = graph.resolver.resolve(actor)
        # If resolver returns a different canonical ID, the actor was resolved
        if canonical != actor:
            resolved_count += 1
        # If actor has edges in graph, they're a known entity
        elif graph.get_node(canonical) is not None:
            resolved_count += 1

    if len(actors) == 0:
        return 0.3

    ratio = resolved_count / len(actors)
    # Scale: all resolved → 0.95, none → 0.3
    return 0.3 + 0.65 * ratio


def compute_risk_coupling(insight: Insight) -> float:
    """r_coupling: proximity to control surfaces.

    Measures whether the finding touches operational risk:
    deployment boundaries, financial transactions, safety-critical paths.

    Derived from:
    - insight type (CONTRADICTION, RISK > OVERLAP > TREND)
    - severity from LLM assessment
    - presence of risk-associated keywords
    - classification modifier (retrieval dampened, discovery boosted)
    - temporal currency modifier (historical/stale dampened)
    """
    insight_type = insight.type.lower()
    severity = insight.details.get("severity", "medium").lower()

    # Base coupling by insight type
    type_coupling = {
        "contradiction": 0.85,
        "risk": 0.80,
        "overlap": 0.55,
        "trend": 0.45,
        "parallel_activity": 0.40,
        "cross_source_correlation": 0.50,
        "recurring_theme": 0.35,
    }
    base = type_coupling.get(insight_type, 0.40)

    # Severity modifier
    severity_mod = {
        "critical": 0.20,
        "high": 0.15,
        "medium": 0.0,
        "low": -0.10,
    }.get(severity, 0.0)

    # Risk keyword boost from summary
    risk_keywords = {
        "deploy", "production", "payment", "billing", "downtime",
        "outage", "security", "auth", "migration", "database",
        "breaking", "regression", "rollback", "hotfix", "incident",
    }
    summary_lower = insight.summary.lower()
    keyword_hits = sum(1 for kw in risk_keywords if kw in summary_lower)
    keyword_boost = min(0.15, keyword_hits * 0.05)

    raw = base + severity_mod + keyword_boost

    # Classification modifier — structural honesty in scoring.
    # Retrieval (re-describing known state) contributes less organizational learning.
    # Discovery (true 2OI→1OI) is the highest value the system can produce.
    classification = insight.details.get("classification", "").lower()
    classification_multiplier = {
        "retrieval": 0.5,
        "amplification": 1.0,
        "correlation": 1.0,
        "discovery": 1.2,
    }.get(classification, 1.0)

    # Temporal currency modifier — active risk scores higher than historical.
    # Historical findings are informative but resolved; stale findings are
    # past concerns with minimal current relevance.
    temporal = insight.details.get("temporal_relevance", "active").lower()
    temporal_multiplier = {
        "active": 1.0,
        "historical": 0.6,
        "stale": 0.2,
    }.get(temporal, 1.0)

    return max(0.05, min(1.0, raw * classification_multiplier * temporal_multiplier))


def compute_communicability(insight: Insight) -> float:
    """γ(f): communicability — can the finding be transmitted with useful precision?

    Paper Equation 4: γ(f) = σ(N*(b_f) - 1)
    where N* is the Crawford-Sobel partition count.

    Approximated here by:
    - Confidence from LLM assessment (high confidence → precise finding)
    - Specificity: does it name specific actors, specific repos, specific PRs?
    - Actionability: does the summary suggest a clear action?

    High communicability = specific, high-confidence, actionable.
    Low communicability = vague, low-confidence, ambiguous.
    """
    confidence = insight.confidence

    # Specificity: count of concrete references
    actors = insight.details.get("actors", [])
    bridges = insight.details.get("bridge_distances", {})
    specificity = min(1.0, len(actors) * 0.2 + len(bridges) * 0.15)

    # Estimate partition count N* from confidence and specificity
    # N* ≈ 1 (babbling) when vague, N* ≈ 4+ when precise
    n_star = 1.0 + 3.0 * (confidence * 0.6 + specificity * 0.4)

    # Apply sigmoid: γ = σ(N* - 1)
    # N*=1 → γ≈0.5 (borderline), N*=3 → γ≈0.88, N*=4 → γ≈0.95
    return _sigmoid(n_star - 1)


def score_finding(
    insight: Insight,
    graph: CommunicationGraph | None = None,
) -> ScoringComponents:
    """Compute S(f) for a finding.

    S(f) = d_bridge × c_entity × r_coupling × γ

    Returns ScoringComponents with individual values and product.
    """
    d_bridge = compute_bridge_distance(insight, graph)
    c_entity = compute_entity_confidence(insight, graph)
    r_coupling = compute_risk_coupling(insight)
    gamma = compute_communicability(insight)

    score = d_bridge * c_entity * r_coupling * gamma

    return ScoringComponents(
        d_bridge=d_bridge,
        c_entity=c_entity,
        r_coupling=r_coupling,
        gamma=gamma,
        score=score,
    )
