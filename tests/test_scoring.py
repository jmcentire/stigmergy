"""Tests for S(f) scoring function — Equation 3."""

from uuid import uuid4

import pytest

from stigmergy.mesh.insights import Insight
from stigmergy.mesh.scoring import (
    ScoringComponents,
    compute_bridge_distance,
    compute_communicability,
    compute_entity_confidence,
    compute_risk_coupling,
    score_finding,
)


def _make_insight(
    type_: str = "OVERLAP",
    summary: str = "Alice and Bob both modifying pricing engine",
    confidence: float = 0.75,
    actors: list | None = None,
    severity: str = "medium",
    bridge_distances: dict | None = None,
    classification: str = "",
    temporal_relevance: str = "",
) -> Insight:
    return Insight(
        agent_id=uuid4(),
        type=type_,
        summary=summary,
        confidence=confidence,
        signal_ids=[uuid4(), uuid4()],
        details={
            "actors": actors or ["alice", "bob"],
            "severity": severity,
            "bridge_distances": bridge_distances or {},
            "classification": classification,
            "temporal_relevance": temporal_relevance,
        },
    )


class TestBridgeDistance:
    def test_no_graph_returns_moderate(self):
        ins = _make_insight()
        d = compute_bridge_distance(ins, None)
        assert d == pytest.approx(0.5)

    def test_single_actor_returns_moderate(self):
        ins = _make_insight(actors=["alice"])
        d = compute_bridge_distance(ins, None)
        assert d == pytest.approx(0.5)


class TestEntityConfidence:
    def test_no_graph_moderate_regardless_of_actors(self):
        """Without graph, entity confidence is moderate prior (0.5)."""
        ins_empty = _make_insight(actors=[])
        ins_full = _make_insight(actors=["alice", "bob"])
        # Without a graph, can't evaluate entity resolution
        assert compute_entity_confidence(ins_empty, None) == pytest.approx(0.5)
        assert compute_entity_confidence(ins_full, None) == pytest.approx(0.5)


class TestRiskCoupling:
    def test_contradiction_high_coupling(self):
        ins = _make_insight(type_="CONTRADICTION", severity="high")
        r = compute_risk_coupling(ins)
        assert r >= 0.8

    def test_trend_low_coupling(self):
        ins = _make_insight(type_="TREND", severity="low")
        r = compute_risk_coupling(ins)
        assert r < 0.5

    def test_risk_keywords_boost(self):
        base_ins = _make_insight(summary="Some general finding")
        boosted_ins = _make_insight(summary="Payment service production deployment breaking")
        r_base = compute_risk_coupling(base_ins)
        r_boosted = compute_risk_coupling(boosted_ins)
        assert r_boosted > r_base

    def test_clamps_to_range(self):
        ins = _make_insight(
            type_="CONTRADICTION",
            severity="critical",
            summary="production payment database migration breaking security incident",
        )
        r = compute_risk_coupling(ins)
        assert 0.0 <= r <= 1.0


class TestCommunicability:
    def test_high_confidence_high_communicability(self):
        ins = _make_insight(
            confidence=0.9,
            actors=["alice", "bob", "charlie"],
            bridge_distances={"a↔b": 3, "b↔c": 2},
        )
        g = compute_communicability(ins)
        assert g > 0.8

    def test_low_confidence_lower_communicability(self):
        ins = _make_insight(confidence=0.3, actors=[], bridge_distances={})
        g = compute_communicability(ins)
        assert g < 0.8

    def test_range(self):
        ins = _make_insight()
        g = compute_communicability(ins)
        assert 0.0 <= g <= 1.0


class TestScoreFinding:
    def test_returns_components(self):
        ins = _make_insight()
        result = score_finding(ins)
        assert isinstance(result, ScoringComponents)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.d_bridge <= 1.0
        assert 0.0 <= result.c_entity <= 1.0
        assert 0.0 <= result.r_coupling <= 1.0
        assert 0.0 <= result.gamma <= 1.0

    def test_multiplicative(self):
        ins = _make_insight()
        result = score_finding(ins)
        expected = result.d_bridge * result.c_entity * result.r_coupling * result.gamma
        assert result.score == pytest.approx(expected, abs=0.001)

    def test_high_risk_scores_higher(self):
        low = _make_insight(type_="TREND", severity="low", confidence=0.3)
        high = _make_insight(
            type_="CONTRADICTION", severity="high", confidence=0.85,
            summary="Production payment service contradiction",
            actors=["alice", "bob", "charlie"],
            bridge_distances={"alice↔bob": 4},
        )
        s_low = score_finding(low)
        s_high = score_finding(high)
        assert s_high.score > s_low.score

    def test_to_dict(self):
        ins = _make_insight()
        result = score_finding(ins)
        d = result.to_dict()
        assert "d_bridge" in d
        assert "c_entity" in d
        assert "r_coupling" in d
        assert "gamma" in d
        assert "score" in d
        assert all(isinstance(v, float) for v in d.values())


class TestTemporalCurrency:
    """Temporal relevance should modify r_coupling scoring."""

    def test_historical_scores_lower_than_active(self):
        active = _make_insight(
            type_="RISK", severity="high",
            temporal_relevance="active",
        )
        historical = _make_insight(
            type_="RISK", severity="high",
            temporal_relevance="historical",
        )
        r_active = compute_risk_coupling(active)
        r_historical = compute_risk_coupling(historical)
        assert r_historical < r_active

    def test_stale_scores_lower_than_historical(self):
        historical = _make_insight(
            type_="RISK", severity="high",
            temporal_relevance="historical",
        )
        stale = _make_insight(
            type_="RISK", severity="high",
            temporal_relevance="stale",
        )
        r_historical = compute_risk_coupling(historical)
        r_stale = compute_risk_coupling(stale)
        assert r_stale < r_historical

    def test_active_no_penalty(self):
        """Active findings should score the same as unspecified (default)."""
        active = _make_insight(type_="RISK", severity="medium", temporal_relevance="active")
        default = _make_insight(type_="RISK", severity="medium", temporal_relevance="")
        r_active = compute_risk_coupling(active)
        r_default = compute_risk_coupling(default)
        assert r_active == pytest.approx(r_default)

    def test_temporal_and_classification_compound(self):
        """Retrieval + stale should score much lower than discovery + active."""
        worst = _make_insight(
            type_="OVERLAP", severity="medium",
            classification="retrieval", temporal_relevance="stale",
        )
        best = _make_insight(
            type_="OVERLAP", severity="medium",
            classification="discovery", temporal_relevance="active",
        )
        r_worst = compute_risk_coupling(worst)
        r_best = compute_risk_coupling(best)
        # retrieval(0.5) * stale(0.2) = 0.1x vs discovery(1.2) * active(1.0) = 1.2x
        assert r_best > r_worst * 3

    def test_stale_sf_lower_than_active_sf(self):
        """Full S(f) score should reflect temporal currency."""
        active = _make_insight(temporal_relevance="active")
        stale = _make_insight(temporal_relevance="stale")
        s_active = score_finding(active)
        s_stale = score_finding(stale)
        assert s_stale.score < s_active.score

    def test_historical_multiplier_value(self):
        """Historical should apply 0.6x multiplier."""
        active = _make_insight(type_="OVERLAP", severity="medium", temporal_relevance="active")
        historical = _make_insight(type_="OVERLAP", severity="medium", temporal_relevance="historical")
        r_active = compute_risk_coupling(active)
        r_historical = compute_risk_coupling(historical)
        # historical = active * 0.6 (after clamping)
        expected = max(0.05, min(1.0, r_active * 0.6))
        assert r_historical == pytest.approx(expected, abs=0.01)


class TestBridgeDistanceWithGraph:
    """Bridge distance with a populated communication graph."""

    def test_connected_actors_nonzero_distance(self):
        """Actors connected via a third party should have measurable distance."""
        from stigmergy.graph.graph import CommunicationGraph
        from stigmergy.graph.identity import IdentityResolver

        resolver = IdentityResolver()
        graph = CommunicationGraph(resolver)

        # alice ← charlie → bob (charlie reviewed both)
        graph.add_edge("charlie", "alice", "pr_review", "repo-a")
        graph.add_edge("charlie", "bob", "pr_review", "repo-b")

        ins = _make_insight(actors=["alice", "bob"])
        d = compute_bridge_distance(ins, graph)
        # alice and bob are 2 hops apart via charlie
        assert d != pytest.approx(0.5)  # not the "no graph" default
        assert d != pytest.approx(0.95)  # not disconnected
        assert 0.0 < d < 1.0

    def test_disconnected_actors_high_distance(self):
        """Actors with no path should get near-maximum distance."""
        from stigmergy.graph.graph import CommunicationGraph
        from stigmergy.graph.identity import IdentityResolver

        resolver = IdentityResolver()
        graph = CommunicationGraph(resolver)

        # Only alice has edges; dave is isolated
        graph.add_edge("alice", "bob", "pr_review", "repo-a")

        ins = _make_insight(actors=["alice", "dave"])
        d = compute_bridge_distance(ins, graph)
        # dave not in graph → inf → 0.95
        assert d == pytest.approx(0.95)

    def test_direct_connection_low_distance(self):
        """Directly connected actors should have low bridge distance."""
        from stigmergy.graph.graph import CommunicationGraph
        from stigmergy.graph.identity import IdentityResolver

        resolver = IdentityResolver()
        graph = CommunicationGraph(resolver)

        graph.add_edge("alice", "bob", "pr_review", "repo-a")
        graph.add_edge("bob", "alice", "pr_comment", "repo-a")

        ins = _make_insight(actors=["alice", "bob"])
        d = compute_bridge_distance(ins, graph)
        # Directly connected, 1 hop, low effective size → low distance
        assert d < 0.5
