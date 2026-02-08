"""Tests for honest finding classification, temporal awareness, and actionable artifacts."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from stigmergy.mesh.insight_store import FindingRegistry, InsightStore
from stigmergy.mesh.insights import Insight
from stigmergy.mesh.scoring import compute_risk_coupling, score_finding
from stigmergy.primitives.schemas import (
    CorrelationInsight,
    RecommendedArtifact,
)


def _make_insight(
    type_: str = "OVERLAP",
    summary: str = "Alice and Bob both modifying pricing engine",
    confidence: float = 0.75,
    actors: list | None = None,
    severity: str = "medium",
    classification: str = "",
    temporal_relevance: str = "",
    artifacts: list | None = None,
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
            "bridge_distances": {"alice↔bob": 3.0},
            "classification": classification,
            "temporal_relevance": temporal_relevance,
            "artifacts": artifacts or [],
        },
    )


# ── Schema validation ─────────────────────────────────────


class TestSchemaFields:
    def test_correlation_insight_has_classification(self):
        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="Test finding",
            severity="medium",
            classification="retrieval",
            temporal_relevance="active",
        )
        assert ci.classification == "retrieval"
        assert ci.temporal_relevance == "active"

    def test_correlation_insight_defaults(self):
        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="Test finding",
            severity="medium",
        )
        assert ci.classification == "correlation"
        assert ci.temporal_relevance == "active"
        assert ci.artifacts == []

    def test_recommended_artifact_model(self):
        art = RecommendedArtifact(
            artifact_type="draft_ticket",
            target="@alice",
            content="Link PLAT-456 and INT-789 under a parent ticket.",
        )
        assert art.artifact_type == "draft_ticket"
        assert art.target == "@alice"
        assert "PLAT-456" in art.content

    def test_correlation_insight_with_artifacts(self):
        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="Test finding",
            severity="medium",
            classification="correlation",
            artifacts=[
                RecommendedArtifact(
                    artifact_type="draft_message",
                    target="#dev-platform",
                    content="@alice and @bob — your PRs overlap.",
                ),
            ],
        )
        assert len(ci.artifacts) == 1
        assert ci.artifacts[0].artifact_type == "draft_message"


# ── Classification in InsightStore persistence ─────────────


class TestClassificationPersistence:
    def test_retrieval_classification_persisted_in_jsonl(self, tmp_path):
        path = tmp_path / "insights.jsonl"
        store = InsightStore(path=path)

        ins = _make_insight(classification="retrieval")
        store.record(ins, score=0.3)
        store.save()

        with open(path) as f:
            data = json.loads(f.readline())
        assert data["classification"] == "retrieval"

    def test_temporal_relevance_persisted_in_jsonl(self, tmp_path):
        path = tmp_path / "insights.jsonl"
        store = InsightStore(path=path)

        ins = _make_insight(temporal_relevance="historical")
        store.record(ins, score=0.3)
        store.save()

        with open(path) as f:
            data = json.loads(f.readline())
        assert data["temporal_relevance"] == "historical"


# ── Classification in FindingRegistry ──────────────────────


class TestFindingRegistryClassification:
    def test_new_finding_stores_classification(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")
        insights = [{
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.75,
            "score": 0.42,
            "details": {"actors": ["alice", "bob"]},
            "classification": "correlation",
        }]
        changed = registry.ingest_insights(insights, "run-001")
        assert changed[0].classification == "correlation"

    def test_classification_persists_across_save(self, tmp_path):
        path = tmp_path / "registry.json"
        registry = FindingRegistry(path=path)

        insights = [{
            "type": "RISK",
            "summary": "Payment at risk",
            "confidence": 0.8,
            "score": 0.55,
            "details": {"actors": ["nick"]},
            "classification": "discovery",
        }]
        registry.ingest_insights(insights, "run-001")
        registry.save()

        registry2 = FindingRegistry(path=path)
        finding = list(registry2._findings.values())[0]
        assert finding.classification == "discovery"

    def test_classification_updated_on_resurface(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insight1 = {
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.75,
            "score": 0.42,
            "details": {"actors": ["alice", "bob"]},
            "classification": "retrieval",
        }
        insight2 = {
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.80,
            "score": 0.50,
            "details": {"actors": ["alice", "bob"]},
            "classification": "correlation",  # upgraded on resurface
        }

        registry.ingest_insights([insight1], "run-001")
        registry.ingest_insights([insight2], "run-002")

        finding = list(registry._findings.values())[0]
        assert finding.classification == "correlation"


# ── Retrieval scores lower ─────────────────────────────────


class TestRetrievalScoresLower:
    def test_retrieval_r_coupling_lower_than_correlation(self):
        retrieval = _make_insight(
            type_="OVERLAP",
            summary="Known pricing overlap",
            severity="medium",
            classification="retrieval",
        )
        correlation = _make_insight(
            type_="OVERLAP",
            summary="Known pricing overlap",
            severity="medium",
            classification="correlation",
        )
        r_retrieval = compute_risk_coupling(retrieval)
        r_correlation = compute_risk_coupling(correlation)
        assert r_retrieval < r_correlation

    def test_discovery_r_coupling_higher_than_correlation(self):
        discovery = _make_insight(
            type_="OVERLAP",
            summary="Known pricing overlap",
            severity="medium",
            classification="discovery",
        )
        correlation = _make_insight(
            type_="OVERLAP",
            summary="Known pricing overlap",
            severity="medium",
            classification="correlation",
        )
        r_discovery = compute_risk_coupling(discovery)
        r_correlation = compute_risk_coupling(correlation)
        assert r_discovery > r_correlation

    def test_retrieval_sf_lower_than_correlation(self):
        retrieval = _make_insight(classification="retrieval")
        correlation = _make_insight(classification="correlation")
        s_ret = score_finding(retrieval)
        s_cor = score_finding(correlation)
        assert s_ret.score < s_cor.score


# ── Retrieval decays faster ────────────────────────────────


class TestRetrievalDecaysFaster:
    def test_retrieval_deferred_after_first_resurface(self, tmp_path):
        """Retrieval findings enter 'deferred' after first resurface (times_surfaced=2).

        Normal findings need 2 resurfacings to reach deferred.
        Retrieval findings reach deferred on the very first resurface
        because repeating known state adds no new information.
        """
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insight = {
            "type": "RISK",
            "summary": "Management fee calculation broken",
            "confidence": 0.7,
            "score": 0.3,
            "details": {"actors": ["nathan"]},
            "classification": "retrieval",
        }

        # First surfacing — new finding starts as active
        changed = registry.ingest_insights([insight], "run-001")
        assert changed[0].decay_stage == "active"
        assert changed[0].times_surfaced == 1

        # First resurface (times_surfaced=2) — retrieval goes straight to deferred
        changed = registry.ingest_insights([insight], "run-002")
        assert changed[0].times_surfaced == 2
        assert changed[0].decay_stage == "deferred"

    def test_correlation_active_after_first_surface(self, tmp_path):
        """Correlation findings stay 'active' after first surfacing."""
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insight = {
            "type": "OVERLAP",
            "summary": "Alice and Bob overlapping",
            "confidence": 0.7,
            "score": 0.3,
            "details": {"actors": ["alice", "bob"]},
            "classification": "correlation",
        }

        changed = registry.ingest_insights([insight], "run-001")
        assert changed[0].decay_stage == "active"

    def test_correlation_needs_two_surfaces_for_deferred(self, tmp_path):
        """Correlation findings need 2 surfacings to reach 'deferred'."""
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insight = {
            "type": "OVERLAP",
            "summary": "Alice and Bob overlapping",
            "confidence": 0.7,
            "score": 0.3,
            "details": {"actors": ["alice", "bob"]},
            "classification": "correlation",
        }

        registry.ingest_insights([insight], "run-001")
        changed = registry.ingest_insights([insight], "run-002")
        assert changed[0].times_surfaced == 2
        assert changed[0].decay_stage == "deferred"


# ── Output display ─────────────────────────────────────────


class TestClassificationInOutput:
    def test_findings_summary_shows_classification_label(self, capsys):
        from stigmergy.cli.output import findings_summary

        ins = _make_insight(classification="correlation", temporal_relevance="active")
        count = findings_summary([ins])
        captured = capsys.readouterr().out
        assert "CORRELATION" in captured
        assert count == 1

    def test_findings_summary_shows_temporal_relevance(self, capsys):
        from stigmergy.cli.output import findings_summary

        ins = _make_insight(temporal_relevance="historical")
        count = findings_summary([ins])
        captured = capsys.readouterr().out
        assert "[historical]" in captured

    def test_findings_summary_shows_retrieval_label(self, capsys):
        from stigmergy.cli.output import findings_summary

        ins = _make_insight(classification="retrieval", temporal_relevance="stale")
        findings_summary([ins])
        captured = capsys.readouterr().out
        assert "RETRIEVAL" in captured
        assert "[stale]" in captured

    def test_artifact_display(self, capsys):
        from stigmergy.cli.output import findings_summary

        ins = _make_insight(
            classification="correlation",
            artifacts=[{
                "artifact_type": "draft_ticket",
                "target": "@alice",
                "content": "Link PLAT-456 and INT-789 under a parent ticket.",
            }],
        )
        findings_summary([ins])
        captured = capsys.readouterr().out
        assert "[draft_ticket]" in captured
        assert "@alice" in captured
        assert "PLAT-456" in captured

    def test_insight_line_shows_classification(self, capsys):
        from stigmergy.cli.output import insight_line

        ins = _make_insight(classification="discovery", temporal_relevance="active")
        insight_line(ins)
        captured = capsys.readouterr().out
        assert "DISCOVERY" in captured
        assert "[active]" in captured


# ── Post-LLM quality filter ─────────────────────────────


class TestPostLLMFilter:
    """Tests for _should_drop_finding() — programmatic guardrail on correlator output."""

    def test_retrieval_dropped(self):
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="SERV-316 Redis provisioning blocked",
            severity="medium",
            classification="retrieval",
            temporal_relevance="active",
            actors=["@devon"],
        )
        assert _should_drop_finding(ci) == "retrieval"

    def test_historical_dropped(self):
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="risk",
            summary="Tax calculation risk in pricing module",
            severity="high",
            classification="correlation",
            temporal_relevance="historical",
            actors=["@alice", "@bob"],
        )
        assert _should_drop_finding(ci) == "temporal:historical"

    def test_stale_dropped(self):
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="risk",
            summary="Old vulnerability already patched",
            severity="medium",
            classification="correlation",
            temporal_relevance="stale",
            actors=["@charlie"],
        )
        assert _should_drop_finding(ci) == "temporal:stale"

    def test_single_actor_overlap_dropped(self):
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="@devon working across pricing and payment repos",
            severity="low",
            classification="correlation",
            temporal_relevance="active",
            actors=["@devon"],
        )
        assert _should_drop_finding(ci) == "single_actor_overlap"

    def test_single_actor_coordination_dropped(self):
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="coordination",
            summary="@alice auditing db access across 4 endpoints",
            severity="low",
            classification="correlation",
            temporal_relevance="active",
            actors=["@alice"],
        )
        assert _should_drop_finding(ci) == "single_actor_overlap"

    def test_single_actor_risk_passes(self):
        """Single-actor risk findings should NOT be dropped — one person can
        introduce real risk (e.g., setting fee to 0.1% instead of 10%)."""
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="risk",
            summary="@devon set management fee to 0.10% instead of 10%",
            severity="high",
            classification="correlation",
            temporal_relevance="active",
            actors=["@devon"],
        )
        assert _should_drop_finding(ci) is None

    def test_multi_actor_overlap_passes(self):
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="@alice and @bob both modifying availability cache",
            severity="high",
            classification="correlation",
            temporal_relevance="active",
            actors=["@alice", "@bob"],
        )
        assert _should_drop_finding(ci) is None

    def test_active_correlation_passes(self):
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="risk",
            summary="Payment webhook timeout converging from 3 directions",
            severity="high",
            classification="correlation",
            temporal_relevance="active",
            actors=["@nick", "@tuan"],
        )
        assert _should_drop_finding(ci) is None

    def test_discovery_passes(self):
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="Two teams building cache invalidation independently",
            severity="high",
            classification="discovery",
            temporal_relevance="active",
            actors=["@alice", "@bob"],
        )
        assert _should_drop_finding(ci) is None

    def test_at_sign_normalization(self):
        """Actors with and without @ prefix for the same person count as one."""
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="Devon working across repos",
            severity="low",
            classification="correlation",
            temporal_relevance="active",
            actors=["@Devon", "devon"],  # same person, different format
        )
        assert _should_drop_finding(ci) == "single_actor_overlap"

    def test_empty_actors_overlap_dropped(self):
        """Overlap with no actors is noise."""
        from stigmergy.primitives.agent import _should_drop_finding
        from stigmergy.primitives.schemas import CorrelationInsight

        ci = CorrelationInsight(
            pattern_type="overlap",
            summary="Multiple changes to the same area",
            severity="low",
            classification="correlation",
            temporal_relevance="active",
            actors=[],
        )
        assert _should_drop_finding(ci) == "single_actor_overlap"
