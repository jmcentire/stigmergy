"""Tests for portfolio risk — compound scoring across related findings."""

import pytest

from stigmergy.attention.portfolio import (
    RiskCluster,
    build_portfolio,
    cluster_findings,
    score_clusters,
)


# ── Helpers ────────────────────────────────────────────────────


def _finding(**kwargs) -> dict:
    defaults = {
        "finding_hash": "f1",
        "summary": "Default finding summary for testing",
        "sf_score": 0.3,
        "actors": [],
        "channels": [],
    }
    defaults.update(kwargs)
    return defaults


# ── Clustering ────────────────────────────────────────────────


class TestClusterFindings:
    def test_shared_actors_cluster(self):
        f1 = _finding(finding_hash="f1", actors=["alice"], summary="Pricing cache race")
        f2 = _finding(finding_hash="f2", actors=["alice"], summary="Payment timeout issue")
        clusters = cluster_findings([f1, f2], actor_threshold=1)
        assert len(clusters) == 1
        assert len(clusters[0].findings) == 2

    def test_shared_terms_cluster(self):
        f1 = _finding(finding_hash="f1", summary="Pricing service cache invalidation error")
        f2 = _finding(finding_hash="f2", summary="Pricing service timeout under heavy load")
        clusters = cluster_findings([f1, f2], term_threshold=2)
        assert len(clusters) == 1
        assert "pricing" in clusters[0].shared_terms or "service" in clusters[0].shared_terms

    def test_no_overlap_no_cluster(self):
        f1 = _finding(finding_hash="f1", actors=["alice"], summary="Frontend layout broken")
        f2 = _finding(finding_hash="f2", actors=["bob"], summary="Backend database migration")
        clusters = cluster_findings([f1, f2], actor_threshold=1, term_threshold=3)
        assert len(clusters) == 0

    def test_transitive_clustering(self):
        """A-B share actors, B-C share terms → A, B, C in same cluster."""
        f1 = _finding(finding_hash="f1", actors=["alice"], summary="Pricing cache error")
        f2 = _finding(finding_hash="f2", actors=["alice"], summary="Service timeout issue")
        f3 = _finding(finding_hash="f3", actors=["bob"], summary="Service timeout failure")
        clusters = cluster_findings([f1, f2, f3], actor_threshold=1, term_threshold=2)
        assert len(clusters) == 1
        assert len(clusters[0].findings) == 3

    def test_single_finding_no_cluster(self):
        f1 = _finding(finding_hash="f1")
        clusters = cluster_findings([f1])
        assert len(clusters) == 0

    def test_multiple_separate_clusters(self):
        f1 = _finding(finding_hash="f1", actors=["alice"], summary="Pricing bug found")
        f2 = _finding(finding_hash="f2", actors=["alice"], summary="Pricing error detected")
        f3 = _finding(finding_hash="f3", actors=["charlie"], summary="Frontend layout broken")
        f4 = _finding(finding_hash="f4", actors=["charlie"], summary="Frontend style regression")
        clusters = cluster_findings([f1, f2, f3, f4], actor_threshold=1, term_threshold=2)
        assert len(clusters) == 2

    def test_shared_channels_cluster(self):
        f1 = _finding(finding_hash="f1", channels=["#dev-pricing"])
        f2 = _finding(finding_hash="f2", channels=["#dev-pricing"])
        clusters = cluster_findings([f1, f2])
        assert len(clusters) == 1


# ── Compound Scoring ──────────────────────────────────────────


class TestScoreClusters:
    def test_compound_score_formula(self):
        """max + sum(others) * decay_factor."""
        cluster = RiskCluster(
            cluster_id="c1",
            findings=[
                _finding(sf_score=0.4),
                _finding(sf_score=0.3),
                _finding(sf_score=0.2),
            ],
        )
        scored = score_clusters([cluster], decay_factor=0.3)
        # 0.4 + (0.3 + 0.2) * 0.3 = 0.4 + 0.15 = 0.55
        assert scored[0].compound_score == pytest.approx(0.55)
        assert scored[0].max_individual_score == pytest.approx(0.4)

    def test_escalation_when_compound_exceeds_but_individual_doesnt(self):
        cluster = RiskCluster(
            cluster_id="c1",
            findings=[
                _finding(sf_score=0.3),
                _finding(sf_score=0.3),
                _finding(sf_score=0.3),
            ],
        )
        scored = score_clusters([cluster], decay_factor=0.3, escalation_threshold=0.4)
        # 0.3 + (0.3 + 0.3) * 0.3 = 0.3 + 0.18 = 0.48
        assert scored[0].compound_score == pytest.approx(0.48)
        assert scored[0].escalated is True

    def test_no_escalation_when_individual_exceeds(self):
        cluster = RiskCluster(
            cluster_id="c1",
            findings=[
                _finding(sf_score=0.5),
                _finding(sf_score=0.3),
            ],
        )
        scored = score_clusters([cluster], decay_factor=0.3, escalation_threshold=0.4)
        # max=0.5 >= 0.4, so not escalated
        assert scored[0].escalated is False

    def test_no_escalation_when_compound_below_threshold(self):
        cluster = RiskCluster(
            cluster_id="c1",
            findings=[
                _finding(sf_score=0.1),
                _finding(sf_score=0.1),
            ],
        )
        scored = score_clusters([cluster], decay_factor=0.3, escalation_threshold=0.4)
        # 0.1 + 0.1 * 0.3 = 0.13 < 0.4
        assert scored[0].escalated is False

    def test_single_finding_cluster_score(self):
        cluster = RiskCluster(
            cluster_id="c1",
            findings=[_finding(sf_score=0.5)],
        )
        scored = score_clusters([cluster])
        assert scored[0].compound_score == pytest.approx(0.5)


# ── Full Pipeline ─────────────────────────────────────────────


class TestBuildPortfolio:
    def test_end_to_end(self):
        findings = [
            _finding(finding_hash="f1", actors=["alice"],
                     summary="Pricing cache race condition detected", sf_score=0.3),
            _finding(finding_hash="f2", actors=["alice"],
                     summary="Pricing timeout under load observed", sf_score=0.25),
            _finding(finding_hash="f3", actors=["bob"],
                     summary="Frontend layout regression", sf_score=0.6),
        ]
        clusters = build_portfolio(
            findings, actor_threshold=1, term_threshold=2,
            decay_factor=0.3, escalation_threshold=0.4,
        )
        # f1 and f2 cluster (shared actor "alice" + shared term "pricing")
        assert len(clusters) >= 1
        alice_cluster = [c for c in clusters if "alice" in c.shared_actors]
        assert len(alice_cluster) == 1
        assert len(alice_cluster[0].findings) == 2

    def test_sorted_by_compound_score(self):
        findings = [
            _finding(finding_hash="f1", actors=["alice"], sf_score=0.5,
                     summary="Pricing cache invalidation detected"),
            _finding(finding_hash="f2", actors=["alice"], sf_score=0.4,
                     summary="Pricing timeout under heavy load"),
            _finding(finding_hash="f3", actors=["bob"], sf_score=0.1,
                     summary="Frontend layout regression observed"),
            _finding(finding_hash="f4", actors=["bob"], sf_score=0.1,
                     summary="Frontend style broken completely"),
        ]
        clusters = build_portfolio(findings, actor_threshold=1, term_threshold=3)
        assert len(clusters) == 2
        assert clusters[0].compound_score >= clusters[1].compound_score

    def test_empty_input(self):
        assert build_portfolio([]) == []

    def test_single_finding(self):
        assert build_portfolio([_finding()]) == []
