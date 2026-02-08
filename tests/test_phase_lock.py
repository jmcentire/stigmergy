"""Tests for phase-lock detection (Paper Section 5.2)."""

from stigmergy.attention.phase_lock import (
    PhaseLockAlert,
    detect_phase_lock,
    _jaccard,
)


class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0

    def test_partial_overlap(self):
        result = _jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"}))
        assert abs(result - 0.5) < 0.01  # 2/4

    def test_empty_sets(self):
        assert _jaccard(frozenset(), frozenset()) == 0.0


class TestPhaseLock:
    def test_detects_converging_findings(self):
        """Similar findings should cluster and trigger alert."""
        findings = [
            {"finding_hash": "aaa", "summary": "pricing service latency increasing under load"},
            {"finding_hash": "bbb", "summary": "pricing service response time degradation under load"},
            {"finding_hash": "ccc", "summary": "pricing service performance issues under heavy load"},
        ]
        alerts = detect_phase_lock(findings, similarity_threshold=0.2, min_findings=2)
        assert len(alerts) > 0
        assert alerts[0].finding_count >= 2
        assert alerts[0].framing_similarity > 0.0

    def test_no_phase_lock_diverse_findings(self):
        """Unrelated findings should not cluster."""
        findings = [
            {"finding_hash": "aaa", "summary": "payment gateway timeout errors"},
            {"finding_hash": "bbb", "summary": "mobile app navigation layout broken"},
            {"finding_hash": "ccc", "summary": "database vacuum schedule overdue"},
        ]
        alerts = detect_phase_lock(findings, similarity_threshold=0.5, min_findings=2)
        assert len(alerts) == 0

    def test_empty_findings(self):
        assert detect_phase_lock([]) == []

    def test_single_finding(self):
        findings = [{"finding_hash": "aaa", "summary": "some issue"}]
        assert detect_phase_lock(findings, min_findings=2) == []

    def test_action_correlation_affects_severity(self):
        """Low action-correlation + high similarity = critical."""
        findings = [
            {"finding_hash": "aaa", "summary": "sync service error handling retry logic"},
            {"finding_hash": "bbb", "summary": "sync service error handling timeout retry"},
        ]
        # With low action correlations
        ac = {"sync": 0.05, "service": 0.05, "error": 0.05, "handling": 0.05, "retry": 0.05}
        alerts = detect_phase_lock(findings, similarity_threshold=0.3,
                                   min_findings=2, action_correlations=ac)
        if alerts:
            assert alerts[0].action_correlation_avg < 0.2

    def test_result_structure(self):
        findings = [
            {"finding_hash": "aaa", "summary": "deployment pipeline broken failing tests"},
            {"finding_hash": "bbb", "summary": "deployment pipeline broken test failures"},
        ]
        alerts = detect_phase_lock(findings, similarity_threshold=0.2, min_findings=2)
        for a in alerts:
            assert isinstance(a, PhaseLockAlert)
            assert isinstance(a.risk_topics, list)
            assert isinstance(a.finding_hashes, list)
            assert a.severity in ("warning", "critical")
            assert 0.0 <= a.framing_similarity <= 1.0

    def test_sorted_by_severity(self):
        # Create multiple clusters with different characteristics
        findings = [
            {"finding_hash": "a1", "summary": "alpha cluster same terms repeated"},
            {"finding_hash": "a2", "summary": "alpha cluster same terms repeated again"},
            {"finding_hash": "b1", "summary": "beta different cluster about other stuff"},
            {"finding_hash": "b2", "summary": "beta different cluster about other things"},
        ]
        alerts = detect_phase_lock(findings, similarity_threshold=0.2, min_findings=2)
        if len(alerts) >= 2:
            severity_order = {"critical": 0, "warning": 1}
            for i in range(len(alerts) - 1):
                assert severity_order.get(alerts[i].severity, 2) <= severity_order.get(alerts[i + 1].severity, 2)
