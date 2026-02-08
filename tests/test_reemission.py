"""Tests for the re-emission engine — event-driven retriggering."""

from dataclasses import dataclass, field

import pytest

from stigmergy.attention.reemission import (
    ReemissionEvent,
    check_compound_risk,
    check_decay_escalation,
    check_new_evidence,
    evaluate_reemissions,
)


# ── Helpers ────────────────────────────────────────────────────


@dataclass
class FakeFinding:
    """Minimal finding stub for testing re-emission triggers."""

    finding_hash: str
    type: str = "risk"
    summary: str = ""
    actors: list = field(default_factory=list)
    max_score: float = 0.5
    decay_stage: str = "deferred"
    times_surfaced: int = 2
    has_state_change: bool = False
    days_since_first: float = 10.0


def _make_finding(**kwargs) -> FakeFinding:
    defaults = {
        "finding_hash": "abc123",
        "summary": "Tax calculation error in pricing service causes overcharge",
        "actors": ["alice", "bob"],
        "max_score": 0.6,
        "decay_stage": "deferred",
        "times_surfaced": 2,
        "days_since_first": 10.0,
    }
    defaults.update(kwargs)
    return FakeFinding(**defaults)


# ── New Evidence Trigger ──────────────────────────────────────


class TestNewEvidence:
    def test_matching_terms_triggers(self):
        finding = _make_finding(
            summary="Tax calculation error in pricing service causes overcharge",
            decay_stage="deferred",
        )
        signal_terms = frozenset(["pricing", "calculation", "service", "refund"])
        events = check_new_evidence([finding], signal_terms, "github", "pricing-repo")
        assert len(events) == 1
        assert events[0].trigger == "new_evidence"
        assert events[0].finding_hash == "abc123"
        assert "pricing" in events[0].trigger_detail.lower()

    def test_no_overlap_no_trigger(self):
        finding = _make_finding(
            summary="Tax calculation error in pricing service causes overcharge",
            decay_stage="deferred",
        )
        signal_terms = frozenset(["unrelated", "deployment", "frontend"])
        events = check_new_evidence([finding], signal_terms, "github", "frontend-repo")
        assert len(events) == 0

    def test_active_findings_ignored(self):
        finding = _make_finding(decay_stage="active")
        signal_terms = frozenset(["pricing", "calculation", "error"])
        events = check_new_evidence([finding], signal_terms, "github", "pricing-repo")
        assert len(events) == 0

    def test_low_score_filtered(self):
        finding = _make_finding(decay_stage="deferred", max_score=0.05)
        signal_terms = frozenset(["pricing", "calculation", "error"])
        events = check_new_evidence(
            [finding], signal_terms, "github", "pricing-repo",
            score_increase_threshold=0.1,
        )
        assert len(events) == 0

    def test_normalized_findings_trigger(self):
        finding = _make_finding(decay_stage="normalized")
        signal_terms = frozenset(["pricing", "calculation", "error", "service"])
        events = check_new_evidence([finding], signal_terms, "slack", "#dev-platform")
        assert len(events) == 1
        assert events[0].trigger == "new_evidence"

    def test_single_term_overlap_insufficient(self):
        finding = _make_finding(
            summary="Tax calculation error in pricing service",
            decay_stage="deferred",
        )
        # Only one term overlaps — below the >= 2 threshold
        signal_terms = frozenset(["pricing", "unrelated", "stuff"])
        events = check_new_evidence([finding], signal_terms, "github", "repo")
        assert len(events) == 0


# ── Compound Risk Trigger ─────────────────────────────────────


class TestCompoundRisk:
    def test_shared_actors_trigger(self):
        f1 = _make_finding(
            finding_hash="f1", summary="Pricing cache race condition",
            actors=["alice"], decay_stage="deferred",
        )
        f2 = _make_finding(
            finding_hash="f2", summary="Payment sync timeout failures",
            actors=["alice"], decay_stage="active",
        )
        events = check_compound_risk([f1, f2], actor_threshold=1)
        assert len(events) == 1
        assert events[0].trigger == "compound_risk"
        assert "alice" in events[0].trigger_detail.lower()

    def test_shared_terms_trigger(self):
        f1 = _make_finding(
            finding_hash="f1",
            summary="Pricing service cache invalidation race condition",
            actors=["alice"], decay_stage="deferred",
        )
        f2 = _make_finding(
            finding_hash="f2",
            summary="Pricing service timeout under load causes stale cache",
            actors=["bob"], decay_stage="active",
        )
        events = check_compound_risk([f1, f2], term_threshold=2)
        assert len(events) == 1
        assert events[0].trigger == "compound_risk"
        assert events[0].related_findings == ["f2"] or events[0].related_findings == ["f1"]

    def test_no_decaying_no_trigger(self):
        f1 = _make_finding(finding_hash="f1", actors=["alice"], decay_stage="active")
        f2 = _make_finding(finding_hash="f2", actors=["alice"], decay_stage="active")
        events = check_compound_risk([f1, f2])
        assert len(events) == 0

    def test_single_finding_no_trigger(self):
        f1 = _make_finding(decay_stage="deferred")
        events = check_compound_risk([f1])
        assert len(events) == 0

    def test_compound_score_sums(self):
        f1 = _make_finding(
            finding_hash="f1", actors=["alice"],
            max_score=0.4, decay_stage="deferred",
        )
        f2 = _make_finding(
            finding_hash="f2", actors=["alice"],
            max_score=0.3, decay_stage="active",
        )
        events = check_compound_risk([f1, f2], actor_threshold=1)
        assert len(events) == 1
        assert events[0].current_score == pytest.approx(0.7)


# ── Decay Escalation Trigger ─────────────────────────────────


class TestDecayEscalation:
    def test_normalized_finding_triggers(self):
        finding = _make_finding(
            decay_stage="normalized", times_surfaced=4,
            max_score=0.6, days_since_first=20.0,
        )
        events = check_decay_escalation([finding])
        assert len(events) == 1
        assert events[0].trigger == "decay_escalation"
        assert "NORMALIZED DEVIANCE" in events[0].trigger_detail

    def test_deferred_approaching_normalized(self):
        finding = _make_finding(
            decay_stage="deferred", times_surfaced=2,
            max_score=0.5, days_since_first=10.0,
        )
        events = check_decay_escalation([finding])
        assert len(events) == 1
        assert "Approaching normalization" in events[0].trigger_detail

    def test_deferred_single_surfacing_no_trigger(self):
        finding = _make_finding(
            decay_stage="deferred", times_surfaced=1,
        )
        events = check_decay_escalation([finding])
        assert len(events) == 0

    def test_low_score_filtered(self):
        finding = _make_finding(
            decay_stage="normalized", max_score=0.1,
        )
        events = check_decay_escalation([finding], min_score=0.2)
        assert len(events) == 0

    def test_active_findings_ignored(self):
        finding = _make_finding(decay_stage="active")
        events = check_decay_escalation([finding])
        assert len(events) == 0

    def test_acknowledged_findings_ignored(self):
        finding = _make_finding(decay_stage="acknowledged")
        events = check_decay_escalation([finding])
        assert len(events) == 0


# ── Integrated Evaluation ─────────────────────────────────────


class TestEvaluateReemissions:
    def test_deduplicates_by_finding_hash(self):
        """Same finding triggers multiple checks — keep highest priority."""
        finding = _make_finding(
            decay_stage="normalized", times_surfaced=4,
            max_score=0.6, days_since_first=20.0,
            summary="Pricing service cache invalidation error causes overcharge",
        )
        signal_terms = frozenset(["pricing", "cache", "invalidation", "error"])
        events = evaluate_reemissions(
            [finding], signal_terms, "github", "pricing-repo",
        )
        # Both new_evidence and decay_escalation match, but dedup keeps
        # decay_escalation (higher priority)
        assert len(events) == 1
        assert events[0].trigger == "decay_escalation"

    def test_multiple_findings_multiple_triggers(self):
        f1 = _make_finding(
            finding_hash="f1",
            decay_stage="normalized", times_surfaced=4,
            max_score=0.6, days_since_first=20.0,
        )
        f2 = _make_finding(
            finding_hash="f2",
            decay_stage="deferred", times_surfaced=2,
            max_score=0.4, days_since_first=8.0,
        )
        events = evaluate_reemissions([f1, f2])
        # f1: decay_escalation (normalized)
        # f2: decay_escalation (deferred, 2x surfaced)
        assert len(events) == 2

    def test_no_findings_no_events(self):
        events = evaluate_reemissions([])
        assert len(events) == 0

    def test_priority_order_in_output(self):
        f1 = _make_finding(
            finding_hash="f1", decay_stage="normalized",
            times_surfaced=4, max_score=0.6, days_since_first=20.0,
            summary="Critical pricing failure in payment pipeline system",
        )
        f2 = _make_finding(
            finding_hash="f2", decay_stage="deferred",
            times_surfaced=2, max_score=0.8, days_since_first=10.0,
            summary="Minor sync delay in event processing pipeline system",
        )
        f3 = _make_finding(
            finding_hash="f3", decay_stage="active",
            actors=["alice"], max_score=0.3,
        )
        signal_terms = frozenset(["pricing", "failure", "payment", "pipeline", "event", "processing", "system"])
        events = evaluate_reemissions(
            [f1, f2, f3], signal_terms, "slack", "#dev",
        )
        # decay_escalation first, then others
        decay_events = [e for e in events if e.trigger == "decay_escalation"]
        other_events = [e for e in events if e.trigger != "decay_escalation"]
        if decay_events and other_events:
            # All decay events should come before non-decay events
            first_other_idx = next(
                i for i, e in enumerate(events) if e.trigger != "decay_escalation"
            )
            last_decay_idx = max(
                i for i, e in enumerate(events) if e.trigger == "decay_escalation"
            )
            assert last_decay_idx < first_other_idx

    def test_empty_signal_terms_skips_evidence_check(self):
        finding = _make_finding(
            decay_stage="deferred", times_surfaced=2,
        )
        events = evaluate_reemissions([finding], signal_terms=None)
        # Only decay_escalation should fire (no new_evidence without terms)
        for event in events:
            assert event.trigger != "new_evidence"
