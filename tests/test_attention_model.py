"""Tests for the probabilistic attention model — P(knows), loss function, calibration."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from stigmergy.attention.exposure import ExposureEvent
from stigmergy.attention.model import (
    AttentionModel,
    AttentionParameters,
    PersonAttentionState,
)
from stigmergy.attention.surfacing import (
    SurfacingDecision,
    compute_surface_value,
    rank_for_person,
)


def _now() -> datetime:
    return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _exposure(
    person_id: str = "alice",
    source: str = "github",
    channel: str = "backend",
    exposure_type: str = "authored",
    timestamp: datetime | None = None,
    terms: frozenset[str] | None = None,
    signal_id: str = "sig-1",
) -> ExposureEvent:
    return ExposureEvent(
        person_id=person_id,
        source=source,
        channel=channel,
        exposure_type=exposure_type,
        timestamp=timestamp or _now() - timedelta(hours=1),
        topic_terms=terms or frozenset({"pricing", "cache", "redis"}),
        signal_id=signal_id,
    )


# ── P(knows) computation ─────────────────────────────────────


class TestPKnows:
    """Test the noisy-OR P(knows) computation."""

    def test_no_exposures_returns_zero(self):
        model = AttentionModel()
        p = model.p_knows("alice", frozenset({"pricing"}), _now())
        assert p == 0.0

    def test_single_authored_recent(self):
        """Person who authored a signal 1 hour ago → P(knows) ≈ 0.95."""
        model = AttentionModel()
        model.record_exposure(_exposure(
            exposure_type="authored",
            timestamp=_now() - timedelta(hours=1),
            terms=frozenset({"pricing", "cache", "redis"}),
        ))
        p = model.p_knows("alice", frozenset({"pricing", "cache", "redis"}), _now())
        # base=0.95 * decay≈1.0 * channel=1.0 * volume=1.0 * overlap=1.0
        assert p > 0.85

    def test_single_authored_decayed(self):
        """Person who authored 30 days ago → P(knows) significantly decayed."""
        model = AttentionModel()
        model.record_exposure(_exposure(
            exposure_type="authored",
            timestamp=_now() - timedelta(days=30),
        ))
        p = model.p_knows("alice", frozenset({"pricing", "cache", "redis"}), _now())
        # 30 days with 14-day half-life: decay ≈ exp(-ln2*30/14) ≈ 0.228
        # base=0.95 * 0.228 ≈ 0.217
        assert p < 0.30
        assert p > 0.10

    def test_multiple_weak_exposures_compound(self):
        """Multiple weak exposures compound via noisy-OR → higher than any individual."""
        model = AttentionModel()
        terms = frozenset({"pricing", "cache"})
        for i in range(5):
            model.record_exposure(_exposure(
                exposure_type="channel_present",  # base=0.20
                timestamp=_now() - timedelta(hours=i + 1),
                terms=terms,
                signal_id=f"sig-{i}",
            ))
        p = model.p_knows("alice", terms, _now())
        # Each p_i ≈ 0.20, noisy-OR of 5: 1 - 0.80^5 ≈ 0.672
        assert p > 0.50
        assert p < 0.85

        # Compare to single exposure
        model2 = AttentionModel()
        model2.record_exposure(_exposure(
            exposure_type="channel_present",
            terms=terms,
        ))
        p_single = model2.p_knows("alice", terms, _now())
        assert p > p_single

    def test_large_channel_discount(self):
        """50-author channel discounts P(knows) significantly."""
        model = AttentionModel()
        model.record_exposure(_exposure(
            source="slack",
            exposure_type="mentioned",
            channel="#all-hands",
            terms=frozenset({"pricing", "update"}),
        ))
        # Set channel to 50 authors (key = source:channel)
        state = model.get_state("alice")
        state.channels_seen["slack:#all-hands"] = 50

        p_big = model.p_knows("alice", frozenset({"pricing", "update"}), _now())

        # Now same with 5-author channel (reference size = 5, no discount)
        model2 = AttentionModel()
        model2.record_exposure(_exposure(
            source="slack",
            exposure_type="mentioned",
            channel="#small-team",
            terms=frozenset({"pricing", "update"}),
        ))
        state2 = model2.get_state("alice")
        state2.channels_seen["slack:#small-team"] = 5

        p_small = model2.p_knows("alice", frozenset({"pricing", "update"}), _now())

        assert p_small > p_big

    def test_high_volume_discount(self):
        """Person tagged on 100 signals/day has reduced attention per signal."""
        model = AttentionModel()
        model.record_exposure(_exposure(
            exposure_type="assigned",
            terms=frozenset({"pricing"}),
        ))
        model.update_signal_volume("alice", 700)  # 100/day

        p_high_vol = model.p_knows("alice", frozenset({"pricing"}), _now())

        model2 = AttentionModel()
        model2.record_exposure(_exposure(
            exposure_type="assigned",
            terms=frozenset({"pricing"}),
        ))
        # Low volume (default 0)

        p_low_vol = model2.p_knows("alice", frozenset({"pricing"}), _now())

        assert p_low_vol > p_high_vol

    def test_topic_overlap_below_threshold_ignored(self):
        """Exposure with topic overlap < 0.1 threshold is ignored."""
        model = AttentionModel()
        model.record_exposure(_exposure(
            exposure_type="authored",
            terms=frozenset({"deployment", "kubernetes", "helm", "cluster",
                             "ingress", "pods", "services", "monitoring",
                             "alerts", "grafana"}),
        ))
        # Finding about pricing — no overlap with deployment terms
        p = model.p_knows("alice", frozenset({"pricing", "revenue", "billing"}), _now())
        assert p == 0.0

    def test_different_persons_independent(self):
        """Exposures for Alice don't affect Bob's P(knows)."""
        model = AttentionModel()
        terms = frozenset({"pricing"})
        model.record_exposure(_exposure(person_id="alice", terms=terms))

        p_alice = model.p_knows("alice", terms, _now())
        p_bob = model.p_knows("bob", terms, _now())

        assert p_alice > 0.5
        assert p_bob == 0.0

    def test_reacted_exposure_type(self):
        """Reacted exposure (emoji reaction) has base weight 0.35."""
        model = AttentionModel()
        terms = frozenset({"pricing", "cache"})
        model.record_exposure(_exposure(
            exposure_type="reacted",
            terms=terms,
        ))
        p = model.p_knows("alice", terms, _now())
        # base=0.35, recent, small channel, low volume
        assert 0.25 < p < 0.45


# ── Surfacing loss function ──────────────────────────────────


class TestSurfacingLossFunction:
    """Test the surfacing decision: importance * P(!knows) - annoyance * P(knows)."""

    def test_high_importance_low_pknows_surfaces(self):
        """High importance + low P(knows) → positive margin → surface."""
        sv = compute_surface_value(importance=0.8, p_knows=0.1)
        assert sv > 0

    def test_low_importance_high_pknows_suppressed(self):
        """Low importance + high P(knows) → negative margin → suppress."""
        sv = compute_surface_value(importance=0.2, p_knows=0.8)
        assert sv < 0

    def test_high_importance_moderate_pknows_still_surfaces(self):
        """Key asymmetry: high importance surfaces even at moderate P(knows)."""
        sv = compute_surface_value(importance=0.9, p_knows=0.5)
        # 1.0 * 0.9 * 0.5 - 0.3 * 0.5 = 0.45 - 0.15 = 0.30
        assert sv > 0.2

    def test_low_importance_moderate_pknows_suppressed(self):
        """Low importance at moderate P(knows) gets suppressed."""
        sv = compute_surface_value(importance=0.15, p_knows=0.5)
        # 1.0 * 0.15 * 0.5 - 0.3 * 0.5 = 0.075 - 0.15 = -0.075
        assert sv < 0

    def test_zero_importance_always_negative(self):
        sv = compute_surface_value(importance=0.0, p_knows=0.0)
        assert sv == 0.0
        sv2 = compute_surface_value(importance=0.0, p_knows=0.5)
        assert sv2 < 0

    def test_custom_weights(self):
        sv = compute_surface_value(
            importance=0.5, p_knows=0.5,
            importance_weight=2.0, annoyance_weight=0.1,
        )
        # 2.0 * 0.5 * 0.5 - 0.1 * 0.5 = 0.5 - 0.05 = 0.45
        assert abs(sv - 0.45) < 0.01


class TestRankForPerson:
    """Test ranking findings for a person."""

    def test_rankings_ordered_by_margin(self):
        model = AttentionModel()
        terms_a = frozenset({"pricing", "cache"})
        terms_b = frozenset({"deployment", "k8s"})

        # Alice knows about pricing but not deployment
        model.record_exposure(_exposure(
            person_id="alice",
            exposure_type="authored",
            terms=terms_a,
        ))

        findings = [
            {
                "finding_hash": "hash-a",
                "summary": "Pricing cache issue",
                "type": "risk",
                "sf_score": 0.7,
                "terms": terms_a,
                "signal_ids": ["sig-1"],
            },
            {
                "finding_hash": "hash-b",
                "summary": "Deployment failure",
                "type": "risk",
                "sf_score": 0.7,
                "terms": terms_b,
                "signal_ids": ["sig-2"],
            },
        ]

        decisions = rank_for_person(findings, "alice", model, _now())
        assert len(decisions) == 2
        # Deployment should rank higher (alice doesn't know about it)
        assert decisions[0].finding_hash == "hash-b"
        assert decisions[0].margin > decisions[1].margin

    def test_end_to_end_alice_vs_bob(self):
        """Alice authored 4 pricing signals, Bob mentioned in 1 large-channel Slack.

        Finding: pricing-related.
        Assert: P(knows|Alice) >> P(knows|Bob)
        Assert: surface_value for Bob > surface_value for Alice
        """
        model = AttentionModel()
        pricing_terms = frozenset({"pricing", "cache", "invalidation", "race", "condition"})

        # Alice authored 4 pricing signals
        for i in range(4):
            model.record_exposure(ExposureEvent(
                person_id="alice",
                source="github",
                channel="pricing-service",
                exposure_type="authored",
                timestamp=_now() - timedelta(days=i),
                topic_terms=pricing_terms,
                signal_id=f"alice-sig-{i}",
            ))

        # Bob mentioned in 1 Slack message in 50-person channel
        model.record_exposure(ExposureEvent(
            person_id="bob",
            source="slack",
            channel="#all-engineering",
            exposure_type="mentioned",
            timestamp=_now() - timedelta(days=2),
            topic_terms=pricing_terms,
            signal_id="bob-sig-1",
        ))
        bob_state = model.get_state("bob")
        bob_state.channels_seen["slack:#all-engineering"] = 50

        p_alice = model.p_knows("alice", pricing_terms, _now())
        p_bob = model.p_knows("bob", pricing_terms, _now())

        assert p_alice > p_bob, f"P(knows|Alice)={p_alice:.3f} should >> P(knows|Bob)={p_bob:.3f}"
        assert p_alice > 0.95
        assert p_bob < 0.30

        # Surface value: Bob should see this finding ranked higher
        finding = {
            "finding_hash": "pricing-race",
            "summary": "Pricing cache invalidation race condition",
            "type": "risk",
            "sf_score": 0.78,
            "terms": pricing_terms,
            "signal_ids": [f"alice-sig-{i}" for i in range(4)] + ["bob-sig-1"],
        }

        bob_decisions = rank_for_person([finding], "bob", model, _now())
        alice_decisions = rank_for_person([finding], "alice", model, _now())

        assert bob_decisions[0].margin > alice_decisions[0].margin


# ── Calibration ──────────────────────────────────────────────


class TestCalibration:
    """Test the feedback calibration loop."""

    def test_already_knew_increases_base_rate(self):
        model = AttentionModel()
        model._get_or_create("alice")

        initial_rate = model.get_state("alice").attention_base_rate
        model.record_feedback("alice", "hash-1", "already_knew")
        new_rate = model.get_state("alice").attention_base_rate

        assert new_rate > initial_rate

    def test_had_no_idea_decreases_base_rate(self):
        model = AttentionModel()
        model._get_or_create("alice")

        initial_rate = model.get_state("alice").attention_base_rate
        model.record_feedback("alice", "hash-1", "had_no_idea")
        new_rate = model.get_state("alice").attention_base_rate

        assert new_rate < initial_rate

    def test_asymmetric_lr_down_faster_than_up(self):
        """had_no_idea moves base_rate further per step than already_knew.

        lr_down=0.20 vs lr_up=0.05 — the system corrects overestimation
        aggressively because the cost of under-informing is invisible.
        """
        # Start two models at the same base rate
        model_up = AttentionModel()
        model_up._get_or_create("alice")
        model_down = AttentionModel()
        model_down._get_or_create("bob")

        initial = 0.5  # default

        model_up.record_feedback("alice", "h1", "already_knew")
        model_down.record_feedback("bob", "h1", "had_no_idea")

        delta_up = abs(model_up.get_state("alice").attention_base_rate - initial)
        delta_down = abs(model_down.get_state("bob").attention_base_rate - initial)

        # Down correction should be ~4x larger than up nudge
        assert delta_down > delta_up * 3

    def test_asymmetric_lr_values(self):
        """Verify the actual lr values produce expected deltas."""
        model = AttentionModel()
        model._get_or_create("alice")
        model._get_or_create("bob")

        # "already_knew" with lr_up=0.05: delta = 0.05 * (0.95 - 0.5) = 0.0225
        model.record_feedback("alice", "h1", "already_knew")
        assert abs(model.get_state("alice").attention_base_rate - 0.5225) < 0.001

        # "had_no_idea" with lr_down=0.20: delta = 0.20 * (0.05 - 0.5) = -0.09
        model.record_feedback("bob", "h1", "had_no_idea")
        assert abs(model.get_state("bob").attention_base_rate - 0.41) < 0.001

    def test_bounded_never_deterministic(self):
        """Base rate bounded [0.05, 0.95] — never becomes deterministic."""
        model = AttentionModel()
        model._get_or_create("alice")

        # Push toward 1.0 with many "already_knew" feedbacks
        for i in range(200):
            model.record_feedback("alice", f"hash-{i}", "already_knew")
        assert model.get_state("alice").attention_base_rate <= 0.95

        # Reset and push toward 0.0
        model.get_state("alice").attention_base_rate = 0.5
        for i in range(100):
            model.record_feedback("alice", f"hash-{i}", "had_no_idea")
        assert model.get_state("alice").attention_base_rate >= 0.05

    def test_multiple_feedbacks_converge(self):
        """Feedback converges toward actual attention footprint."""
        model = AttentionModel()
        model._get_or_create("alice")

        # Alice consistently reports "already_knew" — she reads everything
        # With lr_up=0.05, convergence is slower (intentional: small nudges up)
        for i in range(40):
            model.record_feedback("alice", f"hash-{i}", "already_knew")

        rate = model.get_state("alice").attention_base_rate
        # Should be well above the default 0.5
        assert rate > 0.8

    def test_feedback_counts_tracked(self):
        model = AttentionModel()
        model._get_or_create("alice")
        model.record_feedback("alice", "h1", "already_knew")
        model.record_feedback("alice", "h2", "had_no_idea")
        model.record_feedback("alice", "h3", "already_knew")

        state = model.get_state("alice")
        assert state.feedback_knew == 2
        assert state.feedback_novel == 1


# ── Adaptive C_redundant ─────────────────────────────────────


class TestAdaptiveAnnoyance:
    """Test C_redundant as a function of false-positive rate."""

    def test_no_feedback_returns_base(self):
        """With no feedback, effective annoyance = base * (0.5 + 0.5) = base."""
        model = AttentionModel()
        model._get_or_create("alice")
        eff = model.effective_annoyance_weight("alice")
        assert abs(eff - 0.3) < 0.001  # base * 1.0

    def test_unknown_person_returns_base(self):
        model = AttentionModel()
        eff = model.effective_annoyance_weight("nobody")
        assert abs(eff - 0.3) < 0.001

    def test_all_knew_high_annoyance(self):
        """Person who always already knew → fp_rate=1.0 → C = base * 1.5."""
        model = AttentionModel()
        model._get_or_create("alice")
        for i in range(10):
            model.record_feedback("alice", f"h{i}", "already_knew")

        eff = model.effective_annoyance_weight("alice")
        # base=0.3, fp_rate=1.0 → 0.3 * (0.5 + 1.0) = 0.45
        assert abs(eff - 0.45) < 0.001

    def test_all_novel_low_annoyance(self):
        """Person who never already knew → fp_rate=0.0 → C = base * 0.5."""
        model = AttentionModel()
        model._get_or_create("alice")
        for i in range(10):
            model.record_feedback("alice", f"h{i}", "had_no_idea")

        eff = model.effective_annoyance_weight("alice")
        # base=0.3, fp_rate=0.0 → 0.3 * (0.5 + 0.0) = 0.15
        assert abs(eff - 0.15) < 0.001

    def test_mixed_feedback_moderate(self):
        """50/50 feedback → fp_rate=0.5 → C = base * 1.0."""
        model = AttentionModel()
        model._get_or_create("alice")
        for i in range(5):
            model.record_feedback("alice", f"k{i}", "already_knew")
        for i in range(5):
            model.record_feedback("alice", f"n{i}", "had_no_idea")

        eff = model.effective_annoyance_weight("alice")
        assert abs(eff - 0.3) < 0.001

    def test_adaptive_annoyance_affects_surfacing(self):
        """Person with high fp_rate gets fewer surfaced findings."""
        model = AttentionModel()
        terms = frozenset({"pricing"})

        # Both have identical exposure
        model.record_exposure(_exposure(person_id="alice", terms=terms))
        model.record_exposure(_exposure(person_id="bob", terms=terms))

        # Alice always already knew → high annoyance → harder to surface
        for i in range(10):
            model.record_feedback("alice", f"h{i}", "already_knew")
        # Bob never knew → low annoyance → easier to surface
        for i in range(10):
            model.record_feedback("bob", f"h{i}", "had_no_idea")

        finding = {
            "finding_hash": "h1",
            "summary": "Test finding",
            "type": "risk",
            "sf_score": 0.5,
            "terms": terms,
            "signal_ids": [],
        }

        alice_decisions = rank_for_person([finding], "alice", model, _now())
        bob_decisions = rank_for_person([finding], "bob", model, _now())

        # Bob should have higher margin (lower annoyance cost)
        assert bob_decisions[0].margin > alice_decisions[0].margin


# ── Persistence ──────────────────────────────────────────────


class TestPersistence:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        model = AttentionModel()
        terms = frozenset({"pricing"})
        model.record_exposure(_exposure(
            person_id="alice",
            terms=terms,
        ))
        model.record_feedback("alice", "h1", "already_knew")

        path = tmp_path / "attention.json"
        model.save(path)

        model2 = AttentionModel()
        model2.load(path)

        state = model2.get_state("alice")
        assert state is not None
        assert len(state.exposures) == 1
        assert state.feedback_knew == 1
        assert state.attention_base_rate > 0.5

        # P(knows) should be consistent after load
        p_original = model.p_knows("alice", terms, _now())
        p_loaded = model2.p_knows("alice", terms, _now())
        assert abs(p_original - p_loaded) < 0.01

    def test_load_nonexistent_is_noop(self, tmp_path):
        model = AttentionModel()
        model.load(tmp_path / "nonexistent.json")
        assert model.all_person_ids() == []

    def test_load_corrupt_file_is_noop(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json {{{")
        model = AttentionModel()
        model.load(path)
        assert model.all_person_ids() == []


# ── PersonAttentionState ─────────────────────────────────────


class TestPersonAttentionState:
    """Test bounded exposure list and serialization."""

    def test_bounded_exposures(self):
        state = PersonAttentionState(person_id="alice")
        for i in range(600):
            state.add_exposure(_exposure(signal_id=f"sig-{i}"))
        assert len(state.exposures) == 500

    def test_to_dict_from_dict_roundtrip(self):
        state = PersonAttentionState(
            person_id="alice",
            channels_seen={"slack:#eng": 12},
            signal_volume_7d=35,
            attention_base_rate=0.72,
            feedback_knew=3,
            feedback_novel=1,
        )
        state.add_exposure(_exposure())

        d = state.to_dict()
        restored = PersonAttentionState.from_dict(d)

        assert restored.person_id == "alice"
        assert len(restored.exposures) == 1
        assert restored.channels_seen == {"slack:#eng": 12}
        assert restored.signal_volume_7d == 35
        assert restored.attention_base_rate == 0.72
        assert restored.feedback_knew == 3
        assert restored.feedback_novel == 1
