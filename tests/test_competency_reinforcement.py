"""Tests for competency reinforcement loop — closing the open loop."""

from __future__ import annotations

from uuid import uuid4

import pytest

from stigmergy.mesh.corroboration import (
    QuorumFinding,
    WorkerVerdict,
    reinforce_competencies,
)
from stigmergy.mesh.insights import Insight
from stigmergy.primitives.agent import Agent, CompetencyModel


# ── Helpers ──────────────────────────────────────────────────


def _make_agent(competencies: dict[str, float] | None = None) -> Agent:
    comp = CompetencyModel(competencies or {
        "cross_context_propagation": 0.5,
        "temporal_analysis": 0.5,
        "anomaly_detection": 0.5,
        "similarity_detection": 0.5,
    })
    return Agent(competencies=comp)


def _make_qf(
    agent_id,
    classification: str = "correlation",
    pattern_type: str = "risk",
    verdicts: list[tuple[str, float]] | None = None,
) -> QuorumFinding:
    """Build a QuorumFinding with specified verdicts.

    verdicts: list of (verdict_str, confidence) tuples
    """
    if verdicts is None:
        verdicts = [("corroborate", 0.8), ("corroborate", 0.7)]

    insight = Insight(
        agent_id=agent_id,
        type=pattern_type,
        summary="Test finding",
        confidence=0.75,
        signal_ids=[uuid4()],
        details={"classification": classification},
    )

    worker_verdicts = [
        WorkerVerdict(
            worker_id=uuid4(),
            finding_summary="Test finding",
            verdict=v,
            confidence=c,
            reasoning="test",
            related_context="test",
        )
        for v, c in verdicts
    ]

    return QuorumFinding(
        original=insight,
        verdicts=worker_verdicts,
        corroboration_count=sum(1 for v, _ in verdicts if v in ("corroborate", "partial")),
        contradiction_count=sum(1 for v, _ in verdicts if v == "contradict"),
        total_consulted=len(verdicts),
        quorum_confidence=0.7,
        quorum_met=True,
    )


# ── Tests ────────────────────────────────────────────────────


class TestCompetencyReinforcement:
    def test_corroboration_boosts_competency(self):
        agent = _make_agent()
        initial = agent.competencies.strength("cross_context_propagation")

        qf = _make_qf(
            agent.id,
            classification="correlation",
            verdicts=[("corroborate", 0.8)],
        )
        totals = reinforce_competencies([qf], [agent])

        new_val = agent.competencies.strength("cross_context_propagation")
        assert new_val > initial
        assert "cross_context_propagation" in totals
        assert totals["cross_context_propagation"] > 0

    def test_contradiction_dampens_competency(self):
        agent = _make_agent()
        initial = agent.competencies.strength("cross_context_propagation")

        qf = _make_qf(
            agent.id,
            classification="correlation",
            verdicts=[("contradict", 0.9)],
        )
        totals = reinforce_competencies([qf], [agent])

        new_val = agent.competencies.strength("cross_context_propagation")
        assert new_val < initial
        assert totals["cross_context_propagation"] < 0

    def test_asymmetric_deltas(self):
        """Contradiction costs more than corroboration gains."""
        agent_boost = _make_agent()
        agent_damp = _make_agent()

        qf_boost = _make_qf(
            agent_boost.id,
            classification="correlation",
            verdicts=[("corroborate", 1.0)],
        )
        qf_damp = _make_qf(
            agent_damp.id,
            classification="correlation",
            verdicts=[("contradict", 1.0)],
        )

        totals_boost = reinforce_competencies([qf_boost], [agent_boost])
        totals_damp = reinforce_competencies([qf_damp], [agent_damp])

        # Contradiction delta magnitude > corroboration delta magnitude
        assert abs(totals_damp["cross_context_propagation"]) > abs(totals_boost["cross_context_propagation"])

    def test_partial_verdict_small_positive(self):
        agent = _make_agent()
        initial = agent.competencies.strength("cross_context_propagation")

        qf = _make_qf(
            agent.id,
            classification="correlation",
            verdicts=[("partial", 0.8)],
        )
        totals = reinforce_competencies([qf], [agent])

        new_val = agent.competencies.strength("cross_context_propagation")
        assert new_val > initial
        # Partial boost is smaller than full corroboration
        assert totals["cross_context_propagation"] < 0.02

    def test_no_evidence_zero_delta(self):
        agent = _make_agent()
        initial = agent.competencies.strength("cross_context_propagation")

        qf = _make_qf(
            agent.id,
            classification="correlation",
            verdicts=[("no_evidence", 0.8)],
        )
        totals = reinforce_competencies([qf], [agent])

        assert agent.competencies.strength("cross_context_propagation") == initial
        assert totals == {}

    def test_classification_to_competency_mapping(self):
        """Different classifications map to different competencies."""
        agent = _make_agent()

        # amplification -> temporal_analysis
        qf = _make_qf(agent.id, classification="amplification", verdicts=[("corroborate", 0.8)])
        totals = reinforce_competencies([qf], [agent])
        assert "temporal_analysis" in totals

    def test_risk_pattern_maps_to_anomaly_detection(self):
        agent = _make_agent()
        initial = agent.competencies.strength("anomaly_detection")

        qf = _make_qf(agent.id, classification="risk", verdicts=[("corroborate", 0.8)])
        reinforce_competencies([qf], [agent])

        assert agent.competencies.strength("anomaly_detection") > initial

    def test_overlap_pattern_maps_to_similarity_detection(self):
        agent = _make_agent()
        initial = agent.competencies.strength("similarity_detection")

        qf = _make_qf(agent.id, classification="overlap", verdicts=[("corroborate", 0.8)])
        reinforce_competencies([qf], [agent])

        assert agent.competencies.strength("similarity_detection") > initial

    def test_clamped_to_zero_one(self):
        """Competency weights stay in [0, 1]."""
        agent = _make_agent({"cross_context_propagation": 0.99})

        # Many corroborations shouldn't exceed 1.0
        qf = _make_qf(agent.id, classification="correlation",
                       verdicts=[("corroborate", 1.0)] * 10)
        reinforce_competencies([qf], [agent])
        assert agent.competencies.strength("cross_context_propagation") <= 1.0

        # Many contradictions shouldn't go below 0.0
        agent2 = _make_agent({"cross_context_propagation": 0.01})
        qf2 = _make_qf(agent2.id, classification="correlation",
                        verdicts=[("contradict", 1.0)] * 10)
        reinforce_competencies([qf2], [agent2])
        assert agent2.competencies.strength("cross_context_propagation") >= 0.0

    def test_empty_inputs(self):
        assert reinforce_competencies([], []) == {}
        assert reinforce_competencies([], [_make_agent()]) == {}

    def test_unknown_classification_skipped(self):
        agent = _make_agent()
        initial_weights = dict(agent.competencies.weights)

        qf = _make_qf(agent.id, classification="unknown_type", pattern_type="unknown_pattern",
                       verdicts=[("corroborate", 0.8)])
        totals = reinforce_competencies([qf], [agent])

        # No change — unknown classification doesn't map to any competency
        assert totals == {}
        assert agent.competencies.weights == initial_weights

    def test_multiple_findings_accumulate(self):
        agent = _make_agent()
        initial = agent.competencies.strength("cross_context_propagation")

        findings = [
            _make_qf(agent.id, classification="correlation",
                     verdicts=[("corroborate", 0.8)])
            for _ in range(5)
        ]
        reinforce_competencies(findings, [agent])

        new_val = agent.competencies.strength("cross_context_propagation")
        assert new_val > initial
        # 5 findings * 0.02 * 0.8 = 0.08 increase from seed 0.5
        assert new_val == pytest.approx(0.58, abs=0.01)

    def test_confidence_scales_delta(self):
        """Higher verdict confidence = larger delta."""
        agent_low = _make_agent()
        agent_high = _make_agent()

        qf_low = _make_qf(agent_low.id, classification="correlation",
                           verdicts=[("corroborate", 0.3)])
        qf_high = _make_qf(agent_high.id, classification="correlation",
                            verdicts=[("corroborate", 0.9)])

        reinforce_competencies([qf_low], [agent_low])
        reinforce_competencies([qf_high], [agent_high])

        # Higher confidence should produce larger increase
        assert agent_high.competencies.strength("cross_context_propagation") > \
               agent_low.competencies.strength("cross_context_propagation")
