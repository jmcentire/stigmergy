"""Tests for weighted consensus algorithm.

Key scenarios: agents disagree with different confidence levels and
domain weights. Verify the math produces the right winner.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from stigmergy.core.consensus import consensus, ConsensusResult
from stigmergy.primitives.agent import Agent
from stigmergy.primitives.assessment import Assessment, AssessmentAction


def _make_agent(confidence: float = 0.8, weights: dict | None = None) -> Agent:
    return Agent(
        confidence=confidence,
        weights=weights or {"technical": 0.8, "business": 0.5},
    )


def _make_assessment(agent: Agent, action: AssessmentAction,
                     confidence: float = 0.8, domain: str = "technical",
                     signal_id=None, context_id=None) -> Assessment:
    return Assessment(
        agent_id=agent.id,
        signal_id=signal_id or uuid4(),
        context_id=context_id or uuid4(),
        action=action,
        confidence=confidence,
        domain=domain,
        reasoning="test",
    )


class TestBasicConsensus:
    def test_empty_assessments(self):
        result = consensus([], {})
        assert result.action == AssessmentAction.STORE
        assert result.resolved is False
        assert result.total_weight == 0.0

    def test_single_high_confidence_vote(self):
        agent = _make_agent(confidence=0.9, weights={"technical": 0.9})
        assessment = _make_assessment(agent, AssessmentAction.SURFACE, confidence=0.9)
        result = consensus([assessment], {agent.id: agent}, consensus_threshold=0.6)
        assert result.action == AssessmentAction.SURFACE
        assert result.resolved is True

    def test_single_low_confidence_vote(self):
        agent = _make_agent(confidence=0.3, weights={"technical": 0.3})
        assessment = _make_assessment(agent, AssessmentAction.SURFACE, confidence=0.3)
        result = consensus([assessment], {agent.id: agent}, consensus_threshold=0.6)
        # 0.3 * 0.3 * 0.3 = 0.027 — way below threshold
        assert result.resolved is False


class TestDisagreement:
    """Three agents disagree. Domain weights determine the winner."""

    def test_domain_weight_determines_winner(self):
        """Agent caring about technical domain should dominate technical votes."""
        sig_id = uuid4()
        ctx_id = uuid4()

        # Agent A: high technical weight, votes SURFACE
        agent_a = _make_agent(confidence=0.8, weights={"technical": 0.9})
        # Agent B: low technical weight, votes IGNORE
        agent_b = _make_agent(confidence=0.8, weights={"technical": 0.2})
        # Agent C: medium technical weight, votes STORE
        agent_c = _make_agent(confidence=0.8, weights={"technical": 0.5})

        assessments = [
            _make_assessment(agent_a, AssessmentAction.SURFACE, confidence=0.8,
                           domain="technical", signal_id=sig_id, context_id=ctx_id),
            _make_assessment(agent_b, AssessmentAction.IGNORE, confidence=0.8,
                           domain="technical", signal_id=sig_id, context_id=ctx_id),
            _make_assessment(agent_c, AssessmentAction.STORE, confidence=0.8,
                           domain="technical", signal_id=sig_id, context_id=ctx_id),
        ]

        agents = {a.id: a for a in [agent_a, agent_b, agent_c]}
        result = consensus(assessments, agents, consensus_threshold=0.5)

        # Agent A's vote weight: 0.8 * 0.8 * 0.9 = 0.576
        # Agent B's vote weight: 0.8 * 0.8 * 0.2 = 0.128
        # Agent C's vote weight: 0.8 * 0.8 * 0.5 = 0.320
        # SURFACE wins with 0.576
        assert result.action == AssessmentAction.SURFACE

    def test_business_domain_different_winner(self):
        """Same agents, but business domain — different weights, different winner."""
        sig_id = uuid4()
        ctx_id = uuid4()

        # Agent A: low business weight, votes SURFACE
        agent_a = _make_agent(confidence=0.8, weights={"business": 0.1})
        # Agent B: high business weight, votes BLOCK
        agent_b = _make_agent(confidence=0.8, weights={"business": 0.9})
        # Agent C: medium business weight, votes BLOCK
        agent_c = _make_agent(confidence=0.8, weights={"business": 0.6})

        assessments = [
            _make_assessment(agent_a, AssessmentAction.SURFACE, confidence=0.8,
                           domain="business", signal_id=sig_id, context_id=ctx_id),
            _make_assessment(agent_b, AssessmentAction.BLOCK, confidence=0.8,
                           domain="business", signal_id=sig_id, context_id=ctx_id),
            _make_assessment(agent_c, AssessmentAction.BLOCK, confidence=0.8,
                           domain="business", signal_id=sig_id, context_id=ctx_id),
        ]

        agents = {a.id: a for a in [agent_a, agent_b, agent_c]}
        result = consensus(assessments, agents, consensus_threshold=0.5)

        # Agent A SURFACE weight: 0.8 * 0.8 * 0.1 = 0.064
        # Agent B BLOCK weight: 0.8 * 0.8 * 0.9 = 0.576
        # Agent C BLOCK weight: 0.8 * 0.8 * 0.6 = 0.384
        # BLOCK total: 0.960
        assert result.action == AssessmentAction.BLOCK
        assert result.resolved is True

    def test_confidence_breaks_ties(self):
        """When domain weights are equal, agent confidence decides."""
        sig_id = uuid4()
        ctx_id = uuid4()

        # Same domain weights, different confidence
        agent_hi = _make_agent(confidence=0.9, weights={"technical": 0.5})
        agent_lo = _make_agent(confidence=0.3, weights={"technical": 0.5})

        assessments = [
            _make_assessment(agent_hi, AssessmentAction.SURFACE, confidence=0.9,
                           domain="technical", signal_id=sig_id, context_id=ctx_id),
            _make_assessment(agent_lo, AssessmentAction.IGNORE, confidence=0.3,
                           domain="technical", signal_id=sig_id, context_id=ctx_id),
        ]

        agents = {a.id: a for a in [agent_hi, agent_lo]}
        result = consensus(assessments, agents, consensus_threshold=0.3)

        # Hi: 0.9 * 0.9 * 0.5 = 0.405
        # Lo: 0.3 * 0.3 * 0.5 = 0.045
        assert result.action == AssessmentAction.SURFACE


class TestThresholds:
    def test_uncertainty_zone(self):
        """Score between uncertainty and consensus threshold → flagged for review."""
        agent = _make_agent(confidence=0.6, weights={"technical": 0.7})
        assessment = _make_assessment(agent, AssessmentAction.SURFACE, confidence=0.7)

        result = consensus(
            [assessment],
            {agent.id: agent},
            consensus_threshold=0.5,
            uncertainty_threshold=0.2,
        )
        # Weight: 0.7 * 0.6 * 0.7 = 0.294
        # 0.2 < 0.294 < 0.5 → flagged
        assert result.resolved is False
        assert result.flagged is True

    def test_below_uncertainty_threshold(self):
        """Score below uncertainty threshold → stored silently."""
        agent = _make_agent(confidence=0.2, weights={"technical": 0.2})
        assessment = _make_assessment(agent, AssessmentAction.SURFACE, confidence=0.2)

        result = consensus(
            [assessment],
            {agent.id: agent},
            consensus_threshold=0.5,
            uncertainty_threshold=0.3,
        )
        # Weight: 0.2 * 0.2 * 0.2 = 0.008
        assert result.resolved is False
        assert result.flagged is False

    def test_unknown_agent_ignored(self):
        """Assessment from unknown agent contributes no weight."""
        agent = _make_agent()
        assessment = _make_assessment(agent, AssessmentAction.SURFACE, confidence=0.9)
        # Don't include agent in registry
        result = consensus([assessment], {})
        assert result.total_weight == 0.0


class TestVoteTracking:
    def test_all_participating_agents_tracked(self):
        agents = [_make_agent() for _ in range(3)]
        assessments = [
            _make_assessment(a, AssessmentAction.STORE, confidence=0.5, domain="technical")
            for a in agents
        ]
        agent_dict = {a.id: a for a in agents}
        result = consensus(assessments, agent_dict)
        assert len(result.participating_agents) == 3

    def test_vote_breakdown_present(self):
        agent = _make_agent(confidence=0.9, weights={"technical": 0.9})
        assessment = _make_assessment(agent, AssessmentAction.SURFACE, confidence=0.9)
        result = consensus([assessment], {agent.id: agent})
        assert AssessmentAction.SURFACE in result.votes

    def test_custom_action_flows_through_consensus(self):
        """Agent-chosen action strings (not in enum) flow through consensus."""
        agent = _make_agent(confidence=0.9, weights={"technical": 0.9})
        assessment = _make_assessment(agent, "escalate_to_oncall", confidence=0.9)
        result = consensus([assessment], {agent.id: agent})
        assert result.action == "escalate_to_oncall"
        assert result.resolved
        assert "escalate_to_oncall" in result.votes
