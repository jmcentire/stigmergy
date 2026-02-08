"""Weighted consensus algorithm.

Agent domain weights feed into vote weights. The consensus algorithm
doesn't decide what matters — the agents tell it what they care about.

vote_weight = assessment.confidence * agent.confidence * agent.weights[domain]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from uuid import UUID

from stigmergy.primitives.agent import Agent
from stigmergy.primitives.assessment import Assessment, AssessmentAction


@dataclass
class ConsensusResult:
    action: str
    total_weight: float
    resolved: bool
    flagged: bool = False
    votes: dict[str, float] = field(default_factory=dict)
    participating_agents: list[UUID] = field(default_factory=list)


def consensus(
    assessments: list[Assessment],
    agents: dict[UUID, Agent],
    consensus_threshold: float = 0.6,
    uncertainty_threshold: float = 0.3,
) -> ConsensusResult:
    """Resolve multiple assessments into a single action via weighted vote.

    Pure function. Deterministic given identical inputs.

    Args:
        assessments: All assessments for a signal in a context.
        agents: Agent registry for looking up confidence and domain weights.
        consensus_threshold: Weight above which the winning action is executed.
        uncertainty_threshold: Weight above which we store + flag for review.

    Returns:
        ConsensusResult with the resolved action and vote breakdown.
    """
    if not assessments:
        return ConsensusResult(
            action=AssessmentAction.STORE,
            total_weight=0.0,
            resolved=False,
            votes={},
            participating_agents=[],
        )

    action_weights: dict[str, float] = defaultdict(float)
    participating: list[UUID] = []

    for assessment in assessments:
        agent = agents.get(assessment.agent_id)
        if agent is None:
            continue

        vote_weight = (
            assessment.confidence
            * agent.confidence
            * agent.domain_weight(assessment.domain)
        )

        action_weights[assessment.action] += vote_weight
        participating.append(assessment.agent_id)

    if not action_weights:
        return ConsensusResult(
            action=AssessmentAction.STORE,
            total_weight=0.0,
            resolved=False,
            votes=dict(action_weights),
            participating_agents=participating,
        )

    max_action = max(action_weights, key=lambda a: action_weights[a])
    max_weight = action_weights[max_action]

    if max_weight > consensus_threshold:
        return ConsensusResult(
            action=max_action,
            total_weight=max_weight,
            resolved=True,
            votes=dict(action_weights),
            participating_agents=participating,
        )
    elif max_weight > uncertainty_threshold:
        return ConsensusResult(
            action=AssessmentAction.STORE,
            total_weight=max_weight,
            resolved=False,
            flagged=True,
            votes=dict(action_weights),
            participating_agents=participating,
        )
    else:
        return ConsensusResult(
            action=AssessmentAction.STORE,
            total_weight=max_weight,
            resolved=False,
            flagged=False,
            votes=dict(action_weights),
            participating_agents=participating,
        )
