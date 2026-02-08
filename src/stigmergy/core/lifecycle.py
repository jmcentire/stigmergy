"""Context lifecycle: Fork / Decay.

Agents and contexts are not static. They are born, they evolve, and they die.

Fork: context can no longer adequately serve its signal space.
Decay: context stopped receiving relevant signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from stigmergy.core.energy import ContextStatus, compute_energy
from stigmergy.primitives.agent import Agent, CompetencyModel, RuleSet
from stigmergy.primitives.context import Context
from stigmergy.services.vector_store import InMemoryVectorStore


@dataclass
class ForkResult:
    original: Context
    children: list[Context]
    spawned_agents: list[Agent]


class LifecycleManager:
    """Manages context and agent lifecycle events.

    Not an orchestrator — monitors metrics and triggers events.
    """

    def __init__(
        self,
        *,
        fork_coherence_threshold: float = 0.5,
        fork_volume_threshold: int = 100,
        decay_threshold: float = 0.3,
        archive_threshold: float = 0.05,
    ) -> None:
        self.fork_coherence_threshold = fork_coherence_threshold
        self.fork_volume_threshold = fork_volume_threshold
        self.decay_threshold = decay_threshold
        self.archive_threshold = archive_threshold

    def check_energy(self, context: Context, now: datetime | None = None) -> ContextStatus:
        """Check a context's energy and return its lifecycle status."""
        now = now or datetime.now(timezone.utc)
        elapsed = (now - context.last_signal).total_seconds()
        state = compute_energy(
            base_energy=context.energy,
            decay_rate=context.relevance_decay,
            elapsed_seconds=elapsed,
            decay_threshold=self.decay_threshold,
            archive_threshold=self.archive_threshold,
        )
        return state.status

    def should_fork(self, context: Context, avg_familiarity: float) -> bool:
        """Determine if a context should fork.

        Fork triggers:
        1. Volume overload: signal count exceeds threshold.
        2. Coherence degradation: avg familiarity of incoming signals drops.
        """
        if context.signal_count > self.fork_volume_threshold:
            return True
        if context.signal_count > 10 and avg_familiarity < self.fork_coherence_threshold:
            return True
        return False

    def fork(self, context: Context, agents: list[Agent], n: int = 2) -> ForkResult:
        """Fork a context into n partitions.

        Creates n-1 new contexts. Original keeps first partition.
        Energy is distributed equally. New agents inherit seed competencies.
        """
        children: list[Context] = []
        spawned_agents: list[Agent] = []

        distributed_energy = context.energy / n
        context.energy = distributed_energy
        # Split signal_count so the parent doesn't re-trigger fork immediately
        distributed_count = context.signal_count // n
        context.signal_count = distributed_count

        for i in range(n - 1):
            child = Context(
                relevance_decay=context.relevance_decay,
                relevance_threshold=context.relevance_threshold,
                business_weight=context.business_weight,
                parent_id=context.id,
            )
            child.energy = distributed_energy
            # Establish neighbor relationship
            child.neighbors.add(context.id)
            context.neighbors.add(child.id)
            children.append(child)

            # Spawn agent for new context with seed competencies
            agent = Agent(
                contexts={child.id: 1.0},
                competencies=CompetencyModel(),  # seed defaults
                confidence=0.5,
            )
            child.active_agents.add(agent.id)
            spawned_agents.append(agent)

        return ForkResult(original=context, children=children, spawned_agents=spawned_agents)

    def decay_handoff(self, decaying: Context, forking_neighbor: Context) -> bool:
        """Check if a decaying context should fold into a neighbor's fork.

        If the decaying context's knowledge is near a forking neighbor,
        absorb it rather than letting it die and be rediscovered.
        """
        if forking_neighbor.id not in decaying.neighbors:
            return False
        # The decaying context's terms and knowledge should be folded into
        # the forking neighbor's new partition.
        return True
