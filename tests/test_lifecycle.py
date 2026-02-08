"""Tests for context lifecycle: fork and decay.

Scenario tests as specified in the implementation guide.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stigmergy.core.energy import ContextStatus
from stigmergy.core.lifecycle import LifecycleManager
from stigmergy.primitives.agent import Agent
from stigmergy.primitives.context import Context


@pytest.fixture
def lifecycle():
    return LifecycleManager()


class TestForkTriggers:
    def test_fork_on_volume_overload(self, lifecycle):
        ctx = Context()
        ctx.signal_count = 150  # exceeds default threshold of 100
        assert lifecycle.should_fork(ctx, avg_familiarity=0.7) is True

    def test_no_fork_below_volume(self, lifecycle):
        ctx = Context()
        ctx.signal_count = 50
        assert lifecycle.should_fork(ctx, avg_familiarity=0.7) is False

    def test_fork_on_coherence_degradation(self, lifecycle):
        ctx = Context()
        ctx.signal_count = 20  # above minimum of 10
        assert lifecycle.should_fork(ctx, avg_familiarity=0.3) is True

    def test_no_fork_coherent_context(self, lifecycle):
        ctx = Context()
        ctx.signal_count = 20
        assert lifecycle.should_fork(ctx, avg_familiarity=0.8) is False

    def test_no_fork_too_few_signals(self, lifecycle):
        """Don't fork if not enough signals to judge coherence."""
        ctx = Context()
        ctx.signal_count = 5
        assert lifecycle.should_fork(ctx, avg_familiarity=0.2) is False


class TestFork:
    def test_fork_creates_children(self, lifecycle):
        ctx = Context()
        ctx.energy = 2.0
        agents = [Agent(contexts={ctx.id: 1.0})]
        result = lifecycle.fork(ctx, agents, n=2)

        assert len(result.children) == 1
        assert len(result.spawned_agents) == 1

    def test_fork_distributes_energy(self, lifecycle):
        ctx = Context()
        ctx.energy = 4.0
        result = lifecycle.fork(ctx, [], n=4)

        assert ctx.energy == pytest.approx(1.0)
        for child in result.children:
            assert child.energy == pytest.approx(1.0)

    def test_fork_establishes_neighbor_topology(self, lifecycle):
        ctx = Context()
        ctx.energy = 2.0
        result = lifecycle.fork(ctx, [], n=2)

        child = result.children[0]
        assert child.id in ctx.neighbors
        assert ctx.id in child.neighbors

    def test_fork_sets_parent(self, lifecycle):
        ctx = Context()
        ctx.energy = 2.0
        result = lifecycle.fork(ctx, [], n=2)

        assert result.children[0].parent_id == ctx.id

    def test_spawned_agents_bound_to_children(self, lifecycle):
        ctx = Context()
        ctx.energy = 3.0
        result = lifecycle.fork(ctx, [], n=3)

        for i, agent in enumerate(result.spawned_agents):
            child = result.children[i]
            assert child.id in agent.contexts


class TestDecay:
    def test_active_context(self, lifecycle):
        ctx = Context()
        ctx.energy = 1.0
        ctx.last_signal = datetime.now(timezone.utc)
        status = lifecycle.check_energy(ctx)
        assert status == ContextStatus.ACTIVE

    def test_decaying_context(self, lifecycle):
        ctx = Context()
        ctx.energy = 0.25
        ctx.last_signal = datetime.now(timezone.utc) - timedelta(seconds=1)
        # With energy 0.25 and very recent signal, decay rate matters
        # energy = 0.25 * e^(-0.001 * 1) ≈ 0.2497
        status = lifecycle.check_energy(ctx)
        assert status == ContextStatus.DECAYING

    def test_archived_context(self, lifecycle):
        ctx = Context()
        ctx.energy = 0.01
        ctx.last_signal = datetime.now(timezone.utc) - timedelta(seconds=1)
        status = lifecycle.check_energy(ctx)
        assert status == ContextStatus.ARCHIVED


class TestDecayHandoff:
    def test_handoff_to_forking_neighbor(self, lifecycle):
        decaying = Context()
        neighbor = Context()
        decaying.neighbors.add(neighbor.id)
        assert lifecycle.decay_handoff(decaying, neighbor) is True

    def test_no_handoff_to_non_neighbor(self, lifecycle):
        decaying = Context()
        stranger = Context()
        assert lifecycle.decay_handoff(decaying, stranger) is False
