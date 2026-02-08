"""Tests for agent intelligence monitoring and LLM propagation.

Verifies the critical invariant: every agent in a running mesh must have
LLM intelligence available. Without it, agents fall back to mechanical
heuristics that produce noise instead of insight.

These tests exercise:
1. Intelligence monitoring counters on Agent
2. LLM propagation through the fork lifecycle
3. LLM propagation through mesh worker creation paths
4. Diagnostics reporting
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest

from stigmergy.core.lifecycle import LifecycleManager
from stigmergy.mesh.mesh import Mesh
from stigmergy.mesh.worker import WorkerNode
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.agent import Agent, CompetencyModel
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal


def _run(coro):
    return asyncio.run(coro)


# ── Fake LLM for testing ──────────────────────────────────────────


class FakeLLM:
    """Minimal LLM stub that records calls and returns canned responses."""

    def __init__(self):
        self.extract_calls: list[dict] = []

    async def extract(self, schema, prompt, system, max_tokens=200):
        self.extract_calls.append({
            "schema": schema.__name__ if hasattr(schema, "__name__") else str(schema),
            "prompt_len": len(prompt),
            "max_tokens": max_tokens,
        })
        # Return a minimal valid response for each known schema
        name = schema.__name__ if hasattr(schema, "__name__") else ""
        if name == "SurfaceDecision":
            return _FakeSurfaceDecision()
        if name == "CorrelationBatch":
            return _FakeCorrelationBatch()
        if name == "SignalFacts":
            return _FakeSignalFacts()
        if name == "WorkerProfile":
            return _FakeWorkerProfile()
        raise ValueError(f"Unknown schema: {name}")


@dataclass
class _FakeSurfaceDecision:
    action: str = "store"
    confidence: float = 0.5
    reasoning: str = "test decision"


@dataclass
class _FakeCorrelationBatch:
    insights: list = None
    batch_compression: float = 0.0
    batch_temporal: float = 0.0
    batch_bureaucratic: float = 0.0
    batch_exploration: float = 0.0

    def __post_init__(self):
        if self.insights is None:
            self.insights = []


@dataclass
class _FakeSignalFacts:
    concepts: list = None

    def __post_init__(self):
        if self.concepts is None:
            self.concepts = ["test_concept"]


@dataclass
class _FakeWorkerProfile:
    specialization: str = "test specialization"


def _make_signal(**kwargs) -> Signal:
    defaults = {
        "source": "github",
        "channel": "test-repo",
        "author": "test-user",
        "content": "Test signal content for evaluation",
        "timestamp": datetime.now(timezone.utc),
        "metadata": {},
    }
    defaults.update(kwargs)
    return Signal(**defaults)


# ── Agent Intelligence Monitoring ──────────────────────────────────


class TestAgentDiagnostics:
    """Test the intelligence monitoring counters on Agent."""

    def test_fresh_agent_has_zero_counters(self):
        agent = Agent()
        diag = agent.diagnostics()
        assert diag["reflect"]["llm_calls"] == 0
        assert diag["reflect"]["mechanical_calls"] == 0
        assert diag["reflect"]["llm_skipped_non_batch"] == 0
        assert diag["evaluate"]["llm_calls"] == 0
        assert diag["evaluate"]["mechanical_calls"] == 0

    def test_has_llm_property(self):
        agent_no_llm = Agent()
        agent_with_llm = Agent(llm=FakeLLM())
        assert agent_no_llm.has_llm is False
        assert agent_with_llm.has_llm is True

    def test_diagnostics_reports_has_llm(self):
        agent = Agent(llm=FakeLLM())
        assert agent.diagnostics()["has_llm"] is True

    def test_diagnostics_reports_no_llm(self):
        agent = Agent()
        assert agent.diagnostics()["has_llm"] is False

    def test_intelligence_active_when_llm_only(self):
        agent = Agent(llm=FakeLLM())
        # No calls yet, but LLM present and no mechanical calls
        assert agent.diagnostics()["intelligence_active"] is True

    def test_intelligence_not_active_without_llm(self):
        agent = Agent()
        assert agent.diagnostics()["intelligence_active"] is False


class TestEvaluateCounters:
    """Test that evaluate() correctly increments LLM vs mechanical counters."""

    def test_mechanical_evaluate_increments(self):
        agent = Agent(contexts={uuid4(): 1.0}, confidence=0.5)
        sig = _make_signal()
        ctx_id = list(agent.contexts.keys())[0]
        _run(agent.evaluate(sig, ctx_id, "technical"))
        diag = agent.diagnostics()
        assert diag["evaluate"]["mechanical_calls"] == 1
        assert diag["evaluate"]["llm_calls"] == 0

    def test_llm_evaluate_increments(self):
        llm = FakeLLM()
        agent = Agent(contexts={uuid4(): 1.0}, confidence=0.5, llm=llm)
        sig = _make_signal()
        ctx_id = list(agent.contexts.keys())[0]
        _run(agent.evaluate(sig, ctx_id, "technical"))
        diag = agent.diagnostics()
        assert diag["evaluate"]["llm_calls"] == 1
        assert diag["evaluate"]["mechanical_calls"] == 0
        assert len(llm.extract_calls) == 1

    def test_multiple_evaluates_accumulate(self):
        agent = Agent(contexts={uuid4(): 1.0}, confidence=0.5)
        sig = _make_signal()
        ctx_id = list(agent.contexts.keys())[0]
        for _ in range(5):
            _run(agent.evaluate(sig, ctx_id, "technical"))
        assert agent.diagnostics()["evaluate"]["mechanical_calls"] == 5


class TestReflectCounters:
    """Test that reflect() correctly increments LLM vs mechanical counters."""

    def _make_obs(self, signal=None):
        """Build a minimal ObservationContext for testing."""
        from stigmergy.mesh.insights import ObservationContext, SignalCache
        from stigmergy.mesh.mesh import MeshTrace

        sig = signal or _make_signal()
        trace = MeshTrace(signal_id=sig.id)
        return ObservationContext(
            signal=sig,
            trace=trace,
            spectrum=[],
            cache=SignalCache(capacity=10),
            workers=[],
        )

    def test_mechanical_reflect_increments(self):
        agent = Agent(confidence=0.5)
        obs = self._make_obs()
        _run(agent.reflect(obs))
        diag = agent.diagnostics()
        assert diag["reflect"]["mechanical_calls"] == 1
        assert diag["reflect"]["llm_calls"] == 0

    def test_llm_reflect_skipped_on_non_batch(self):
        """LLM-equipped agent skips reflect on non-batch signals (counter 1-9)."""
        agent = Agent(confidence=0.5, llm=FakeLLM())
        obs = self._make_obs()
        # First call: counter becomes 1, not a multiple of 10
        _run(agent.reflect(obs))
        diag = agent.diagnostics()
        assert diag["reflect"]["llm_skipped_non_batch"] == 1
        assert diag["reflect"]["llm_calls"] == 0
        assert diag["reflect"]["mechanical_calls"] == 0

    def test_llm_reflect_fires_on_batch_interval(self):
        """LLM-equipped agent fires correlator every 10th signal."""
        from stigmergy.mesh.insights import SignalCache
        from stigmergy.mesh.mesh import MeshTrace

        llm = FakeLLM()
        agent = Agent(confidence=0.5, llm=llm)

        # Populate cache with signals so the correlator has data to work with
        cache = SignalCache(capacity=50)
        for i in range(5):
            s = _make_signal(content=f"Signal {i} for testing correlator batch")
            t = MeshTrace(signal_id=s.id)
            cache.add(s, t, [0.5], [uuid4()])

        from stigmergy.mesh.insights import ObservationContext

        # Run 10 reflects — the 10th should fire LLM
        for i in range(10):
            sig = _make_signal(content=f"Batch test signal {i}")
            trace = MeshTrace(signal_id=sig.id)
            obs = ObservationContext(
                signal=sig, trace=trace, spectrum=[0.5],
                cache=cache, workers=[],
            )
            _run(agent.reflect(obs))

        diag = agent.diagnostics()
        assert diag["reflect"]["llm_calls"] == 1
        assert diag["reflect"]["llm_skipped_non_batch"] == 9
        assert diag["reflect"]["mechanical_calls"] == 0
        # Verify the LLM was actually called
        assert any(c["schema"] == "CorrelationBatch" for c in llm.extract_calls)

    def test_no_mechanical_when_llm_present(self):
        """Critical invariant: LLM-equipped agents NEVER use mechanical reflect."""
        agent = Agent(confidence=0.5, llm=FakeLLM())
        obs = self._make_obs()
        for _ in range(25):
            _run(agent.reflect(obs))
        diag = agent.diagnostics()
        assert diag["reflect"]["mechanical_calls"] == 0
        assert diag["intelligence_active"] is True


# ── LLM Propagation Through Fork ──────────────────────────────────


class TestForkLLMPropagation:
    """Test that forked agents receive LLM from the mesh."""

    def test_lifecycle_fork_creates_agents_without_llm(self):
        """LifecycleManager.fork() intentionally creates naked agents.
        The mesh is responsible for injecting LLM after fork."""
        lifecycle = LifecycleManager()
        ctx = Context()
        ctx.energy = 2.0
        ctx.signal_count = 200
        result = lifecycle.fork(ctx, [], n=2)
        # Forked agent has no LLM (lifecycle doesn't know about LLM)
        assert len(result.spawned_agents) == 1
        assert result.spawned_agents[0]._llm is None

    def test_mesh_check_forks_injects_llm(self):
        """When mesh forks a worker, spawned agents must receive LLM."""
        llm = FakeLLM()
        agents = AgentRegistry()
        mesh = Mesh(agents, llm=llm, max_workers=10, worker_capacity=50)

        # Create a worker that will trigger fork (high signal count)
        ctx = Context(capacity=50)
        ctx.signal_count = 200  # way over capacity
        worker = WorkerNode(ctx, llm=llm)
        mesh.add_worker(worker)

        # Register an agent for this context
        agent = Agent(contexts={ctx.id: 1.0}, llm=llm)
        agents.register(agent)

        # Trigger fork via lifecycle check
        mesh._check_forks()

        # All agents in registry should have LLM
        for a in agents.all_agents():
            assert a._llm is not None, f"Agent {a.id} missing LLM after fork"
            assert a.has_llm, f"Agent {a.id} has_llm should be True"

    def test_mesh_fork_child_workers_get_llm(self):
        """Forked child workers must receive LLM from mesh."""
        llm = FakeLLM()
        agents = AgentRegistry()
        mesh = Mesh(agents, llm=llm, max_workers=10, worker_capacity=50)

        ctx = Context(capacity=50)
        ctx.signal_count = 200
        worker = WorkerNode(ctx, llm=llm)
        mesh.add_worker(worker)

        agent = Agent(contexts={ctx.id: 1.0}, llm=llm)
        agents.register(agent)

        initial_workers = mesh.worker_count
        mesh._check_forks()

        # Should have spawned a child worker
        assert mesh.worker_count > initial_workers

        # All workers must have LLM
        for w in mesh.workers:
            assert w._llm is not None, f"Worker {w.id} missing LLM"


class TestSpawnWorkerLLMPropagation:
    """Test LLM propagation through mesh.spawn_worker()."""

    def test_spawn_worker_inherits_llm(self):
        llm = FakeLLM()
        agents = AgentRegistry()
        mesh = Mesh(agents, llm=llm)
        worker = mesh.spawn_worker()
        assert worker._llm is llm

    def test_spawn_worker_no_llm_when_mesh_has_none(self):
        agents = AgentRegistry()
        mesh = Mesh(agents, llm=None)
        worker = mesh.spawn_worker()
        assert worker._llm is None


class TestGapDetectionLLMPropagation:
    """Test that gap-spawned workers get LLM."""

    def test_gap_spawned_worker_has_llm(self):
        """Workers spawned by gap detection must have LLM."""
        llm = FakeLLM()
        agents = AgentRegistry()
        mesh = Mesh(agents, llm=llm, gap_threshold=0.5, max_workers=10)

        # Add a seed worker
        ctx = Context(capacity=200)
        worker = WorkerNode(ctx, llm=llm)
        mesh.add_worker(worker)

        # Create a signal and trace that triggers gap detection
        sig = _make_signal()
        from stigmergy.mesh.mesh import MeshTrace
        from stigmergy.structures.bloom import CountingBloomFilter
        trace = MeshTrace(signal_id=sig.id)
        # All scores below gap_threshold → triggers gap
        trace.familiarity_scores[worker.id] = 0.01

        signal_bloom = CountingBloomFilter(capacity=10000)
        signal_bloom.add_many(sig.terms)

        initial_count = mesh.worker_count
        _run(mesh._handle_gap(sig, signal_bloom, trace))

        assert mesh.worker_count > initial_count
        # The new worker should have LLM
        new_workers = [w for w in mesh.workers if w.id != worker.id]
        for w in new_workers:
            assert w._llm is llm, f"Gap-spawned worker {w.id} missing LLM"


# ── Integration: All Agents Must Have LLM ──────────────────────────


class TestAllAgentsInvariant:
    """Integration test: after mesh operations, all agents must have LLM."""

    def test_no_naked_agents_after_fork_cycle(self):
        """Simulate a full fork cycle and verify no agents are naked."""
        llm = FakeLLM()
        agents = AgentRegistry()
        mesh = Mesh(agents, llm=llm, max_workers=20, worker_capacity=30)

        # Create multiple workers that will fork
        for i in range(3):
            ctx = Context(capacity=30)
            ctx.signal_count = 100  # triggers fork
            ctx.terms = {f"term_{i}_{j}" for j in range(5)}
            ctx.term_bloom.add_many(ctx.terms)
            worker = WorkerNode(ctx, llm=llm)
            mesh.add_worker(worker)

            agent = Agent(contexts={ctx.id: 1.0}, llm=llm)
            agents.register(agent)

        # Run fork checks
        mesh._check_forks()

        # Invariant: every agent has LLM
        naked = [a for a in agents.all_agents() if not a.has_llm]
        assert len(naked) == 0, (
            f"{len(naked)} agents missing LLM after fork: "
            f"{[str(a.id)[:8] for a in naked]}"
        )

        # Invariant: every worker has LLM
        naked_workers = [w for w in mesh.workers if w._llm is None]
        assert len(naked_workers) == 0, (
            f"{len(naked_workers)} workers missing LLM after fork"
        )

    def test_diagnostics_all_intelligence_active(self):
        """After proper setup, all agents report intelligence_active."""
        llm = FakeLLM()
        agents = AgentRegistry()
        mesh = Mesh(agents, llm=llm, max_workers=10, worker_capacity=30)

        ctx = Context(capacity=30)
        ctx.signal_count = 100
        worker = WorkerNode(ctx, llm=llm)
        mesh.add_worker(worker)

        agent = Agent(contexts={ctx.id: 1.0}, llm=llm)
        agents.register(agent)

        mesh._check_forks()

        for a in agents.all_agents():
            diag = a.diagnostics()
            assert diag["has_llm"] is True, f"Agent {diag['agent_id']} missing LLM"
            # No mechanical calls should have happened yet
            assert diag["reflect"]["mechanical_calls"] == 0
            assert diag["evaluate"]["mechanical_calls"] == 0


# ── Intelligence Report Output ──────────────────────────────────


class TestIntelligenceReport:
    """Test the intelligence_report output function."""

    def test_report_no_crash_empty(self, capsys):
        from stigmergy.cli.output import intelligence_report
        intelligence_report([])
        captured = capsys.readouterr()
        assert captured.out == ""  # no output for empty list

    def test_report_shows_all_green(self, capsys):
        from stigmergy.cli.output import intelligence_report
        diags = [
            Agent(llm=FakeLLM()).diagnostics(),
            Agent(llm=FakeLLM()).diagnostics(),
        ]
        intelligence_report(diags)
        captured = capsys.readouterr()
        assert "2/2 agents have LLM" in captured.out
        assert "intelligence" in captured.out.lower()

    def test_report_warns_on_naked_agents(self, capsys):
        from stigmergy.cli.output import intelligence_report
        diags = [
            Agent(llm=FakeLLM()).diagnostics(),
            Agent().diagnostics(),  # no LLM
        ]
        intelligence_report(diags)
        captured = capsys.readouterr()
        assert "1/2 agents have LLM" in captured.out
        assert "MISSING" in captured.out or "WARNING" in captured.out

    def test_report_shows_mechanical_counts(self, capsys):
        from stigmergy.cli.output import intelligence_report
        agent = Agent()
        agent._mechanical_reflect_count = 42
        agent._mechanical_evaluate_count = 10
        diags = [agent.diagnostics()]
        intelligence_report(diags)
        captured = capsys.readouterr()
        assert "42" in captured.out
        assert "mechanical" in captured.out.lower()

    def test_report_shows_dedup_stats(self, capsys):
        from stigmergy.cli.output import intelligence_report
        agent = Agent(llm=FakeLLM())
        agent._llm_reflect_count = 10
        agent._batch_dedup_count = 25
        diags = [agent.diagnostics()]
        intelligence_report(diags)
        captured = capsys.readouterr()
        assert "batch-dedup" in captured.out.lower() or "dedup" in captured.out.lower()
        assert "25" in captured.out


# ── Correlator Batch Dedup ─────────────────────────────────────────


class TestCorrelatorBatchDedup:
    """Test that the correlator skips LLM calls on redundant batches."""

    def _make_obs_with_cache(self, signal, cache):
        from stigmergy.mesh.insights import ObservationContext
        from stigmergy.mesh.mesh import MeshTrace
        trace = MeshTrace(signal_id=signal.id)
        return ObservationContext(
            signal=signal, trace=trace, spectrum=[0.5],
            cache=cache, workers=[],
        )

    def test_fresh_agent_has_empty_batch_state(self):
        agent = Agent()
        assert agent._last_batch_ids == frozenset()
        assert agent._last_batch_insights == []
        assert agent._batch_dedup_count == 0

    def test_first_batch_always_calls_llm(self):
        """The first correlator batch should always call the LLM."""
        from stigmergy.mesh.insights import SignalCache
        from stigmergy.mesh.mesh import MeshTrace

        llm = FakeLLM()
        agent = Agent(confidence=0.5, llm=llm)

        cache = SignalCache(capacity=50)
        # Populate cache with signals
        signals = []
        for i in range(5):
            s = _make_signal(content=f"Signal {i} about database migration")
            t = MeshTrace(signal_id=s.id)
            cache.add(s, t, [0.5], [uuid4()])
            signals.append(s)

        # Run to batch interval (10th call fires correlator)
        for i in range(10):
            sig = signals[i % len(signals)]
            obs = self._make_obs_with_cache(sig, cache)
            _run(agent.reflect(obs))

        assert agent._llm_reflect_count == 1
        assert agent._batch_dedup_count == 0
        assert agent._last_batch_ids != frozenset()  # batch cached

    def test_identical_batch_skips_llm(self):
        """When signal batch hasn't changed, skip the LLM call."""
        from stigmergy.mesh.insights import SignalCache
        from stigmergy.mesh.mesh import MeshTrace

        llm = FakeLLM()
        agent = Agent(confidence=0.5, llm=llm)

        cache = SignalCache(capacity=50)
        signals = []
        for i in range(5):
            s = _make_signal(content=f"Signal {i} about database migration")
            t = MeshTrace(signal_id=s.id)
            cache.add(s, t, [0.5], [uuid4()])
            signals.append(s)

        # First batch (call 10) — calls LLM
        for i in range(10):
            sig = signals[i % len(signals)]
            obs = self._make_obs_with_cache(sig, cache)
            _run(agent.reflect(obs))

        assert agent._llm_reflect_count == 1
        first_batch = agent._last_batch_ids
        assert len(first_batch) > 0, "Batch IDs should be cached after first LLM call"

        # Second batch (call 20) — same signals in cache, should dedup
        # Use the same signal as the 10th call to get identical temporal window
        for i in range(9):
            sig = signals[i % len(signals)]
            obs = self._make_obs_with_cache(sig, cache)
            _run(agent.reflect(obs))
        # 20th call — this one fires the correlator
        sig = signals[4]  # same signal as the 10th call fired on
        obs = self._make_obs_with_cache(sig, cache)
        _run(agent.reflect(obs))

        assert agent._llm_reflect_count == 1  # still 1, not 2 — dedup prevented second LLM call
        assert agent._batch_dedup_count == 1
        # Verify LLM was only called once for CorrelationBatch
        correlation_calls = [c for c in llm.extract_calls if c["schema"] == "CorrelationBatch"]
        assert len(correlation_calls) == 1

    def test_new_signals_bypass_dedup(self):
        """When significant new signals arrive, the LLM should fire again."""
        from stigmergy.mesh.insights import SignalCache
        from stigmergy.mesh.mesh import MeshTrace

        llm = FakeLLM()
        agent = Agent(confidence=0.5, llm=llm)

        cache = SignalCache(capacity=50)
        # First batch of signals
        for i in range(5):
            s = _make_signal(content=f"Signal {i} about database migration")
            t = MeshTrace(signal_id=s.id)
            cache.add(s, t, [0.5], [uuid4()])

        # First batch fires
        for i in range(10):
            obs = self._make_obs_with_cache(_make_signal(), cache)
            _run(agent.reflect(obs))

        assert agent._llm_reflect_count == 1

        # Add a lot of new signals (>20% turnover)
        for i in range(10):
            s = _make_signal(content=f"NEW signal {i} about pricing engine refactor")
            t = MeshTrace(signal_id=s.id)
            cache.add(s, t, [0.5], [uuid4()])

        # Second batch fires — new signals should bypass dedup
        for i in range(10):
            obs = self._make_obs_with_cache(_make_signal(), cache)
            _run(agent.reflect(obs))

        assert agent._llm_reflect_count == 2  # LLM called again
        assert agent._batch_dedup_count == 0  # no dedup happened

    def test_diagnostics_includes_dedup_count(self):
        agent = Agent()
        agent._batch_dedup_count = 15
        diag = agent.diagnostics()
        assert diag["reflect"]["batch_dedup_skips"] == 15
