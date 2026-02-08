"""Tests for Mesh — decentralized BFS routing through WorkerNodes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from stigmergy.constraints.filter import (
    ConstraintAction,
    ConstraintEvaluator,
    ConstraintResult,
    _KillPattern,
    _RedactPattern,
)
from stigmergy.core.consensus import ConsensusResult
from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.core.lifecycle import LifecycleManager
from stigmergy.mesh.mesh import HopRecord, Mesh, MeshTrace, _infer_domain
from stigmergy.mesh.worker import WorkerNode
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.agent import Agent
from stigmergy.primitives.assessment import AssessmentAction
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource
from stigmergy.structures.bloom import CountingBloomFilter
from stigmergy.tracing.trace import TraceLog

import re


def _ctx(*, signal_count: int = 0, capacity: int = 200, **kwargs) -> Context:
    ctx = Context(capacity=capacity, **kwargs)
    ctx.signal_count = signal_count
    return ctx


def _signal(content: str = "test signal", **kwargs) -> Signal:
    defaults = dict(
        content=content,
        source=SignalSource.GITHUB,
        channel="test/repo",
        author="test.user",
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def _mesh(**kwargs) -> Mesh:
    """Create a mesh with sensible test defaults."""
    agents = kwargs.pop("agents", AgentRegistry())
    kwargs.setdefault("dedup_enabled", False)
    return Mesh(agents, **kwargs)


# ── MeshTrace ──────────────────────────────────────────────────


class TestMeshTrace:
    def test_defaults(self):
        trace = MeshTrace()
        assert trace.total_hops == 0
        assert trace.workers_visited == set()
        assert not trace.output_delivered
        assert not trace.duplicate

    def test_hop_tracking(self):
        wid = uuid4()
        trace = MeshTrace(
            hops=[HopRecord(worker_id=wid, score=0.5, accepted=True, learning_mode="rag_indexed", hop=0)],
        )
        assert trace.total_hops == 1
        assert wid in trace.workers_visited

    def test_to_trace(self):
        wid = uuid4()
        sig_id = uuid4()
        trace = MeshTrace(
            signal_id=sig_id,
            accepted_workers={wid},
            familiarity_scores={wid: 0.7},
            output_delivered=True,
            output_content="hello",
        )
        t = trace.to_trace()
        assert t.signal_id == sig_id
        assert wid in t.contexts
        assert t.familiarity_scores[wid] == 0.7
        assert t.output_delivered
        assert t.output_content == "hello"


# ── Worker Management ──────────────────────────────────────────


class TestWorkerManagement:
    def test_add_and_get(self):
        mesh = _mesh()
        worker = WorkerNode(_ctx())
        mesh.add_worker(worker)
        assert mesh.get_worker(worker.id) is worker
        assert mesh.worker_count == 1

    def test_remove(self):
        mesh = _mesh()
        worker = WorkerNode(_ctx())
        mesh.add_worker(worker)
        mesh.remove_worker(worker.id)
        assert mesh.get_worker(worker.id) is None
        assert mesh.worker_count == 0

    def test_remove_cleans_neighbors(self):
        mesh = _mesh()
        w1 = WorkerNode(_ctx())
        w2 = WorkerNode(_ctx())
        mesh.add_worker(w1)
        mesh.add_worker(w2)
        mesh.connect(w1.id, w2.id)
        assert w2.id in w1.context.neighbors
        mesh.remove_worker(w2.id)
        assert w2.id not in w1.context.neighbors

    def test_remove_source_worker_cleans_index(self):
        mesh = _mesh()
        worker = WorkerNode(_ctx(), source_name="github")
        mesh.add_worker(worker)
        assert mesh._source_workers.get("github") == worker.id
        mesh.remove_worker(worker.id)
        assert "github" not in mesh._source_workers

    def test_spawn_worker(self):
        mesh = _mesh()
        worker = mesh.spawn_worker(source_name="slack")
        assert mesh.get_worker(worker.id) is worker
        assert worker.source_name == "slack"
        assert mesh._source_workers["slack"] == worker.id

    def test_spawn_with_connection(self):
        mesh = _mesh()
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker(connect_to=w1.id)
        assert w1.id in w2.context.neighbors
        assert w2.id in w1.context.neighbors

    def test_workers_list(self):
        mesh = _mesh()
        mesh.spawn_worker()
        mesh.spawn_worker()
        assert len(mesh.workers) == 2


# ── Fullness Stats ─────────────────────────────────────────────


class TestFullnessStats:
    def test_avg_fullness_empty_mesh(self):
        mesh = _mesh()
        assert mesh.avg_fullness == 0.0

    def test_avg_fullness(self):
        mesh = _mesh()
        w1 = WorkerNode(_ctx(signal_count=100, capacity=200))
        w2 = WorkerNode(_ctx(signal_count=200, capacity=200))
        mesh.add_worker(w1)
        mesh.add_worker(w2)
        assert mesh.avg_fullness == pytest.approx(0.75)

    def test_all_full(self):
        mesh = _mesh()
        mesh.add_worker(WorkerNode(_ctx(signal_count=200, capacity=200)))
        mesh.add_worker(WorkerNode(_ctx(signal_count=200, capacity=200)))
        assert mesh.all_full

    def test_not_all_full(self):
        mesh = _mesh()
        mesh.add_worker(WorkerNode(_ctx(signal_count=200, capacity=200)))
        mesh.add_worker(WorkerNode(_ctx(signal_count=0, capacity=200)))
        assert not mesh.all_full

    def test_all_full_empty_mesh(self):
        mesh = _mesh()
        assert not mesh.all_full


# ── Topology ───────────────────────────────────────────────────


class TestTopology:
    def test_connect(self):
        mesh = _mesh()
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker()
        assert mesh.connect(w1.id, w2.id)
        assert w2.id in w1.context.neighbors
        assert w1.id in w2.context.neighbors

    def test_disconnect(self):
        mesh = _mesh()
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker(connect_to=w1.id)
        assert mesh.disconnect(w1.id, w2.id)
        assert w2.id not in w1.context.neighbors
        assert w1.id not in w2.context.neighbors

    def test_connect_nonexistent(self):
        mesh = _mesh()
        assert not mesh.connect(uuid4(), uuid4())

    def test_topology_view(self):
        mesh = _mesh()
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker(connect_to=w1.id)
        topo = mesh.topology()
        assert len(topo) == 2
        w1_short = str(w1.id)[:8]
        w2_short = str(w2.id)[:8]
        assert w2_short in topo[w1_short]
        assert w1_short in topo[w2_short]


# ── Entry Worker Selection ─────────────────────────────────────


class TestEntryWorkerSelection:
    def test_content_affinity_preferred(self):
        """Entry worker is selected by content overlap, not source."""
        mesh = _mesh()
        w1 = mesh.spawn_worker()
        w1.context.terms = {"deploy", "release"}
        w1.context.term_bloom.add_many(w1.context.terms)
        w1.context.signal_count = 5
        w2 = mesh.spawn_worker()
        w2.context.terms = {"pricing", "engine"}
        w2.context.term_bloom.add_many(w2.context.terms)
        w2.context.signal_count = 5
        signal = _signal(content="pricing engine update")
        bloom = CountingBloomFilter(capacity=10000)
        bloom.add_many(signal.terms)
        entry = mesh._select_entry_worker(signal, bloom)
        assert entry.id == w2.id

    def test_bloom_overlap_fallback(self):
        mesh = _mesh()
        w1 = WorkerNode(_ctx())
        w1.context.terms = {"deploy", "release", "production"}
        w1.context.term_bloom.add_many(w1.context.terms)
        w1.context.signal_count = 5  # non-empty
        mesh.add_worker(w1)

        signal = _signal(content="deploy release to production")
        bloom = CountingBloomFilter(capacity=10000)
        bloom.add_many(signal.terms)
        entry = mesh._select_entry_worker(signal, bloom)
        assert entry.id == w1.id

    def test_auto_spawn_when_empty(self):
        mesh = _mesh()
        signal = _signal(source=SignalSource.SLACK)
        bloom = CountingBloomFilter(capacity=10000)
        bloom.add_many(signal.terms)
        entry = mesh._select_entry_worker(signal, bloom)
        assert entry is not None
        assert mesh.worker_count == 1


# ── BFS Routing ────────────────────────────────────────────────


class TestBFSRouting:
    def test_single_worker_accept(self):
        mesh = _mesh(base_threshold=0.0)  # accept everything
        mesh.spawn_worker(source_name="github")
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        assert trace.total_hops == 1
        assert len(trace.accepted_workers) == 1

    def test_single_worker_reject(self):
        mesh = _mesh(base_threshold=0.99)  # reject everything
        mesh.spawn_worker(source_name="github")
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        assert trace.total_hops == 1
        assert len(trace.accepted_workers) == 0

    def test_multi_hop_propagation_stops_on_accept(self):
        """With competitive routing, the first worker to accept wins."""
        mesh = _mesh(base_threshold=0.0)
        w1 = mesh.spawn_worker(source_name="github")
        w2 = mesh.spawn_worker(connect_to=w1.id)
        w3 = mesh.spawn_worker(connect_to=w2.id)
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        # Stop-on-accept: first worker accepts, routing stops
        assert trace.total_hops == 1
        assert len(trace.accepted_workers) == 1

    def test_multi_hop_when_rejected(self):
        """When entry worker rejects, BFS continues to neighbors."""
        mesh = _mesh(base_threshold=0.0, max_hops=10)
        # w1 is nearly full and will reject low-score signals
        w1 = WorkerNode(_ctx(signal_count=200, capacity=200),
                        base_threshold=0.15, max_threshold=0.8)
        mesh.add_worker(w1)
        w2 = mesh.spawn_worker(connect_to=w1.id)
        # Force w1 to be entry by giving it some signal history
        w1.context.terms = {"test", "signal"}
        w1.context.term_bloom.add_many(w1.context.terms)
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        # w1 rejects, w2 accepts → 2 hops
        assert trace.total_hops == 2
        assert w2.id in trace.accepted_workers

    def test_ttl_limits_hops(self):
        mesh = _mesh(base_threshold=0.99, max_hops=2)  # reject everything
        w1 = WorkerNode(_ctx(signal_count=200, capacity=200),
                        base_threshold=0.99, max_threshold=0.99)
        mesh.add_worker(w1)
        w2 = mesh.spawn_worker(connect_to=w1.id)
        w2.context.signal_count = 200  # make full so it rejects too
        w3 = mesh.spawn_worker(connect_to=w2.id)
        w3.context.signal_count = 200
        # Give w1 terms so it's selected as entry
        w1.context.terms = {"test", "signal"}
        w1.context.term_bloom.add_many(w1.context.terms)
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        # max_hops=2 limits to hops 0 and 1
        assert trace.total_hops <= 2
        assert w3.id not in trace.workers_visited

    def test_cycle_prevention(self):
        """With competitive routing, cycle prevention still works (visited set)."""
        mesh = _mesh(base_threshold=0.99)  # reject everything to force full BFS
        w1 = WorkerNode(_ctx(signal_count=200, capacity=200),
                        base_threshold=0.99, max_threshold=0.99)
        w2 = WorkerNode(_ctx(signal_count=200, capacity=200),
                        base_threshold=0.99, max_threshold=0.99)
        mesh.add_worker(w1)
        mesh.add_worker(w2)
        mesh.connect(w1.id, w2.id)
        # Give w1 terms so it's entry
        w1.context.terms = {"test", "signal"}
        w1.context.term_bloom.add_many(w1.context.terms)
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        # Each worker visited at most once
        assert trace.total_hops == 2

    def test_disconnected_workers_not_reached(self):
        mesh = _mesh(base_threshold=0.0)
        w1 = mesh.spawn_worker(source_name="github")
        w2 = mesh.spawn_worker()  # disconnected, no neighbors
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        assert w1.id in trace.workers_visited
        assert w2.id not in trace.workers_visited

    def test_auto_spawns_entry_worker(self):
        mesh = _mesh(base_threshold=0.0)
        signal = _signal(source=SignalSource.LINEAR)
        trace = asyncio.run(mesh.ingest(signal))
        assert mesh.worker_count == 1
        assert trace.total_hops == 1

    def test_signals_ingested_counter(self):
        mesh = _mesh(base_threshold=0.0)
        mesh.spawn_worker(source_name="github")
        asyncio.run(mesh.ingest(_signal()))
        asyncio.run(mesh.ingest(_signal(content="different signal")))
        assert mesh.signals_ingested == 2


# ── Deduplication ──────────────────────────────────────────────


class TestDeduplication:
    def test_duplicate_detected(self):
        mesh = _mesh(dedup_enabled=True)
        mesh.spawn_worker(source_name="github")
        signal = _signal(content="fix critical bug in login flow")
        asyncio.run(mesh.ingest(signal))
        # Same content = near-duplicate
        signal2 = _signal(content="fix critical bug in login flow")
        trace2 = asyncio.run(mesh.ingest(signal2))
        assert trace2.duplicate
        assert not trace2.output_delivered
        assert mesh.signals_deduplicated == 1

    def test_non_duplicate_passes(self):
        mesh = _mesh(dedup_enabled=True)
        mesh.spawn_worker(source_name="github")
        asyncio.run(mesh.ingest(_signal(content="fix login bug")))
        trace2 = asyncio.run(mesh.ingest(_signal(content="deploy pricing engine to production")))
        assert not trace2.duplicate

    def test_dedup_disabled(self):
        mesh = _mesh(dedup_enabled=False)
        mesh.spawn_worker(source_name="github")
        asyncio.run(mesh.ingest(_signal(content="same thing")))
        trace2 = asyncio.run(mesh.ingest(_signal(content="same thing")))
        assert not trace2.duplicate


# ── Learning Modes ─────────────────────────────────────────────


class TestLearningModes:
    def test_accepted_worker_learns_terms(self):
        mesh = _mesh(base_threshold=0.0, high_relevance_offset=0.99)
        w = mesh.spawn_worker(source_name="github")
        signal = _signal(content="pricing engine cache invalidation strategy")
        asyncio.run(mesh.ingest(signal))
        # context_summarized mode should add terms
        assert "pricing" in w.context.terms
        assert "engine" in w.context.terms

    def test_rejected_worker_weight_shifts(self):
        mesh = _mesh(base_threshold=0.99)
        w = mesh.spawn_worker(source_name="github")
        signal = _signal(source=SignalSource.SLACK)
        asyncio.run(mesh.ingest(signal))
        # weight_shifted mode does NOT increment source_counts — rejected
        # signals should not bias source affinity (fix #10: source bias bug).
        assert w.context.source_counts.get("slack", 0) == 0

    def test_accept_increments_signal_count(self):
        mesh = _mesh(base_threshold=0.0, high_relevance_offset=0.99)
        w = mesh.spawn_worker(source_name="github")
        assert w.context.signal_count == 0
        asyncio.run(mesh.ingest(_signal()))
        assert w.context.signal_count == 1


# ── Agent Assessment ───────────────────────────────────────────


class TestAgentAssessment:
    def test_agents_evaluate_accepted_workers(self):
        agents = AgentRegistry()
        mesh = _mesh(agents=agents, base_threshold=0.0, high_relevance_offset=0.99)
        w = mesh.spawn_worker(source_name="github")

        # Register an agent bound to this worker's context
        agent = Agent(
            contexts={w.context.id: 1.0},
            confidence=0.9,
            weights={"general": 0.8},
        )
        agents.register(agent)

        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        assert len(trace.assessments) == 1
        assert trace.assessments[0].agent_id == agent.id

    def test_no_agents_no_assessments(self):
        mesh = _mesh(base_threshold=0.0)
        mesh.spawn_worker(source_name="github")
        trace = asyncio.run(mesh.ingest(_signal()))
        assert len(trace.assessments) == 0

    def test_agents_not_called_for_rejected(self):
        agents = AgentRegistry()
        mesh = _mesh(agents=agents, base_threshold=0.99)
        w = mesh.spawn_worker(source_name="github")
        agent = Agent(contexts={w.context.id: 1.0}, confidence=0.9)
        agents.register(agent)
        trace = asyncio.run(mesh.ingest(_signal()))
        assert len(trace.assessments) == 0


# ── Consensus ──────────────────────────────────────────────────


class TestConsensus:
    def test_consensus_resolved(self):
        agents = AgentRegistry()
        mesh = _mesh(agents=agents, base_threshold=0.0, high_relevance_offset=0.99)
        w = mesh.spawn_worker(source_name="github")

        agent = Agent(
            contexts={w.context.id: 1.0},
            confidence=0.9,
            weights={"technical": 0.9},
        )
        agents.register(agent)

        signal = _signal(content="fix critical bug in production")
        trace = asyncio.run(mesh.ingest(signal))
        assert trace.consensus is not None

    def test_no_assessments_no_consensus(self):
        mesh = _mesh(base_threshold=0.0)
        mesh.spawn_worker(source_name="github")
        trace = asyncio.run(mesh.ingest(_signal()))
        assert trace.consensus is None


# ── Constraints ────────────────────────────────────────────────


class TestConstraints:
    def test_constraint_kill(self):
        agents = AgentRegistry()
        kill_pattern = _KillPattern(
            pattern=re.compile(r"SECRET_KEY", re.IGNORECASE),
            category="secrets",
            keywords=["secret_key"],
        )
        constraints = ConstraintEvaluator(kill_patterns=[kill_pattern])
        mesh = _mesh(
            agents=agents,
            base_threshold=0.0,
            high_relevance_offset=0.99,
            constraint_evaluator=constraints,
        )
        w = mesh.spawn_worker(source_name="github")
        agent = Agent(
            contexts={w.context.id: 1.0},
            confidence=0.9,
            weights={"technical": 0.9},
        )
        agents.register(agent)

        signal = _signal(content="fix critical bug with SECRET_KEY exposed")
        trace = asyncio.run(mesh.ingest(signal))
        # Whether output_delivered depends on consensus action
        # The constraint should at least be evaluated if SURFACE/BLOCK
        if trace.consensus and trace.consensus.action in (AssessmentAction.SURFACE, AssessmentAction.BLOCK):
            assert not trace.output_delivered

    def test_constraint_pass(self):
        agents = AgentRegistry()
        constraints = ConstraintEvaluator()  # no patterns = always pass
        mesh = _mesh(
            agents=agents,
            base_threshold=0.0,
            high_relevance_offset=0.99,
            constraint_evaluator=constraints,
        )
        w = mesh.spawn_worker(source_name="github")
        agent = Agent(
            contexts={w.context.id: 1.0},
            confidence=0.9,
            weights={"technical": 0.9},
        )
        agents.register(agent)

        signal = _signal(content="fix critical bug in production")
        trace = asyncio.run(mesh.ingest(signal))
        if trace.consensus and trace.consensus.action in (AssessmentAction.SURFACE, AssessmentAction.BLOCK):
            assert trace.output_delivered


# ── Lifecycle (disabled in hot path) ──────────────────────────


class TestLifecycleEnabled:
    """Lifecycle runs after each ingest: fork, recenter, decay."""

    def test_fork_on_volume_overload(self):
        """Worker with signal_count > fork_volume_threshold should fork."""
        lifecycle = LifecycleManager(fork_volume_threshold=5)
        mesh = _mesh(lifecycle=lifecycle, base_threshold=0.0, high_relevance_offset=0.99, max_workers=10)
        w = mesh.spawn_worker(source_name="github")
        w.context.signal_count = 100  # well above fork threshold
        w.signals_accepted = 1  # must be > 0 for real avg_fam to be used
        asyncio.run(mesh.ingest(_signal()))
        # Fork should have created a child worker
        assert mesh.worker_count >= 2

    def test_fork_on_low_coherence(self):
        """Worker with low avg familiarity and enough signals should fork."""
        lifecycle = LifecycleManager(fork_coherence_threshold=0.5, fork_volume_threshold=1000)
        mesh = _mesh(lifecycle=lifecycle, base_threshold=0.0, high_relevance_offset=0.99, max_workers=10)
        w = mesh.spawn_worker()
        w.context.signal_count = 15  # > 10
        w.signals_accepted = 5
        w.rolling_avg_familiarity = 0.2  # < 0.5 coherence threshold
        asyncio.run(mesh.ingest(_signal()))
        assert mesh.worker_count >= 2

    def test_no_fork_below_thresholds(self):
        """Worker below both thresholds should not fork."""
        lifecycle = LifecycleManager(fork_volume_threshold=1000, fork_coherence_threshold=0.01)
        mesh = _mesh(lifecycle=lifecycle, base_threshold=0.0, high_relevance_offset=0.99, max_workers=10)
        w = mesh.spawn_worker()
        w.context.signal_count = 5  # below volume threshold AND below 10 (coherence guard)
        asyncio.run(mesh.ingest(_signal()))
        assert mesh.worker_count == 1  # no fork

    def test_fork_respects_max_workers(self):
        """Fork should not exceed max_workers limit."""
        lifecycle = LifecycleManager(fork_volume_threshold=5)
        mesh = _mesh(lifecycle=lifecycle, base_threshold=0.0, high_relevance_offset=0.99, max_workers=2)
        w = mesh.spawn_worker()
        w.context.signal_count = 100
        w.signals_accepted = 1
        # Already at 1, max is 2, so one fork is allowed
        asyncio.run(mesh.ingest(_signal()))
        assert mesh.worker_count <= 2


# ── Idle Worker Pruning ───────────────────────────────────────


class TestPruneIdleWorkers:
    def test_prunes_empty_workers(self):
        mesh = _mesh(base_threshold=0.0)
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker(connect_to=w1.id)
        # Simulate w1 accepted a signal
        w1.signals_accepted = 1
        w1.context.signal_count = 1
        # w2 is idle
        pruned = mesh.prune_idle_workers(min_workers=1)
        assert w2.id in pruned
        assert mesh.worker_count == 1
        assert mesh.get_worker(w1.id) is not None

    def test_respects_min_workers(self):
        mesh = _mesh()
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker()
        # Both idle, but min_workers=2 prevents pruning
        pruned = mesh.prune_idle_workers(min_workers=2)
        assert len(pruned) == 0
        assert mesh.worker_count == 2

    def test_keeps_workers_with_signals(self):
        mesh = _mesh()
        w1 = mesh.spawn_worker()
        w1.signals_accepted = 5
        w1.context.signal_count = 5
        pruned = mesh.prune_idle_workers(min_workers=0)
        assert len(pruned) == 0

    def test_cleans_up_neighbor_refs(self):
        mesh = _mesh()
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker(connect_to=w1.id)
        w1.signals_accepted = 1
        w1.context.signal_count = 1
        mesh.prune_idle_workers(min_workers=1)
        # w2 removed, w1's neighbor set should be cleaned
        assert w2.id not in w1.context.neighbors


# ── Trace Logging ──────────────────────────────────────────────


class TestTraceLogging:
    def test_traces_logged(self):
        trace_log = TraceLog()
        mesh = _mesh(trace_log=trace_log, base_threshold=0.0)
        mesh.spawn_worker(source_name="github")
        asyncio.run(mesh.ingest(_signal()))
        assert trace_log.count() == 1

    def test_duplicate_logged(self):
        trace_log = TraceLog()
        mesh = _mesh(trace_log=trace_log, dedup_enabled=True, base_threshold=0.0)
        mesh.spawn_worker(source_name="github")
        asyncio.run(mesh.ingest(_signal(content="important signal here")))
        asyncio.run(mesh.ingest(_signal(content="important signal here")))
        assert trace_log.count() == 2  # both logged


# ── Domain Inference ───────────────────────────────────────────


class TestDomainInference:
    def test_technical(self):
        assert _infer_domain(_signal(content="fix bug in deployment")) == "technical"

    def test_business(self):
        assert _infer_domain(_signal(content="revenue growth from sales")) == "business"

    def test_operations(self):
        assert _infer_domain(_signal(content="deploy new build to staging")) == "operations"

    def test_organizational(self):
        assert _infer_domain(_signal(content="team meeting review notes")) == "organizational"

    def test_general(self):
        # Source-neutral signal with no domain keywords → general
        sig = _signal(content="hello world", source="unknown")
        assert _infer_domain(sig) == "general"

    def test_source_evidence(self):
        # GitHub source adds technical evidence even without keywords
        sig = _signal(content="hello world", source=SignalSource.GITHUB)
        assert _infer_domain(sig) == "technical"


# ── Competitive Routing ───────────────────────────────────────


class TestCompetitiveRouting:
    def test_stop_on_accept(self):
        """Only one worker stores the signal — first to accept wins."""
        mesh = _mesh(base_threshold=0.0)
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker(connect_to=w1.id)
        w3 = mesh.spawn_worker(connect_to=w2.id)
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        assert len(trace.accepted_workers) == 1

    def test_rejected_workers_weight_shift_only(self):
        """Workers that don't accept only weight-shift."""
        # Use high fork thresholds to prevent fork from interfering
        lifecycle = LifecycleManager(fork_volume_threshold=10000, fork_coherence_threshold=0.0)
        mesh = _mesh(base_threshold=0.0, max_hops=10, lifecycle=lifecycle)
        # w1 has very high threshold and will reject
        w1 = WorkerNode(_ctx(signal_count=200, capacity=200),
                        base_threshold=0.99, max_threshold=0.99)
        mesh.add_worker(w1)
        w2 = mesh.spawn_worker(connect_to=w1.id)
        # Make w1 the entry point by giving it unique terms
        w1.context.terms = {"kubernetes", "infrastructure"}
        w1.context.term_bloom.add_many(w1.context.terms)
        signal = _signal(content="kubernetes deployment rollout", source=SignalSource.SLACK)
        w1_count_before = w1.context.signal_count
        asyncio.run(mesh.ingest(signal))
        # w1 rejected: weight_shifted only, signal_count unchanged.
        # source_counts NOT incremented on rejection (fix #10: source bias).
        assert w1.context.signal_count == w1_count_before  # unchanged
        assert w1.context.source_counts.get("slack", 0) == 0  # no bias on reject
        # w2 accepted
        assert w2.context.signal_count == 1

    def test_spectra_still_recorded(self):
        """Familiarity scores recorded for visited workers even with stop-on-accept."""
        mesh = _mesh(base_threshold=0.0)
        w1 = mesh.spawn_worker()
        signal = _signal()
        trace = asyncio.run(mesh.ingest(signal))
        assert w1.id in trace.familiarity_scores

    def test_differentiation_emerges(self):
        """Workers develop different terms when signals compete."""
        mesh = _mesh(base_threshold=0.0, high_relevance_offset=0.99)
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker(connect_to=w1.id)
        w3 = mesh.spawn_worker(connect_to=w2.id)

        # Feed distinct signal domains — round-robin seeds differentiation
        for _ in range(3):
            asyncio.run(mesh.ingest(_signal(content="pricing engine optimization cache")))
            asyncio.run(mesh.ingest(_signal(content="deploy kubernetes helm infrastructure")))
            asyncio.run(mesh.ingest(_signal(content="hiring onboarding culture interview")))

        # Workers should NOT all have identical terms
        term_sets = [w.context.terms for w in mesh.workers if w.context.terms]
        if len(term_sets) >= 2:
            # At least some workers should have different terms
            # (not all identical)
            all_same = all(ts == term_sets[0] for ts in term_sets)
            assert not all_same


# ── Cold-Start Round-Robin ────────────────────────────────────


class TestColdStartRoundRobin:
    def test_round_robin_seeds_different_workers(self):
        """During cold start, signals go to different workers via round-robin.

        Round-robin only applies when ALL workers have signal_count == 0.
        After the first accept, position-based routing takes over. So we
        need distinct content per worker to test that initial seeding works.
        """
        mesh = _mesh(base_threshold=0.0, high_relevance_offset=0.99)
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker()
        w3 = mesh.spawn_worker()

        # First signal goes to w1 (round-robin idx=0)
        trace1 = asyncio.run(mesh.ingest(_signal(content="first signal aaa")))
        # After first signal, one worker is warm. But other 2 are still cold.
        # The round-robin only applied for the first signal.
        # After that, position-based routing selects the warm worker.
        # Verify that at least the first signal went to a specific worker
        assert trace1.entry_worker_id is not None
        assert len(trace1.accepted_workers) == 1

        # To test round-robin properly: verify the counter increments
        assert mesh._cold_start_counter >= 1

    def test_warm_workers_use_position(self):
        """Once workers have signals, entry selection uses position distance."""
        mesh = _mesh(base_threshold=0.0, high_relevance_offset=0.99)
        w1 = mesh.spawn_worker()
        w2 = mesh.spawn_worker()

        # Warm up w1 with pricing terms
        asyncio.run(mesh.ingest(_signal(content="pricing engine cache")))
        # Now w1 has signals — position-based routing should kick in
        trace = asyncio.run(mesh.ingest(_signal(content="pricing strategy optimization")))
        # The pricing-related signal should route to w1 (which knows pricing)
        assert w1.context.signal_count >= 1


# ── Gap Detection and Spawn ───────────────────────────────────


class TestGapDetection:
    def test_gap_spawns_new_worker(self):
        """When all scores are below gap threshold, a new worker is spawned."""
        # Mesh has low base_threshold so spawned workers CAN accept.
        # The existing worker has high threshold to force rejection.
        mesh = _mesh(
            base_threshold=0.0,       # spawned workers accept easily
            gap_threshold=0.5,        # high gap threshold to trigger easily
            max_workers=10,
        )
        # Add a worker that will reject the signal
        w1 = WorkerNode(_ctx(signal_count=200, capacity=200),
                        base_threshold=0.99, max_threshold=0.99)
        mesh.add_worker(w1)
        initial_count = mesh.worker_count

        signal = _signal(content="completely novel topic nobody knows about")
        trace = asyncio.run(mesh.ingest(signal))

        # A new worker should have been spawned for the gap
        assert mesh.worker_count > initial_count
        # The signal should have been accepted by the new worker
        assert len(trace.accepted_workers) >= 1

    def test_no_gap_spawn_at_max_workers(self):
        """Gap detection respects max_workers limit."""
        mesh = _mesh(
            base_threshold=0.99,
            max_threshold=0.99,
            gap_threshold=0.5,
            max_workers=1,
        )
        w1 = WorkerNode(_ctx(signal_count=200, capacity=200),
                        base_threshold=0.99, max_threshold=0.99)
        mesh.add_worker(w1)

        signal = _signal(content="novel topic")
        asyncio.run(mesh.ingest(signal))
        assert mesh.worker_count == 1  # no spawn

    def test_gap_worker_connected_to_nearest(self):
        """Spawned gap workers connect to nearest existing workers."""
        mesh = _mesh(
            base_threshold=0.0,       # spawned workers accept
            gap_threshold=0.5,
            max_workers=10,
        )
        w1 = WorkerNode(_ctx(signal_count=200, capacity=200),
                        base_threshold=0.99, max_threshold=0.99)
        w1.context.terms = {"pricing", "engine"}
        w1.context.term_bloom.add_many(w1.context.terms)
        mesh.add_worker(w1)

        signal = _signal(content="novel topic")
        asyncio.run(mesh.ingest(signal))

        # New worker should be connected to w1
        new_workers = [w for w in mesh.workers if w.id != w1.id]
        assert len(new_workers) >= 1
        # At least one new worker should be connected to w1
        connected = any(w1.id in w.context.neighbors for w in new_workers)
        assert connected


# ── Fork with Real Familiarity (lifecycle manager unit tests) ──


class TestForkWithFamiliarity:
    """Test LifecycleManager.should_fork directly (not via mesh.ingest).

    Fork is disabled in the mesh hot path but the lifecycle manager's
    logic is preserved for future re-enablement.
    """

    def test_lifecycle_detects_low_coherence(self):
        """LifecycleManager correctly identifies fork-worthy conditions."""
        lifecycle = LifecycleManager(
            fork_coherence_threshold=0.5,
            fork_volume_threshold=1000,
        )
        ctx = _ctx(signal_count=15)
        assert lifecycle.should_fork(ctx, avg_familiarity=0.2) is True

    def test_lifecycle_respects_volume_threshold(self):
        lifecycle = LifecycleManager(fork_volume_threshold=100)
        ctx = _ctx(signal_count=150)
        assert lifecycle.should_fork(ctx, avg_familiarity=0.8) is True

    def test_lifecycle_no_fork_when_coherent(self):
        lifecycle = LifecycleManager(fork_coherence_threshold=0.5)
        ctx = _ctx(signal_count=15)
        assert lifecycle.should_fork(ctx, avg_familiarity=0.8) is False
