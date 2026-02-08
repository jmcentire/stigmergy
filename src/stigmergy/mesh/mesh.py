"""Mesh: decentralized signal routing through self-organizing workers.

Theoretical foundation: an ART (Adaptive Resonance Theory) network
operating over organizational signals. Each WorkerNode is an ART
category. The routing algorithm is the ART search procedure:

  For each signal I:
    1. Present I to the nearest category (entry worker)
    2. Compute match |I ∩ w_J|/|I| via familiarity()
    3. If match ≥ ρ (vigilance): resonance — category learns from I
    4. If match < ρ: mismatch — propagate to next category (BFS hop)
    5. If no category resonates: gap detection → spawn new category

The gap_threshold parameter is the ART complement coding analog:
it prevents category collapse by detecting uncovered signal space
and creating new categories (workers) to fill gaps.

Routing IS classification. No global coordinator. Workers fill up,
raising their vigilance ρ, creating natural backpressure and
exponential time-value decay for free.

Signal flow:
  1. Dedup (SimHash)
  2. Embed (if needed)
  3. Select entry worker (nearest in signal space)
  4. BFS through mesh — each worker runs ART resonance check
  5. Agents evaluate in accepted workers
  6. Consensus (same pure function)
  7. Constraint gate (same evaluator)
  8. Lifecycle check (gap spawn, fork/merge/decay)
  9. Return MeshTrace
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from stigmergy.constraints.filter import ConstraintAction, ConstraintEvaluator

if TYPE_CHECKING:
    from stigmergy.graph.extractor import EdgeExtractor
    from stigmergy.graph.graph import CommunicationGraph
from stigmergy.core.consensus import ConsensusResult, consensus
from stigmergy.core.energy import ContextStatus
from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.core.lifecycle import LifecycleManager
from stigmergy.mesh.topology import (
    detect_gap,
    select_peers,
    signal_position_distance,
)
from stigmergy.mesh.worker import ReceiveResult, WorkerNode
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.assessment import Assessment, AssessmentAction
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal
from stigmergy.services.embedding import EmbeddingService
from stigmergy.services.llm import LLMService
from stigmergy.structures.bloom import CountingBloomFilter
from stigmergy.structures.simhash import SimHash, SimHashIndex
from stigmergy.tracing.trace import Trace, TraceLog

logger = logging.getLogger(__name__)


@dataclass
class HopRecord:
    """Record of a single hop in the mesh routing."""

    worker_id: UUID
    score: float
    accepted: bool
    learning_mode: str
    hop: int


@dataclass
class MeshTrace:
    """Full decision chain for a signal through the mesh."""

    id: UUID = field(default_factory=uuid4)
    signal_id: UUID | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hops: list[HopRecord] = field(default_factory=list)
    entry_worker_id: UUID | None = None
    accepted_workers: set[UUID] = field(default_factory=set)
    familiarity_scores: dict[UUID, float] = field(default_factory=dict)
    assessments: list[Assessment] = field(default_factory=list)
    consensus: ConsensusResult | None = None
    constraint_eval: object | None = None
    output_delivered: bool = False
    output_content: str | None = None
    duplicate: bool = False
    insights: list = field(default_factory=list)  # list[Insight] — populated by agents

    @property
    def total_hops(self) -> int:
        return len(self.hops)

    @property
    def workers_visited(self) -> set[UUID]:
        return {h.worker_id for h in self.hops}

    def to_trace(self) -> Trace:
        """Convert to a standard Trace for the trace log."""
        return Trace(
            id=self.id,
            signal_id=self.signal_id,
            timestamp=self.timestamp,
            contexts=self.accepted_workers,
            familiarity_scores=self.familiarity_scores,
            assessments=self.assessments,
            consensus=self.consensus,
            constraint_eval=self.constraint_eval,
            output_delivered=self.output_delivered,
            output_content=self.output_content,
        )


class Mesh:
    """ART network for decentralized organizational signal routing.

    Workers are ART categories (Contexts wrapped with vigilance criteria).
    Signals enter at the nearest category and propagate via BFS.
    Each worker runs the ART resonance check — accept on match,
    propagate on mismatch. Full workers raise vigilance, creating
    natural backpressure (complement coding analog).

    Key ART parameters:
    - base_threshold: minimum vigilance ρ (empty workers)
    - max_threshold: maximum vigilance ρ (full workers)
    - gap_threshold: complement coding — spawns new category when no
      existing category resonates (all scores below this threshold)
    """

    def __init__(
        self,
        agents: AgentRegistry,
        *,
        embedding_service: EmbeddingService | None = None,
        constraint_evaluator: ConstraintEvaluator | None = None,
        trace_log: TraceLog | None = None,
        lifecycle: LifecycleManager | None = None,
        max_hops: int = 10,
        max_workers: int = 50,
        familiarity_weights: FamiliarityWeights | None = None,
        consensus_threshold: float = 0.6,
        uncertainty_threshold: float = 0.3,
        dedup_enabled: bool = True,
        dedup_threshold: int = 3,
        worker_capacity: int = 200,
        base_threshold: float = 0.15,
        max_threshold: float = 0.8,
        threshold_curve: float = 2.0,
        high_relevance_offset: float = 0.2,
        gap_threshold: float = 0.08,
        recenter_interval: int = 20,
        llm: LLMService | None = None,
    ) -> None:
        self._agents = agents
        self._embedding = embedding_service
        self._constraints = constraint_evaluator or ConstraintEvaluator()
        self._trace_log = trace_log or TraceLog()
        self._lifecycle = lifecycle or LifecycleManager()
        self._max_hops = max_hops
        self._max_workers = max_workers
        self._weights = familiarity_weights or FamiliarityWeights()
        self._consensus_threshold = consensus_threshold
        self._uncertainty_threshold = uncertainty_threshold
        self._worker_capacity = worker_capacity
        self._base_threshold = base_threshold
        self._max_threshold = max_threshold
        self._threshold_curve = threshold_curve
        self._high_relevance_offset = high_relevance_offset
        self._gap_threshold = gap_threshold
        self._recenter_interval = recenter_interval
        self._llm = llm  # agent intelligence — passed to workers

        # Worker registry
        self._workers: dict[UUID, WorkerNode] = {}
        self._source_workers: dict[str, UUID] = {}  # source_name -> worker_id

        # Dedup
        self._dedup_enabled = dedup_enabled
        self._dedup_index = SimHashIndex(distance_threshold=dedup_threshold)

        # Cold-start round-robin counter
        self._cold_start_counter: int = 0

        # Communication graph (optional — set via set_graph())
        self._graph: CommunicationGraph | None = None
        self._edge_extractor: EdgeExtractor | None = None

        # Attention model (optional — for per-person P(knows) tracking)
        self._attention_model = None
        self._exposure_extractor = None

        # Stability tracker (optional — for Lyapunov validation)
        self._stability_tracker = None

        # Stats
        self.signals_ingested: int = 0
        self.signals_deduplicated: int = 0

    def set_graph(self, graph: CommunicationGraph) -> None:
        """Attach a communication graph for edge extraction during ingest."""
        from stigmergy.graph.extractor import EdgeExtractor as _EdgeExtractor

        self._graph = graph
        self._edge_extractor = _EdgeExtractor(graph)

    def set_attention_model(self, attention_model, exposure_extractor) -> None:
        """Attach an attention model for per-person P(knows) tracking."""
        self._attention_model = attention_model
        self._exposure_extractor = exposure_extractor

    def set_stability_tracker(self, tracker) -> None:
        """Attach a StabilityTracker for Lyapunov validation."""
        self._stability_tracker = tracker

    # ── Worker Management ──────────────────────────────────────────

    def add_worker(self, worker: WorkerNode) -> None:
        """Register a worker in the mesh."""
        self._workers[worker.id] = worker
        if worker.source_name:
            self._source_workers[worker.source_name] = worker.id

    def remove_worker(self, worker_id: UUID) -> None:
        """Remove a worker from the mesh and clean up topology."""
        worker = self._workers.pop(worker_id, None)
        if worker is None:
            return
        # Remove from source index
        if worker.source_name and self._source_workers.get(worker.source_name) == worker_id:
            del self._source_workers[worker.source_name]
        # Remove from neighbor sets
        for other in self._workers.values():
            other.context.neighbors.discard(worker_id)

    def get_worker(self, worker_id: UUID) -> WorkerNode | None:
        return self._workers.get(worker_id)

    def spawn_worker(
        self,
        *,
        source_name: str | None = None,
        connect_to: UUID | None = None,
    ) -> WorkerNode:
        """Create a new worker and add it to the mesh.

        If connect_to is given, establishes a neighbor relationship.
        """
        ctx = Context(capacity=self._worker_capacity)
        worker = WorkerNode(
            ctx,
            base_threshold=self._base_threshold,
            max_threshold=self._max_threshold,
            threshold_curve=self._threshold_curve,
            high_relevance_offset=self._high_relevance_offset,
            source_name=source_name,
            llm=self._llm,
        )
        if connect_to and connect_to in self._workers:
            neighbor = self._workers[connect_to]
            ctx.neighbors.add(neighbor.context.id)
            neighbor.context.neighbors.add(ctx.id)
        self.add_worker(worker)
        if self._stability_tracker is not None:
            self._stability_tracker.record_lifecycle(
                "spawn",
                worker_id=str(worker.id)[:8],
                detail={"source": source_name or "gap"},
            )
        return worker

    @property
    def workers(self) -> list[WorkerNode]:
        return list(self._workers.values())

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    @property
    def avg_fullness(self) -> float:
        if not self._workers:
            return 0.0
        return sum(w.fullness for w in self._workers.values()) / len(self._workers)

    @property
    def all_full(self) -> bool:
        return bool(self._workers) and all(w.is_full for w in self._workers.values())

    def _adaptive_gap_threshold(self) -> float:
        """Compute adaptive gap threshold based on mesh maturity.

        During cold starts, workers have few signals and low vocabulary,
        causing all familiarity scores to be low. A fixed gap threshold
        (e.g. 0.08) triggers premature worker spawning — every signal
        looks like uncovered territory when workers haven't learned yet.

        Adaptive behavior:
        - When avg signals/worker < 10: raise threshold to 0.0 (suppress spawning)
        - When avg signals/worker 10-50: linearly interpolate toward configured value
        - When avg signals/worker > 50: use configured gap_threshold
        """
        if not self._workers:
            return self._gap_threshold

        avg_signals = sum(
            w.context.signal_count for w in self._workers.values()
        ) / len(self._workers)

        # Cold start suppression: don't spawn until workers have enough data
        cold_start_min = 10
        maturity_target = 50

        if avg_signals < cold_start_min:
            return 0.0  # Suppress gap spawning entirely during cold start
        elif avg_signals < maturity_target:
            # Linear ramp from 0 to configured threshold
            progress = (avg_signals - cold_start_min) / (maturity_target - cold_start_min)
            return self._gap_threshold * progress
        else:
            return self._gap_threshold

    # ── Entry Worker Selection ─────────────────────────────────────

    def _select_entry_worker(
        self, signal: Signal, signal_bloom: CountingBloomFilter
    ) -> WorkerNode:
        """Select the entry point for a signal by position distance.

        Routes based on which worker is closest in signal space. During
        cold-start (all workers empty), uses round-robin to seed initial
        differentiation.

        Priority:
        1. Nearest worker by position distance (among workers with signals)
        2. Round-robin (cold-start — no worker has signals yet)
        3. Auto-spawn a new worker
        """
        # Find workers that have processed signals
        warm_workers = [w for w in self._workers.values() if w.context.signal_count > 0]

        if warm_workers:
            # Use position distance for fast routing
            warm_workers.sort(
                key=lambda w: signal_position_distance(signal_bloom, w.position),
            )
            return warm_workers[0]

        # Cold start: all workers empty — round-robin to seed differentiation
        if self._workers:
            worker_list = list(self._workers.values())
            idx = self._cold_start_counter % len(worker_list)
            self._cold_start_counter += 1
            return worker_list[idx]

        # No workers at all — spawn one
        return self.spawn_worker()

    # ── Core Routing ───────────────────────────────────────────────

    async def ingest(self, signal: Signal) -> MeshTrace:
        """Route a signal through the mesh via BFS.

        Each worker makes a local accept/reject decision. Accepted workers
        ingest the signal. Rejected workers just weight-shift. After routing,
        agents evaluate and consensus resolves the outcome.
        """
        trace = MeshTrace(signal_id=signal.id)

        # 1. Dedup
        if self._dedup_enabled:
            terms = list(signal.terms)
            if terms:
                fingerprint = SimHash.fingerprint(terms)
                if self._dedup_index.has_near_duplicate(fingerprint):
                    self.signals_deduplicated += 1
                    trace.duplicate = True
                    trace.output_delivered = False
                    self._trace_log.append(trace.to_trace())
                    return trace
                self._dedup_index.add(str(signal.id), fingerprint)

        # 2. Embed (if needed and embedding service available)
        if not signal.embeddings and self._embedding:
            signal_embeddings = await self._embedding.embed_all(signal.content)
            signal = signal.model_copy(update={"embeddings": signal_embeddings})

        # 3. Build signal bloom for fast overlap scoring
        # Must match Context.term_bloom capacity (default 10000) for Jaccard estimation
        signal_bloom = CountingBloomFilter(capacity=10000)
        signal_bloom.add_many(signal.terms)

        # 4. Select entry worker
        # Track whether this is a cold-start round-robin assignment
        # (multiple workers, all empty — round-robin seeding differentiation)
        all_cold = (
            len(self._workers) > 1
            and all(w.context.signal_count == 0 for w in self._workers.values())
        )
        entry = self._select_entry_worker(signal, signal_bloom)
        trace.entry_worker_id = entry.id

        routed = False

        # 5. Cold-start force-accept: when ALL workers are empty and we're
        # doing round-robin seeding, normal scoring returns 0.0. Force the
        # entry worker to accept, seeding initial differentiation.
        if all_cold:
            entry._context_summarize(signal, 0.1)
            entry.signals_accepted += 1
            entry.signals_received += 1
            entry._position_dirty = True
            entry._update_familiarity_ema(0.1)
            entry.recenter_counter += 1
            trace.hops.append(HopRecord(
                worker_id=entry.id,
                score=0.1,
                accepted=True,
                learning_mode="context_summarized",
                hop=0,
            ))
            trace.accepted_workers.add(entry.id)
            trace.familiarity_scores[entry.id] = 0.1
            routed = True

        # 6. Competitive routing — BFS through neighbors, stop on first accept.
        # Only the best-fit worker stores the signal. Others just weight-shift.
        # This forces differentiation: workers specialize through competition.
        # (Skipped when cold-start force-accept already handled the signal.)
        if not routed:
            queue: deque[tuple[WorkerNode, int]] = deque([(entry, 0)])
            visited: set[UUID] = set()

            while queue:
                worker, hop = queue.popleft()
                if worker.id in visited or hop >= self._max_hops:
                    continue
                visited.add(worker.id)

                # Score signal against this worker
                centroid = await worker.context.vector_store.get_centroid("semantic")
                score = worker.score_signal(
                    signal, self._weights, centroid=centroid, signal_bloom=signal_bloom
                )
                trace.familiarity_scores[worker.id] = score

                # Worker's local decision
                result = await worker.receive(signal, score)
                trace.hops.append(HopRecord(
                    worker_id=worker.id,
                    score=score,
                    accepted=result.accepted,
                    learning_mode=result.learning_mode,
                    hop=hop,
                ))

                if result.accepted:
                    trace.accepted_workers.add(worker.id)
                    # Stop-on-accept: first worker to accept wins the signal.
                    # This is what forces differentiation — no duplicate storage.
                    routed = True
                    break

                # Enqueue neighbors (sorted by bloom overlap for better routing)
                neighbors = []
                for neighbor_id in worker.context.neighbors:
                    if neighbor_id not in visited:
                        neighbor_worker = self._workers.get(neighbor_id)
                        if neighbor_worker:
                            neighbors.append(neighbor_worker)

                if neighbors:
                    neighbors.sort(
                        key=lambda w: _safe_bloom_overlap(w.context, signal_bloom),
                        reverse=True,
                    )
                    for n in neighbors:
                        queue.append((n, hop + 1))

            # Fallback: if no worker accepted and there are unvisited workers,
            # try them sorted by content affinity. Prevents topology dead-ends.
            if not routed and len(visited) < len(self._workers):
                unvisited = [
                    w for w in self._workers.values() if w.id not in visited
                ]
                unvisited.sort(
                    key=lambda w: _safe_bloom_overlap(w.context, signal_bloom),
                    reverse=True,
                )
                for worker in unvisited[:3]:  # try top 3 unvisited
                    hop = len(visited)
                    if hop >= self._max_hops:
                        break
                    visited.add(worker.id)
                    centroid = await worker.context.vector_store.get_centroid("semantic")
                    score = worker.score_signal(
                        signal, self._weights, centroid=centroid, signal_bloom=signal_bloom
                    )
                    trace.familiarity_scores[worker.id] = score
                    result = await worker.receive(signal, score)
                    trace.hops.append(HopRecord(
                        worker_id=worker.id,
                        score=score,
                        accepted=result.accepted,
                        learning_mode=result.learning_mode,
                        hop=hop,
                    ))
                    if result.accepted:
                        trace.accepted_workers.add(worker.id)
                        routed = True
                        break  # found a home

            # Gap detection: if no worker accepted and all scores are below
            # the gap threshold, spawn a new worker for this uncovered region.
            # Adaptive threshold: suppress gap spawning during cold starts when
            # workers have too few signals to generate meaningful scores.
            if not routed and self.worker_count < self._max_workers:
                effective_gap = self._adaptive_gap_threshold()
                if detect_gap(trace.familiarity_scores, threshold=effective_gap):
                    await self._handle_gap(signal, signal_bloom, trace)

        self.signals_ingested += 1

        # 6. Agent assessment in accepted workers
        all_assessments: list[Assessment] = []
        for wid in trace.accepted_workers:
            worker = self._workers[wid]
            context_agents = self._agents.for_context(worker.context.id)
            for agent in context_agents:
                domain = _infer_domain(signal)
                assessment = await agent.evaluate(signal, worker.context.id, domain)
                all_assessments.append(assessment)

        trace.assessments = all_assessments

        # 7. Consensus
        if all_assessments:
            consensus_result = consensus(
                all_assessments,
                self._agents.as_dict(),
                consensus_threshold=self._consensus_threshold,
                uncertainty_threshold=self._uncertainty_threshold,
            )
            trace.consensus = consensus_result

            # 8. Constraint check (if surfacing or blocking)
            if consensus_result.action in (AssessmentAction.SURFACE, AssessmentAction.BLOCK):
                constraint_result = self._constraints.evaluate(signal.content)
                trace.constraint_eval = constraint_result

                if constraint_result.action == ConstraintAction.KILL:
                    trace.output_delivered = False
                elif constraint_result.action == ConstraintAction.REDACT:
                    trace.output_content = constraint_result.content
                    trace.output_delivered = True
                else:
                    trace.output_content = signal.content
                    trace.output_delivered = True
            else:
                trace.output_delivered = False
        else:
            trace.output_delivered = False

        # 9. Lifecycle checks — pass signal timestamp so energy decay uses
        # signal time (not wall clock). During backfill, wall-clock elapsed
        # time is tiny but signal-time gaps may span days.
        self._check_lifecycle(signal_time=signal.timestamp)

        # 10. Log trace
        self._trace_log.append(trace.to_trace())

        # 11. Extract communication graph edges
        if self._edge_extractor is not None:
            self._edge_extractor.extract(signal)

        # 11.5 Extract exposure events for attention model
        if self._exposure_extractor is not None:
            exposure_events = self._exposure_extractor.extract(signal)
            if self._attention_model is not None:
                self._attention_model.record_exposures(exposure_events)

        # 12. Stability tracking (Lyapunov validation)
        if self._stability_tracker is not None:
            self._stability_tracker.record(self, trace)

        return trace

    # ── Lifecycle ──────────────────────────────────────────────────

    def _check_lifecycle(self, signal_time: datetime | None = None) -> None:
        """Run lifecycle checks after each signal.

        Order matters:
          1. Fork overloaded/incoherent workers (creates new workers)
          2. Recenter topology (rebalances neighbor sets after forks)
          3. Decay idle workers (prunes workers with depleted energy)

        Each method has its own guards — _check_forks respects max_workers,
        _recenter_workers only fires every recenter_interval accepts,
        _check_decay protects recently-active workers.

        Args:
            signal_time: The signal's timestamp. Used for energy decay so
                that backfill (weeks of signals in minutes) properly reflects
                inter-signal time gaps rather than wall-clock elapsed time.
        """
        self._check_forks()
        self._recenter_workers()
        self._check_decay(signal_time=signal_time)

    def _check_forks(self) -> None:
        """Fork workers that are full or incoherent.

        Children inherit the parent's terms, bloom filter, and source/author
        counts so they can participate in routing immediately. The child starts
        with signal_count=0 (low fullness → low threshold), making it more
        accepting than the parent. Signals the parent rejects flow to the
        child via BFS, creating natural differentiation over time.

        Critical: the parent's signal_count is NOT reduced during fork.
        Reducing it lowers the parent's threshold, making it MORE accepting
        — the opposite of what fork is for. Instead, the parent stays at
        high fullness (high threshold, selective), and a cooldown prevents
        immediate re-triggering.
        """
        if self.worker_count >= self._max_workers:
            return

        for worker in list(self._workers.values()):
            if self.worker_count >= self._max_workers:
                break

            # Fork cooldown: skip workers that recently forked.
            # Cooldown decrements once per lifecycle check (once per signal).
            cooldown = getattr(worker, '_fork_cooldown', 0)
            if cooldown > 0:
                worker._fork_cooldown = cooldown - 1
                continue

            # Use real rolling avg familiarity instead of stub 1.0.
            # Workers that haven't accepted anything this session get avg_fam=1.0
            # so the coherence trigger doesn't fire on stale persisted state.
            if worker.signals_accepted == 0:
                avg_fam = 1.0
            else:
                avg_fam = worker.rolling_avg_familiarity
            if not self._lifecycle.should_fork(worker.context, avg_familiarity=avg_fam):
                continue

            # Save parent state before fork (lifecycle.fork modifies it)
            parent_ctx = worker.context
            original_signal_count = parent_ctx.signal_count

            # Fork via lifecycle manager
            agents = self._agents.for_context(worker.context.id)
            fork_result = self._lifecycle.fork(worker.context, agents)

            # Restore parent's signal_count — keep it at high fullness so it
            # stays selective. The child starts at 0, creating a threshold gap
            # that routes overflow signals to the child via BFS.
            parent_ctx.signal_count = original_signal_count

            # Cooldown: don't fork this worker again for a while.
            # This gives the child time to accumulate and differentiate.
            worker._fork_cooldown = parent_ctx.capacity  # ~200 signals

            for child in fork_result.children:
                # Seed child with parent's knowledge so it can participate
                # in routing. Without this, the child has empty bloom/terms
                # and is invisible to entry selection and scores too low to
                # accept via BFS. The child's low fullness (signal_count=0)
                # gives it a lower threshold than the parent, so it accepts
                # signals the parent rejects — creating differentiation.
                child.terms = set(parent_ctx.terms)
                child.term_bloom.add_many(parent_ctx.terms)
                child.source_counts = dict(parent_ctx.source_counts)
                child.author_counts = dict(parent_ctx.author_counts)

                child_worker = WorkerNode(
                    child,
                    base_threshold=self._base_threshold,
                    max_threshold=self._max_threshold,
                    threshold_curve=self._threshold_curve,
                    high_relevance_offset=self._high_relevance_offset,
                    llm=self._llm,
                )
                # Inherit fork cooldown — prevent cascade forking where
                # a child immediately triggers its own fork.
                child_worker._fork_cooldown = parent_ctx.capacity
                self.add_worker(child_worker)

            for agent in fork_result.spawned_agents:
                # Propagate LLM, annotations, and circuit breaker to forked agents —
                # without this, forked agents fall back to mechanical
                # heuristics instead of using agent intelligence.
                if self._llm is not None:
                    agent._llm = self._llm
                    # Inherit annotation store and circuit breaker from a sibling agent
                    siblings = self._agents.for_context(worker.context.id)
                    if siblings:
                        if hasattr(siblings[0], '_annotations'):
                            agent._annotations = siblings[0]._annotations
                        if hasattr(siblings[0], '_circuit_breaker'):
                            agent._circuit_breaker = siblings[0]._circuit_breaker
                self._agents.register(agent)

            if self._stability_tracker is not None:
                self._stability_tracker.record_lifecycle(
                    "fork",
                    worker_id=str(worker.id)[:8],
                    detail={
                        "children": [str(c.id)[:8] for c in fork_result.children],
                        "avg_fam": round(avg_fam, 4),
                        "signal_count": original_signal_count,
                    },
                )

            logger.info(
                "Worker %s forked → %d children (signal_count=%d, avg_fam=%.3f, cooldown=%d)",
                str(worker.id)[:8],
                len(fork_result.children),
                original_signal_count,
                avg_fam,
                worker._fork_cooldown,
            )

    def _recenter_workers(self) -> None:
        """Periodic peer recentering for workers that have accepted enough signals.

        Each worker independently rebalances its neighbor set based on
        position distance to other workers. Runs every recenter_interval
        accepted signals per worker.
        """
        if len(self._workers) < 3:
            return  # need at least 3 workers for recentering to matter

        all_positions = [w.position for w in self._workers.values()]

        for worker in self._workers.values():
            if worker.recenter_counter < self._recenter_interval:
                continue

            # Time to recenter this worker
            worker.recenter_counter = 0
            new_peer_ids = select_peers(worker.position, all_positions)

            # Update neighbor set — disconnect old, connect new
            old_neighbors = set(worker.context.neighbors)
            new_neighbors = {pid for pid in new_peer_ids if pid in self._workers}

            # Remove old connections not in new set
            for old_id in old_neighbors - new_neighbors:
                other = self._workers.get(old_id)
                if other:
                    other.context.neighbors.discard(worker.id)
            # Add new connections
            for new_id in new_neighbors - old_neighbors:
                other = self._workers.get(new_id)
                if other:
                    other.context.neighbors.add(worker.id)

            worker.context.neighbors = new_neighbors

            logger.debug(
                "Worker %s recentered: %d → %d peers",
                str(worker.id)[:8],
                len(old_neighbors),
                len(new_neighbors),
            )

    def _check_decay(self, signal_time: datetime | None = None) -> None:
        """Archive workers with depleted energy.

        Protects workers that have accepted signals recently — a worker
        that just processed something should never be immediately archived.

        Args:
            signal_time: If provided, use signal timestamp for decay
                calculation instead of wall clock. This matters during
                backfill where wall-clock time is compressed.
        """
        now = signal_time or datetime.now(timezone.utc)
        to_remove: list[UUID] = []
        for worker in self._workers.values():
            # Protect workers that accepted signals in the last 60 seconds
            elapsed = (now - worker.context.last_signal).total_seconds()
            if elapsed < 60:
                continue
            status = self._lifecycle.check_energy(worker.context, now=now)
            if status == ContextStatus.ARCHIVED:
                to_remove.append(worker.id)
                logger.info("Worker %s archived (energy depleted)", str(worker.id)[:8])

        for wid in to_remove:
            worker = self._workers.get(wid)
            if self._stability_tracker is not None:
                self._stability_tracker.record_lifecycle(
                    "archive",
                    worker_id=str(wid)[:8],
                    detail={
                        "signal_count": worker.context.signal_count if worker else 0,
                        "energy": round(worker.context.energy, 4) if worker else 0.0,
                        "reason": "energy_depleted",
                    },
                )
            self.remove_worker(wid)

    async def _handle_gap(
        self, signal: Signal, signal_bloom: CountingBloomFilter, trace: MeshTrace
    ) -> None:
        """Spawn a new worker for uncovered signal territory.

        When no existing worker matches a signal, the signal is in a gap.
        A new worker is created and connected to the 2 nearest existing
        workers (by position distance to the signal).
        """
        # Find 2 nearest workers to connect to
        connect_targets: list[UUID] = []
        if self._workers:
            scored = [
                (w, signal_position_distance(signal_bloom, w.position))
                for w in self._workers.values()
            ]
            scored.sort(key=lambda x: x[1])
            connect_targets = [w.id for w, _ in scored[:2]]

        # Spawn new worker
        first_connect = connect_targets[0] if connect_targets else None
        new_worker = self.spawn_worker(connect_to=first_connect)

        # Connect to second nearest if available
        if len(connect_targets) > 1:
            self.connect(new_worker.id, connect_targets[1])

        # Force-ingest the signal into the new worker. This is the signal that
        # caused the gap — the new worker exists specifically for this domain.
        # Normal scoring would return 0.0 (empty worker), so we bypass the
        # threshold and directly context-summarize.
        new_worker._context_summarize(signal, 0.1)
        new_worker.signals_accepted += 1
        new_worker.signals_received += 1
        new_worker._position_dirty = True
        new_worker._update_familiarity_ema(0.1)
        new_worker.recenter_counter += 1

        trace.hops.append(HopRecord(
            worker_id=new_worker.id,
            score=0.1,
            accepted=True,
            learning_mode="context_summarized",
            hop=trace.total_hops,
        ))
        trace.accepted_workers.add(new_worker.id)
        trace.familiarity_scores[new_worker.id] = 0.1

        logger.info(
            "Gap detected — spawned worker %s, connected to %d peers",
            str(new_worker.id)[:8],
            len(connect_targets),
        )

    # ── Pruning ────────────────────────────────────────────────────

    def prune_idle_workers(self, min_workers: int = 1) -> list[UUID]:
        """Remove workers that accepted zero signals this session.

        Simple rule: if you didn't earn your keep during backfill,
        you're removed. Replaces complex energy-based decay.

        Returns list of removed worker IDs.
        """
        to_remove = [
            w.id for w in self._workers.values()
            if w.signals_accepted == 0 and w.context.signal_count == 0
        ]
        # Never prune below min_workers
        keep_count = max(min_workers, len(self._workers) - len(to_remove))
        removable = len(self._workers) - keep_count
        to_remove = to_remove[:removable]

        for wid in to_remove:
            self.remove_worker(wid)
            logger.info("Pruned idle worker %s (0 signals accepted)", str(wid)[:8])

        return to_remove

    # ── Topology ───────────────────────────────────────────────────

    def connect(self, worker_a_id: UUID, worker_b_id: UUID) -> bool:
        """Establish a neighbor relationship between two workers."""
        a = self._workers.get(worker_a_id)
        b = self._workers.get(worker_b_id)
        if a is None or b is None:
            return False
        a.context.neighbors.add(worker_b_id)
        b.context.neighbors.add(worker_a_id)
        return True

    def disconnect(self, worker_a_id: UUID, worker_b_id: UUID) -> bool:
        """Remove a neighbor relationship between two workers."""
        a = self._workers.get(worker_a_id)
        b = self._workers.get(worker_b_id)
        if a is None or b is None:
            return False
        a.context.neighbors.discard(worker_b_id)
        b.context.neighbors.discard(worker_a_id)
        return True

    def topology(self) -> dict[str, list[str]]:
        """Get the mesh topology as a dict of worker_id -> neighbor_ids."""
        result: dict[str, list[str]] = {}
        for worker in self._workers.values():
            wid = str(worker.id)[:8]
            neighbors = [
                str(nid)[:8]
                for nid in worker.context.neighbors
                if nid in self._workers
            ]
            result[wid] = neighbors
        return result


def _safe_bloom_overlap(context: Context, signal_bloom: CountingBloomFilter) -> float:
    """Estimate bloom overlap, returning 0.0 if filters are incompatible."""
    try:
        return context.fast_term_overlap(signal_bloom)
    except ValueError:
        return 0.0


"""Domain hint table — data, not code.

Each domain has soft signals: keywords, sources, channels. The scoring
is additive (every match contributes weight), not exclusive (first-match-wins).
Agents can expand this table with new domains and evidence patterns.
With a real LLM, agents reason about domain directly — this table becomes
prompt conditioning rather than the decision itself.

A fresh pair of eyes reading any domain string should understand it
without needing shared context. Domain names are self-describing.
"""
_DOMAIN_HINTS: dict[str, dict[str, set[str]]] = {
    "technical": {
        "keywords": {"bug", "error", "fail", "crash", "fix", "merge", "refactor",
                     "api", "endpoint", "test", "code", "pr", "commit", "branch"},
        "sources": {"github"},
        "channels": {"engineering", "dev", "code-review", "backend", "frontend"},
    },
    "business": {
        "keywords": {"revenue", "sales", "customer", "market", "growth", "churn",
                     "conversion", "pricing", "forecast", "contract"},
        "sources": set(),
        "channels": {"sales", "marketing", "leadership", "revenue"},
    },
    "operations": {
        "keywords": {"deploy", "release", "rollback", "ci", "build", "infra",
                     "incident", "outage", "monitoring", "alert", "pipeline"},
        "sources": {"deploy"},
        "channels": {"ops", "devops", "incidents", "infrastructure"},
    },
    "organizational": {
        "keywords": {"hire", "team", "meeting", "review", "onboarding", "process",
                     "standup", "retro", "sprint", "planning"},
        "sources": set(),
        "channels": {"general", "hr", "hiring", "culture"},
    },
}


def _infer_domain(signal: Signal) -> str:
    """Infer domain via soft scoring against a hint table.

    Stub for agent reasoning. With an LLM, agents decide domain directly.
    In stub mode, we score against keyword/source/channel evidence.

    Multiple signals are additive — "deploy fix for customer bug" scores
    across technical, operations, and business instead of snapping to
    whichever keyword list matched first.
    """
    terms = signal.terms
    scores: dict[str, float] = {}

    for domain, hints in _DOMAIN_HINTS.items():
        score = 0.0
        # Keyword evidence — each matching term adds weight
        kw_matches = len(terms & hints["keywords"])
        score += kw_matches * 0.2
        # Source evidence
        if signal.source in hints["sources"]:
            score += 0.3
        # Channel evidence
        ch = signal.channel.lower()
        if any(c in ch for c in hints["channels"]):
            score += 0.2
        # Metadata evidence (labels, tags)
        labels = signal.metadata.get("labels", [])
        if isinstance(labels, list):
            label_terms = {str(lb).lower() for lb in labels}
            label_hits = len(label_terms & hints["keywords"])
            score += label_hits * 0.1
        scores[domain] = score

    if not scores or max(scores.values()) == 0:
        return "general"

    return max(scores, key=lambda d: scores[d])
