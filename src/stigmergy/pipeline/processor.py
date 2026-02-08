"""Pipeline: the full signal flow.

Signal → Dedup → Ingestion → Classification → Active Inquiry → Disposition →
Consensus → Constraints → Output

Optimized with:
- LSH pre-filter: skip irrelevant contexts entirely (O(1) vs O(n) scan)
- SimHash dedup: detect near-duplicate signals in O(1) before processing
- Bloom filter: fast term overlap in familiarity scoring
- Batched vector ops: matrix cosine similarity
- Metrics: every stage instrumented for observability

No orchestrator. Agents pull from queues. Consensus is collected, not directed.
"""

from __future__ import annotations

import logging
import time
from uuid import UUID

from stigmergy.connectors.metrics import MetricsCollector
from stigmergy.constraints.filter import ConstraintAction, ConstraintEvaluator
from stigmergy.core.consensus import ConsensusResult, consensus
from stigmergy.core.familiarity import FamiliarityWeights, familiarity
from stigmergy.primitives.agent import Agent
from stigmergy.primitives.assessment import Assessment, AssessmentAction
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal
from stigmergy.services.embedding import EmbeddingService
from stigmergy.services.llm import LLMService
from stigmergy.services.token_budget import TokenBudget
from stigmergy.structures.bloom import CountingBloomFilter
from stigmergy.structures.lsh import LSHIndex
from stigmergy.structures.simhash import SimHash, SimHashIndex
from stigmergy.tracing.trace import Trace, TraceLog

logger = logging.getLogger(__name__)


class ContextRegistry:
    """Registry of active contexts with LSH index for fast pre-filtering."""

    def __init__(self, embedding_dims: int = 384) -> None:
        self._contexts: dict[UUID, Context] = {}
        self._lsh_index = LSHIndex(dimensions=embedding_dims)

    def register(self, context: Context) -> None:
        self._contexts[context.id] = context

    def unregister(self, context_id: UUID) -> None:
        self._contexts.pop(context_id, None)
        self._lsh_index.remove(str(context_id))

    def get(self, context_id: UUID) -> Context | None:
        return self._contexts.get(context_id)

    def active(self) -> list[Context]:
        return list(self._contexts.values())

    def update_lsh(self, context_id: UUID, centroid: list[float]) -> None:
        """Update the LSH index with a context's centroid."""
        self._lsh_index.update(str(context_id), centroid)

    def candidate_contexts(self, signal_embedding: list[float], top_k: int = 20) -> list[Context]:
        """Use LSH to find candidate contexts for a signal. O(1) avg case."""
        if len(self._lsh_index) == 0:
            return self.active()  # fallback to all if LSH empty
        candidates = self._lsh_index.query_ids(signal_embedding, top_k=top_k)
        result = []
        for cid_str in candidates:
            try:
                ctx = self._contexts.get(UUID(cid_str))
                if ctx:
                    result.append(ctx)
            except ValueError:
                continue
        # Always include contexts not in LSH yet (new, no centroid)
        indexed = {str(ctx.id) for ctx in result}
        for ctx in self._contexts.values():
            if str(ctx.id) not in indexed and str(ctx.id) not in self._lsh_index:
                result.append(ctx)
        return result


class AgentRegistry:
    """Registry of active agents."""

    def __init__(self) -> None:
        self._agents: dict[UUID, Agent] = {}
        self._context_index: dict[UUID, set[UUID]] = {}  # context_id -> agent_ids

    def register(self, agent: Agent) -> None:
        self._agents[agent.id] = agent
        for ctx_id in agent.contexts:
            self._context_index.setdefault(ctx_id, set()).add(agent.id)

    def unregister(self, agent_id: UUID) -> None:
        agent = self._agents.pop(agent_id, None)
        if agent:
            for ctx_id in agent.contexts:
                ctx_agents = self._context_index.get(ctx_id)
                if ctx_agents:
                    ctx_agents.discard(agent_id)

    def get(self, agent_id: UUID) -> Agent | None:
        return self._agents.get(agent_id)

    def as_dict(self) -> dict[UUID, Agent]:
        return dict(self._agents)

    def for_context(self, context_id: UUID) -> list[Agent]:
        """O(k) where k is agents in context, instead of O(n) full scan."""
        agent_ids = self._context_index.get(context_id, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def all_agents(self) -> list[Agent]:
        """All registered agents."""
        return list(self._agents.values())


class Pipeline:
    """Full signal processing pipeline with observability and fast-path optimizations.

    Signal → Dedup → Classify (LSH pre-filter) → Assess → Consensus → Constraints → Output
    """

    def __init__(
        self,
        contexts: ContextRegistry,
        agents: AgentRegistry,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        constraint_evaluator: ConstraintEvaluator,
        trace_log: TraceLog,
        metrics: MetricsCollector | None = None,
        token_budget: TokenBudget | None = None,
        familiarity_weights: FamiliarityWeights | None = None,
        consensus_threshold: float = 0.6,
        uncertainty_threshold: float = 0.3,
        dedup_enabled: bool = True,
        dedup_threshold: int = 3,
    ) -> None:
        self.contexts = contexts
        self.agents = agents
        self.embedding = embedding_service
        self.llm = llm_service
        self.constraints = constraint_evaluator
        self.traces = trace_log
        self.metrics = metrics or MetricsCollector()
        self.token_budget = token_budget
        self.familiarity_weights = familiarity_weights or FamiliarityWeights()
        self.consensus_threshold = consensus_threshold
        self.uncertainty_threshold = uncertainty_threshold

        # SimHash dedup index
        self._dedup_enabled = dedup_enabled
        self._dedup_index = SimHashIndex(distance_threshold=dedup_threshold)

    async def process_signal(self, signal: Signal) -> Trace:
        """Process a single signal through the full pipeline.

        Returns the trace for this signal's journey through the system.
        """
        pipeline_start = time.monotonic()
        trace = Trace(signal_id=signal.id)
        self.metrics.count("pipeline.signals_received")
        self.metrics.count("pipeline.signals_received", source=signal.source)

        # 0. Dedup check
        if self._dedup_enabled:
            dedup_start = time.monotonic()
            terms = list(signal.terms)
            fingerprint = SimHash.fingerprint(terms)
            if self._dedup_index.has_near_duplicate(fingerprint):
                self.metrics.count("pipeline.duplicates_detected")
                self.metrics.count("pipeline.duplicates_detected", source=signal.source)
                trace.output_delivered = False
                self.traces.append(trace)
                self.metrics.timer("pipeline.dedup_ms", (time.monotonic() - dedup_start) * 1000)
                return trace
            self._dedup_index.add(str(signal.id), fingerprint)
            self.metrics.timer("pipeline.dedup_ms", (time.monotonic() - dedup_start) * 1000)

        # 1. Embed the signal if not already embedded
        embed_start = time.monotonic()
        if not signal.embeddings:
            signal_embeddings = await self.embedding.embed_all(signal.content)
        else:
            signal_embeddings = dict(signal.embeddings)
        signal_with_embeddings = signal.model_copy(update={"embeddings": signal_embeddings})
        self.metrics.timer("pipeline.embed_ms", (time.monotonic() - embed_start) * 1000)

        # 2. Classification — LSH pre-filter, then familiarity scoring
        classify_start = time.monotonic()

        # Build a bloom filter for signal terms (for fast overlap estimation)
        signal_bloom = CountingBloomFilter(capacity=1000)
        signal_bloom.add_many(signal.terms)

        # LSH pre-filter: only evaluate candidate contexts
        semantic_embedding = signal_embeddings.get("semantic")
        if semantic_embedding:
            candidate_contexts = self.contexts.candidate_contexts(semantic_embedding)
        else:
            candidate_contexts = self.contexts.active()

        self.metrics.gauge("pipeline.candidate_contexts", len(candidate_contexts))
        self.metrics.gauge("pipeline.total_contexts", len(self.contexts.active()))

        relevant_contexts: list[tuple[Context, float]] = []
        for context in candidate_contexts:
            centroid = await context.vector_store.get_centroid("semantic")
            score = familiarity(
                signal_with_embeddings,
                context,
                weights=self.familiarity_weights,
                centroid=centroid,
            )
            trace.familiarity_scores[context.id] = score
            self.metrics.histogram("pipeline.familiarity_score", score)
            self.metrics.histogram("pipeline.familiarity_score", score, context=str(context.id))

            if score > context.relevance_threshold:
                relevant_contexts.append((context, score))
                await context.ingest_signal(
                    signal_id=str(signal.id),
                    embeddings=signal_embeddings,
                    terms=signal.terms,
                    source=signal.source,
                    author=signal.author,
                    score=score,
                    metadata=dict(signal.metadata),
                )
                # Update LSH index with new centroid
                new_centroid = await context.vector_store.get_centroid("semantic")
                if new_centroid:
                    self.contexts.update_lsh(context.id, new_centroid)

        trace.contexts = {ctx.id for ctx, _ in relevant_contexts}
        self.metrics.timer("pipeline.classify_ms", (time.monotonic() - classify_start) * 1000)
        self.metrics.histogram("pipeline.contexts_matched", len(relevant_contexts))

        # 3. Agent assessment
        assess_start = time.monotonic()
        all_assessments: list[Assessment] = []
        for context, score in relevant_contexts:
            context_agents = self.agents.for_context(context.id)
            for agent in context_agents:
                domain = self._infer_domain(signal)
                assessment = await agent.evaluate(signal_with_embeddings, context.id, domain)
                all_assessments.append(assessment)
                self.metrics.count("pipeline.assessments_produced", domain=domain, action=assessment.action)

        trace.assessments = all_assessments
        self.metrics.timer("pipeline.assess_ms", (time.monotonic() - assess_start) * 1000)

        # 4. Consensus
        consensus_start = time.monotonic()
        if all_assessments:
            result = consensus(
                all_assessments,
                self.agents.as_dict(),
                consensus_threshold=self.consensus_threshold,
                uncertainty_threshold=self.uncertainty_threshold,
            )
            trace.consensus = result
            self.metrics.count("pipeline.consensus_resolved" if result.resolved else "pipeline.consensus_unresolved")
            self.metrics.histogram("pipeline.consensus_weight", result.total_weight)

            # 5. Constraint check (if surfacing or blocking)
            if result.action in (AssessmentAction.SURFACE, AssessmentAction.BLOCK):
                constraint_start = time.monotonic()
                output_content = signal.content
                constraint_result = self.constraints.evaluate(output_content)
                trace.constraint_eval = constraint_result
                self.metrics.count("pipeline.constraint_result", action=constraint_result.action)
                self.metrics.timer("pipeline.constraint_ms", (time.monotonic() - constraint_start) * 1000)

                if constraint_result.action == ConstraintAction.KILL:
                    logger.info("Output killed by constraint: %s", constraint_result.category)
                    trace.output_delivered = False
                elif constraint_result.action == ConstraintAction.REDACT:
                    logger.info("Output redacted: %s", constraint_result.category)
                    trace.output_content = constraint_result.content
                    trace.output_delivered = True
                else:
                    trace.output_content = output_content
                    trace.output_delivered = True
            else:
                trace.output_delivered = False
        else:
            trace.output_delivered = False

        self.metrics.timer("pipeline.consensus_ms", (time.monotonic() - consensus_start) * 1000)

        # 6. Log trace
        self.traces.append(trace)
        total_ms = (time.monotonic() - pipeline_start) * 1000
        self.metrics.timer("pipeline.total_ms", total_ms)
        self.metrics.count("pipeline.signals_processed")
        self.metrics.count("pipeline.signals_processed", source=signal.source)

        return trace

    def _infer_domain(self, signal: Signal) -> str:
        """Infer domain via soft scoring against evidence.

        Stub for agent reasoning. With an LLM, agents decide domain directly.
        Uses the shared hint table from mesh.mesh module. Scores are additive
        across keyword, source, channel, and metadata evidence.
        """
        from stigmergy.mesh.mesh import _infer_domain as _mesh_infer_domain
        return _mesh_infer_domain(signal)

    def get_metrics_summary(self) -> dict:
        """Get a summary of pipeline performance metrics."""
        return {
            "signals_processed": self.metrics.counter_value("pipeline.signals_processed"),
            "duplicates_detected": self.metrics.counter_value("pipeline.duplicates_detected"),
            "latency": self.metrics.summary("pipeline.total_ms"),
            "classify_latency": self.metrics.summary("pipeline.classify_ms"),
            "consensus_resolved": self.metrics.counter_value("pipeline.consensus_resolved"),
            "consensus_unresolved": self.metrics.counter_value("pipeline.consensus_unresolved"),
            "constraints": {
                "kills": self.metrics.counter_value("pipeline.constraint_result", action="kill"),
                "redacts": self.metrics.counter_value("pipeline.constraint_result", action="redact"),
                "passes": self.metrics.counter_value("pipeline.constraint_result", action="pass"),
            },
            "familiarity_scores": self.metrics.summary("pipeline.familiarity_score"),
        }
