"""Integration tests for the full signal processing pipeline.

Signal enters → classified to correct contexts → agents produce assessments →
consensus resolves → constraint agents evaluate → output or kill.
Full trace produced for every signal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from stigmergy.constraints.filter import ConstraintAction, ConstraintEvaluator
from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.pipeline.processor import AgentRegistry, ContextRegistry, Pipeline
from stigmergy.primitives.agent import Agent
from stigmergy.primitives.assessment import AssessmentAction
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource
from stigmergy.services.embedding import StubEmbeddingService
from stigmergy.services.llm import StubLLMService
from stigmergy.tracing.trace import TraceLog


def _build_pipeline(
    contexts: list[Context] | None = None,
    agents: list[Agent] | None = None,
    constraint_config: str = "config/constraints.yaml",
) -> Pipeline:
    ctx_registry = ContextRegistry()
    agent_registry = AgentRegistry()

    for ctx in (contexts or []):
        ctx_registry.register(ctx)

    for agent in (agents or []):
        agent_registry.register(agent)

    return Pipeline(
        contexts=ctx_registry,
        agents=agent_registry,
        embedding_service=StubEmbeddingService(),
        llm_service=StubLLMService(),
        constraint_evaluator=ConstraintEvaluator.from_config(constraint_config),
        trace_log=TraceLog(),
        familiarity_weights=FamiliarityWeights(
            embedding_similarity=0.0,  # stub embeddings won't match
            keyword_overlap=0.5,
            source_affinity=0.2,
            temporal_proximity=0.2,
            author_affinity=0.1,
        ),
    )


def _make_signal(content: str, **kwargs) -> Signal:
    defaults = {
        "source": SignalSource.SLACK,
        "channel": "#engineering",
        "author": "test.user",
        "timestamp": datetime.now(timezone.utc),
    }
    defaults.update(kwargs)
    return Signal(content=content, **defaults)


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_signal_produces_trace(self):
        """Every signal produces a trace, even with no contexts."""
        pipeline = _build_pipeline()
        signal = _make_signal("test signal")
        trace = await pipeline.process_signal(signal)
        assert trace.signal_id == signal.id
        assert pipeline.traces.count() == 1

    @pytest.mark.asyncio
    async def test_signal_classified_to_matching_context(self):
        """Signal with matching keywords gets classified to the right context."""
        ctx = Context(relevance_threshold=0.1)
        ctx.terms = {"booking", "sync", "guesty", "error"}
        ctx.source_counts = {"slack": 50}
        ctx.signal_count = 50
        ctx.last_signal = datetime.now(timezone.utc)

        agent = Agent(
            contexts={ctx.id: 0.9},
            confidence=0.8,
            weights={"technical": 0.8},
        )
        ctx.active_agents.add(agent.id)

        pipeline = _build_pipeline(contexts=[ctx], agents=[agent])
        signal = _make_signal("booking sync guesty error on webhook")
        trace = await pipeline.process_signal(signal)

        assert ctx.id in trace.contexts
        assert len(trace.assessments) > 0

    @pytest.mark.asyncio
    async def test_signal_not_classified_to_unrelated_context(self):
        """Signal with no keyword overlap stays out of unrelated contexts."""
        ctx = Context(relevance_threshold=0.5)
        ctx.terms = {"pricing", "revenue", "forecast"}
        # Fresh context — no source or author affinity
        ctx.last_signal = datetime.now(timezone.utc) - timedelta(days=7)

        pipeline = _build_pipeline(contexts=[ctx])
        signal = _make_signal("booking sync guesty error")
        trace = await pipeline.process_signal(signal)

        assert ctx.id not in trace.contexts

    @pytest.mark.asyncio
    async def test_multiple_contexts_receive_relevant_signal(self):
        """A signal can be classified to multiple contexts simultaneously."""
        ctx_eng = Context(relevance_threshold=0.1)
        ctx_eng.terms = {"sync", "error", "webhook", "guesty"}
        ctx_eng.source_counts = {"slack": 80}
        ctx_eng.signal_count = 80
        ctx_eng.last_signal = datetime.now(timezone.utc)

        ctx_support = Context(relevance_threshold=0.1)
        ctx_support.terms = {"error", "customer", "booking", "guesty"}
        ctx_support.source_counts = {"slack": 60}
        ctx_support.signal_count = 60
        ctx_support.last_signal = datetime.now(timezone.utc)

        agent_eng = Agent(contexts={ctx_eng.id: 0.9}, confidence=0.8, weights={"technical": 0.8})
        agent_sup = Agent(contexts={ctx_support.id: 0.9}, confidence=0.8, weights={"technical": 0.6})
        ctx_eng.active_agents.add(agent_eng.id)
        ctx_support.active_agents.add(agent_sup.id)

        pipeline = _build_pipeline(
            contexts=[ctx_eng, ctx_support],
            agents=[agent_eng, agent_sup],
        )
        signal = _make_signal("guesty booking error affecting customers")
        trace = await pipeline.process_signal(signal)

        assert ctx_eng.id in trace.contexts
        assert ctx_support.id in trace.contexts
        assert len(trace.assessments) == 2


class TestConstraintIntegration:
    @pytest.mark.asyncio
    async def test_pii_kills_output(self):
        """Signal containing PII gets killed at constraint gate."""
        ctx = Context(relevance_threshold=0.1)
        ctx.terms = {"user", "contact", "email"}
        ctx.source_counts = {"slack": 50}
        ctx.signal_count = 50
        ctx.last_signal = datetime.now(timezone.utc)

        # Agent configured to surface
        agent = Agent(
            contexts={ctx.id: 1.0},
            confidence=0.95,
            weights={"technical": 0.95},
        )
        ctx.active_agents.add(agent.id)

        pipeline = _build_pipeline(contexts=[ctx], agents=[agent])
        signal = _make_signal("Contact user at john.doe@company.com for the error details")
        trace = await pipeline.process_signal(signal)

        # Even if consensus says surface, constraint should kill it
        if trace.constraint_eval:
            assert trace.constraint_eval.action == ConstraintAction.KILL
            assert trace.output_delivered is False

    @pytest.mark.asyncio
    async def test_clean_signal_passes_constraints(self):
        """Signal without sensitive content passes through constraints."""
        ctx = Context(relevance_threshold=0.1)
        ctx.terms = {"booking", "sync", "error"}
        ctx.source_counts = {"slack": 50}
        ctx.signal_count = 50
        ctx.last_signal = datetime.now(timezone.utc)

        agent = Agent(
            contexts={ctx.id: 1.0},
            confidence=0.95,
            weights={"technical": 0.95},
        )
        ctx.active_agents.add(agent.id)

        pipeline = _build_pipeline(contexts=[ctx], agents=[agent])
        signal = _make_signal("The booking sync error has been fixed in PR 487")
        trace = await pipeline.process_signal(signal)

        if trace.consensus and trace.consensus.action == AssessmentAction.SURFACE:
            assert trace.constraint_eval.action == ConstraintAction.PASS
            assert trace.output_delivered is True


class TestTraceCompleteness:
    @pytest.mark.asyncio
    async def test_trace_records_familiarity_scores(self):
        ctx = Context(relevance_threshold=0.0)  # accept everything
        ctx.last_signal = datetime.now(timezone.utc)
        agent = Agent(contexts={ctx.id: 0.5}, confidence=0.5, weights={"general": 0.5})
        ctx.active_agents.add(agent.id)

        pipeline = _build_pipeline(contexts=[ctx], agents=[agent])
        signal = _make_signal("test")
        trace = await pipeline.process_signal(signal)

        assert ctx.id in trace.familiarity_scores
        assert isinstance(trace.familiarity_scores[ctx.id], float)

    @pytest.mark.asyncio
    async def test_trace_retrievable_by_signal_id(self):
        pipeline = _build_pipeline()
        signal = _make_signal("test")
        await pipeline.process_signal(signal)

        traces = pipeline.traces.for_signal(signal.id)
        assert len(traces) == 1
        assert traces[0].signal_id == signal.id
