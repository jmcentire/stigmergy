"""End-to-end scenario tests demonstrating the full system.

These tests simulate realistic organizational signal flows and verify
the system behaves correctly at every stage. Each scenario exercises
the full pipeline: ingestion → classification → assessment → consensus →
constraints → output, with metrics and traces.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stigmergy.adapters.github import MockGitHubAdapter
from stigmergy.adapters.linear import MockLinearAdapter
from stigmergy.adapters.slack import MockSlackAdapter
from stigmergy.connectors.metrics import MetricsCollector
from stigmergy.constraints.filter import ConstraintAction, ConstraintEvaluator
from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.pipeline.processor import AgentRegistry, ContextRegistry, Pipeline
from stigmergy.primitives.agent import Agent
from stigmergy.primitives.assessment import AssessmentAction
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource
from stigmergy.services.embedding import StubEmbeddingService
from stigmergy.services.llm import StubLLMService
from stigmergy.services.token_budget import TokenBudget
from stigmergy.tracing.trace import TraceLog


def _build_system() -> tuple[Pipeline, list[Context], list[Agent]]:
    """Build a complete system with realistic contexts and agents."""
    metrics = MetricsCollector()
    ctx_registry = ContextRegistry()
    agent_registry = AgentRegistry()

    # Create contexts representing organizational knowledge domains
    ctx_engineering = Context(relevance_threshold=0.15, business_weight=3.0)
    ctx_engineering.terms = {"bug", "error", "fix", "pr", "merge", "code", "deploy",
                             "sync", "api", "webhook", "cache", "pms", "integration",
                             "booking", "availability", "pricing"}
    ctx_engineering.source_counts = {"slack": 60, "github": 30, "linear": 10}
    ctx_engineering.author_counts = {"bob.martinez": 30, "alice.chen": 25,
                                     "dave.kim": 20, "grace.liu": 15}
    ctx_engineering.signal_count = 100
    ctx_engineering.last_signal = datetime.now(timezone.utc)

    ctx_support = Context(relevance_threshold=0.15, business_weight=2.0)
    ctx_support.terms = {"customer", "issue", "booking", "double", "complaint",
                         "support", "ticket", "resolution", "property", "guest"}
    ctx_support.source_counts = {"slack": 70, "linear": 30}
    ctx_support.author_counts = {"eve.santos": 50, "frank.reyes": 20}
    ctx_support.signal_count = 70
    ctx_support.last_signal = datetime.now(timezone.utc)

    ctx_leadership = Context(relevance_threshold=0.2, business_weight=5.0)
    ctx_leadership.terms = {"revenue", "market", "strategy", "growth", "expansion",
                            "quarterly", "pipeline", "forecast", "coastal", "region"}
    ctx_leadership.source_counts = {"slack": 40, "docs": 30}
    ctx_leadership.author_counts = {"carol.park": 60}
    ctx_leadership.signal_count = 40
    ctx_leadership.last_signal = datetime.now(timezone.utc)

    # Register contexts
    for ctx in [ctx_engineering, ctx_support, ctx_leadership]:
        ctx_registry.register(ctx)

    # Create agents bound to contexts
    agent_eng = Agent(
        contexts={ctx_engineering.id: 0.95},
        confidence=0.85,
        weights={"technical": 0.9, "business": 0.2, "operations": 0.7, "general": 0.3},
    )
    ctx_engineering.active_agents.add(agent_eng.id)

    agent_sup = Agent(
        contexts={ctx_support.id: 0.9},
        confidence=0.8,
        weights={"technical": 0.4, "business": 0.7, "operations": 0.3, "general": 0.5},
    )
    ctx_support.active_agents.add(agent_sup.id)

    agent_lead = Agent(
        contexts={ctx_leadership.id: 0.95},
        confidence=0.9,
        weights={"technical": 0.3, "business": 0.95, "operations": 0.5, "general": 0.4},
    )
    ctx_leadership.active_agents.add(agent_lead.id)

    # Cross-context agent (an executive-level agent that watches multiple areas)
    agent_exec = Agent(
        contexts={
            ctx_engineering.id: 0.7,
            ctx_leadership.id: 0.9,
            ctx_support.id: 0.5,
        },
        confidence=0.85,
        weights={"technical": 0.8, "business": 0.9, "operations": 0.6, "general": 0.3},
    )
    for ctx in [ctx_engineering, ctx_support, ctx_leadership]:
        ctx.active_agents.add(agent_exec.id)

    for agent in [agent_eng, agent_sup, agent_lead, agent_exec]:
        agent_registry.register(agent)

    pipeline = Pipeline(
        contexts=ctx_registry,
        agents=agent_registry,
        embedding_service=StubEmbeddingService(),
        llm_service=StubLLMService(),
        constraint_evaluator=ConstraintEvaluator.from_config("config/constraints.yaml"),
        trace_log=TraceLog(),
        metrics=metrics,
        familiarity_weights=FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.5,
            source_affinity=0.2,
            temporal_proximity=0.2,
            author_affinity=0.1,
        ),
        consensus_threshold=0.4,
        uncertainty_threshold=0.15,
    )

    contexts = [ctx_engineering, ctx_support, ctx_leadership]
    agents = [agent_eng, agent_sup, agent_lead, agent_exec]
    return pipeline, contexts, agents


class TestMultiSourceIngestion:
    """Signals from Slack, Linear, and GitHub flow through the pipeline."""

    @pytest.mark.asyncio
    async def test_ingest_all_mock_sources(self):
        pipeline, contexts, agents = _build_system()

        slack = MockSlackAdapter()
        linear = MockLinearAdapter()
        github = MockGitHubAdapter()

        slack_signals = await slack.emit_all()
        linear_signals = await linear.emit_all()
        github_signals = await github.emit_all()

        all_signals = slack_signals + linear_signals + github_signals
        assert len(all_signals) > 10  # we have realistic mock data

        for signal in all_signals:
            trace = await pipeline.process_signal(signal)
            assert trace.signal_id == signal.id

        # Verify metrics
        total = pipeline.metrics.counter_value("pipeline.signals_processed")
        assert total == len(all_signals)

        # Verify traces were recorded
        assert pipeline.traces.count() == len(all_signals)


class TestSignalRouting:
    """Signals route to the correct contexts based on content."""

    @pytest.mark.asyncio
    async def test_engineering_signal_routes_to_engineering(self):
        pipeline, contexts, _ = _build_system()
        ctx_eng = contexts[0]

        signal = Signal(
            content="The booking sync is failing for external PMS with 500 errors on the webhook callback",
            source=SignalSource.SLACK,
            channel="#engineering",
            author="bob.martinez",
            timestamp=datetime.now(timezone.utc),
        )
        trace = await pipeline.process_signal(signal)
        assert ctx_eng.id in trace.contexts

    @pytest.mark.asyncio
    async def test_support_signal_routes_to_support(self):
        pipeline, contexts, _ = _build_system()
        ctx_sup = contexts[1]

        signal = Signal(
            content="Customer at the lakefront property is reporting double bookings again, third complaint this month",
            source=SignalSource.SLACK,
            channel="#support",
            author="eve.santos",
            timestamp=datetime.now(timezone.utc),
        )
        trace = await pipeline.process_signal(signal)
        assert ctx_sup.id in trace.contexts

    @pytest.mark.asyncio
    async def test_cross_cutting_signal_routes_to_multiple_contexts(self):
        pipeline, contexts, _ = _build_system()

        signal = Signal(
            content="Customer booking error is a bug in the sync code, need a fix deployed",
            source=SignalSource.SLACK,
            channel="#engineering",
            author="bob.martinez",
            timestamp=datetime.now(timezone.utc),
        )
        trace = await pipeline.process_signal(signal)
        # Should hit both engineering and support contexts
        matched_ids = trace.contexts
        assert len(matched_ids) >= 2

    @pytest.mark.asyncio
    async def test_irrelevant_signal_routes_nowhere_interesting(self):
        pipeline, contexts, _ = _build_system()

        signal = Signal(
            content="Anyone want to grab lunch? Thinking tacos today",
            source=SignalSource.SLACK,
            channel="#general",
            author="frank.reyes",
            timestamp=datetime.now(timezone.utc),
        )
        trace = await pipeline.process_signal(signal)
        # Low-relevance signal shouldn't match high-threshold contexts
        # May match some due to source/temporal affinity but scores should be low
        for ctx_id, score in trace.familiarity_scores.items():
            assert score < 0.5  # nothing should score high


class TestConstraintsInPipeline:
    """Constraint agents kill or redact sensitive content."""

    @pytest.mark.asyncio
    async def test_pii_blocked_in_full_pipeline(self):
        pipeline, _, _ = _build_system()

        signal = Signal(
            content="Customer SSN is 123-45-6789 and they're having booking errors",
            source=SignalSource.SLACK,
            channel="#support",
            author="eve.santos",
            timestamp=datetime.now(timezone.utc),
        )
        trace = await pipeline.process_signal(signal)
        # If the signal reached consensus with SURFACE action,
        # the constraint should have killed it
        if trace.constraint_eval:
            assert trace.constraint_eval.action == ConstraintAction.KILL
            assert trace.output_delivered is False

    @pytest.mark.asyncio
    async def test_credentials_blocked(self):
        pipeline, _, _ = _build_system()

        signal = Signal(
            content="Here's the api_key = sk_test_FAKE0000000000000000000000000 for the sync fix",
            source=SignalSource.SLACK,
            channel="#engineering",
            author="bob.martinez",
            timestamp=datetime.now(timezone.utc),
        )
        trace = await pipeline.process_signal(signal)
        if trace.constraint_eval:
            assert trace.constraint_eval.action == ConstraintAction.KILL


class TestDuplicateDetection:
    """SimHash dedup catches near-duplicate signals."""

    @pytest.mark.asyncio
    async def test_exact_duplicate_detected(self):
        pipeline, _, _ = _build_system()
        content = "The booking sync is failing for external PMS properties with 500 errors"

        signal1 = Signal(
            content=content,
            source=SignalSource.SLACK,
            channel="#engineering",
            author="bob.martinez",
            timestamp=datetime.now(timezone.utc),
        )
        signal2 = Signal(
            content=content,
            source=SignalSource.SLACK,
            channel="#engineering",
            author="bob.martinez",
            timestamp=datetime.now(timezone.utc),
        )

        await pipeline.process_signal(signal1)
        trace2 = await pipeline.process_signal(signal2)
        # Second signal should be caught as duplicate
        dupes = pipeline.metrics.counter_value("pipeline.duplicates_detected")
        assert dupes >= 1


class TestMetricsObservability:
    """Pipeline emits metrics at every stage for observability."""

    @pytest.mark.asyncio
    async def test_pipeline_metrics_emitted(self):
        pipeline, _, _ = _build_system()

        # Distinct content per signal to avoid SimHash dedup
        topics = [
            "The booking sync webhook handler is returning 500 errors for external PMS",
            "Cache invalidation race condition in the availability endpoint",
            "PR #482 merged: pricing engine v3 dynamic rate adjustments",
            "Deploy rollback needed for the API gateway load balancer config",
            "Database migration script failing on the v2Booking column rename",
            "Unit test coverage dropped below 80% on the sync service",
            "Third-party connector integration webhook callback URL needs updating",
            "Redis connection pool exhaustion under peak traffic load",
            "CI pipeline flaky test in the property listing search module",
            "API rate limiting configuration update for the third-party connector",
        ]
        signals = [
            Signal(
                content=topics[i],
                source=SignalSource.SLACK,
                channel="#engineering",
                author="bob.martinez",
                timestamp=datetime.now(timezone.utc),
            )
            for i in range(10)
        ]

        for signal in signals:
            await pipeline.process_signal(signal)

        summary = pipeline.get_metrics_summary()
        assert summary["signals_processed"] == 10.0
        assert summary["latency"]["count"] == 10
        assert summary["latency"]["mean"] >= 0  # non-negative timing

    @pytest.mark.asyncio
    async def test_familiarity_scores_tracked(self):
        pipeline, _, _ = _build_system()

        signal = Signal(
            content="The booking sync error needs a fix in the webhook handler",
            source=SignalSource.SLACK,
            channel="#engineering",
            author="bob.martinez",
            timestamp=datetime.now(timezone.utc),
        )
        await pipeline.process_signal(signal)

        summary = pipeline.get_metrics_summary()
        assert summary["familiarity_scores"]["count"] > 0


class TestTokenBudgetIntegration:
    """Token budget affects pipeline behavior."""

    @pytest.mark.asyncio
    async def test_budget_tracking(self):
        pipeline, contexts, _ = _build_system()
        budget = TokenBudget(daily_cap=100000, reserve_pool=10000)
        budget.allocate({ctx.id: ctx.business_weight for ctx in contexts})
        pipeline.token_budget = budget

        signal = Signal(
            content="Pricing engine v3 showing improvement in dynamic pricing accuracy",
            source=SignalSource.SLACK,
            channel="#data-science",
            author="bob.martinez",
            timestamp=datetime.now(timezone.utc),
        )
        await pipeline.process_signal(signal)
        # Pipeline should still work with budget attached
        assert pipeline.traces.count() == 1


class TestTraceAudit:
    """Every decision is traceable."""

    @pytest.mark.asyncio
    async def test_full_trace_chain(self):
        pipeline, _, _ = _build_system()

        signal = Signal(
            content="Deploy to production scheduled for 3pm. Pricing and sync services.",
            source=SignalSource.SLACK,
            channel="#deploys",
            author="grace.liu",
            timestamp=datetime.now(timezone.utc),
        )
        trace = await pipeline.process_signal(signal)

        # Trace should have full decision chain
        assert trace.signal_id == signal.id
        assert len(trace.familiarity_scores) > 0  # scored against contexts
        # Assessments may or may not exist depending on context matching

    @pytest.mark.asyncio
    async def test_traces_queryable_by_signal(self):
        pipeline, _, _ = _build_system()

        signal = Signal(
            content="Bug fix for availability cache race condition",
            source=SignalSource.GITHUB,
            channel="acme-org/backend",
            author="alice.chen",
            timestamp=datetime.now(timezone.utc),
        )
        await pipeline.process_signal(signal)

        traces = pipeline.traces.for_signal(signal.id)
        assert len(traces) == 1
        assert traces[0].signal_id == signal.id
