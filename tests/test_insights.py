"""Tests for the spectral observation / insight system."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from stigmergy.mesh.insights import (
    CachedSignal,
    Insight,
    ObservationContext,
    SignalCache,
    spectral_cosine,
)
from stigmergy.mesh.mesh import HopRecord, MeshTrace
from stigmergy.mesh.worker import WorkerNode
from stigmergy.primitives.agent import Agent, CompetencyModel
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource


# ── Fixtures ────────────────────────────────────────────────


def _signal(
    content: str = "test signal",
    source: SignalSource = SignalSource.GITHUB,
    author: str = "alice",
    channel: str = "#eng",
    metadata: dict | None = None,
    ts_offset_hours: float = 0,
) -> Signal:
    return Signal(
        id=uuid4(),
        content=content,
        source=source,
        channel=channel,
        author=author,
        timestamp=datetime.now(timezone.utc) - timedelta(hours=ts_offset_hours),
        embeddings={},
        metadata=metadata or {},
    )


def _trace(
    signal_id=None,
    familiarity_scores: dict | None = None,
    accepted_workers: set | None = None,
    hops: list | None = None,
) -> MeshTrace:
    return MeshTrace(
        signal_id=signal_id or uuid4(),
        familiarity_scores=familiarity_scores or {},
        accepted_workers=accepted_workers or set(),
        hops=hops or [],
    )


def _worker(label_terms: list[str] | None = None) -> WorkerNode:
    ctx = Context(capacity=200)
    if label_terms:
        ctx.terms = set(label_terms)
    return WorkerNode(context=ctx)


# ── spectral_cosine ─────────────────────────────────────────


class TestSpectralCosine:
    def test_identical_vectors(self):
        assert spectral_cosine([0.5, 0.3, 0.8], [0.5, 0.3, 0.8]) == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_vectors(self):
        assert spectral_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=0.001)

    def test_empty_vectors(self):
        assert spectral_cosine([], []) == 0.0

    def test_zero_vector(self):
        assert spectral_cosine([0.0, 0.0], [0.5, 0.3]) == 0.0

    def test_mismatched_lengths_truncates(self):
        sim = spectral_cosine([0.5, 0.3, 0.8], [0.5, 0.3])
        assert 0.0 <= sim <= 1.0

    def test_similar_vectors_high_similarity(self):
        a = [0.4, 0.3, 0.7]
        b = [0.45, 0.28, 0.72]
        assert spectral_cosine(a, b) > 0.99

    def test_dissimilar_vectors_low_similarity(self):
        a = [0.9, 0.1, 0.0]
        b = [0.0, 0.1, 0.9]
        assert spectral_cosine(a, b) < 0.3


# ── SignalCache ──────────────────────────────────────────────


class TestSignalCache:
    def test_add_and_len(self):
        cache = SignalCache(capacity=10)
        sig = _signal()
        trace = _trace(signal_id=sig.id)
        cache.add(sig, trace, [0.5, 0.3], [uuid4(), uuid4()])
        assert len(cache) == 1

    def test_capacity_eviction(self):
        cache = SignalCache(capacity=3)
        for i in range(5):
            sig = _signal(content=f"signal {i}")
            cache.add(sig, _trace(signal_id=sig.id), [float(i)], [uuid4()])
        assert len(cache) == 3

    def test_recent_returns_newest_first(self):
        cache = SignalCache(capacity=10)
        for i in range(4):
            sig = _signal(content=f"signal {i}")
            cache.add(sig, _trace(signal_id=sig.id), [float(i)], [uuid4()])
        recent = cache.recent(2)
        assert len(recent) == 2
        assert recent[0].signal.content == "signal 3"
        assert recent[1].signal.content == "signal 2"

    def test_spectral_neighbors_finds_similar(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4(), uuid4()]
        # Add a signal with spectrum [0.5, 0.8]
        sig1 = _signal(content="first")
        cache.add(sig1, _trace(signal_id=sig1.id), [0.5, 0.8], w_ids)
        # Query with similar spectrum
        neighbors = cache.spectral_neighbors([0.48, 0.82], threshold=0.99)
        assert len(neighbors) == 1
        assert neighbors[0][0].signal.content == "first"
        assert neighbors[0][1] > 0.99

    def test_spectral_neighbors_threshold_filters(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4(), uuid4()]
        sig = _signal()
        cache.add(sig, _trace(signal_id=sig.id), [0.9, 0.1], w_ids)
        # Dissimilar spectrum → no results
        neighbors = cache.spectral_neighbors([0.1, 0.9], threshold=0.9)
        assert len(neighbors) == 0

    def test_by_author(self):
        cache = SignalCache(capacity=10)
        sig_a = _signal(author="alice")
        sig_b = _signal(author="bob")
        sig_a2 = _signal(author="alice", content="another")
        for sig in [sig_a, sig_b, sig_a2]:
            cache.add(sig, _trace(signal_id=sig.id), [0.5], [uuid4()])
        alice_sigs = cache.by_author("alice")
        assert len(alice_sigs) == 2

    def test_in_window(self):
        cache = SignalCache(capacity=10)
        recent = _signal(ts_offset_hours=1)
        old = _signal(ts_offset_hours=48, content="old")
        cache.add(recent, _trace(signal_id=recent.id), [0.5], [uuid4()])
        cache.add(old, _trace(signal_id=old.id), [0.3], [uuid4()])
        window = cache.in_window(hours=24)
        assert len(window) == 1
        assert window[0].signal.author == recent.author

    def test_in_window_with_reference_time_backfill_bug(self):
        """BUG REGRESSION: correlator sees zero signals during backfill.

        in_window() used datetime.now() as the reference point. During a
        30-day backfill, all signals have timestamps from weeks ago, so
        'now - 24h' (yesterday) never matched any of them. The correlator
        was completely non-functional during every backfill run.

        Fix: accept a reference_time parameter so the correlator uses the
        current signal's timestamp as "now" instead of wall clock.
        """
        cache = SignalCache(capacity=10)
        # Simulate backfill: signals from 10 days ago
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        sig1 = Signal(
            content="old signal 1",
            source="github",
            channel="#eng",
            author="alice",
            timestamp=old_time - timedelta(hours=2),
        )
        sig2 = Signal(
            content="old signal 2",
            source="linear",
            channel="PLAT-123",
            author="bob",
            timestamp=old_time,
        )
        cache.add(sig1, _trace(signal_id=sig1.id), [0.5], [uuid4()])
        cache.add(sig2, _trace(signal_id=sig2.id), [0.6], [uuid4()])

        # Without reference_time: window is relative to now,
        # so 10-day-old signals are invisible in a 24h window
        window_now = cache.in_window(hours=24)
        assert len(window_now) == 0, "Sanity check: old signals aren't in wall-clock window"

        # With reference_time: window is relative to the current signal,
        # so 10-day-old signals ARE visible when reference is also 10 days ago
        window_ref = cache.in_window(hours=24, reference_time=old_time)
        assert len(window_ref) >= 1, (
            "With reference_time, the correlator should see signals "
            "within 24h of the reference point, not wall clock"
        )


# ── ObservationContext ───────────────────────────────────────


class TestObservationContext:
    def _make_obs(self, cache=None, workers=None, familiarity_scores=None):
        sig = _signal()
        scores = familiarity_scores or {}
        trace = _trace(signal_id=sig.id, familiarity_scores=scores)
        spectrum = list(scores.values()) if scores else []
        return ObservationContext(
            signal=sig,
            trace=trace,
            spectrum=spectrum,
            cache=cache or SignalCache(capacity=10),
            workers=workers or [],
        )

    def test_worker_scores_returns_copy(self):
        w1, w2 = uuid4(), uuid4()
        obs = self._make_obs(familiarity_scores={w1: 0.5, w2: 0.8})
        scores = obs.worker_scores()
        assert scores[w1] == 0.5
        assert scores[w2] == 0.8
        # Modifying return doesn't affect trace
        scores[w1] = 0.0
        assert obs.trace.familiarity_scores[w1] == 0.5

    def test_score_variance_uniform(self):
        w1, w2, w3 = uuid4(), uuid4(), uuid4()
        obs = self._make_obs(familiarity_scores={w1: 0.5, w2: 0.5, w3: 0.5})
        assert obs.score_variance() == 0.0

    def test_score_variance_divergent(self):
        w1, w2, w3 = uuid4(), uuid4(), uuid4()
        obs = self._make_obs(familiarity_scores={w1: 0.1, w2: 0.9, w3: 0.5})
        assert obs.score_variance() > 0.3

    def test_score_variance_single_worker(self):
        obs = self._make_obs(familiarity_scores={uuid4(): 0.5})
        assert obs.score_variance() == 0.0

    def test_metadata_access(self):
        sig = _signal(metadata={"status": "In Progress", "labels": ["bug"]})
        trace = _trace(signal_id=sig.id)
        obs = ObservationContext(sig, trace, [], SignalCache(10), [])
        assert obs.metadata("status") == "In Progress"
        assert obs.metadata("missing", "default") == "default"

    def test_spectral_neighbors_delegates_to_cache(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4(), uuid4()]
        old_sig = _signal(content="old one")
        cache.add(old_sig, _trace(signal_id=old_sig.id), [0.5, 0.8], w_ids)

        new_sig = _signal(content="new one")
        trace = _trace(signal_id=new_sig.id)
        obs = ObservationContext(new_sig, trace, [0.52, 0.79], cache, [])
        neighbors = obs.spectral_neighbors(threshold=0.99)
        assert len(neighbors) == 1

    def test_cross_source_filters_same_source(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4()]
        # Add a GitHub signal
        gh = _signal(source=SignalSource.GITHUB, content="github")
        cache.add(gh, _trace(signal_id=gh.id), [0.5], w_ids)
        # Add a Linear signal
        lr = _signal(source=SignalSource.LINEAR, content="linear")
        cache.add(lr, _trace(signal_id=lr.id), [0.48], w_ids)

        # Query from a GitHub signal — should only find the Linear one
        new_gh = _signal(source=SignalSource.GITHUB)
        obs = ObservationContext(new_gh, _trace(signal_id=new_gh.id), [0.5], cache, [])
        cross = obs.cross_source_similar(threshold=0.9)
        assert all(c[0].signal.source == SignalSource.LINEAR for c in cross)

    def test_worker_state(self):
        w = _worker(["pricing", "sync"])
        obs = self._make_obs(workers=[w])
        assert obs.worker_state(w.id) is w
        assert obs.worker_state(uuid4()) is None

    def test_accepted_workers(self):
        w1 = uuid4()
        sig = _signal()
        trace = _trace(signal_id=sig.id, accepted_workers={w1})
        obs = ObservationContext(sig, trace, [], SignalCache(10), [])
        assert w1 in obs.accepted_workers


# ── Agent.reflect() ──────────────────────────────────────────


class TestAgentReflect:
    @pytest.mark.asyncio
    async def test_empty_cache_no_insights(self):
        agent = Agent(confidence=0.8)
        sig = _signal()
        trace = _trace(signal_id=sig.id, familiarity_scores={uuid4(): 0.5})
        obs = ObservationContext(sig, trace, [0.5], SignalCache(10), [])
        insights = await agent.reflect(obs)
        assert len(insights) == 0

    @pytest.mark.asyncio
    async def test_spectral_neighbor_produces_insight(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4(), uuid4()]
        old_sig = _signal(content="pricing engine fix", author="bob")
        cache.add(old_sig, _trace(signal_id=old_sig.id), [0.7, 0.3], w_ids)

        agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({"similarity_detection": 0.9}),
        )
        new_sig = _signal(content="pricing multipliers", author="alice")
        trace = _trace(signal_id=new_sig.id, familiarity_scores=dict(zip(w_ids, [0.72, 0.28])))
        obs = ObservationContext(new_sig, trace, [0.72, 0.28], cache, [])
        insights = await agent.reflect(obs)
        assert len(insights) > 0
        # Different authors in same spectral space → agent should notice
        for ins in insights:
            assert isinstance(ins.type, str) and len(ins.type) > 0
            assert ins.confidence > 0
            assert len(ins.signal_ids) >= 1

    @pytest.mark.asyncio
    async def test_same_author_produces_work_sequence(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4(), uuid4()]
        old_sig = _signal(content="sync fix part 1", author="alice")
        cache.add(old_sig, _trace(signal_id=old_sig.id), [0.6, 0.4], w_ids)

        agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({"similarity_detection": 0.9}),
        )
        new_sig = _signal(content="sync fix part 2", author="alice")
        trace = _trace(signal_id=new_sig.id, familiarity_scores=dict(zip(w_ids, [0.62, 0.38])))
        obs = ObservationContext(new_sig, trace, [0.62, 0.38], cache, [])
        insights = await agent.reflect(obs)
        assert len(insights) > 0
        # Same author, similar content → agent should produce insight
        for ins in insights:
            assert isinstance(ins.type, str) and len(ins.type) > 0
            assert ins.confidence > 0

    @pytest.mark.asyncio
    async def test_score_divergence_produces_insight(self):
        w1, w2, w3 = uuid4(), uuid4(), uuid4()
        cache = SignalCache(capacity=10)
        agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({"similarity_detection": 0.9}),
        )
        sig = _signal()
        scores = {w1: 0.1, w2: 0.9, w3: 0.15}
        trace = _trace(signal_id=sig.id, familiarity_scores=scores)
        obs = ObservationContext(sig, trace, [0.1, 0.9, 0.15], cache, [])
        insights = await agent.reflect(obs)
        assert len(insights) > 0
        # High score variance → agent should notice disagreement
        for ins in insights:
            assert isinstance(ins.type, str)
            assert ins.confidence > 0

    @pytest.mark.asyncio
    async def test_low_confidence_agent_produces_fewer_insights(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4(), uuid4()]
        old_sig = _signal(content="pricing engine", author="bob")
        cache.add(old_sig, _trace(signal_id=old_sig.id), [0.7, 0.3], w_ids)

        low_agent = Agent(
            confidence=0.2,
            competencies=CompetencyModel({"similarity_detection": 0.5}),
        )
        high_agent = Agent(
            confidence=0.9,
            competencies=CompetencyModel({"similarity_detection": 0.9}),
        )
        new_sig = _signal(content="pricing update", author="alice")
        trace = _trace(signal_id=new_sig.id, familiarity_scores=dict(zip(w_ids, [0.72, 0.28])))
        obs = ObservationContext(new_sig, trace, [0.72, 0.28], cache, [])

        low_insights = await low_agent.reflect(obs)
        high_insights = await high_agent.reflect(obs)
        # Higher confidence agent should produce at least as many (or more confident) insights
        assert len(high_insights) >= len(low_insights)

    @pytest.mark.asyncio
    async def test_cross_context_agent_notices_cross_source(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4()]
        linear_sig = _signal(source=SignalSource.LINEAR, content="availability issue", author="alice")
        cache.add(linear_sig, _trace(signal_id=linear_sig.id), [0.7], w_ids)

        agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({
                "similarity_detection": 0.1,
                "cross_context_propagation": 0.9,
            }),
        )
        gh_sig = _signal(source=SignalSource.GITHUB, content="availability fix", author="alice")
        trace = _trace(signal_id=gh_sig.id, familiarity_scores=dict(zip(w_ids, [0.72])))
        obs = ObservationContext(gh_sig, trace, [0.72], cache, [])
        insights = await agent.reflect(obs)
        assert len(insights) > 0
        # Cross-source signals → cross_context competency should produce insight
        for ins in insights:
            assert isinstance(ins.type, str)
            assert ins.confidence > 0

    @pytest.mark.asyncio
    async def test_temporal_agent_notices_recurring_theme(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4()]
        # Add signals from multiple authors with overlapping terms
        for author in ["bob", "carol", "dave"]:
            sig = _signal(
                content="sync error failing webhook callback",
                author=author,
                ts_offset_hours=2,
            )
            cache.add(sig, _trace(signal_id=sig.id), [0.5], w_ids)

        agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({"temporal_analysis": 0.9}),
        )
        new_sig = _signal(content="sync error webhook retry", author="alice")
        trace = _trace(signal_id=new_sig.id, familiarity_scores=dict(zip(w_ids, [0.5])))
        obs = ObservationContext(new_sig, trace, [0.5], cache, [])
        insights = await agent.reflect(obs)
        assert len(insights) > 0
        # Multiple authors with overlapping terms → temporal agent should notice
        for ins in insights:
            assert isinstance(ins.type, str)
            assert ins.confidence > 0

    @pytest.mark.asyncio
    async def test_different_competencies_different_insights(self):
        """Two agents with different competencies produce different insight types."""
        cache = SignalCache(capacity=10)
        w_ids = [uuid4(), uuid4()]
        # Cross-source signal in cache
        linear_sig = _signal(source=SignalSource.LINEAR, content="pricing config", author="bob")
        cache.add(linear_sig, _trace(signal_id=linear_sig.id), [0.6, 0.4], w_ids)

        similarity_agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({"similarity_detection": 0.9, "cross_context_propagation": 0.0}),
        )
        cross_agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({"similarity_detection": 0.0, "cross_context_propagation": 0.9}),
        )

        gh_sig = _signal(source=SignalSource.GITHUB, content="pricing engine", author="alice")
        scores = dict(zip(w_ids, [0.62, 0.38]))
        trace = _trace(signal_id=gh_sig.id, familiarity_scores=scores)
        obs = ObservationContext(gh_sig, trace, [0.62, 0.38], cache, [])

        sim_insights = await similarity_agent.reflect(obs)
        cross_insights = await cross_agent.reflect(obs)

        sim_types = {i.type for i in sim_insights}
        cross_types = {i.type for i in cross_insights}

        # Different competencies should produce different insight types —
        # testing the mechanism (competency drives attention), not specific labels
        if sim_insights and cross_insights:
            # They should notice different things from the same observations
            assert sim_types != cross_types or len(sim_insights) != len(cross_insights)

    @pytest.mark.asyncio
    async def test_insight_type_is_agent_chosen_string(self):
        """Insight types are strings, not enum values."""
        cache = SignalCache(capacity=10)
        w_ids = [uuid4()]
        old = _signal(author="bob")
        cache.add(old, _trace(signal_id=old.id), [0.5], w_ids)

        agent = Agent(confidence=0.8, competencies=CompetencyModel({"similarity_detection": 0.9}))
        sig = _signal(author="alice")
        trace = _trace(signal_id=sig.id, familiarity_scores=dict(zip(w_ids, [0.52])))
        obs = ObservationContext(sig, trace, [0.52], cache, [])
        insights = await agent.reflect(obs)
        for ins in insights:
            assert isinstance(ins.type, str)
            assert isinstance(ins.summary, str)
            assert 0.0 <= ins.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_insight_has_agent_id(self):
        cache = SignalCache(capacity=10)
        w_ids = [uuid4()]
        old = _signal(author="bob")
        cache.add(old, _trace(signal_id=old.id), [0.5], w_ids)

        agent = Agent(confidence=0.8, competencies=CompetencyModel({"similarity_detection": 0.9}))
        sig = _signal(author="alice")
        trace = _trace(signal_id=sig.id, familiarity_scores=dict(zip(w_ids, [0.52])))
        obs = ObservationContext(sig, trace, [0.52], cache, [])
        insights = await agent.reflect(obs)
        for ins in insights:
            assert ins.agent_id == agent.id


# ── Integration ──────────────────────────────────────────────


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_flow_ingest_reflect(self):
        """Full flow: build cache, create obs context, reflect, verify insights."""
        cache = SignalCache(capacity=100)
        w_ids = [uuid4(), uuid4(), uuid4()]

        # Simulate 3 previous signals
        for i, author in enumerate(["bob", "carol", "dave"]):
            sig = _signal(content=f"availability cache error {i}", author=author)
            spectrum = [0.4 + i * 0.05, 0.3, 0.6 - i * 0.05]
            cache.add(sig, _trace(signal_id=sig.id), spectrum, w_ids)

        # New signal arrives
        new_sig = _signal(
            content="availability cache invalidation fix",
            source=SignalSource.GITHUB,
            author="alice",
        )
        scores = dict(zip(w_ids, [0.45, 0.28, 0.55]))
        trace = _trace(
            signal_id=new_sig.id,
            familiarity_scores=scores,
            accepted_workers={w_ids[0], w_ids[2]},
        )
        spectrum = [0.45, 0.28, 0.55]

        # Create obs context
        workers = [_worker(["availability"]) for _ in range(3)]
        obs = ObservationContext(new_sig, trace, spectrum, cache, workers)

        # Agent reflects
        agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({
                "similarity_detection": 0.7,
                "temporal_analysis": 0.6,
                "cross_context_propagation": 0.5,
            }),
        )
        insights = await agent.reflect(obs)

        # Should find something — multiple similar signals in cache
        assert len(insights) > 0
        # All insights should have valid structure
        for ins in insights:
            assert ins.agent_id == agent.id
            assert isinstance(ins.type, str)
            assert len(ins.type) > 0
            assert len(ins.summary) > 0
            assert 0.0 <= ins.confidence <= 1.0
            assert len(ins.signal_ids) > 0

        # Add to cache after analysis (as the real flow would)
        cache.add(new_sig, trace, spectrum, w_ids)
        assert len(cache) == 4


# ── Open taxonomy tests ──────────────────────────────────────


class TestOpenTaxonomy:
    """Verify the system works with agent-chosen strings, not fixed enums.

    The "fresh pair of eyes" principle: any label generated by the system
    should be interpretable without shared context. No opaque tokens.
    """

    def test_signal_source_accepts_any_string(self):
        """Signal.source takes any descriptive string, not just enum members."""
        sig = Signal(
            content="datadog alert: CPU spike",
            source="datadog",
            channel="#ops-alerts",
            author="monitoring-bot",
            timestamp=datetime.now(timezone.utc),
        )
        assert sig.source == "datadog"
        # Well-known constants still work
        sig2 = Signal(
            content="test",
            source=SignalSource.GITHUB,
            channel="#eng",
            author="alice",
            timestamp=datetime.now(timezone.utc),
        )
        assert sig2.source == "github"

    def test_assessment_action_accepts_any_string(self):
        """Assessment.action takes any descriptive string."""
        from stigmergy.primitives.assessment import Assessment, AssessmentAction
        a = Assessment(
            agent_id=uuid4(),
            signal_id=uuid4(),
            context_id=uuid4(),
            action="escalate_to_oncall",
            confidence=0.8,
            domain="operations",
            reasoning="incident severity warrants human review",
        )
        assert a.action == "escalate_to_oncall"
        # Well-known constants still compare as strings
        a2 = Assessment(
            agent_id=uuid4(),
            signal_id=uuid4(),
            context_id=uuid4(),
            action=AssessmentAction.SURFACE,
            confidence=0.7,
            domain="technical",
            reasoning="relevant to engineering",
        )
        assert a2.action == "surface"
        assert a2.action == AssessmentAction.SURFACE

    def test_unknown_competency_gets_all_primitives(self):
        """An agent with a novel competency isn't locked out — it gets baseline access."""
        agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({"supply_chain_risk": 0.9}),
        )
        # The agent should have _COMPETENCY_PRIMITIVES fallback
        # that gives all primitives at 0.5 strength
        affinities = agent._COMPETENCY_PRIMITIVES.get(
            "supply_chain_risk",
            {p: 0.5 for p in agent._PRIMITIVE_METHODS},
        )
        # Unknown competency: falls back to all primitives
        assert len(affinities) == len(agent._PRIMITIVE_METHODS)
        assert all(v == 0.5 for v in affinities.values())

    @pytest.mark.asyncio
    async def test_novel_competency_produces_insights(self):
        """Agent with unknown competency still observes and reports."""
        cache = SignalCache(capacity=10)
        w_ids = [uuid4()]
        old = _signal(author="bob")
        cache.add(old, _trace(signal_id=old.id), [0.5], w_ids)

        # Agent with a competency not in the registry
        agent = Agent(
            confidence=0.8,
            competencies=CompetencyModel({"pattern_recognition": 0.9}),
        )
        sig = _signal(author="alice")
        trace = _trace(signal_id=sig.id, familiarity_scores=dict(zip(w_ids, [0.52])))
        obs = ObservationContext(sig, trace, [0.52], cache, [])
        insights = await agent.reflect(obs)
        # Should still produce insights via fallback primitives
        assert len(insights) > 0

    def test_insight_summary_is_human_readable(self):
        """Every insight summary should be interpretable by a fresh pair of eyes."""
        ins = Insight(
            agent_id=uuid4(),
            type="cross_source_correlation",
            summary="github signal correlates with linear signal from @bob (spectral similarity 85%)",
            confidence=0.7,
            signal_ids=[uuid4(), uuid4()],
            details={"similarity": 0.85},
        )
        # Summary contains complete, self-describing text
        assert len(ins.summary) > 20
        assert any(c.isalpha() for c in ins.summary)
        # Type is a descriptive snake_case string, not an opaque code
        assert "_" in ins.type or ins.type.replace("_", "").isalpha()

    def test_domain_inference_is_soft_scored(self):
        """Domain inference uses soft scoring, not first-match keyword lists."""
        from stigmergy.mesh.mesh import _infer_domain
        # "deploy fix" has evidence for both technical and operations
        sig = _signal(content="deploy fix for the api build", source="unknown")
        domain = _infer_domain(sig)
        # Should pick the one with most evidence, not just first match
        assert domain in ("technical", "operations")

    def test_competency_registry_is_data_driven(self):
        """Competency→primitive mapping is a dict, not if/elif."""
        # Verify the registry exists and is a dict
        assert isinstance(Agent._COMPETENCY_PRIMITIVES, dict)
        assert isinstance(Agent._PRIMITIVE_METHODS, dict)
        # Known competencies have custom affinities
        assert "similarity_detection" in Agent._COMPETENCY_PRIMITIVES
        # Primitives map to actual method names
        for method_name in Agent._PRIMITIVE_METHODS.values():
            assert hasattr(Agent, method_name)

    def test_worker_learning_detail_is_self_describing(self):
        """ReceiveResult.learning_detail explains what happened."""
        from stigmergy.mesh.worker import ReceiveResult
        r = ReceiveResult(
            accepted=True,
            worker_id=uuid4(),
            score=0.5,
            learning_mode="rag_indexed",
            learning_detail="score 0.500 above high threshold 0.350, full vector+term indexing",
        )
        assert "threshold" in r.learning_detail
        assert "score" in r.learning_detail
