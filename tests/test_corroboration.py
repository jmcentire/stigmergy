"""Tests for Phase 2: cross-signal finding corroboration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from stigmergy.mesh.corroboration import (
    CorroborationEngine,
    CorroborationPayload,
    CorroborationResult,
    FindingRoute,
    FindingRouter,
    QuorumFinding,
    WorkerVerdict,
    run_corroboration,
)
from stigmergy.mesh.insights import Insight
from stigmergy.mesh.worker import WorkerNode
from stigmergy.primitives.context import Context
from stigmergy.services.llm import StubLLMService


# ── Helpers ──────────────────────────────────────────────────


def _make_insight(
    type_: str = "risk",
    summary: str = "Three separate PMS sync failures reported",
    confidence: float = 0.75,
    actors: list | None = None,
) -> Insight:
    return Insight(
        agent_id=uuid4(),
        type=type_,
        summary=summary,
        confidence=confidence,
        signal_ids=[uuid4(), uuid4()],
        details={
            "actors": actors or ["@alice", "@bob"],
            "severity": "high",
            "unknowns": "Do these share a common code path?",
        },
    )


def _make_worker(terms: set[str] | None = None, signal_count: int = 50) -> WorkerNode:
    ctx = Context()
    if terms is not None:
        ctx.terms = terms
    else:
        ctx.terms = {"sync", "pms", "hostaway", "booking", "calendar"}
    ctx.signal_count = signal_count
    return WorkerNode(context=ctx)


# ── FindingRouter tests ──────────────────────────────────────


class TestFindingRouter:
    def test_route_findings_to_relevant_workers(self):
        router = FindingRouter(relevance_threshold=0.01)

        findings = [_make_insight(
            summary="PMS sync failures in hostaway booking pipeline",
        )]
        workers = [
            _make_worker({"sync", "pms", "hostaway", "booking", "pipeline"}),
            _make_worker({"pricing", "engine", "rate", "multiplier", "tax"}),
            _make_worker({"frontend", "react", "component", "layout", "css"}),
        ]

        routes = router.route(findings, workers)

        assert len(routes) >= 1
        # The PMS worker should have the highest relevance
        pms_routes = [r for r in routes if r.worker_id == workers[0].id]
        assert len(pms_routes) == 1
        assert pms_routes[0].relevance > 0

    def test_route_empty_findings(self):
        router = FindingRouter()
        routes = router.route([], [_make_worker()])
        assert routes == []

    def test_route_empty_workers(self):
        router = FindingRouter()
        routes = router.route([_make_insight()], [])
        assert routes == []

    def test_min_workers_enforced(self):
        """Even if relevance is low, at least min_workers get the finding."""
        router = FindingRouter(relevance_threshold=0.99, min_workers=2)

        findings = [_make_insight(summary="completely unrelated topic")]
        workers = [
            _make_worker({"alpha", "beta", "gamma"}),
            _make_worker({"delta", "epsilon", "zeta"}),
            _make_worker({"eta", "theta", "iota"}),
        ]

        routes = router.route(findings, workers)
        # Should have at least min_workers routes even with high threshold
        assert len(routes) >= 2

    def test_max_workers_enforced(self):
        """No more than max_workers should receive a finding."""
        router = FindingRouter(relevance_threshold=0.0, max_workers=3)

        findings = [_make_insight(summary="sync pms booking calendar availability")]
        workers = [_make_worker({"sync", "pms", "booking"}) for _ in range(10)]

        routes = router.route(findings, workers)
        assert len(routes) <= 3

    def test_multiple_findings_routed_independently(self):
        router = FindingRouter(relevance_threshold=0.01, max_workers=2)

        findings = [
            _make_insight(summary="PMS sync failures hostaway booking"),
            _make_insight(summary="Pricing engine tax calculation error"),
        ]
        workers = [
            _make_worker({"sync", "pms", "hostaway", "booking"}),
            _make_worker({"pricing", "engine", "tax", "calculation"}),
        ]

        routes = router.route(findings, workers)
        # Both findings should be routed
        finding_summaries = {r.finding.summary for r in routes}
        assert len(finding_summaries) == 2

    def test_actors_used_as_terms(self):
        """Actors from findings should be included in term extraction."""
        router = FindingRouter(relevance_threshold=0.01)

        findings = [_make_insight(
            summary="Deployment issue",
            actors=["@alice", "@bob"],
        )]
        workers = [
            _make_worker({"alice", "bob", "team"}),
            _make_worker({"charlie", "dave", "other"}),
        ]

        routes = router.route(findings, workers)
        # Worker with matching actors should have higher relevance
        alice_routes = [r for r in routes if r.worker_id == workers[0].id]
        charlie_routes = [r for r in routes if r.worker_id == workers[1].id]
        if alice_routes and charlie_routes:
            assert alice_routes[0].relevance >= charlie_routes[0].relevance


class TestBuildPayloads:
    def test_payloads_bundled_per_worker(self):
        router = FindingRouter()

        w1 = _make_worker({"sync", "pms"})
        w2 = _make_worker({"pricing", "engine"})
        f1 = _make_insight(summary="sync failure")
        f2 = _make_insight(summary="pricing bug")

        routes = [
            FindingRoute(finding=f1, worker_id=w1.id, relevance=0.5),
            FindingRoute(finding=f2, worker_id=w1.id, relevance=0.3),
            FindingRoute(finding=f2, worker_id=w2.id, relevance=0.6),
        ]

        payloads = router.build_payloads(routes, [w1, w2])
        assert len(payloads) == 2

        # w1 should have 2 findings
        w1_payload = [p for p in payloads if p.worker_id == w1.id][0]
        assert len(w1_payload.findings) == 2

        # w2 should have 1 finding
        w2_payload = [p for p in payloads if p.worker_id == w2.id][0]
        assert len(w2_payload.findings) == 1

    def test_payload_deduplicates_findings(self):
        router = FindingRouter()

        w1 = _make_worker({"sync"})
        f1 = _make_insight(summary="same finding")

        # Same finding routed twice to same worker
        routes = [
            FindingRoute(finding=f1, worker_id=w1.id, relevance=0.5),
            FindingRoute(finding=f1, worker_id=w1.id, relevance=0.6),
        ]

        payloads = router.build_payloads(routes, [w1])
        assert len(payloads) == 1
        assert len(payloads[0].findings) == 1

    def test_payload_captures_worker_terms(self):
        router = FindingRouter()

        terms = {"sync", "pms", "hostaway", "booking", "calendar"}
        w1 = _make_worker(terms, signal_count=100)
        f1 = _make_insight()

        routes = [FindingRoute(finding=f1, worker_id=w1.id, relevance=0.5)]
        payloads = router.build_payloads(routes, [w1])

        assert payloads[0].worker_signal_count == 100
        assert set(payloads[0].worker_terms) == terms


# ── CorroborationEngine tests ────────────────────────────────


class TestCorroborationEngine:
    def test_mechanical_corroboration_high_relevance(self):
        engine = CorroborationEngine()
        worker = _make_worker({"sync", "pms"})

        payload = CorroborationPayload(
            worker_id=worker.id,
            worker_terms=["sync", "pms"],
            worker_signal_count=50,
            findings=[_make_insight()],
            relevance_scores={_make_insight().summary[:64]: 0.15},
        )
        # Rebuild with correct key
        f = _make_insight()
        payload.findings = [f]
        payload.relevance_scores = {f.summary[:64]: 0.15}

        result = engine._corroborate_mechanical(payload, worker)

        assert not result.llm_used
        assert len(result.verdicts) == 1
        assert result.verdicts[0].verdict == "corroborate"
        assert result.verdicts[0].confidence > 0.3

    def test_mechanical_corroboration_low_relevance(self):
        engine = CorroborationEngine()
        worker = _make_worker({"unrelated", "terms"})

        f = _make_insight()
        payload = CorroborationPayload(
            worker_id=worker.id,
            worker_terms=["unrelated"],
            worker_signal_count=10,
            findings=[f],
            relevance_scores={f.summary[:64]: 0.01},
        )

        result = engine._corroborate_mechanical(payload, worker)

        assert result.verdicts[0].verdict == "no_evidence"

    def test_mechanical_corroboration_medium_relevance(self):
        engine = CorroborationEngine()
        worker = _make_worker({"sync"})

        f = _make_insight()
        payload = CorroborationPayload(
            worker_id=worker.id,
            worker_terms=["sync"],
            worker_signal_count=30,
            findings=[f],
            relevance_scores={f.summary[:64]: 0.05},
        )

        result = engine._corroborate_mechanical(payload, worker)

        assert result.verdicts[0].verdict == "partial"

    @pytest.mark.asyncio
    async def test_corroborate_with_stub_llm(self):
        engine = CorroborationEngine()
        llm = StubLLMService()
        worker = _make_worker({"sync", "pms"})

        f = _make_insight()
        payload = CorroborationPayload(
            worker_id=worker.id,
            worker_terms=["sync", "pms"],
            worker_signal_count=50,
            findings=[f],
            relevance_scores={f.summary[:64]: 0.15},
        )

        results = await engine.corroborate([payload], [worker], llm=llm)

        assert len(results) == 1
        # Stub LLM returns default CorroborationBatch (empty assessments)
        # so it should fall through or produce stub verdicts
        assert isinstance(results[0], CorroborationResult)

    @pytest.mark.asyncio
    async def test_corroborate_falls_back_to_mechanical(self):
        engine = CorroborationEngine()
        worker = _make_worker({"sync", "pms"})

        f = _make_insight()
        payload = CorroborationPayload(
            worker_id=worker.id,
            worker_terms=["sync", "pms"],
            worker_signal_count=50,
            findings=[f],
            relevance_scores={f.summary[:64]: 0.15},
        )

        # No LLM provided
        results = await engine.corroborate([payload], [worker], llm=None)

        assert len(results) == 1
        assert not results[0].llm_used
        assert len(results[0].verdicts) == 1


# ── Quorum aggregation tests ────────────────────────────────


class TestQuorumAggregation:
    def test_quorum_met_with_enough_corroborations(self):
        engine = CorroborationEngine(quorum_size=2, quorum_threshold=0.5)
        f = _make_insight()

        w1_id, w2_id, w3_id = uuid4(), uuid4(), uuid4()
        results = [
            CorroborationResult(
                worker_id=w1_id,
                verdicts=[WorkerVerdict(
                    worker_id=w1_id, finding_summary=f.summary[:200],
                    verdict="corroborate", confidence=0.8,
                    reasoning="Seen 3 sync failures", related_context="",
                )],
                llm_used=True,
            ),
            CorroborationResult(
                worker_id=w2_id,
                verdicts=[WorkerVerdict(
                    worker_id=w2_id, finding_summary=f.summary[:200],
                    verdict="corroborate", confidence=0.7,
                    reasoning="Hostaway API errors", related_context="",
                )],
                llm_used=True,
            ),
            CorroborationResult(
                worker_id=w3_id,
                verdicts=[WorkerVerdict(
                    worker_id=w3_id, finding_summary=f.summary[:200],
                    verdict="no_evidence", confidence=0.3,
                    reasoning="No related context", related_context="",
                )],
                llm_used=True,
            ),
        ]

        quorum = engine.aggregate_quorum([f], results)

        assert len(quorum) == 1
        qf = quorum[0]
        assert qf.quorum_met is True
        assert qf.corroboration_count == 2
        assert qf.total_consulted == 3
        assert qf.quorum_confidence >= 0.5

    def test_quorum_not_met_insufficient_corroborations(self):
        engine = CorroborationEngine(quorum_size=3, quorum_threshold=0.5)
        f = _make_insight()

        w1_id, w2_id = uuid4(), uuid4()
        results = [
            CorroborationResult(
                worker_id=w1_id,
                verdicts=[WorkerVerdict(
                    worker_id=w1_id, finding_summary=f.summary[:200],
                    verdict="corroborate", confidence=0.8,
                    reasoning="Seen sync failures", related_context="",
                )],
                llm_used=True,
            ),
            CorroborationResult(
                worker_id=w2_id,
                verdicts=[WorkerVerdict(
                    worker_id=w2_id, finding_summary=f.summary[:200],
                    verdict="no_evidence", confidence=0.3,
                    reasoning="No context", related_context="",
                )],
                llm_used=True,
            ),
        ]

        quorum = engine.aggregate_quorum([f], results)

        assert len(quorum) == 1
        assert quorum[0].quorum_met is False
        assert quorum[0].corroboration_count == 1

    def test_quorum_not_met_low_confidence(self):
        engine = CorroborationEngine(quorum_size=2, quorum_threshold=0.8)
        f = _make_insight()

        w1_id, w2_id = uuid4(), uuid4()
        results = [
            CorroborationResult(
                worker_id=w1_id,
                verdicts=[WorkerVerdict(
                    worker_id=w1_id, finding_summary=f.summary[:200],
                    verdict="corroborate", confidence=0.4,
                    reasoning="Weak evidence", related_context="",
                )],
                llm_used=True,
            ),
            CorroborationResult(
                worker_id=w2_id,
                verdicts=[WorkerVerdict(
                    worker_id=w2_id, finding_summary=f.summary[:200],
                    verdict="corroborate", confidence=0.3,
                    reasoning="Very weak", related_context="",
                )],
                llm_used=True,
            ),
        ]

        quorum = engine.aggregate_quorum([f], results)

        assert len(quorum) == 1
        # 2 corroborations but low confidence
        assert quorum[0].corroboration_count == 2
        assert quorum[0].quorum_confidence < 0.8
        assert quorum[0].quorum_met is False

    def test_contradiction_counted(self):
        engine = CorroborationEngine(quorum_size=2)
        f = _make_insight()

        w1_id, w2_id = uuid4(), uuid4()
        results = [
            CorroborationResult(
                worker_id=w1_id,
                verdicts=[WorkerVerdict(
                    worker_id=w1_id, finding_summary=f.summary[:200],
                    verdict="corroborate", confidence=0.7,
                    reasoning="Supports finding", related_context="",
                )],
                llm_used=True,
            ),
            CorroborationResult(
                worker_id=w2_id,
                verdicts=[WorkerVerdict(
                    worker_id=w2_id, finding_summary=f.summary[:200],
                    verdict="contradict", confidence=0.8,
                    reasoning="My context shows the opposite", related_context="",
                )],
                llm_used=True,
            ),
        ]

        quorum = engine.aggregate_quorum([f], results)

        assert quorum[0].contradiction_count == 1
        assert quorum[0].corroboration_count == 1

    def test_partial_verdicts_contribute_at_half_weight(self):
        engine = CorroborationEngine(quorum_size=2, quorum_threshold=0.3)
        f = _make_insight()

        w1_id, w2_id = uuid4(), uuid4()
        results = [
            CorroborationResult(
                worker_id=w1_id,
                verdicts=[WorkerVerdict(
                    worker_id=w1_id, finding_summary=f.summary[:200],
                    verdict="partial", confidence=0.6,
                    reasoning="Some evidence", related_context="",
                )],
                llm_used=True,
            ),
            CorroborationResult(
                worker_id=w2_id,
                verdicts=[WorkerVerdict(
                    worker_id=w2_id, finding_summary=f.summary[:200],
                    verdict="corroborate", confidence=0.8,
                    reasoning="Strong evidence", related_context="",
                )],
                llm_used=True,
            ),
        ]

        quorum = engine.aggregate_quorum([f], results)

        qf = quorum[0]
        assert qf.corroboration_count == 2  # partial counts
        assert qf.quorum_met is True
        # Confidence should reflect half-weight for partial
        # (0.6 * 0.5 + 0.8 * 1.0) / 2 = 0.55
        assert 0.4 < qf.quorum_confidence < 0.8

    def test_multiple_findings_tracked_independently(self):
        engine = CorroborationEngine(quorum_size=2)

        f1 = _make_insight(summary="PMS sync failures")
        f2 = _make_insight(summary="Pricing calculation errors")

        w1_id = uuid4()
        results = [
            CorroborationResult(
                worker_id=w1_id,
                verdicts=[
                    WorkerVerdict(
                        worker_id=w1_id, finding_summary=f1.summary[:200],
                        verdict="corroborate", confidence=0.8,
                        reasoning="Seen sync issues", related_context="",
                    ),
                    WorkerVerdict(
                        worker_id=w1_id, finding_summary=f2.summary[:200],
                        verdict="no_evidence", confidence=0.2,
                        reasoning="No pricing context", related_context="",
                    ),
                ],
                llm_used=True,
            ),
        ]

        quorum = engine.aggregate_quorum([f1, f2], results)

        assert len(quorum) == 2
        f1_q = [q for q in quorum if q.original.summary == f1.summary][0]
        f2_q = [q for q in quorum if q.original.summary == f2.summary][0]
        assert f1_q.corroboration_count == 1
        assert f2_q.corroboration_count == 0

    def test_no_findings_no_quorum(self):
        engine = CorroborationEngine()
        quorum = engine.aggregate_quorum([], [])
        assert quorum == []


# ── End-to-end orchestrator tests ────────────────────────────


class TestRunCorroboration:
    @pytest.mark.asyncio
    async def test_full_pipeline_mechanical(self):
        """End-to-end: findings → route → corroborate → quorum (no LLM)."""
        findings = [
            _make_insight(summary="PMS sync failures in hostaway booking pipeline"),
            _make_insight(summary="Pricing engine tax calculation regression"),
        ]
        workers = [
            _make_worker({"sync", "pms", "hostaway", "booking", "pipeline"}, signal_count=100),
            _make_worker({"pricing", "engine", "tax", "calculation", "rate"}, signal_count=80),
            _make_worker({"frontend", "react", "component", "layout", "css"}, signal_count=60),
        ]

        quorum_findings, metrics = await run_corroboration(
            findings, workers,
            llm=None,
            min_workers=2,
            max_workers=3,
            quorum_size=2,
        )

        assert metrics["findings_input"] == 2
        assert metrics["workers_consulted"] >= 1
        assert metrics["llm_calls"] == 0
        assert metrics["mechanical_calls"] >= 1
        assert len(quorum_findings) == 2

    @pytest.mark.asyncio
    async def test_empty_findings_returns_empty(self):
        quorum, metrics = await run_corroboration([], [_make_worker()])
        assert quorum == []
        assert metrics["findings_input"] == 0

    @pytest.mark.asyncio
    async def test_empty_workers_returns_empty(self):
        quorum, metrics = await run_corroboration([_make_insight()], [])
        assert quorum == []
        assert metrics["workers_consulted"] == 0

    @pytest.mark.asyncio
    async def test_metrics_accuracy(self):
        findings = [_make_insight(summary="sync failure pms hostaway")]
        workers = [
            _make_worker({"sync", "pms", "hostaway"}),
            _make_worker({"pricing", "engine"}),
        ]

        _, metrics = await run_corroboration(
            findings, workers,
            llm=None,
            min_workers=2,
        )

        assert metrics["findings_input"] == 1
        assert metrics["routes_created"] >= 2
        assert metrics["quorum_size"] == 2  # default
        assert "quorum_met" in metrics

    @pytest.mark.asyncio
    async def test_single_worker_no_quorum(self):
        """With only 1 worker, quorum of 2 is impossible."""
        findings = [_make_insight(summary="sync failure pms hostaway")]
        workers = [_make_worker({"sync", "pms", "hostaway"})]

        quorum_findings, metrics = await run_corroboration(
            findings, workers,
            llm=None,
            min_workers=1,
            max_workers=1,
            quorum_size=2,
        )

        assert metrics["workers_consulted"] == 1
        if quorum_findings:
            assert quorum_findings[0].quorum_met is False


# ── Edge cases ───────────────────────────────────────────────


class TestEdgeCases:
    def test_finding_with_no_text_terms(self):
        """A finding with only short words should still route (via actors)."""
        router = FindingRouter(min_workers=1)
        findings = [_make_insight(
            summary="A B C",
            actors=["@hostaway", "@quentin"],
        )]
        workers = [_make_worker({"hostaway", "quentin", "sync"})]

        routes = router.route(findings, workers)
        # Actors should provide routing terms
        assert len(routes) >= 1

    def test_worker_with_no_terms_skipped(self):
        """Workers with empty term sets shouldn't crash."""
        router = FindingRouter(min_workers=0)
        findings = [_make_insight()]
        workers = [_make_worker(set(), signal_count=0)]

        routes = router.route(findings, workers)
        # Worker has no terms, so no overlap possible
        assert len(routes) == 0

    def test_unanimous_no_evidence(self):
        """All workers say no_evidence — quorum should not be met."""
        engine = CorroborationEngine(quorum_size=2)
        f = _make_insight()

        results = []
        for _ in range(3):
            wid = uuid4()
            results.append(CorroborationResult(
                worker_id=wid,
                verdicts=[WorkerVerdict(
                    worker_id=wid, finding_summary=f.summary[:200],
                    verdict="no_evidence", confidence=0.2,
                    reasoning="Nothing relevant", related_context="",
                )],
                llm_used=False,
            ))

        quorum = engine.aggregate_quorum([f], results)
        assert quorum[0].quorum_met is False
        assert quorum[0].corroboration_count == 0

    def test_unanimous_contradiction(self):
        """All workers contradict — quorum not met, contradiction counted."""
        engine = CorroborationEngine(quorum_size=2)
        f = _make_insight()

        results = []
        for _ in range(3):
            wid = uuid4()
            results.append(CorroborationResult(
                worker_id=wid,
                verdicts=[WorkerVerdict(
                    worker_id=wid, finding_summary=f.summary[:200],
                    verdict="contradict", confidence=0.8,
                    reasoning="Context shows opposite", related_context="",
                )],
                llm_used=False,
            ))

        quorum = engine.aggregate_quorum([f], results)
        assert quorum[0].quorum_met is False
        assert quorum[0].contradiction_count == 3
        assert quorum[0].corroboration_count == 0
