"""Phase 2: Cross-signal finding corroboration.

Findings from Phase 1 are routed back through the mesh as stimuli.
Each worker evaluates findings against its accumulated context — the
same ART mechanism used for signals, but applied to findings.

Architecture:
  1. FindingRouter: local classifier (term overlap / familiarity) routes
     each finding to the K workers whose context is most relevant.
     No LLM needed — reuses the same ART match function.
  2. CorroborationPayload: bundles all findings for a given worker into
     a single LLM call. Cost model: 1 call per worker, not per finding.
  3. QuorumAggregator: findings corroborated by N workers (quorum)
     become cross-signal findings with higher confidence.

The workers don't match the org chart or teams. They are positioned
in n-dimensional topic space and assess findings against accumulated
context. A finding about PMS sync goes to the PMS worker AND the
infrastructure worker AND the integrations worker — multiple contexts
evaluate it independently.

Cost: ~1% overhead (W additional LLM calls where W = worker count,
vs ~700 Phase 1 calls for a typical run).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from stigmergy.core.familiarity import _keyword_overlap

if TYPE_CHECKING:
    from stigmergy.mesh.insights import Insight
    from stigmergy.mesh.worker import WorkerNode
    from stigmergy.services.llm import LLMService

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────


@dataclass
class FindingRoute:
    """A finding routed to a specific worker with relevance score."""

    finding: Insight
    worker_id: UUID
    relevance: float  # term-overlap score [0, 1]


@dataclass
class CorroborationPayload:
    """Batched findings for a single worker's LLM call."""

    worker_id: UUID
    worker_terms: list[str]  # top terms for context
    worker_signal_count: int
    findings: list[Insight]
    relevance_scores: dict[str, float]  # finding_summary_hash -> relevance


@dataclass
class WorkerVerdict:
    """A single worker's assessment of a single finding."""

    worker_id: UUID
    finding_summary: str
    verdict: str  # "corroborate", "contradict", "no_evidence", "partial"
    confidence: float
    reasoning: str
    related_context: str  # what the worker knows that's relevant


@dataclass
class CorroborationResult:
    """All verdicts from a single worker on its payload of findings."""

    worker_id: UUID
    verdicts: list[WorkerVerdict]
    llm_used: bool  # True if LLM called, False if mechanical


@dataclass
class QuorumFinding:
    """A finding that achieved quorum — corroborated by multiple workers."""

    original: Insight
    verdicts: list[WorkerVerdict]  # all worker verdicts for this finding
    corroboration_count: int  # how many workers corroborated
    contradiction_count: int
    total_consulted: int
    quorum_confidence: float  # aggregated confidence
    quorum_met: bool


# ── Finding Router ───────────────────────────────────────────


class FindingRouter:
    """Routes findings to relevant workers using local term overlap.

    Same ART principle: |I ∩ w_J| / |I| where I is the finding's terms
    and w_J is the worker's accumulated vocabulary. No LLM needed.

    A finding is routed to a worker if term overlap exceeds the threshold.
    Each finding goes to at least min_workers (if enough workers exist)
    and at most max_workers.
    """

    def __init__(
        self,
        *,
        relevance_threshold: float = 0.02,
        min_workers: int = 2,
        max_workers: int = 5,
    ) -> None:
        self._relevance_threshold = relevance_threshold
        self._min_workers = min_workers
        self._max_workers = max_workers

    def route(
        self,
        findings: list[Insight],
        workers: list[WorkerNode],
    ) -> list[FindingRoute]:
        """Route findings to relevant workers.

        Returns a list of FindingRoute objects, one per (finding, worker) pair.
        """
        if not findings or not workers:
            return []

        routes: list[FindingRoute] = []

        for finding in findings:
            finding_terms = self._extract_terms(finding)
            if not finding_terms:
                continue

            # Score against each worker's vocabulary
            scored: list[tuple[WorkerNode, float]] = []
            for worker in workers:
                worker_terms = worker.context.terms
                if not worker_terms:
                    continue
                overlap = _keyword_overlap(finding_terms, worker_terms)
                scored.append((worker, overlap))

            # Sort by overlap descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Take top workers above threshold, respecting min/max
            selected: list[tuple[WorkerNode, float]] = []
            for worker, score in scored:
                if len(selected) >= self._max_workers:
                    break
                if score >= self._relevance_threshold or len(selected) < self._min_workers:
                    selected.append((worker, score))

            for worker, score in selected:
                routes.append(FindingRoute(
                    finding=finding,
                    worker_id=worker.id,
                    relevance=score,
                ))

        return routes

    def build_payloads(
        self,
        routes: list[FindingRoute],
        workers: list[WorkerNode],
    ) -> list[CorroborationPayload]:
        """Bundle routes into per-worker payloads for batched LLM calls."""
        worker_map = {w.id: w for w in workers}
        payloads: dict[UUID, CorroborationPayload] = {}
        seen_summaries: dict[UUID, set[str]] = {}  # per-worker dedup set

        for route in routes:
            wid = route.worker_id
            if wid not in payloads:
                worker = worker_map.get(wid)
                if worker is None:
                    continue
                all_terms = worker.context.terms
                top_terms = sorted(all_terms)[:30] if all_terms else []
                payloads[wid] = CorroborationPayload(
                    worker_id=wid,
                    worker_terms=top_terms,
                    worker_signal_count=worker.context.signal_count,
                    findings=[],
                    relevance_scores={},
                )
                seen_summaries[wid] = set()

            payload = payloads[wid]
            # Deduplicate findings within a payload — O(1) set lookup
            summary = route.finding.summary
            if summary not in seen_summaries[wid]:
                seen_summaries[wid].add(summary)
                payload.findings.append(route.finding)
                payload.relevance_scores[summary[:64]] = route.relevance

        return list(payloads.values())

    def _extract_terms(self, finding: Insight) -> set[str]:
        """Extract terms from a finding for routing."""
        text = finding.summary.lower()
        # Extract actors as terms too
        actors = finding.details.get("actors", [])
        actor_terms = {a.lower().strip("@") for a in actors if a}

        # Basic term extraction: split on non-alpha, filter short words
        words = set()
        for word in text.split():
            cleaned = word.strip(".,;:!?()[]{}\"'`-–—/\\@#")
            if len(cleaned) >= 3 and cleaned.isalpha():
                words.add(cleaned)

        return words | actor_terms


# ── Corroboration Engine ─────────────────────────────────────


class CorroborationEngine:
    """Runs Phase 2: finding corroboration across workers.

    For each payload, sends the worker's accumulated context plus
    the findings to the LLM for assessment. Falls back to mechanical
    verdict when no LLM is available.
    """

    def __init__(
        self,
        *,
        quorum_size: int = 2,
        quorum_threshold: float = 0.6,
    ) -> None:
        self._quorum_size = quorum_size
        self._quorum_threshold = quorum_threshold

    async def corroborate(
        self,
        payloads: list[CorroborationPayload],
        workers: list[WorkerNode],
        llm: LLMService | None = None,
        comm_graph: Any = None,
    ) -> list[CorroborationResult]:
        """Execute corroboration across all workers.

        One LLM call per payload (per worker). Returns results for all workers.
        """
        worker_map = {w.id: w for w in workers}
        results: list[CorroborationResult] = []

        for payload in payloads:
            worker = worker_map.get(payload.worker_id)
            if worker is None:
                continue

            if llm is not None:
                try:
                    result = await self._corroborate_agent(payload, worker, llm)
                    results.append(result)
                    continue
                except Exception as exc:
                    logger.warning(
                        "Corroboration LLM failed for worker %s: %s",
                        str(payload.worker_id)[:8], exc,
                    )

            # Mechanical fallback
            result = self._corroborate_mechanical(payload, worker)
            results.append(result)

        return results

    async def _corroborate_agent(
        self,
        payload: CorroborationPayload,
        worker: WorkerNode,
        llm: LLMService,
    ) -> CorroborationResult:
        """LLM-powered corroboration: worker assesses findings against context."""
        from stigmergy.primitives.schemas import CorroborationBatch
        from stigmergy.services.agent_prompts import (
            CORROBORATOR_SYSTEM,
            corroborator_prompt,
        )

        prompt = corroborator_prompt(
            worker_terms=payload.worker_terms,
            worker_signal_count=payload.worker_signal_count,
            findings=[
                {
                    "type": f.type,
                    "summary": f.summary,
                    "actors": f.details.get("actors", []),
                    "confidence": f.confidence,
                    "unknowns": f.details.get("unknowns", ""),
                }
                for f in payload.findings
            ],
        )

        batch = await llm.extract(
            CorroborationBatch,
            prompt,
            CORROBORATOR_SYSTEM,
            max_tokens=600,
        )

        verdicts: list[WorkerVerdict] = []
        for assessment in batch.assessments:
            # Match assessment back to finding by index
            finding_idx = min(assessment.finding_index, len(payload.findings) - 1)
            finding = payload.findings[max(0, finding_idx)]
            verdicts.append(WorkerVerdict(
                worker_id=payload.worker_id,
                finding_summary=finding.summary[:200],
                verdict=assessment.verdict,
                confidence=assessment.confidence,
                reasoning=assessment.reasoning,
                related_context=assessment.related_context,
            ))

        return CorroborationResult(
            worker_id=payload.worker_id,
            verdicts=verdicts,
            llm_used=True,
        )

    def _corroborate_mechanical(
        self,
        payload: CorroborationPayload,
        worker: WorkerNode,
    ) -> CorroborationResult:
        """Mechanical fallback: term-overlap based verdict."""
        verdicts: list[WorkerVerdict] = []

        for finding in payload.findings:
            relevance = payload.relevance_scores.get(finding.summary[:64], 0.0)

            # Higher relevance = more likely to corroborate
            if relevance >= 0.1:
                verdict = "corroborate"
                confidence = min(0.7, 0.3 + relevance * 2)
            elif relevance >= 0.03:
                verdict = "partial"
                confidence = 0.3
            else:
                verdict = "no_evidence"
                confidence = 0.2

            verdicts.append(WorkerVerdict(
                worker_id=payload.worker_id,
                finding_summary=finding.summary[:200],
                verdict=verdict,
                confidence=confidence,
                reasoning=f"mechanical: term_overlap={relevance:.3f}",
                related_context=f"worker has {worker.context.signal_count} signals, "
                                f"{len(worker.context.terms)} terms",
            ))

        return CorroborationResult(
            worker_id=payload.worker_id,
            verdicts=verdicts,
            llm_used=False,
        )

    def aggregate_quorum(
        self,
        findings: list[Insight],
        results: list[CorroborationResult],
    ) -> list[QuorumFinding]:
        """Aggregate worker verdicts into quorum findings.

        A finding achieves quorum when >= quorum_size workers corroborate it
        with aggregate confidence >= quorum_threshold.
        """
        # Collect all verdicts per finding (by summary prefix match)
        finding_verdicts: dict[str, list[WorkerVerdict]] = {}
        finding_map: dict[str, Insight] = {}

        for finding in findings:
            key = finding.summary[:200]
            finding_verdicts[key] = []
            finding_map[key] = finding

        for result in results:
            for verdict in result.verdicts:
                key = verdict.finding_summary
                if key in finding_verdicts:
                    finding_verdicts[key].append(verdict)

        quorum_findings: list[QuorumFinding] = []

        for key, verdicts in finding_verdicts.items():
            if key not in finding_map:
                continue

            original = finding_map[key]
            corroborate_count = sum(
                1 for v in verdicts if v.verdict in ("corroborate", "partial")
            )
            contradict_count = sum(
                1 for v in verdicts if v.verdict == "contradict"
            )
            total = len(verdicts)

            # Quorum confidence: weighted average of corroborating verdicts
            corroborating = [v for v in verdicts if v.verdict in ("corroborate", "partial")]
            if corroborating:
                # Partial verdicts contribute at half weight
                weighted_conf = sum(
                    v.confidence * (1.0 if v.verdict == "corroborate" else 0.5)
                    for v in corroborating
                )
                quorum_confidence = weighted_conf / len(corroborating)
            else:
                quorum_confidence = 0.0

            quorum_met = (
                corroborate_count >= self._quorum_size
                and quorum_confidence >= self._quorum_threshold
            )

            quorum_findings.append(QuorumFinding(
                original=original,
                verdicts=verdicts,
                corroboration_count=corroborate_count,
                contradiction_count=contradict_count,
                total_consulted=total,
                quorum_confidence=quorum_confidence,
                quorum_met=quorum_met,
            ))

        return quorum_findings


# ── Top-level orchestrator ───────────────────────────────────


async def run_corroboration(
    findings: list[Insight],
    workers: list[WorkerNode],
    *,
    llm: LLMService | None = None,
    comm_graph: Any = None,
    relevance_threshold: float = 0.02,
    min_workers: int = 2,
    max_workers: int = 5,
    quorum_size: int = 2,
    quorum_threshold: float = 0.6,
) -> tuple[list[QuorumFinding], dict[str, Any]]:
    """Run the full Phase 2 corroboration pipeline.

    Returns:
        (quorum_findings, metrics) where metrics tracks routing/LLM stats.
    """
    if not findings or not workers:
        return [], {"findings_input": 0, "workers_consulted": 0}

    router = FindingRouter(
        relevance_threshold=relevance_threshold,
        min_workers=min_workers,
        max_workers=max_workers,
    )
    engine = CorroborationEngine(
        quorum_size=quorum_size,
        quorum_threshold=quorum_threshold,
    )

    # Step 1: Route findings to workers (local, no LLM)
    routes = router.route(findings, workers)

    # Step 2: Bundle into per-worker payloads
    payloads = router.build_payloads(routes, workers)

    # Step 3: Execute corroboration (1 LLM call per worker)
    results = await engine.corroborate(payloads, workers, llm=llm, comm_graph=comm_graph)

    # Step 4: Aggregate into quorum findings
    quorum_findings = engine.aggregate_quorum(findings, results)

    # Metrics
    llm_calls = sum(1 for r in results if r.llm_used)
    mechanical_calls = sum(1 for r in results if not r.llm_used)
    quorum_met_count = sum(1 for q in quorum_findings if q.quorum_met)

    metrics = {
        "findings_input": len(findings),
        "routes_created": len(routes),
        "workers_consulted": len(payloads),
        "llm_calls": llm_calls,
        "mechanical_calls": mechanical_calls,
        "quorum_findings": len(quorum_findings),
        "quorum_met": quorum_met_count,
        "quorum_size": quorum_size,
        "quorum_threshold": quorum_threshold,
    }

    logger.info(
        "Phase 2 corroboration: %d findings → %d routes → %d workers → "
        "%d quorum (%d met threshold)",
        len(findings), len(routes), len(payloads),
        len(quorum_findings), quorum_met_count,
    )

    return quorum_findings, metrics


# ── Competency Reinforcement ─────────────────────────────────
#
# Maps corroboration verdicts to competency deltas, closing the
# open loop: agents learn from whether their findings survive
# independent assessment.

# Verdict → base delta (asymmetric: false findings cost more)
_VERDICT_DELTAS: dict[str, float] = {
    "corroborate": 0.02,
    "partial": 0.005,
    "contradict": -0.03,
    "no_evidence": 0.0,
}

# Finding classification → competency name
_CLASSIFICATION_COMPETENCY: dict[str, str] = {
    "correlation": "cross_context_propagation",
    "discovery": "cross_context_propagation",
    "amplification": "temporal_analysis",
    "trend": "temporal_analysis",
    "risk": "anomaly_detection",
    "overlap": "similarity_detection",
}


def reinforce_competencies(
    quorum_findings: list[QuorumFinding],
    agents: list,
) -> dict[str, float]:
    """Apply competency deltas based on corroboration outcomes.

    Maps each finding's classification to a competency name, then
    applies verdict-weighted deltas via CompetencyModel.reinforce().

    Effective learning rate: ~0.006-0.027 per finding (base delta *
    verdict confidence). From seed 0.5 → takes ~20 corroborated
    findings to reach 0.9. Gradual, not sudden.

    Returns a summary of total deltas applied per competency.
    """
    if not quorum_findings or not agents:
        return {}

    agent_map = {a.id: a for a in agents}
    totals: dict[str, float] = {}

    for qf in quorum_findings:
        # Determine which competency this finding exercises
        classification = qf.original.details.get("classification", "")
        pattern_type = qf.original.type
        competency = _CLASSIFICATION_COMPETENCY.get(
            classification,
            _CLASSIFICATION_COMPETENCY.get(pattern_type, ""),
        )
        if not competency:
            continue

        # Find the agent that produced this finding
        agent = agent_map.get(qf.original.agent_id)
        if agent is None:
            continue

        # Aggregate verdict signal across all consulting workers
        for verdict in qf.verdicts:
            base_delta = _VERDICT_DELTAS.get(verdict.verdict, 0.0)
            if base_delta == 0.0:
                continue
            # Scale by verdict confidence
            effective_delta = base_delta * verdict.confidence
            agent.competencies.reinforce(competency, effective_delta)
            totals[competency] = totals.get(competency, 0.0) + effective_delta

    if totals:
        logger.info(
            "Competency reinforcement: %s",
            ", ".join(f"{k}={v:+.4f}" for k, v in totals.items()),
        )

    return totals
