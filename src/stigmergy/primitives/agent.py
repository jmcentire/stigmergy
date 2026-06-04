"""Agent: THE one class. There is no type hierarchy.

An agent bound to "a leader's attention" and an agent bound to "TICKET-7372"
and an agent bound to "output compliance" are instantiated from this same
class with different configurations. They differ in context bindings,
learned competencies, and constraints — not in kind.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from stigmergy.primitives.assessment import Assessment, AssessmentAction
from stigmergy.primitives.signal import Signal

if TYPE_CHECKING:
    from stigmergy.mesh.insights import Insight, ObservationContext

logger = logging.getLogger(__name__)


def _should_drop_finding(ci) -> str | None:
    """Post-LLM quality filter for correlator findings.

    Returns a reason string if the finding should be dropped, None if it passes.
    Programmatic enforcement of correlator prompt instructions — LLMs don't
    always follow instructions, so this is the guardrail.
    """
    # Drop retrieval: re-describing a known ticket is not a finding
    if ci.classification == "retrieval":
        return "retrieval"

    # Drop historical/stale: resolved items aren't active concerns
    if ci.temporal_relevance in ("historical", "stale"):
        return f"temporal:{ci.temporal_relevance}"

    # Drop single-actor "overlap"/"coordination": one person working
    # across repos already knows what they're doing
    unique_actors = set(
        a.lstrip("@").lower() for a in ci.actors
    ) if ci.actors else set()
    if (
        len(unique_actors) <= 1
        and ci.pattern_type in ("overlap", "coordination", "duplicate_work")
    ):
        return "single_actor_overlap"

    return None


class RuleSet:
    """Constraints governing agent behavior.

    Rules are data, not code. An agent bound to a compliance context
    has rules about what content to kill. An agent bound to a knowledge
    context has rules about relevance thresholds.
    """

    def __init__(self, rules: dict[str, Any] | None = None) -> None:
        self.rules: dict[str, Any] = rules or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.rules.get(key, default)


class CompetencyModel:
    """Learned competency weights. Not typed — emergent from signal processing.

    Starts with seed weights. Evolves based on feedback from consensus
    outcomes and downstream signals. Competency names are self-describing
    strings — agents can develop new competencies beyond the seeds.

    Each competency maps to observation primitives via the Agent's
    _COMPETENCY_PRIMITIVES registry. Unknown competencies get baseline
    access to all primitives, not zero access.
    """

    def __init__(self, seeds: dict[str, float] | None = None) -> None:
        self.weights: dict[str, float] = seeds or {
            "similarity_detection": 0.5,
            "temporal_analysis": 0.5,
            "cross_context_propagation": 0.5,
        }

    def strength(self, competency: str) -> float:
        return self.weights.get(competency, 0.0)

    def reinforce(self, competency: str, delta: float) -> None:
        current = self.weights.get(competency, 0.0)
        self.weights[competency] = max(0.0, min(1.0, current + delta))


class Agent:
    """The single agent class. All agents are instances of this.

    When an LLM service is provided, evaluate() and reflect() use agent
    intelligence via schema-constrained prompts. When no LLM is available
    (stub mode, budget exhausted), they fall back to the heuristic
    implementations below.

    The LLM methods are the "real" behavior. The heuristics are the
    mechanical fallback — they exist for testing and cost control, not
    as the intended architecture.
    """

    def __init__(
        self,
        *,
        id: UUID | None = None,
        contexts: dict[UUID, float] | None = None,
        competencies: CompetencyModel | None = None,
        confidence: float = 0.5,
        rules: RuleSet | None = None,
        weights: dict[str, float] | None = None,
        state: dict[str, Any] | None = None,
        llm: Any | None = None,  # LLMService — Any to avoid circular import
    ) -> None:
        self.id = id or uuid4()
        self.contexts: dict[UUID, float] = contexts or {}  # context_id -> affinity_weight
        self.competencies = competencies or CompetencyModel()
        self.confidence = confidence
        self.rules = rules or RuleSet()
        self.weights: dict[str, float] = weights or {}  # domain -> concern weight
        self.state: dict[str, Any] = state or {}
        self._llm = llm  # fast/cheap tier — surfacer (evaluate) + mechanical fallbacks
        # Quality tier for the batched correlator (reflect). When None, reflect
        # falls back to self._llm. Injected by run_cmd / propagated on fork.
        self._correlator_llm = None
        self._annotations = None  # AnnotationStore — injected after construction
        self._discovery_store = None  # DiscoveryStore — injected by run_cmd for mesh context
        self._compression_tracker = None  # CompressionTracker — injected by run_cmd
        self._reflect_counter: int = 0
        self._reflect_interval: int = 10  # correlator fires every N signals

        # Circuit breaker — injected by run_cmd, trips on consecutive LLM failures
        self._circuit_breaker = None

        # Correlator batch dedup — skip LLM call when signal batch is mostly the same
        self._last_batch_ids: frozenset[UUID] = frozenset()
        self._last_batch_insights: list = []  # cached Insight list from last correlator call
        self._batch_dedup_count: int = 0  # how many LLM calls we skipped via dedup
        self._batch_overlap_threshold: float = 0.8  # skip if overlap exceeds this

        # Intelligence monitoring — track LLM vs mechanical execution paths
        self._llm_reflect_count: int = 0
        self._mechanical_reflect_count: int = 0
        self._llm_reflect_skipped: int = 0  # non-batch signals (LLM present, not batch interval)
        self._llm_evaluate_count: int = 0
        self._mechanical_evaluate_count: int = 0

    def reset_batch_state(self) -> None:
        """Clear correlator batch cache to restore epistemic independence.

        Call at the start of each poll cycle so the correlator re-evaluates
        patterns from scratch rather than returning cached insights from
        the previous cycle. Without this, >80% batch overlap triggers
        cache return and the same findings re-emit as echoes.
        """
        self._last_batch_ids = frozenset()
        self._last_batch_insights = []

    def bind(self, context_id: UUID, affinity: float = 1.0) -> None:
        """Bind this agent to a context with given affinity weight."""
        self.contexts[context_id] = max(0.0, min(1.0, affinity))

    def unbind(self, context_id: UUID) -> None:
        """Release this agent from a context."""
        self.contexts.pop(context_id, None)

    def domain_weight(self, domain: str) -> float:
        """How much this agent cares about a given domain of concern."""
        return self.weights.get(domain, 0.0)

    async def evaluate(self, signal: Signal, context_id: UUID, domain: str) -> Assessment:
        """Evaluate a signal: surface, store, or ignore.

        With LLM: the surfacer agent reads the signal and decides based
        on its constitutional prompt — what's worth a human's attention.
        Without LLM: mechanical heuristic based on confidence * affinity.
        """
        # Agent-powered surfacing
        if self._llm is not None:
            # Backoff if we've had consecutive failures (rate limit recovery)
            if self._circuit_breaker is not None:
                await self._circuit_breaker.cooldown()
            try:
                from stigmergy.primitives.schemas import SurfaceDecision
                from stigmergy.services.agent_prompts import (
                    SURFACER_SYSTEM,
                    surfacer_prompt,
                )

                signal_summary = (
                    f"[{signal.source}/{signal.channel}] @{signal.author}: "
                    f"{signal.content[:500]}"
                )
                # Sense the pheromone trail: what have other agents
                # already noticed about this resource?
                prior_context = ""
                if self._annotations is not None:
                    prior_context = self._annotations.summary_for(
                        signal.source, signal.channel,
                    )
                decision = await self._llm.extract(
                    SurfaceDecision,
                    surfacer_prompt(
                        signal_summary=signal_summary,
                        worker_specialization=domain,
                        recent_context=prior_context,
                    ),
                    SURFACER_SYSTEM,
                    max_tokens=200,
                )
                action_map = {
                    "surface": AssessmentAction.SURFACE,
                    "store": AssessmentAction.STORE,
                    "ignore": AssessmentAction.IGNORE,
                }
                self._llm_evaluate_count += 1
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                return Assessment(
                    agent_id=self.id,
                    signal_id=signal.id,
                    context_id=context_id,
                    action=action_map.get(decision.action, decision.action),
                    confidence=decision.confidence,
                    domain=domain,
                    reasoning=decision.reasoning,
                )
            except Exception as exc:
                logger.warning("Surfacer LLM failed: %s", exc)
                logger.debug("Surfacer traceback", exc_info=True)
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()

        # Mechanical fallback
        self._mechanical_evaluate_count += 1
        affinity = self.contexts.get(context_id, 0.0)
        eval_confidence = self.confidence * affinity
        action = AssessmentAction.STORE
        if eval_confidence > 0.7 and self.domain_weight(domain) > 0.5:
            action = AssessmentAction.SURFACE
        elif eval_confidence < 0.2:
            action = AssessmentAction.IGNORE

        return Assessment(
            agent_id=self.id,
            signal_id=signal.id,
            context_id=context_id,
            action=action,
            confidence=eval_confidence,
            domain=domain,
            reasoning=f"affinity={affinity:.2f} confidence={self.confidence:.2f} domain_weight={self.domain_weight(domain):.2f}",
        )

    async def reflect(self, obs: ObservationContext) -> list[Insight]:
        """Find cross-signal patterns through this agent's lens.

        With LLM: the correlator agent reads a batch of recent signal
        summaries and reasons about patterns — overlap, contradiction,
        risk, trends. The prompt defines what to look for.
        Without LLM: mechanical observation primitives (similarity,
        temporal, cross-source).
        """
        from stigmergy.mesh.insights import Insight

        self._reflect_counter += 1

        # Agent-powered correlation — batched, not per-signal.
        # The correlator is expensive and reasons over the same cache
        # window, so running it every signal just repeats the same work.
        if self._llm is not None:
            if self._reflect_counter % self._reflect_interval == 0:
                # Backoff if we've had consecutive failures (rate limit recovery)
                if self._circuit_breaker is not None:
                    await self._circuit_breaker.cooldown()
                try:
                    result = await self._reflect_agent(obs)
                    # _llm_reflect_count is incremented inside _reflect_agent
                    # only on real LLM calls (not batch dedup cache hits)
                    if self._circuit_breaker is not None:
                        self._circuit_breaker.record_success()
                    return result
                except Exception as exc:
                    logger.warning("Correlator LLM failed: %s", exc)
                    logger.debug("Correlator traceback", exc_info=True)
                    if self._circuit_breaker is not None:
                        self._circuit_breaker.record_failure()
            # When LLM is available, skip mechanical on non-batch signals.
            # The correlator handles pattern detection; mechanical observations
            # (parallel_activity, cross_source_correlation) are noisy.
            self._llm_reflect_skipped += 1
            return []

        # Mechanical fallback — only when no LLM available
        self._mechanical_reflect_count += 1
        return self._reflect_mechanical(obs)

    async def _reflect_agent(self, obs: ObservationContext) -> list[Insight]:
        """Correlator agent: LLM reasons over recent signal batch."""
        from stigmergy.mesh.insights import Insight
        from stigmergy.primitives.schemas import CorrelationBatch
        from stigmergy.services.agent_prompts import (
            CORRELATOR_SYSTEM,
            correlator_prompt,
        )

        # Build signal summaries from two sources:
        # 1. Temporal window (7 days) — catches chronologically nearby signals
        # 2. Spectral neighbors — catches structurally similar signals regardless
        #    of time (the "Foo scenario": PR did Foo last week, ticket says do Foo)
        recent = obs.temporal_cluster(window_hours=168)  # 7 days
        spectral = obs.spectral_neighbors(threshold=0.6)

        # Merge and deduplicate by signal ID
        seen_ids: set = set()
        combined = []
        for cached in recent:
            if cached.signal.id not in seen_ids:
                seen_ids.add(cached.signal.id)
                combined.append(cached)
        for cached, _sim in spectral:
            if cached.signal.id not in seen_ids:
                seen_ids.add(cached.signal.id)
                combined.append(cached)

        if len(combined) < 2:
            return []

        # Batch dedup: if this batch overlaps >80% with the last one we sent
        # to the LLM, the correlator will rediscover the same patterns.
        # Return the cached insights instead of burning an LLM call.
        current_batch_ids = frozenset(seen_ids)
        if self._last_batch_ids:
            overlap = len(current_batch_ids & self._last_batch_ids)
            max_size = max(len(current_batch_ids), len(self._last_batch_ids))
            if max_size > 0 and overlap / max_size >= self._batch_overlap_threshold:
                self._batch_dedup_count += 1
                logger.debug(
                    "Correlator batch dedup: %d/%d signals overlap (%.0f%%), "
                    "returning %d cached insights",
                    overlap, max_size, overlap / max_size * 100,
                    len(self._last_batch_insights),
                )
                return self._last_batch_insights

        batch_signals = combined[:40]
        summaries = []
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        for cached in batch_signals:
            s = cached.signal
            meta = s.metadata or {}
            event_type = meta.get("event_type", "")
            status = meta.get("status") or meta.get("state") or ""
            priority = meta.get("priority", "")

            # Temporal currency: show how long ago state changes happened
            temporal_tag = ""
            state_upper = str(status).upper() if status else ""
            if state_upper in ("MERGED", "CLOSED"):
                # Try to compute "Xd ago" from merged_at or closed_at
                age_ts = meta.get("merged_at") or meta.get("closedAt") or meta.get("closed_at")
                if age_ts:
                    try:
                        from datetime import datetime as _dt, timezone as _tz
                        if isinstance(age_ts, str):
                            age_dt = _dt.fromisoformat(age_ts.replace("Z", "+00:00"))
                        else:
                            age_dt = age_ts
                        days_ago = (now - age_dt).total_seconds() / 86400
                        if days_ago < 1:
                            temporal_tag = f"{state_upper} <1d ago"
                        else:
                            temporal_tag = f"{state_upper} {int(days_ago)}d ago"
                    except (ValueError, TypeError):
                        temporal_tag = state_upper
                else:
                    temporal_tag = state_upper
            elif state_upper in ("OPEN", "IN PROGRESS", "IN_PROGRESS"):
                temporal_tag = "OPEN"
            elif state_upper in ("DONE", "RESOLVED", "COMPLETE", "COMPLETED"):
                temporal_tag = state_upper
            elif status:
                temporal_tag = str(status)

            # For Slack: show thread age if old
            if s.source and s.source.lower() == "slack":
                try:
                    thread_age = (now - s.timestamp).total_seconds() / 86400
                    if thread_age > 1:
                        temporal_tag = f"thread from {int(thread_age)}d ago"
                except (TypeError, AttributeError):
                    pass

            context_parts = []
            if event_type:
                context_parts.append(event_type)
            if temporal_tag:
                context_parts.append(temporal_tag)
            elif status:
                context_parts.append(str(status))
            if priority:
                context_parts.append(f"P{priority}")
            context_str = ", ".join(context_parts)
            if context_str:
                context_str = f"{context_str}, "
            summaries.append(
                f"[{s.source}/{s.channel}] @{s.author} "
                f"({context_str}{s.timestamp.strftime('%Y-%m-%d %H:%M')}): "
                f"{s.content[:500]}"
            )

        # Pheromone feedback: gather prior correlator annotations for
        # resources in the current batch so the LLM can avoid re-emitting
        # findings that are already known.
        prior_findings = ""
        if self._annotations is not None:
            seen_keys: set[str] = set()
            prior_lines: list[str] = []
            for cached in batch_signals:
                s = cached.signal
                from stigmergy.mesh.annotations import resource_key as _rkey
                rk = _rkey(s.source, s.channel)
                if rk in seen_keys:
                    continue
                seen_keys.add(rk)
                anns = self._annotations.get(s.source, s.channel)
                for ann in anns:
                    if ann.get("agent_type") == "correlator":
                        conf = ann.get("confidence", 0)
                        summary = ann.get("summary", "")
                        if summary:
                            prior_lines.append(
                                f"- [{s.source}/{s.channel}] (confidence={conf:.2f}): {summary}"
                            )
            if prior_lines:
                # Cap to avoid prompt bloat
                prior_findings = "\n".join(prior_lines[:20])

        # Build bridge distance context from communication graph
        graph_context = ""
        authors = sorted({c.signal.author for c in batch_signals if c.signal.author})
        if len(authors) >= 2:
            distances = []
            for i, a in enumerate(authors):
                for b in authors[i + 1:]:
                    d = obs.bridge_distance(a, b)
                    if d != float("inf") and d > 0:
                        distances.append((a, b, d))
            if distances:
                distances.sort(key=lambda x: x[2], reverse=True)
                lines = [f"  @{a} ↔ @{b}: {d:.0f} hops" for a, b, d in distances[:15]]
                graph_context = "\n".join(lines)

        # Assemble mesh context from discovery store (meta-modeler observations)
        mesh_context = ""
        if self._discovery_store is not None:
            discoveries = self._discovery_store.sense(limit=5)
            if discoveries:
                mesh_lines = []
                for d in discoveries:
                    dtype = d.get("observation_type", "unknown")
                    strength = d.get("decay_weight", 0.0)
                    summary = d.get("summary", "")[:200]
                    mesh_lines.append(f"- [{dtype}] (strength={strength:.2f}): {summary}")
                mesh_context = "\n".join(mesh_lines)

        # Correlator runs on the quality tier when configured; the surfacer and
        # mechanical paths stay on the cheap self._llm.
        correlator_llm = self._correlator_llm or self._llm
        batch = await correlator_llm.extract(
            CorrelationBatch,
            correlator_prompt(
                summaries,
                graph_context=graph_context,
                prior_findings=prior_findings,
                mesh_context=mesh_context,
            ),
            CORRELATOR_SYSTEM,
            max_tokens=800,
        )
        self._llm_reflect_count += 1

        # ── Agent-assessed Normalized Deviance ──
        # The correlator reports batch-level ND metrics alongside findings.
        # Feed these to the CompressionTracker keyed by channel.
        if self._compression_tracker is not None:
            channel = obs.signal.channel or "unknown"
            # Batch-level ND from agent judgment (primary path)
            if batch.batch_compression > 0 or batch.batch_temporal != 0 or batch.batch_bureaucratic > 0:
                from stigmergy.attention.linguistic import CompressionSignals
                agent_signals = CompressionSignals.from_agent(
                    compression_level=batch.batch_compression,
                    temporal_direction=batch.batch_temporal,
                    bureaucratic_index=batch.batch_bureaucratic,
                    exploration_signal=batch.batch_exploration,
                    text=obs.signal.content,
                )
                self._compression_tracker.update(
                    channel,
                    agent_signals.compression_index,
                    ld=agent_signals.lexical_diversity,
                    se=agent_signals.shannon_entropy,
                )

        # Convert CorrelationInsight -> Insight
        insights: list[Insight] = []
        dropped = 0
        for ci in batch.insights:
            # ── Post-LLM quality filter ──
            drop_reason = _should_drop_finding(ci)
            if drop_reason:
                dropped += 1
                continue

            # Parse signal_ids back to UUIDs (best-effort)
            signal_ids: list[UUID] = []
            for sid in ci.signal_ids:
                try:
                    signal_ids.append(UUID(sid))
                except ValueError:
                    pass

            # Use the LLM's continuous confidence when available,
            # fall back to severity category for older schema responses
            confidence = ci.confidence if ci.confidence > 0 else {
                "high": 0.9, "medium": 0.6, "low": 0.3,
            }.get(ci.severity, 0.5)
            # Compute bridge distances between actors in this insight
            actor_distances = {}
            if len(ci.actors) >= 2:
                for ai, actor_a in enumerate(ci.actors):
                    for actor_b in ci.actors[ai + 1:]:
                        d = obs.bridge_distance(actor_a, actor_b)
                        if d != float("inf"):
                            actor_distances[f"{actor_a}↔{actor_b}"] = d

            # Convert RecommendedArtifact models to dicts for serialization
            artifacts_data = []
            for art in (ci.artifacts or []):
                artifacts_data.append({
                    "artifact_type": art.artifact_type,
                    "target": art.target,
                    "content": art.content,
                })

            insights.append(Insight(
                agent_id=self.id,
                type=ci.pattern_type,
                summary=ci.summary,
                confidence=confidence,
                signal_ids=signal_ids or [obs.signal.id],
                details={
                    "actors": ci.actors,
                    "severity": ci.severity,
                    "bridge_distances": actor_distances,
                    "unknowns": ci.unknowns,
                    "recommended_action": ci.recommended_action,
                    "inaction_risk": ci.inaction_risk,
                    "classification": ci.classification,
                    "temporal_relevance": ci.temporal_relevance,
                    "artifacts": artifacts_data,
                    # Agent-assessed ND per finding (falls back to batch if 0)
                    "compression_index": ci.compression_level or batch.batch_compression,
                    "temporal_orientation": ci.temporal_direction or batch.batch_temporal,
                    "bureaucratic_density": ci.bureaucratic_index or batch.batch_bureaucratic,
                    "exploration_signal": ci.exploration_signal or batch.batch_exploration,
                },
            ))

            # Deposit pheromone: annotate the current signal's resource
            # so future surfacer calls can sense this correlation
            if self._annotations is not None:
                from stigmergy.mesh.annotations import Annotation
                self._annotations.annotate(
                    obs.signal.source,
                    obs.signal.channel,
                    Annotation(
                        agent_type="correlator",
                        summary=ci.summary,
                        confidence=confidence,
                        details={
                            "pattern_type": ci.pattern_type,
                            "severity": ci.severity,
                            "unknowns": ci.unknowns,
                            "recommended_action": ci.recommended_action,
                            "inaction_risk": ci.inaction_risk,
                            "classification": ci.classification,
                            "temporal_relevance": ci.temporal_relevance,
                        },
                    ),
                )

        if dropped:
            import logging
            logging.getLogger("stigmergy.agent").debug(
                "Post-LLM filter dropped %d/%d findings (retrieval/historical/single-actor)",
                dropped, len(batch.insights),
            )

        # Cache this batch for dedup on subsequent calls
        self._last_batch_ids = current_batch_ids
        self._last_batch_insights = insights

        return insights

    def _reflect_mechanical(self, obs: ObservationContext) -> list[Insight]:
        """Mechanical fallback: competency-weighted observation primitives."""
        from stigmergy.mesh.insights import Insight

        insights: list[Insight] = []
        attention = self.confidence
        min_report_threshold = self.rules.get("insight_threshold", 0.25)

        for competency, weight in self.competencies.weights.items():
            strength = weight * attention
            if strength < 0.1:
                continue

            observations = self._observe(competency, obs, strength)
            for obs_type, obs_summary, obs_confidence, obs_signals, obs_details in observations:
                if obs_confidence >= min_report_threshold:
                    insights.append(Insight(
                        agent_id=self.id,
                        type=obs_type,
                        summary=obs_summary,
                        confidence=min(1.0, obs_confidence),
                        signal_ids=obs_signals,
                        details=obs_details,
                    ))

        # Deduplicate: keep highest confidence per (type, signal_set) pair
        best: dict[tuple[str, frozenset[UUID]], Insight] = {}
        for ins in insights:
            key = (ins.type, frozenset(ins.signal_ids))
            if key not in best or ins.confidence > best[key].confidence:
                best[key] = ins

        return list(best.values())

    # Observation primitive registry — data, not code.
    #
    # Maps competency names to the observation primitives they prefer
    # and the relative attention weight for each. This is a starting point;
    # agents expand and customize this through their own state.
    #
    # Every competency gets access to ALL primitives — the weights just
    # control how much attention each gets. Unknown competencies get
    # all primitives at baseline strength.
    _COMPETENCY_PRIMITIVES: dict[str, dict[str, float]] = {
        "similarity_detection":       {"similarity": 1.0, "temporal": 0.3, "cross_source": 0.2},
        "temporal_analysis":          {"temporal": 1.0, "similarity": 0.3, "cross_source": 0.2},
        "cross_context_propagation":  {"cross_source": 1.0, "similarity": 0.3, "temporal": 0.2},
        "anomaly_detection":          {"similarity": 0.8, "temporal": 0.8, "cross_source": 0.5},
        "risk_assessment":            {"cross_source": 0.8, "temporal": 0.6, "similarity": 0.5},
    }

    # Primitive method registry — maps primitive names to method names.
    # New primitives are added here; competencies reference them by name.
    _PRIMITIVE_METHODS: dict[str, str] = {
        "similarity": "_observe_similarity",
        "temporal": "_observe_temporal",
        "cross_source": "_observe_cross_context",
    }

    def _observe(
        self,
        competency: str,
        obs: ObservationContext,
        strength: float,
    ) -> list[tuple[str, str, float, list[UUID], dict[str, Any]]]:
        """Use observation primitives gated by competency.

        Returns raw (type, summary, confidence, signal_ids, details) tuples.
        The agent names the type and writes the summary — no enum constrains it.

        Dispatch is data-driven via _COMPETENCY_PRIMITIVES. Known competencies
        have tuned affinities. Unknown competencies get all primitives at
        baseline strength — they're not locked out, just untuned.
        """
        results: list[tuple[str, str, float, list[UUID], dict[str, Any]]] = []

        # Look up primitive affinities for this competency.
        # Unknown competencies get all primitives at 0.5 strength.
        affinities = self._COMPETENCY_PRIMITIVES.get(
            competency,
            {p: 0.5 for p in self._PRIMITIVE_METHODS},
        )

        for primitive_name, affinity in affinities.items():
            effective = strength * affinity
            if effective < 0.1:
                continue
            method_name = self._PRIMITIVE_METHODS.get(primitive_name)
            if method_name is None:
                continue
            method = getattr(self, method_name, None)
            if method is None:
                continue
            results.extend(method(obs, effective))

        return results

    def _observe_similarity(
        self, obs: ObservationContext, strength: float,
    ) -> list[tuple[str, str, float, list[UUID], dict[str, Any]]]:
        """Observe spectral similarity patterns."""
        results: list[tuple[str, str, float, list[UUID], dict[str, Any]]] = []

        neighbors = obs.spectral_neighbors(threshold=0.5, limit=10)

        # If ALL neighbors are extremely similar (>0.95), the mesh isn't
        # differentiating — report that once instead of N "parallel activity" lines.
        if len(neighbors) >= 2 and all(sim > 0.95 for _, sim in neighbors):
            unique_authors = {c.signal.author for c, _ in neighbors} - {obs.signal.author}
            if unique_authors:
                results.append((
                    "undifferentiated_signals",
                    f"Mesh sees {len(neighbors)+1} signals as near-identical — "
                    f"workers may need more diverse signal history to differentiate "
                    f"({', '.join(f'@{a}' for a in sorted(unique_authors))})",
                    strength * 0.3,  # low confidence — it's a mesh state observation
                    [obs.signal.id] + [c.signal.id for c, _ in neighbors[:3]],
                    {"neighbor_count": len(neighbors), "avg_similarity": sum(s for _, s in neighbors) / len(neighbors)},
                ))
            return results  # don't add per-pair noise

        # Cap observations to avoid N^2 explosion as cache fills
        seen_authors: set[str] = set()
        for cached, sim in neighbors:
            cs = cached.signal
            if cs.author in seen_authors:
                continue  # one observation per unique author
            seen_authors.add(cs.author)

            same_author = cs.author == obs.signal.author
            if same_author:
                confidence = strength * sim * 0.6
                results.append((
                    "work_sequence",
                    f"@{cs.author} working through related area "
                    f"(spectral similarity {sim:.0%})",
                    confidence,
                    [obs.signal.id, cs.id],
                    {"similarity": sim, "author": cs.author},
                ))
            else:
                confidence = strength * sim * 0.9
                results.append((
                    "parallel_activity",
                    f"@{obs.signal.author} and @{cs.author} in same signal space "
                    f"(spectral similarity {sim:.0%})",
                    confidence,
                    [obs.signal.id, cs.id],
                    {"similarity": sim, "author_a": obs.signal.author, "author_b": cs.author},
                ))

        # Score variance — the signal confused the mesh
        variance = obs.score_variance()
        if variance > 0.15 and len(obs.trace.familiarity_scores) >= 3:
            confidence = strength * min(1.0, variance / 0.3)
            scores = obs.trace.familiarity_scores
            high_w = max(scores, key=scores.get)  # type: ignore[arg-type]
            low_w = min(scores, key=scores.get)  # type: ignore[arg-type]
            results.append((
                "classification_divergence",
                f"Workers disagree: scores range {min(scores.values()):.3f}-{max(scores.values()):.3f} "
                f"(stdev={variance:.3f})",
                confidence,
                [obs.signal.id],
                {"stdev": variance, "high_worker": str(high_w), "low_worker": str(low_w)},
            ))

        return results

    def _observe_temporal(
        self, obs: ObservationContext, strength: float,
    ) -> list[tuple[str, str, float, list[UUID], dict[str, Any]]]:
        """Observe temporal clustering patterns."""
        results: list[tuple[str, str, float, list[UUID], dict[str, Any]]] = []

        cluster = obs.temporal_cluster(window_hours=24)
        if len(cluster) < 2:
            return results

        # Group by unique term overlap to find recurring themes
        sig_terms = obs.signal.terms
        if not sig_terms:
            return results

        matching_authors: set[str] = set()
        matching_ids: list[UUID] = []
        for cached in cluster:
            other_terms = cached.signal.terms
            if not other_terms:
                continue
            intersection = sig_terms & other_terms
            union = sig_terms | other_terms
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard >= 0.25:
                matching_authors.add(cached.signal.author)
                matching_ids.append(cached.signal.id)

        # Multiple authors hitting the same area = recurring theme
        if len(matching_authors) >= 2 and len(matching_ids) >= 2:
            confidence = strength * min(1.0, len(matching_ids) / 5.0)
            results.append((
                "recurring_theme",
                f"'{', '.join(sorted(sig_terms)[:5])}' appearing across "
                f"{len(matching_authors)} authors in 24h",
                confidence,
                [obs.signal.id] + matching_ids[:4],
                {"authors": sorted(matching_authors), "match_count": len(matching_ids)},
            ))

        return results

    def _observe_cross_context(
        self, obs: ObservationContext, strength: float,
    ) -> list[tuple[str, str, float, list[UUID], dict[str, Any]]]:
        """Observe cross-source correlation patterns."""
        results: list[tuple[str, str, float, list[UUID], dict[str, Any]]] = []

        cross = obs.cross_source_similar(threshold=0.5, limit=10)
        for cached, sim in cross:
            cs = cached.signal
            confidence = strength * sim * 0.8

            # Cross-source correlation: same signal space, different origin
            results.append((
                "cross_source_correlation",
                f"{obs.signal.source} signal correlates with "
                f"{cs.source} signal from @{cs.author} "
                f"(spectral similarity {sim:.0%})",
                confidence,
                [obs.signal.id, cs.id],
                {
                    "similarity": sim,
                    "source_a": obs.signal.source,
                    "source_b": cs.source,
                },
            ))

        return results

    @property
    def has_llm(self) -> bool:
        """Whether this agent has LLM intelligence available."""
        return self._llm is not None

    def diagnostics(self) -> dict[str, Any]:
        """Return intelligence monitoring diagnostics for this agent.

        Tracks whether the agent is exercising its intelligence (LLM path)
        or falling back to mechanical heuristics. The design intent is that
        ALL agents should use LLM when available — mechanical is only for
        testing and budget exhaustion.
        """
        total_reflect = self._llm_reflect_count + self._mechanical_reflect_count + self._llm_reflect_skipped
        total_evaluate = self._llm_evaluate_count + self._mechanical_evaluate_count
        return {
            "agent_id": str(self.id)[:8],
            "has_llm": self.has_llm,
            "reflect": {
                "llm_calls": self._llm_reflect_count,
                "mechanical_calls": self._mechanical_reflect_count,
                "llm_skipped_non_batch": self._llm_reflect_skipped,
                "batch_dedup_skips": self._batch_dedup_count,
                "total": total_reflect,
            },
            "evaluate": {
                "llm_calls": self._llm_evaluate_count,
                "mechanical_calls": self._mechanical_evaluate_count,
                "total": total_evaluate,
            },
            "intelligence_active": self.has_llm and self._mechanical_reflect_count == 0 and self._mechanical_evaluate_count == 0,
        }

    def update_confidence(self, delta: float) -> None:
        """Adjust confidence based on consensus feedback."""
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
