"""`stigmergy run` — bootstrap pipeline from config and process signals.

Load config -> Bootstrap pipeline -> Connect sources -> Backfill -> Poll loop
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal as signal_mod
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid5, NAMESPACE_DNS

import yaml

from stigmergy.adapters.github import MockGitHubAdapter
from stigmergy.adapters.grafana import MockGrafanaAdapter
from stigmergy.adapters.linear import MockLinearAdapter
from stigmergy.adapters.slack import MockSlackAdapter
from stigmergy.cli.budget import DollarBudgetTracker
from stigmergy.cli.config_schema import StigmergyConfig
from stigmergy.cli.output import (
    InsightDeduplicator,
    NoveltyTracker,
    budget_line,
    error,
    findings_summary,
    heading,
    info,
    insight_line,
    intervention_line,
    mesh_lifecycle_event,
    mesh_poll_header,
    mesh_scan_summary,
    mesh_signal_detail,
    mesh_signal_line,
    mesh_topology,
    policy_summary,
    separator,
    session_summary,
    signal_line,
    success,
    identity_summary,
    insight_funnel,
    intelligence_report,
    corroboration_summary,
    decay_summary,
    attention_findings_summary,
    normalized_deviance_summary,
    portfolio_summary,
    reemission_summary,
    warn,
)
from stigmergy.connectors.metrics import MetricsCollector
from stigmergy.constraints.filter import ConstraintEvaluator
from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.pipeline.processor import AgentRegistry, ContextRegistry, Pipeline
from stigmergy.primitives.agent import Agent, CompetencyModel
from stigmergy.primitives.assessment import AssessmentAction
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal
from stigmergy.services.embedding import StubEmbeddingService
from stigmergy.services.llm import StubLLMService
from stigmergy.tracing.trace import TraceLog


CONFIG_PATH = Path(".stigmergy") / "config.yaml"
STATE_PATH = Path(".stigmergy") / "state.yaml"
DEDUP_INDEX_PATH = Path(".stigmergy") / "dedup_index.json"

# Ceiling on how far back an automated (no --since) backfill will reach.
# Without this, an interrupted run leaves a checkpoint whose timestamp can be
# months old; each subsequent run then re-fetches the entire backlog (tens of
# thousands of signals), drains only ~1 day of it before the launchd timeout,
# and never converges — so its findings freeze. Clamping `since` to a recent
# window keeps fetches cheap, lets a run actually complete (which clears the
# checkpoint), and keeps the daily output fresh. Override via env for a manual
# catch-up; explicit `--since` bypasses the clamp entirely.
MAX_BACKFILL_DAYS = int(os.environ.get("STIGMERGY_MAX_BACKFILL_DAYS", "7"))

# Approximate tokens per word (GPT/Claude tokenizers average ~1.3 tokens/word)
_TOKENS_PER_WORD = 1.3
# Estimated output tokens per agent assessment (action + confidence + reasoning)
_OUTPUT_TOKENS_PER_ASSESSMENT = 50


def _load_config() -> StigmergyConfig:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} not found. Run `stigmergy init` first.")
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return StigmergyConfig(**data)


def _load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


def _save_checkpoint(
    mesh,
    ctx_map: dict,
    annotation_store,
    comm_graph,
    policy_engine,
    last_signal_ts: datetime,
    signals_processed: int,
    total_signals: int,
    budget,
    alias_store=None,
    stability_tracker=None,
    attention_model=None,
) -> None:
    """Save incremental progress so interrupted runs can resume.

    Called every N signals during processing. On next run without --clean,
    the checkpoint timestamp determines where to resume from.
    """
    from stigmergy.primitives.context import Context

    # Save context learning
    contexts = [w.context for w in mesh.workers]
    _save_context_state(contexts, {k: UUID(v) if isinstance(v, str) else v for k, v in ctx_map.items()})

    # Save annotations and graph
    annotation_store.save()
    comm_graph.save(Path(".stigmergy/graph.json"))
    if alias_store is not None:
        alias_store.save()
    if stability_tracker is not None:
        stability_tracker.flush()
    if attention_model is not None:
        attention_model.save(Path(".stigmergy/attention.json"))

    # Save dedup index so resumed runs skip already-processed signals
    if hasattr(mesh, '_dedup_index'):
        mesh._dedup_index.save(DEDUP_INDEX_PATH)

    # Save checkpoint in state — also advance last_run so even a hard kill
    # (SIGKILL, OOM) doesn't lose all progress
    state = _load_state()
    state["checkpoint"] = {
        "timestamp": last_signal_ts.isoformat(),
        "signals_processed": signals_processed,
        "total_signals": total_signals,
        "spend_usd": round(budget.daily_spend, 4),
    }
    state["last_run"] = last_signal_ts.isoformat()
    _save_state(state)


def _stable_uuid(name: str) -> UUID:
    """Generate a stable UUID from a context/agent name for persistence."""
    return uuid5(NAMESPACE_DNS, f"stigmergy.{name}")


# Cold contexts need lower thresholds until they accumulate enough signal history
# for source/author/temporal affinity to contribute meaningfully.
_COLD_START_THRESHOLD_FACTOR = 0.25  # Use 25% of configured threshold when cold
_COLD_START_MIN_SIGNALS = 20  # Signals needed before switching to configured threshold


def _estimate_signal_tokens(signal: Signal, num_assessments: int) -> tuple[int, int]:
    """Estimate input/output tokens for processing a signal.

    Even in stub mode, this gives meaningful budget tracking so users can
    see what real usage would cost before switching to a live LLM.
    """
    word_count = len(signal.content.split())
    input_tokens = int(word_count * _TOKENS_PER_WORD) * max(num_assessments, 1)
    output_tokens = _OUTPUT_TOKENS_PER_ASSESSMENT * max(num_assessments, 1)
    return input_tokens, output_tokens


def _print_coupling_summary(fingerprint_store) -> None:
    """Print structural coupling detections at end of run."""
    if fingerprint_store.count < 10:
        return  # Need enough fingerprints for meaningful comparison

    # Hidden couplings: high activation similarity, low term overlap
    couplings = fingerprint_store.detect_couplings(
        min_activation_similarity=0.5,
        max_term_overlap=0.3,
        recent_n=500,
    )
    # Cross-source corroborations
    cross_source = fingerprint_store.detect_cross_source_consistency(
        recent_n=500,
        min_similarity=0.4,
    )

    if not couplings and not cross_source:
        info(f"  Fingerprints: {fingerprint_store.count} recorded, no hidden couplings detected")
        return

    separator()
    info(f"  Fingerprints: {fingerprint_store.count} | "
         f"workers tracked: {len(fingerprint_store._worker_universe)}")

    if couplings:
        hidden = [c for c in couplings if c.is_hidden_coupling]
        if hidden:
            heading(f"  Hidden Couplings ({len(hidden)} detected)")
            for c in hidden[:5]:  # Show top 5
                src_info = f"{c.source_a}/{c.channel_a} <-> {c.source_b}/{c.channel_b}"
                info(f"    activation={c.activation_similarity:.2f} "
                     f"vocab={c.term_overlap:.2f} "
                     f"strength={c.coupling_strength:.2f} "
                     f"shared_workers={len(c.shared_workers)} "
                     f"  {src_info}")

    if cross_source:
        heading(f"  Cross-Source Corroborations ({len(cross_source)})")
        for c in cross_source[:5]:
            info(f"    {c.source_a}<->{c.source_b} "
                 f"sim={c.activation_similarity:.2f} "
                 f"  {c.description}")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration (e.g. '5m00s', '1h15m')."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    h, rem = divmod(s, 3600)
    return f"{h}h{rem // 60:02d}m"


import shutil

def _archive_run_outputs() -> Path | None:
    """Copy key output files to a timestamped run archive.

    Creates .stigmergy/runs/{YYYYMMDD_HHMMSS}/ and copies all output
    files there. This prevents overwriting across runs and provides
    a complete snapshot for cross-run comparison.
    """
    stigmergy_dir = Path(".stigmergy")
    if not stigmergy_dir.exists():
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = stigmergy_dir / "runs" / ts
    archive_dir.mkdir(parents=True, exist_ok=True)

    files_to_archive = [
        "state.yaml",
        "annotations.json",
        "graph.json",
        "insights.jsonl",
        "finding_registry.json",
        "stability.jsonl",
        "routing_history.json",
        "budgets.json",
        "traces.jsonl",
        "corroboration.json",
        "run.log",
    ]
    for fname in files_to_archive:
        src = stigmergy_dir / fname
        if src.exists():
            shutil.copy2(src, archive_dir / fname)

    return archive_dir


async def _reflect_and_cache(
    sig: Signal,
    trace,
    mesh,
    agent_registry: AgentRegistry,
    signal_cache,
    comm_graph=None,
    known_finding_hashes: set[str] | None = None,
) -> list:
    """Compute spectrum, run agent reflection, cache signal, return insights."""
    from stigmergy.mesh.insights import ObservationContext

    # Build spectrum: ordered familiarity scores across all workers
    worker_ids = [w.id for w in mesh.workers]
    spectrum = [trace.familiarity_scores.get(wid, 0.0) for wid in worker_ids]

    # Build observation context
    obs = ObservationContext(
        signal=sig,
        trace=trace,
        spectrum=spectrum,
        cache=signal_cache,
        workers=list(mesh.workers),
        graph=comm_graph,
    )

    # Each agent reflects through its competency lens
    all_insights = []
    for agent in agent_registry.all_agents():
        insights = await agent.reflect(obs)
        all_insights.extend(insights)

    # Cross-agent dedup: keep highest confidence per (type, signal_set)
    from uuid import UUID as _UUID
    raw_count = len(all_insights)
    best: dict[tuple[str, frozenset], object] = {}
    for ins in all_insights:
        key = (ins.type, frozenset(ins.signal_ids))
        if key not in best or ins.confidence > best[key].confidence:
            best[key] = ins
    all_insights = list(best.values())

    # Entity-based dedup: when multiple agents emit findings about the same
    # situation (same PRs, tickets, or actor pairs) under different types
    # (overlap vs dependency vs coordination), keep only the highest-confidence.
    import re as _re_dedup
    _entity_best: dict[str, object] = {}
    for ins in all_insights:
        _summary_lower = (ins.summary or "").lower()
        _ent_prs = sorted(set(_re_dedup.findall(r'pr\s*#(\d+)', _summary_lower)))
        _ent_tickets = sorted(set(_re_dedup.findall(r'[a-z]+-\d+', _summary_lower)))
        _ent_mentions = sorted(set(_re_dedup.findall(r'@([\w-]+)', _summary_lower)))
        # Entity key: if the same PR numbers or ticket IDs appear, it's the same situation
        _ent_key_parts = _ent_prs + _ent_tickets
        if len(_ent_key_parts) >= 2:
            _ent_key = "|".join(_ent_key_parts[:6])
            if _ent_key not in _entity_best or ins.confidence > _entity_best[_ent_key].confidence:
                _entity_best[_ent_key] = ins
        elif _ent_mentions and len(_ent_mentions) >= 2:
            # Fall back to actor pair if no specific identifiers
            _ent_key = "actors:" + "|".join(_ent_mentions[:4])
            if _ent_key not in _entity_best or ins.confidence > _entity_best[_ent_key].confidence:
                _entity_best[_ent_key] = ins
        else:
            # No identifiers to dedup on — keep as-is (use unique key)
            _entity_best[f"_solo_{id(ins)}"] = ins
    all_insights = list(_entity_best.values())

    # Finding-level dedup: suppress insights whose content hash is already known.
    # This prevents the correlator from re-surfacing the same pattern every time
    # it sees signals in the same domain (e.g. a user's 30+ audit tickets in one area).
    if known_finding_hashes is not None:
        from stigmergy.mesh.insight_store import _finding_hash
        novel = []
        for ins in all_insights:
            actors = ins.details.get("actors", []) if ins.details else []
            fh = _finding_hash(ins.type, ins.summary, actors)
            if fh not in known_finding_hashes:
                novel.append(ins)
        all_insights = novel

    # Attach to trace
    trace.insights = all_insights

    # Add to cache AFTER reflection (so signal doesn't match itself)
    signal_cache.add(sig, trace, spectrum, worker_ids)

    return all_insights, raw_count


def _save_context_state(contexts: list[Context], ctx_map: dict[str, UUID]) -> None:
    """Persist learned context state so the system improves across sessions."""
    state = _load_state()
    name_for_id = {str(v): k for k, v in ctx_map.items()}
    ctx_state: dict[str, dict] = {}
    for ctx in contexts:
        name = name_for_id.get(str(ctx.id))
        if not name:
            continue
        # Persist learned knowledge only — NOT operational state.
        # signal_count and last_signal are session-specific and cause
        # stale fullness/threshold/fork triggers on reload.
        ctx_state[name] = {
            "source_counts": dict(ctx.source_counts),
            "author_counts": dict(ctx.author_counts),
            "learned_terms": sorted(ctx.terms),
        }
    state["context_state"] = ctx_state
    _save_state(state)


def bootstrap(config: StigmergyConfig) -> tuple[
    Pipeline,
    list[Context],
    list[Agent],
    dict[str, UUID],
]:
    """Build pipeline from config. Mirrors _build_system() in test_scenario.py."""
    state = _load_state()
    ctx_map: dict[str, UUID] = state.get("context_ids", {})

    metrics = MetricsCollector()
    ctx_registry = ContextRegistry()
    agent_registry = AgentRegistry()

    # 1. Create contexts
    contexts: list[Context] = []
    for name, ctx_cfg in config.contexts.items():
        ctx_id = UUID(ctx_map[name]) if name in ctx_map else _stable_uuid(name)
        ctx_map[name] = str(ctx_id)

        # Restore persisted state if available
        saved_ctx = state.get("context_state", {}).get(name, {})
        # Cold start: use saved signal_count if available, else treat as cold.
        # (Pipeline mode still uses signal_count for threshold adjustment.)
        is_cold = saved_ctx.get("signal_count", 0) < _COLD_START_MIN_SIGNALS

        # Cold contexts use a reduced threshold so signals can route in before
        # the context has enough history for affinity scoring to work.
        threshold = ctx_cfg.relevance_threshold
        if is_cold:
            threshold = ctx_cfg.relevance_threshold * _COLD_START_THRESHOLD_FACTOR

        ctx = Context(
            id=ctx_id,
            relevance_threshold=threshold,
            business_weight=ctx_cfg.business_weight,
        )
        ctx.terms = set(ctx_cfg.terms)

        # Restore learned state from previous runs
        if saved_ctx:
            ctx.source_counts = dict(saved_ctx.get("source_counts", {}))
            ctx.author_counts = dict(saved_ctx.get("author_counts", {}))
            ctx.signal_count = saved_ctx.get("signal_count", 0)
            ctx.terms.update(saved_ctx.get("learned_terms", []))

        ctx_registry.register(ctx)
        contexts.append(ctx)

    # 2. Create agents
    agents: list[Agent] = []
    for name, agent_cfg in config.agents.items():
        # Resolve context names to UUIDs
        ctx_bindings: dict[UUID, float] = {}
        for ctx_name, affinity in agent_cfg.contexts.items():
            if ctx_name in ctx_map:
                ctx_bindings[UUID(ctx_map[ctx_name])] = affinity

        competencies = CompetencyModel(agent_cfg.competencies) if agent_cfg.competencies else None
        agent = Agent(
            id=_stable_uuid(f"agent.{name}"),
            contexts=ctx_bindings,
            competencies=competencies,
            confidence=agent_cfg.confidence,
            weights=agent_cfg.weights,
        )

        # Register agent with contexts
        for ctx_id in ctx_bindings:
            ctx_obj = ctx_registry.get(ctx_id)
            if ctx_obj:
                ctx_obj.active_agents.add(agent.id)

        agent_registry.register(agent)
        agents.append(agent)

    # 3. Resolve LLM service
    llm_service = _resolve_llm(config)

    # Inject LLM and annotation store into agents.
    # Only real LLM, not stub. Stub means mechanical fallback.
    from stigmergy.mesh.annotations import AnnotationStore
    annotation_store = AnnotationStore()
    if not isinstance(llm_service, StubLLMService):
        for agent in agents:
            agent._llm = llm_service
            agent._annotations = annotation_store

    # 4. Build pipeline
    fw = config.pipeline.familiarity_weights
    pipeline = Pipeline(
        contexts=ctx_registry,
        agents=agent_registry,
        embedding_service=StubEmbeddingService(),
        llm_service=llm_service,
        constraint_evaluator=ConstraintEvaluator.from_config("config/constraints.yaml"),
        trace_log=TraceLog(),
        metrics=metrics,
        familiarity_weights=FamiliarityWeights(
            embedding_similarity=fw.embedding_similarity,
            keyword_overlap=fw.keyword_overlap,
            source_affinity=fw.source_affinity,
            temporal_proximity=fw.temporal_proximity,
            author_affinity=fw.author_affinity,
        ),
        consensus_threshold=config.pipeline.consensus_threshold,
        uncertainty_threshold=config.pipeline.uncertainty_threshold,
        dedup_enabled=config.pipeline.dedup_enabled,
    )

    # 5. Build budget
    budget_tracker = DollarBudgetTracker(
        daily_cap_usd=config.budget.daily_cap_usd,
        hourly_cap_usd=config.budget.hourly_cap_usd,
        input_cost_per_million=config.budget.input_cost_per_million,
        output_cost_per_million=config.budget.output_cost_per_million,
        warn_at_percent=config.budget.warn_at_percent,
    )
    budget_tracker.set_model_pricing(config.llm.model)
    if not isinstance(llm_service, StubLLMService):
        budget_tracker.set_llm(llm_service)
    context_weights = {ctx.id: ctx.business_weight for ctx in contexts}
    token_budget = budget_tracker.build_token_budget(context_weights)
    pipeline.token_budget = token_budget

    # Persist context ID mappings
    state["context_ids"] = ctx_map
    _save_state(state)

    return pipeline, contexts, agents, ctx_map


def _resolve_llm(config: StigmergyConfig):
    """Resolve LLM service from config."""
    if config.llm.provider == "anthropic":
        try:
            from stigmergy.cli.anthropic_llm import AnthropicLLMService
            return AnthropicLLMService(model=config.llm.model)
        except ImportError:
            warn("anthropic package not installed. Falling back to stub.")
            warn("Install with: pip install -e '.[cli]'")
            return StubLLMService()
        except ValueError as exc:
            warn(str(exc))
            warn("Falling back to stub LLM (heuristic-only mode).")
            return StubLLMService()
    return StubLLMService()


def _resolve_fast_llm(config: StigmergyConfig, quality_llm):
    """Resolve the cheap per-signal LLM tier (worker extraction + surfacer).

    Returns a separate client on config.llm.fast_model when it differs from the
    quality model; otherwise returns the quality client unchanged (single-tier).
    Falls back to the quality client on any construction error.
    """
    fast_model = getattr(config.llm, "fast_model", "") or ""
    if (
        quality_llm is None
        or not fast_model
        or fast_model == config.llm.model
        or config.llm.provider != "anthropic"
    ):
        return quality_llm
    try:
        from stigmergy.cli.anthropic_llm import AnthropicLLMService
        fast = AnthropicLLMService(model=fast_model)
        info(f"  Tiered LLM: fast={fast_model} (per-signal) / quality={config.llm.model} (correlator)")
        return fast
    except Exception as exc:
        warn(f"Could not build fast LLM ({fast_model}): {exc}. Using quality model for all calls.")
        return quality_llm


async def _preflight_llm(llm) -> None:
    """Verify LLM is reachable before starting a run.

    A run without LLM intelligence produces only mechanical heuristic
    noise — it's worse than useless because it burns budget and time.
    Fail hard here so the user can fix the issue before wasting a run.
    """
    info("Preflight: verifying LLM connectivity...")
    try:
        await llm.probe()
        success("Preflight: LLM reachable")
    except Exception as exc:
        error(f"Preflight FAILED: LLM unreachable — {exc}")
        error("Without LLM, agents fall back to mechanical heuristics (useless run).")
        error("Check ANTHROPIC_API_KEY and network connectivity, then retry.")
        raise SystemExit(1) from exc


class _LLMCircuitBreaker:
    """Trips after consecutive LLM failures, aborting the run.

    The system MUST use agent intelligence. If the LLM becomes
    unreachable mid-run, continuing just produces mechanical noise.

    Includes an async cooldown: after 3+ consecutive failures, sleeps
    with exponential backoff to let rate limits recover before the
    next attempt.
    """

    def __init__(self, threshold: int = 15) -> None:
        self._consecutive_failures: int = 0
        self._threshold = threshold
        self._total_failures: int = 0
        self._logger = logging.getLogger(__name__)

    def record_success(self) -> None:
        if self._consecutive_failures > 0:
            self._logger.info(
                "LLM recovered after %d consecutive failures",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._total_failures += 1
        self._logger.warning(
            "LLM failure %d/%d (total: %d)",
            self._consecutive_failures,
            self._threshold,
            self._total_failures,
        )
        if self._consecutive_failures >= self._threshold:
            raise SystemExit(
                f"CIRCUIT BREAKER: {self._consecutive_failures} consecutive LLM failures. "
                f"Aborting — mechanical fallback produces noise, not insight. "
                f"Check API key, network, and rate limits. "
                f"See .stigmergy/run.log for details."
            )

    async def cooldown(self) -> None:
        """Exponential backoff after consecutive failures.

        Call this before the next LLM attempt to give rate limits
        time to recover. Sleeps 0s for <=2 failures, then 2^(n-3)
        seconds up to 30s.
        """
        if self._consecutive_failures < 3:
            return
        delay = min(30, 2 ** (self._consecutive_failures - 3))
        self._logger.info("Circuit breaker cooldown: %.1fs before next LLM attempt", delay)
        await asyncio.sleep(delay)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def total_failures(self) -> int:
        return self._total_failures


# Global circuit breaker — shared across all agents in a run
_circuit_breaker: _LLMCircuitBreaker | None = None


def _gh_progress(repo: str, current: int, total: int, count: int) -> None:
    """Progress callback for GitHub adapter."""
    bar_width = 20
    filled = int(bar_width * current / total)
    bar = "=" * filled + "-" * (bar_width - filled)
    suffix = f" {count} signals" if count else ""
    print(f"\r  [{bar}] {current}/{total} repos  {repo:<40s}{suffix}", end="", flush=True)
    if current == total:
        print()  # newline at end


def _gh_error(repo: str, msg: str) -> None:
    """Error callback for GitHub adapter."""
    # Skip non-actionable errors: private repos, disabled features
    if "Could not resolve" in msg or "Not Found" in msg:
        return
    if "disabled issues" in msg:
        return
    warn(f"  {repo}: {msg[:120]}")


def _linear_progress(team: str, current: int, total: int, count: int) -> None:
    """Progress callback for Linear adapter."""
    bar_width = 20
    filled = int(bar_width * current / total)
    bar = "=" * filled + "-" * (bar_width - filled)
    suffix = f" {count} issues" if count else ""
    print(f"\r  [{bar}] {current}/{total} teams  {team:<30s}{suffix}", end="", flush=True)
    if current == total:
        print()  # newline at end


def _linear_error(source: str, msg: str) -> None:
    """Error callback for Linear adapter."""
    warn(f"  linear ({source}): {msg[:120]}")


def _slack_progress(channel: str, current: int, total: int, count: int) -> None:
    """Progress callback for Slack adapter."""
    bar_width = 20
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)
    suffix = f" {count} messages" if count else ""
    print(f"\r  [{bar}] {current}/{total} channels  {channel:<30s}{suffix}", end="", flush=True)
    if current == total:
        print()


def _slack_error(source: str, msg: str) -> None:
    """Error callback for Slack adapter."""
    warn(f"  slack ({source}): {msg[:120]}")


def _grafana_progress(phase: str, current: int, total: int, count: int) -> None:
    """Progress callback for Grafana adapter."""
    bar_width = 20
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)
    suffix = f" {count} signals" if count else ""
    print(f"\r  [{bar}] {current}/{total} phases  {phase:<20s}{suffix}", end="", flush=True)
    if current == total:
        print()


def _grafana_error(source: str, msg: str) -> None:
    """Error callback for Grafana adapter."""
    warn(f"  grafana ({source}): {msg[:120]}")


def _build_sources(config: StigmergyConfig, live: bool) -> list[tuple[str, object, bool]]:
    """Build source adapters from config. Returns (name, adapter, is_live)."""
    sources: list[tuple[str, object, bool]] = []

    if config.sources.github.enabled:
        if live and config.sources.github.mode == "live":
            try:
                from stigmergy.cli.github_live import LiveGitHubAdapter
                adapter = LiveGitHubAdapter(
                    repos=config.sources.github.repos,
                    progress_callback=_gh_progress,
                    error_callback=_gh_error,
                )
                sources.append(("github", adapter, True))
            except ImportError:
                warn("Live GitHub adapter unavailable, using mock.")
                sources.append(("github", MockGitHubAdapter(), False))
        else:
            sources.append(("github", MockGitHubAdapter(), False))

    if config.sources.linear.enabled:
        if live and config.sources.linear.mode == "live":
            try:
                from stigmergy.cli.linear_live import LiveLinearAdapter
                teams = [{"name": t.name, "id": t.id} for t in config.sources.linear.teams]
                api_key = config.sources.linear.api_key or os.environ.get("LINEAR_API_KEY", "")
                adapter = LiveLinearAdapter(
                    teams=teams,
                    api_key=api_key,
                    progress_callback=_linear_progress,
                    error_callback=_linear_error,
                )
                sources.append(("linear", adapter, True))
            except ImportError:
                warn("Live Linear adapter unavailable, using mock.")
                sources.append(("linear", MockLinearAdapter(), False))
        else:
            sources.append(("linear", MockLinearAdapter(), False))

    if config.sources.slack.enabled:
        if live and config.sources.slack.mode == "live":
            try:
                from stigmergy.cli.slack_live import LiveSlackAdapter
                adapter = LiveSlackAdapter(
                    channels=config.sources.slack.channels,
                    bot_token=config.sources.slack.bot_token or os.environ.get("SLACK_BOT_TOKEN", ""),
                    progress_callback=_slack_progress,
                    error_callback=_slack_error,
                )
                cached = adapter.load_user_cache()
                if cached:
                    info(f"  Loaded {cached} cached Slack user identities")
                sources.append(("slack", adapter, True))
            except ImportError:
                warn("Live Slack adapter unavailable, using mock.")
                sources.append(("slack", MockSlackAdapter(), False))
        else:
            sources.append(("slack", MockSlackAdapter(), False))
    elif config.sources.slack.channels:
        ch_count = len(config.sources.slack.channels)
        info(f"  Slack: configured ({ch_count} channels) but disabled — set enabled: true and SLACK_BOT_TOKEN to activate")

    if config.sources.grafana.enabled:
        if live and config.sources.grafana.mode == "live":
            try:
                from stigmergy.cli.grafana_live import LiveGrafanaAdapter
                adapter = LiveGrafanaAdapter(
                    base_url=config.sources.grafana.base_url,
                    api_key=config.sources.grafana.api_key or os.environ.get("GRAFANA_API_KEY", ""),
                    dashboards=config.sources.grafana.dashboards,
                    tempo_services=config.sources.grafana.tempo_services,
                    anomaly_stddev_threshold=config.sources.grafana.anomaly_stddev_threshold,
                    progress_callback=_grafana_progress,
                    error_callback=_grafana_error,
                )
                sources.append(("grafana", adapter, True))
            except ImportError:
                warn("Live Grafana adapter unavailable, using mock.")
                sources.append(("grafana", MockGrafanaAdapter(), False))
        else:
            sources.append(("grafana", MockGrafanaAdapter(), False))

    return sources


async def _run_once(config: StigmergyConfig, live: bool) -> int:
    """Single-pass: backfill all sources, process, print results, exit."""
    heading("stigmergy run --once")
    print()

    pipeline, contexts, agents, ctx_map = bootstrap(config)
    sources = _build_sources(config, live)

    budget = DollarBudgetTracker(
        daily_cap_usd=config.budget.daily_cap_usd,
        hourly_cap_usd=config.budget.hourly_cap_usd,
        input_cost_per_million=config.budget.input_cost_per_million,
        output_cost_per_million=config.budget.output_cost_per_million,
        warn_at_percent=config.budget.warn_at_percent,
    )
    budget.set_model_pricing(config.llm.model)

    # Counters
    surfaced = 0
    stored = 0
    ignored = 0
    blocked = 0
    duplicates = 0
    total = 0

    # Wider window on first run (no state yet), narrower on subsequent runs
    state = _load_state()
    last_run = state.get("last_run")
    if last_run:
        since = datetime.fromisoformat(last_run)
    else:
        since = datetime.now(timezone.utc) - timedelta(days=7)

    for source_name, adapter, _is_live in sources:
        info(f"Connecting {source_name}{'  (live)' if _is_live else '  (mock)'}...")
        try:
            await adapter.connect()
        except (ConnectionError, OSError) as e:
            warn(f"  {source_name}: connection failed ({e}), skipping")
            continue

        info(f"Backfilling {source_name} since {since.date()}...")
        source_count = 0
        async for sig in adapter.backfill(since):
            source_count += 1
            # Budget check
            if not budget.can_spend():
                warn("Budget exhausted. Stopping.")
                break

            for w in budget.check_warnings():
                warn(w)

            trace = await pipeline.process_signal(sig)
            total += 1

            # Estimate and record token cost for budget tracking
            budget.sync_from_llm()

            # Determine action from trace
            action = "ignore"
            confidence = 0.0
            if trace.consensus:
                action = trace.consensus.action
                confidence = trace.consensus.total_weight

            # Check if dedup caught it
            if not trace.contexts and not trace.assessments and not trace.consensus:
                if pipeline.metrics.counter_value("pipeline.duplicates_detected") > duplicates:
                    action = "duplicate"
                    duplicates = int(pipeline.metrics.counter_value("pipeline.duplicates_detected"))

            if not trace.output_delivered and trace.constraint_eval:
                action = "block"

            # Count
            if action == "surface" and trace.output_delivered:
                surfaced += 1
            elif action == "block":
                blocked += 1
            elif action == "store":
                stored += 1
            else:
                ignored += 1

            # Print
            if config.output.surface_only and action != "surface":
                continue

            signal_line(
                source=sig.source,
                channel=sig.channel,
                author=sig.author,
                action=action,
                confidence=confidence,
                content_preview=sig.content,
            )

            if config.output.show_traces and trace.familiarity_scores:
                # Reverse-map context UUIDs to names
                id_to_name = {UUID(v): k for k, v in ctx_map.items()}
                for ctx_id, score in trace.familiarity_scores.items():
                    ctx_name = id_to_name.get(ctx_id, str(ctx_id)[:8])
                    info(f"    familiarity({ctx_name}) = {score:.3f}")

        info(f"  {source_name}: {source_count} signals ingested")

    # Print budget
    budget_line(budget.daily_spend, budget.daily_cap_usd, budget.hourly_spend, budget.hourly_cap_usd)

    # Print summary
    session_summary(
        signals_processed=total,
        surfaced=surfaced,
        stored=stored,
        ignored=ignored,
        blocked=blocked,
        duplicates=duplicates,
        total_cost=budget.total_spend,
    )

    # Save state — context learning persists across runs
    _save_context_state(contexts, {k: UUID(v) for k, v in ctx_map.items()})
    state = _load_state()
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["last_sources"] = [name for name, _, _ in sources]
    state["last_summary"] = {
        "signals_processed": total,
        "surfaced": surfaced,
        "stored": stored,
        "ignored": ignored,
        "blocked": blocked,
        "duplicates": duplicates,
    }
    _save_state(state)

    return 0


async def _run_loop(config: StigmergyConfig, live: bool) -> int:
    """Continuous poll loop. Ctrl+C to stop."""
    heading("stigmergy run (continuous)")
    info("Press Ctrl+C to stop")
    print()

    pipeline, contexts, agents, ctx_map = bootstrap(config)
    sources = _build_sources(config, live)

    budget = DollarBudgetTracker(
        daily_cap_usd=config.budget.daily_cap_usd,
        hourly_cap_usd=config.budget.hourly_cap_usd,
        input_cost_per_million=config.budget.input_cost_per_million,
        output_cost_per_million=config.budget.output_cost_per_million,
        warn_at_percent=config.budget.warn_at_percent,
    )
    budget.set_model_pricing(config.llm.model)

    counters = {"surfaced": 0, "stored": 0, "ignored": 0, "blocked": 0, "duplicates": 0, "total": 0}
    id_to_name = {UUID(v): k for k, v in ctx_map.items()}
    stop = asyncio.Event()

    def handle_sigint():
        if stop.is_set():
            # Second Ctrl+C — force exit
            raise KeyboardInterrupt
        stop.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal_mod.SIGINT, handle_sigint)

    # Connect all sources
    has_live = False
    failed_sources: set[str] = set()
    for source_name, adapter, is_live in sources:
        info(f"Connecting {source_name}{'  (live)' if is_live else '  (mock)'}...")
        try:
            await adapter.connect()
        except (ConnectionError, OSError) as e:
            warn(f"  {source_name}: connection failed ({e}), skipping")
            failed_sources.add(source_name)
            continue
        if is_live:
            has_live = True

    if not has_live:
        warn("All sources are mocks — will process once then wait for Ctrl+C.")
        warn("Use --live for continuous polling with real sources.")

    # Track poll schedules: (name, adapter, next_poll_at, interval, is_live, polled_once)
    poll_schedule: list[list] = []
    for source_name, adapter, is_live in sources:
        if source_name in failed_sources:
            continue
        interval = 300
        if source_name == "github":
            interval = config.sources.github.poll_interval
        elif source_name == "linear":
            interval = config.sources.linear.poll_interval
        poll_schedule.append([source_name, adapter, time.monotonic(), interval, is_live, False])

    last_backfill = datetime.now(timezone.utc) - timedelta(days=1)

    while not stop.is_set():
        now_mono = time.monotonic()

        for entry in poll_schedule:
            source_name, adapter, next_poll, interval, is_live, polled_once = entry

            if now_mono < next_poll:
                continue

            # Mock sources only poll once — replaying the same data just dedupes
            if not is_live and polled_once:
                continue

            if not budget.can_spend():
                warn("Budget exhausted. Waiting for reset...")
                break

            info(f"Polling {source_name}...")
            async for sig in adapter.backfill(last_backfill):
                for w in budget.check_warnings():
                    warn(w)

                trace = await pipeline.process_signal(sig)
                counters["total"] += 1

                # Estimate and record token cost for budget tracking
                budget.sync_from_llm()

                action = "ignore"
                confidence = 0.0
                if trace.consensus:
                    action = trace.consensus.action
                    confidence = trace.consensus.total_weight

                if not trace.output_delivered and trace.constraint_eval:
                    action = "block"

                if action == "surface" and trace.output_delivered:
                    counters["surfaced"] += 1
                elif action == "block":
                    counters["blocked"] += 1
                elif action == "store":
                    counters["stored"] += 1
                else:
                    counters["ignored"] += 1

                if not config.output.surface_only or action == "surface":
                    signal_line(
                        source=sig.source,
                        channel=sig.channel,
                        author=sig.author,
                        action=action,
                        confidence=confidence,
                        content_preview=sig.content,
                    )

            entry[2] = now_mono + interval  # next_poll
            entry[5] = True  # polled_once
            last_backfill = datetime.now(timezone.utc)

        # If all sources are mocked and polled, just wait for Ctrl+C
        all_done = all(not e[4] and e[5] for e in poll_schedule)
        if all_done:
            info("All mock sources processed. Waiting for Ctrl+C...")
            try:
                await stop.wait()
            except asyncio.CancelledError:
                pass
            break

        # Sleep until next poll or Ctrl+C
        active_polls = [e[2] for e in poll_schedule if e[4] or not e[5]]
        if active_polls:
            next_time = min(active_polls)
            sleep_for = max(0.1, next_time - time.monotonic())
        else:
            sleep_for = 10

        try:
            await asyncio.wait_for(stop.wait(), timeout=sleep_for)
        except asyncio.TimeoutError:
            pass

    # Print summary on exit
    budget_line(budget.daily_spend, budget.daily_cap_usd, budget.hourly_spend, budget.hourly_cap_usd)
    session_summary(
        signals_processed=counters["total"],
        surfaced=counters["surfaced"],
        stored=counters["stored"],
        ignored=counters["ignored"],
        blocked=counters["blocked"],
        duplicates=counters["duplicates"],
        total_cost=budget.total_spend,
    )

    # Save state — context learning persists across runs
    _save_context_state(contexts, {k: UUID(v) for k, v in ctx_map.items()})
    state = _load_state()
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["last_sources"] = [name for name, _, _ in sources]
    state["last_summary"] = counters
    _save_state(state)

    return 0


def bootstrap_mesh(config: StigmergyConfig) -> tuple:
    """Build a Mesh from config, reusing agents and familiarity weights.

    Returns (mesh, agent_registry, budget_tracker).
    """
    from stigmergy.core.familiarity import FamiliarityWeights
    from stigmergy.core.lifecycle import LifecycleManager
    from stigmergy.mesh.insights import ObservationContext, SignalCache
    from stigmergy.mesh.mesh import Mesh
    from stigmergy.mesh.worker import WorkerNode

    state = _load_state()
    ctx_map: dict[str, UUID] = state.get("context_ids", {})

    agent_registry = AgentRegistry()

    # Create agents from config
    agents_list: list[Agent] = []
    for name, agent_cfg in config.agents.items():
        ctx_bindings: dict[UUID, float] = {}
        for ctx_name, affinity in agent_cfg.contexts.items():
            if ctx_name in ctx_map:
                ctx_bindings[UUID(ctx_map[ctx_name])] = affinity
        competencies = CompetencyModel(agent_cfg.competencies) if agent_cfg.competencies else None
        agent = Agent(
            id=_stable_uuid(f"agent.{name}"),
            contexts=ctx_bindings,
            competencies=competencies,
            confidence=agent_cfg.confidence,
            weights=agent_cfg.weights,
        )
        agent_registry.register(agent)
        agents_list.append(agent)

    # Familiarity weights
    fw = config.pipeline.familiarity_weights
    weights = FamiliarityWeights(
        embedding_similarity=fw.embedding_similarity,
        keyword_overlap=fw.keyword_overlap,
        source_affinity=fw.source_affinity,
        temporal_proximity=fw.temporal_proximity,
        author_affinity=fw.author_affinity,
    )

    # Mesh config
    mc = config.mesh
    wc = mc.worker

    # Resolve LLM — agent intelligence for workers and agents.
    # Only pass real LLM; stub mode means mechanical fallback.
    llm_service = _resolve_llm(config)
    mesh_llm = llm_service if not isinstance(llm_service, StubLLMService) else None
    # Tiered models: fast_llm (cheap) handles the high-volume per-signal calls
    # (worker fact-extraction + surfacer); mesh_llm (quality) handles the
    # batched correlator/reflect, corroboration, and meta-modeler. When no
    # fast_model is configured, fast_llm IS mesh_llm (single tier).
    fast_llm = _resolve_fast_llm(config, mesh_llm)

    # Inject LLM, annotation store, and circuit breaker into agents.
    # The circuit breaker aborts the run if LLM becomes unreachable mid-flight.
    from stigmergy.mesh.annotations import AnnotationStore
    annotation_store = AnnotationStore()
    global _circuit_breaker
    _circuit_breaker = _LLMCircuitBreaker(threshold=10) if mesh_llm is not None else None
    # Shared CompressionTracker — agents feed ND assessments into it
    from stigmergy.attention.linguistic import CompressionTracker as _AgentCompTracker
    _agent_comp_tracker = _AgentCompTracker()

    # Discovery store — cross-session self-knowledge (independent of meta-modeler)
    from stigmergy.mesh.discovery_store import DiscoveryStore
    discovery_store = DiscoveryStore()

    # Health pulse collector — mechanical mesh vitals
    from stigmergy.mesh.health_pulse import HealthPulseCollector
    health_pulse_collector = HealthPulseCollector()

    # Meta-modeler — LLM-powered mesh self-assessment (opt-in)
    meta_modeler = None
    mm_config = getattr(config, "meta_modeler", None)
    if mm_config is not None and mm_config.enabled and mesh_llm is not None:
        from stigmergy.mesh.meta_modeler import MetaModeler
        meta_modeler = MetaModeler(
            mesh_llm,
            discovery_store,
            fire_interval=mm_config.fire_interval,
            max_output_tokens=mm_config.max_output_tokens,
            perturbation_enabled=mm_config.perturbation_enabled,
        )

    # Unity field engine — mechanical self-regulation (opt-in)
    field_engine = None
    unity_config = getattr(config, "unity", None)
    if unity_config is not None and unity_config.enabled:
        from stigmergy.unity.field_config import FieldConfig
        from stigmergy.unity.field_engine import FieldEngine
        fc = FieldConfig(**unity_config.model_dump())
        field_engine = FieldEngine(fc)
        # Warm restart: restore PID integral, calibration, eigenmonitor history
        if field_engine.load_state():
            info("  Restored unity field engine state from previous session")

    if mesh_llm is not None:
        for agent in agents_list:
            agent._llm = fast_llm          # surfacer/evaluate — high-volume, cheap tier
            agent._correlator_llm = mesh_llm  # reflect/correlator — batched, quality tier
            agent._annotations = annotation_store
            agent._circuit_breaker = _circuit_breaker
            agent._compression_tracker = _agent_comp_tracker
            agent._discovery_store = discovery_store

    # Build mesh
    mesh = Mesh(
        agent_registry,
        embedding_service=StubEmbeddingService(),
        constraint_evaluator=ConstraintEvaluator.from_config("config/constraints.yaml"),
        trace_log=TraceLog(),
        lifecycle=LifecycleManager(fork_volume_threshold=wc.capacity),
        max_hops=mc.max_hops,
        max_workers=mc.max_workers,
        familiarity_weights=weights,
        consensus_threshold=config.pipeline.consensus_threshold,
        uncertainty_threshold=config.pipeline.uncertainty_threshold,
        dedup_enabled=config.pipeline.dedup_enabled,
        worker_capacity=wc.capacity,
        base_threshold=wc.base_threshold,
        max_threshold=wc.max_threshold,
        threshold_curve=wc.threshold_curve,
        high_relevance_offset=wc.high_relevance_offset,
        llm=fast_llm,
    )
    # Quality tier for the batched correlator/corroboration and for propagation
    # to forked agents' reflect() (see Mesh fork logic). Workers use mesh._llm
    # (fast) for per-signal fact extraction.
    mesh._correlator_llm = mesh_llm

    # Warm-start dedup index from previous run
    if DEDUP_INDEX_PATH.exists() and config.pipeline.dedup_enabled:
        loaded = mesh._dedup_index.load(DEDUP_INDEX_PATH)
        if loaded:
            info(f"  Loaded dedup index: {len(mesh._dedup_index)} fingerprints from previous run")

    # Pre-seed workers from configured contexts
    for name, ctx_cfg in config.contexts.items():
        ctx_id = UUID(ctx_map[name]) if name in ctx_map else _stable_uuid(name)
        ctx_map[name] = str(ctx_id)

        saved_ctx = state.get("context_state", {}).get(name, {})
        ctx = Context(id=ctx_id, capacity=wc.capacity, business_weight=ctx_cfg.business_weight)
        ctx.terms = set(ctx_cfg.terms)

        if saved_ctx:
            ctx.source_counts = dict(saved_ctx.get("source_counts", {}))
            ctx.author_counts = dict(saved_ctx.get("author_counts", {}))
            # DESIGN DECISION: signal_count intentionally NOT restored.
            #
            # Workers start operationally fresh each session (0 fullness,
            # low thresholds) but retain learned terms and affinities from
            # previous sessions. This means:
            #
            # Benefit: workers are maximally accepting at session start,
            # so signals don't bounce through BFS and spawn unnecessary
            # gap workers. The learned vocabulary (terms + bloom) ensures
            # signals still route to the right workers.
            #
            # Tradeoff: the fullness-based adaptive threshold (which rises
            # with signal_count / capacity) resets to its base value each
            # session. A worker that was highly selective (high fullness)
            # in the previous run becomes accepting again. This is
            # intentional — fresh sessions should re-earn selectivity
            # based on current signal volume, not inherit stale thresholds.
            #
            # If signal_count WERE restored, workers would start at high
            # thresholds, reject most signals, trigger excessive gap
            # spawning, and produce a fragmented mesh on first run.
            ctx.terms.update(saved_ctx.get("learned_terms", []))

        # Populate bloom from all terms (config + learned) for overlap scoring
        ctx.term_bloom.add_many(ctx.terms)
        ctx.last_signal = datetime.now(timezone.utc)

        worker = WorkerNode(
            ctx,
            base_threshold=wc.base_threshold,
            max_threshold=wc.max_threshold,
            threshold_curve=wc.threshold_curve,
            high_relevance_offset=wc.high_relevance_offset,
            llm=mesh_llm,
        )
        mesh.add_worker(worker)

        # Bind agents to this worker's context
        for agent in agents_list:
            if ctx_id in agent.contexts:
                ctx.active_agents.add(agent.id)

    # Connect all seeded workers as neighbors (mesh topology)
    worker_ids = [w.id for w in mesh.workers]
    for i, wid_a in enumerate(worker_ids):
        for wid_b in worker_ids[i + 1:]:
            mesh.connect(wid_a, wid_b)

    # Warm restart: restore self-organized state from previous session.
    # Workers get routing stats, agents get evolved competencies,
    # compression tracker gets ND trend memory.
    from stigmergy.mesh.state_store import (
        load_mesh_state,
        restore_agent_state,
        restore_compression_tracker,
        restore_worker_state,
    )
    _saved_mesh_state = load_mesh_state()
    if _saved_mesh_state:
        for w in mesh.workers:
            restore_worker_state(w, _saved_mesh_state)
        for agent in agents_list:
            restore_agent_state(agent, _saved_mesh_state)
        restore_compression_tracker(_agent_comp_tracker, _saved_mesh_state)

    # Budget — auto-detect pricing from model, track actual API token counts
    budget = DollarBudgetTracker(
        daily_cap_usd=config.budget.daily_cap_usd,
        hourly_cap_usd=config.budget.hourly_cap_usd,
        input_cost_per_million=config.budget.input_cost_per_million,
        output_cost_per_million=config.budget.output_cost_per_million,
        warn_at_percent=config.budget.warn_at_percent,
    )
    budget.set_model_pricing(config.llm.model)
    if mesh_llm is not None:
        # Track each tier at its own model's pricing. add_priced_llm dedupes by
        # client identity, so a single-tier setup (fast_llm is mesh_llm) registers once.
        budget.add_priced_llm(mesh_llm, config.llm.model)
        budget.add_priced_llm(fast_llm, getattr(config.llm, "fast_model", "") or config.llm.model)

    # Signal cache for cross-signal observation
    signal_cache = SignalCache(capacity=500)

    # Policy engine: trace store + structure graph + activity budgets
    from stigmergy.policy.budgets import ActivityBudget, BudgetConfig
    from stigmergy.policy.engine import PolicyEngine
    from stigmergy.policy.structure import StructureGraph
    from stigmergy.policy.traces import TraceStore

    trace_store = TraceStore()
    structure_graph = StructureGraph()

    # Auto-discover from repos directory if available.
    # Search upward from CWD — stigmergy lives at tools/stigmergy/, repos at ../../repos/
    repos_dir = None
    repo_names = set()
    if config.sources.github.enabled:
        repo_names = {r.split("/")[-1] for r in config.sources.github.repos}
    for candidate in [Path("repos"), Path("../../repos"), Path("../repos")]:
        if candidate.is_dir() and any((candidate / name).is_dir() for name in repo_names):
            repos_dir = candidate
            break

    if repos_dir is not None:
        edge_count = structure_graph.discover_from_repos(repos_dir)
        if edge_count > 0:
            structure_graph.save()
    else:
        structure_graph.load()  # Try cached graph

    # Ensure all configured repos are nodes
    if config.sources.github.enabled:
        structure_graph.discover_from_config(config.sources.github.repos)

    activity_budget = ActivityBudget()
    policy_engine = PolicyEngine(
        graph=structure_graph,
        budget=activity_budget,
        trace_store=trace_store,
        llm=fast_llm,
    )

    # Spectral fingerprint store — per-signal activation vectors for
    # vocabulary-independent structural coupling detection
    from stigmergy.mesh.fingerprints import FingerprintStore
    fingerprint_store = FingerprintStore()

    # Communication graph — person-to-person edges extracted from signals.
    # Enables bridge distance (Discovery), constraint (Normalized Deviance),
    # and structural hole (Channel migration) detection.
    from stigmergy.graph.graph import CommunicationGraph
    from stigmergy.identity import IdentityResolver
    from stigmergy.identity.store import AliasStore
    from stigmergy.identity.providers.static import StaticCSVProvider, StaticLinearMdProvider
    from stigmergy.identity.providers.github import GitHubOrgProvider
    from stigmergy.identity.providers.linear import LinearUsersProvider
    from stigmergy.identity.providers.slack import SlackUsersProvider

    # Build identity providers — static first (authoritative), then live
    providers: list = []
    for candidate in [Path("config"), Path("../../config"), Path("../config")]:
        team_csv = candidate / "team_roster.csv"
        alias_csv = candidate / "email_aliases.csv"
        linear_md = candidate / "linear-reference.md"
        if team_csv.exists():
            providers.append(StaticCSVProvider(team_csv, alias_csv if alias_csv.exists() else None))
            if linear_md.exists():
                providers.append(StaticLinearMdProvider(linear_md))
            break

    # Supplementary local identities (non-team members, display name variants)
    local_csv = Path("config/local_identities.csv")
    if local_csv.exists():
        providers.append(StaticCSVProvider(local_csv))

    # Live providers — enabled by config or environment
    identity_config = config.identity
    if identity_config.github_org:
        providers.append(GitHubOrgProvider(identity_config.github_org))
    providers.append(LinearUsersProvider(config.sources.linear.api_key or None))
    providers.append(SlackUsersProvider(config.sources.slack.bot_token or None))

    # Alias store for runtime-learned aliases
    alias_store: AliasStore | None = None
    if identity_config.persist_learned_aliases:
        alias_store = AliasStore()

    resolver = IdentityResolver(providers=providers, store=alias_store)

    comm_graph = CommunicationGraph(resolver)
    graph_path = Path(".stigmergy/graph.json")
    comm_graph.load(graph_path)
    mesh.set_graph(comm_graph)

    # Attention model — per-person P(knows) and surfacing decisions
    attention_model = None
    if config.attention.enabled:
        from stigmergy.attention.exposure import ExposureExtractor
        from stigmergy.attention.model import AttentionModel, AttentionParameters

        attn_params = AttentionParameters(
            decay_half_life_days=config.attention.decay_half_life_days,
            channel_size_reference=config.attention.channel_size_reference,
            volume_discount_half=config.attention.volume_discount_half,
            importance_weight=config.attention.importance_weight,
            annoyance_weight=config.attention.annoyance_weight,
            calibration_lr_up=config.attention.calibration_lr_up,
            calibration_lr_down=config.attention.calibration_lr_down,
        )
        attention_model = AttentionModel(resolver=resolver, params=attn_params)
        attn_path = Path(".stigmergy/attention.json")
        attention_model.load(attn_path)
        exposure_extractor = ExposureExtractor(resolver=resolver)
        mesh.set_attention_model(attention_model, exposure_extractor)

    # Stability tracker — Lyapunov validation (Section 4.4)
    from stigmergy.mesh.stability import StabilityTracker
    stability_tracker = StabilityTracker()
    mesh.set_stability_tracker(stability_tracker)

    # Persist context IDs
    state["context_ids"] = ctx_map
    _save_state(state)

    return mesh, agent_registry, budget, ctx_map, signal_cache, annotation_store, policy_engine, fingerprint_store, comm_graph, alias_store, stability_tracker, attention_model, discovery_store, health_pulse_collector, meta_modeler, field_engine


def _refresh_repos_if_available(config: StigmergyConfig) -> None:
    """Git-fetch all repos if a repos directory is found."""
    from stigmergy.cli.discovery import refresh_repos

    repo_names = set()
    if config.sources.github.enabled:
        repo_names = {r.split("/")[-1] for r in config.sources.github.repos}
    repos_dir = None
    for candidate in [Path("repos"), Path("../../repos"), Path("../repos")]:
        if candidate.is_dir() and any((candidate / name).is_dir() for name in repo_names):
            repos_dir = candidate
            break
    if repos_dir is None:
        return

    def _progress(name: str, done: int, total: int) -> None:
        bar_width = 20
        filled = int(bar_width * done / total)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r  [{bar}] {done}/{total} repos fetched  {name:<30s}", end="", flush=True)
        if done == total:
            print()

    info("Refreshing git repos...")
    results = refresh_repos(repos_dir, progress_callback=_progress)
    ok = sum(1 for v in results.values() if v)
    fail = len(results) - ok
    if fail:
        warn(f"  {fail}/{len(results)} repos failed to fetch")
    else:
        info(f"  {ok} repos fetched successfully")


def _resolve_backfill_since(
    state: dict,
    since_days: int | None,
    now: datetime | None = None,
) -> tuple[datetime, list[str]]:
    """Resolve the backfill 'since' cutoff (pure — no I/O, for testability).

    Priority: checkpoint (resume an interrupted run) > --since flag > last_run
    > 30-day default. On the automated path (since_days is None) the result is
    clamped to MAX_BACKFILL_DAYS so a stale checkpoint can't drag the window
    months back and stall the run. An explicit --since is honored verbatim.

    Returns (since, notes) where notes are human-readable lines the caller
    should info()-log.
    """
    now = now or datetime.now(timezone.utc)
    notes: list[str] = []
    checkpoint = state.get("checkpoint")

    def _resume_note(cp: dict) -> str:
        return (
            f"Resuming from checkpoint: {cp['signals_processed']}/{cp['total_signals']} "
            f"signals processed (${cp.get('spend_usd', 0):.2f} spent)"
        )

    if checkpoint and since_days is not None:
        # --since sets the window, but resume from the checkpoint if it is newer.
        since = now - timedelta(days=since_days)
        resume_ts = datetime.fromisoformat(checkpoint["timestamp"])
        if resume_ts > since:
            notes.append(_resume_note(checkpoint))
            since = resume_ts
    elif since_days is not None:
        since = now - timedelta(days=since_days)
    elif checkpoint:
        since = datetime.fromisoformat(checkpoint["timestamp"])
        notes.append(_resume_note(checkpoint))
    else:
        last_run = state.get("last_run")
        since = datetime.fromisoformat(last_run) if last_run else now - timedelta(days=30)

    if since_days is None:
        floor = now - timedelta(days=MAX_BACKFILL_DAYS)
        if since < floor:
            notes.append(
                f"Clamping backfill window: checkpoint/last_run was {since.date()}, "
                f"using {floor.date()} (MAX_BACKFILL_DAYS={MAX_BACKFILL_DAYS}). "
                f"Older unprocessed signals are skipped so the run can converge."
            )
            since = floor

    return since, notes


# Short pure-acknowledgement phrases — meaningless on their own. Compared after
# lowercasing and stripping surrounding punctuation/emoji.
_ACK_PHRASES = frozenset({
    "ok", "okay", "k", "kk", "kthx", "thanks", "thank you", "thx", "ty", "tysm",
    "ty!", "np", "no problem", "yes", "yep", "yeah", "ya", "no", "nope", "sure",
    "sounds good", "sgtm", "lgtm", "+1", "-1", "done", "done!", "ditto", "same",
    "agreed", "agree", "nice", "cool", "great", "awesome", "lol", "haha", "ack",
    "acked", "got it", "gotit", "yw", "ditto.", "+1!", "this", "indeed", "right",
})


def _is_noise_signal(sig) -> bool:
    """True if a signal is obvious low-value chaff that should skip the mesh.

    Conservative: only drops empties, pure emoji/punctuation, short pure-acks,
    and Slack system leftovers. Anything with real words and length stays.
    """
    content = (getattr(sig, "content", "") or "").strip()
    if not content:
        return True
    # Slack system leftovers (defence in depth; the adapter drops most).
    meta = getattr(sig, "metadata", None) or {}
    if meta.get("subtype") in (
        "channel_join", "channel_leave", "bot_add", "bot_remove",
        "channel_topic", "channel_purpose", "channel_name",
        "pinned_item", "unpinned_item",
    ):
        return True
    # No alphanumeric characters at all — pure emoji / reaction / punctuation.
    if not any(c.isalnum() for c in content):
        return True
    # Short pure-acknowledgements.
    normalized = content.lower().strip(" \t\n.!?,:;)(_*~`\"'")
    if len(content) <= 16 and normalized in _ACK_PHRASES:
        return True
    return False


async def _run_once_mesh(config: StigmergyConfig, live: bool, since_days: int | None = None) -> int:
    """Single-pass mesh mode: backfill sources, route through mesh, print results."""
    from stigmergy.mesh.temporal import TemporalScanner

    # Handle SIGTERM (from timeout(1)) — set a flag instead of raising,
    # so the signal routing loop can break cooperatively and save state.
    # Raising KeyboardInterrupt from a signal handler inside asyncio.run()
    # cancels all tasks before the inner except block can save checkpoints.
    _sigterm_received = False
    _original_sigterm = signal_mod.getsignal(signal_mod.SIGTERM)

    def _sigterm_handler(signum, frame):
        nonlocal _sigterm_received
        _sigterm_received = True

    signal_mod.signal(signal_mod.SIGTERM, _sigterm_handler)

    heading("stigmergy run --once (mesh)")
    print()

    _refresh_repos_if_available(config)

    mesh, agent_registry, budget, ctx_map, signal_cache, annotation_store, policy_engine, fingerprint_store, comm_graph, alias_store, stability_tracker, attention_model, discovery_store, health_pulse_collector, meta_modeler, field_engine = bootstrap_mesh(config)
    # Extract compression tracker from agents (created in bootstrap, attached to each agent)
    _all_agents = list(agent_registry.all_agents())
    _agent_comp_tracker = getattr(_all_agents[0], "_compression_tracker", None) if _all_agents else None
    sources = _build_sources(config, live)

    # Preflight: verify LLM is reachable before committing to a full run.
    # Without LLM, agents fall back to mechanical heuristics — a useless run.
    if mesh._llm is not None:
        await _preflight_llm(mesh._llm)

    graph_info = policy_engine._graph.summary() if policy_engine._graph.node_count > 0 else "no structure graph"
    info(f"Mesh initialized: {mesh.worker_count} workers, {len(config.agents)} agents, {graph_info}")
    print()

    # Determine the "since" cutoff for signal fetching (resume > --since >
    # last_run > 30d), clamped to MAX_BACKFILL_DAYS on the automated path.
    # See _resolve_backfill_since for the rules (unit-tested there).
    state = _load_state()
    prev_sources = set(state.get("last_sources", []))
    since, _since_notes = _resolve_backfill_since(state, since_days)
    for _note in _since_notes:
        info(_note)

    # Collect all signals from all sources
    all_signals: list[Signal] = []
    for source_name, adapter, _is_live in sources:
        info(f"Connecting {source_name}{'  (live)' if _is_live else '  (mock)'}...")
        try:
            await adapter.connect()
        except (ConnectionError, OSError) as e:
            warn(f"  {source_name}: connection failed ({e}), skipping")
            continue
        # New sources get a wider initial backfill (7 days) instead of
        # the narrow last_run window, which may be only minutes old.
        if source_name not in prev_sources and since_days is None and prev_sources:
            source_since = datetime.now(timezone.utc) - timedelta(days=7)
            info(f"Backfilling {source_name} since {source_since.date()} (first run for this source)...")
        else:
            source_since = since
            info(f"Backfilling {source_name} since {source_since.date()}...")
        source_count = 0
        async for sig in adapter.backfill(source_since):
            all_signals.append(sig)
            source_count += 1
        extras = []
        if hasattr(adapter, "bot_skipped") and adapter.bot_skipped > 0:
            extras.append(f"{adapter.bot_skipped} bot filtered")
        if hasattr(adapter, "_stats") and adapter._stats:
            s = adapter._stats
            if s.get("thread_replies"):
                extras.append(f"{s['thread_replies']} from threads")
            if s.get("filtered"):
                extras.append(f"{s['filtered']} empty/system filtered")
        extra_note = f" ({', '.join(extras)})" if extras else ""
        info(f"  {source_name}: {source_count} signals fetched{extra_note}")

    # Noise pre-filter — drop low-value signals (empty, pure-emoji/reaction,
    # short acks like "ok"/"thanks"/"lgtm", system leftovers) BEFORE they incur
    # any per-signal LLM cost. Conservative by design: only obvious chaff. This
    # is the bulk cost lever on chatty Slack corpora. Disable with
    # STIGMERGY_DISABLE_PREFILTER=1.
    if not os.environ.get("STIGMERGY_DISABLE_PREFILTER"):
        _pre = len(all_signals)
        all_signals = [s for s in all_signals if not _is_noise_signal(s)]
        _dropped = _pre - len(all_signals)
        if _dropped:
            _pct = _dropped * 100 // max(_pre, 1)
            info(f"  Pre-filter: dropped {_dropped} noise signals ({_pct}%); {len(all_signals)} remain")

    info(f"\nRouting {len(all_signals)} signals through mesh...")
    separator()

    # Sort chronologically (oldest first) — workers learn progressively,
    # keeping temporal_proximity high. Newest-first kills older signal scores.
    all_signals.sort(key=lambda s: s.timestamp)

    # === CAUCUS PHASE ===
    # Seed all workers before normal routing so entry selection distributes
    # properly. Without this, the first signal makes one worker "warm" and
    # that worker wins every subsequent entry selection (starvation).
    from stigmergy.mesh.caucus import Caucus, CaucusConfig
    from stigmergy.mesh.state_store import load_mesh_state, restore_run_stats

    _prev_run_stats = restore_run_stats(load_mesh_state())
    _caucus_elapsed = 0.0
    if _prev_run_stats and _prev_run_stats.timestamp_iso:
        try:
            _prev_ts = datetime.fromisoformat(_prev_run_stats.timestamp_iso)
            _caucus_elapsed = (datetime.now(timezone.utc) - _prev_ts).total_seconds()
        except (ValueError, TypeError):
            pass

    _caucus = Caucus(
        mesh, mesh._weights, CaucusConfig(),
        prev_stats=_prev_run_stats, elapsed_seconds=_caucus_elapsed,
    )
    _caucus_size = _caucus.compute_size(len(all_signals))

    if _caucus_size > 0 and len(all_signals) >= _caucus_size and mesh.worker_count > 1:
        _caucus_signals = all_signals[:_caucus_size]
        _caucus_result = _caucus.run(_caucus_signals)
        info(f"  Caucus: seeded {mesh.worker_count} workers with {_caucus_result.caucus_size} signals")
        if _caucus_result.suggested_worker_count is not None:
            info(f"  Caucus suggests {_caucus_result.suggested_worker_count} workers")
    # All signals still go through mesh.ingest() below — caucus only seeds terms

    # Feed signals to mesh
    tracker = NoveltyTracker()
    insight_dedup = InsightDeduplicator()
    traces = []
    all_insights: list = []  # Collect all insights for Key Findings summary
    raw_insight_total = 0  # Track pre-dedup count for funnel metrics
    total_signals = len(all_signals)
    _mesh_start_time = time.monotonic()
    _checkpoint_interval = 50  # Save progress every N signals
    _last_checkpoint_idx = 0

    # Cadence monitor — detects worker silence and injects synthetic signals
    from stigmergy.mesh.stability import CadenceMonitor
    cadence_monitor = CadenceMonitor(mesh)
    _last_processed_ts: datetime | None = None
    stopped_reason = "complete"
    try:
      for sig_idx, sig in enumerate(all_signals, 1):
        if _sigterm_received:
            warn("SIGTERM received (timeout). Saving progress...")
            stopped_reason = "sigterm"
            break

        if not budget.can_spend():
            warn("Budget exhausted. Stopping.")
            stopped_reason = "budget_exhausted"
            break

        for w in budget.check_warnings():
            warn(w)

        # Progress indicator on every signal — diagnose hangs
        elapsed = time.monotonic() - _mesh_start_time
        rate = sig_idx / max(elapsed, 0.01)
        eta_min = int((total_signals - sig_idx) / rate / 60) if rate > 0 else 0
        print(f"\r  [{sig_idx}/{total_signals}] ${budget.daily_spend:.2f} spent | ~{eta_min}m remaining   ", end="", flush=True)

        trace = await mesh.ingest(sig)
        traces.append(trace)

        # Record spectral fingerprint (activation vector across workers)
        if not trace.duplicate and trace.familiarity_scores:
            from stigmergy.mesh.fingerprints import fingerprint_from_trace
            fingerprint_store.record(fingerprint_from_trace(trace, sig))

        # Track actual API token spend (not estimates)
        budget.sync_from_llm()

        # Agent reflection — spectral observation. Skip near-duplicates: the
        # signal was already routed and reflected when first seen, so reflecting
        # again is pure LLM spend for a finding we already have.
        insights, _raw = [], 0
        if not trace.duplicate:
            insights, _raw = await _reflect_and_cache(sig, trace, mesh, agent_registry, signal_cache, comm_graph)
            raw_insight_total += _raw
            budget.sync_from_llm()  # Reflect also makes LLM calls

        # Policy evaluation — structural signals + agent judgment
        from stigmergy.policy.traces import TraceEvent as PolicyTraceEvent
        policy_interventions = []
        trace_event = PolicyTraceEvent.from_signal(sig)
        try:
            policy_interventions = await policy_engine.evaluate(trace_event)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise  # Re-raise KeyboardInterrupt, CancelledError, SystemExit

        # Determine best learning mode for display
        if trace.duplicate:
            mode = "duplicate"
        elif trace.hops:
            # Use the best learning mode from accepted workers
            accepted_hops = [h for h in trace.hops if h.accepted]
            if accepted_hops:
                mode = accepted_hops[0].learning_mode
            else:
                mode = trace.hops[0].learning_mode
        else:
            mode = "none"

        # Novelty-aware output: detail first, compact next, then suppress
        category = f"{sig.source}:{mode}"
        novelty = tracker.check(category)

        if novelty != "suppress" and (not config.output.surface_only or (trace.consensus and trace.consensus.action == AssessmentAction.SURFACE)):
            if novelty == "detail":
                worker_labels = {str(w.id)[:8]: w.label[:20] for w in mesh.workers}
                consensus_action = trace.consensus.action if trace.consensus else None
                consensus_weight = trace.consensus.total_weight if trace.consensus else 0.0
                mesh_signal_detail(
                    source=sig.source,
                    channel=sig.channel,
                    author=sig.author,
                    content_preview=sig.content,
                    hops=trace.hops,
                    consensus_action=consensus_action,
                    consensus_weight=consensus_weight,
                    output_delivered=trace.output_delivered,
                    worker_labels=worker_labels,
                )
            else:
                mesh_signal_line(
                    source=sig.source,
                    channel=sig.channel,
                    author=sig.author,
                    hops=trace.total_hops,
                    workers_accepted=len(trace.accepted_workers),
                    learning_mode=mode,
                    content_preview=sig.content,
                )
            for intv in policy_interventions:
                intervention_line(intv)

        # Insights: display each unique insight inline once (dedup),
        # but always collect all instances for the end-of-run report.
        for ins in insights:
            if insight_dedup.should_display(ins):
                insight_line(ins)
        all_insights.extend(insights)

        # Track progress for checkpoint
        _last_processed_ts = sig.timestamp

        # Periodic checkpoint — save progress so interrupted runs can resume
        if sig_idx - _last_checkpoint_idx >= _checkpoint_interval:
            _last_checkpoint_idx = sig_idx

            # Cadence monitor: check for worker silence at each checkpoint
            cadence_signals = cadence_monitor.check(sig_idx)
            for csig in cadence_signals:
                ctrace = await mesh.ingest(csig)
                traces.append(ctrace)
                # Run agent reflection so the correlator can reason about silence
                c_insights, c_raw = await _reflect_and_cache(
                    csig, ctrace, mesh, agent_registry, signal_cache, comm_graph,
                )
                raw_insight_total += c_raw
                all_insights.extend(c_insights)
                for cins in c_insights:
                    insight_line(cins)
                info(f"  [cadence] {csig.content[:80]}...")

            # Health pulse: collect mesh vitals at each checkpoint
            from stigmergy.mesh.insight_store import FindingRegistry as _FReg
            _finding_reg = _FReg() if sig_idx == _checkpoint_interval else getattr(_save_checkpoint, '_finding_reg', None)
            pulse = health_pulse_collector.collect(
                sig_idx, mesh,
                stability_tracker=stability_tracker,
                agents=list(agent_registry.all_agents()),
                finding_registry=_finding_reg,
                compression_tracker=_agent_comp_tracker if '_agent_comp_tracker' in dir() else None,
            )

            # Unity field engine: mechanical self-regulation (when enabled)
            # Runs before meta-modeler so CERTX state can enrich the prompt.
            if field_engine is not None:
                _spectral_snapshot = stability_tracker if stability_tracker is not None else None
                _field_state = field_engine.tick(
                    pulse=pulse, mesh=mesh,
                    agents=list(agent_registry.all_agents()),
                    stability_tracker=stability_tracker,
                    spectral=_spectral_snapshot,
                    compression_tracker=_agent_comp_tracker if '_agent_comp_tracker' in dir() else None,
                )
                if meta_modeler is not None and hasattr(meta_modeler, "set_field_state"):
                    meta_modeler.set_field_state(_field_state)

            # Meta-modeler: evaluate mesh health (when enabled)
            if meta_modeler is not None:
                assessment = await meta_modeler.evaluate(pulse)
                if assessment is not None:
                    meta_modeler.apply_recommendations(
                        assessment, agents=list(agent_registry.all_agents()),
                    )

            _save_checkpoint(
                mesh, ctx_map, annotation_store, comm_graph,
                policy_engine, _last_processed_ts, sig_idx, total_signals,
                budget, alias_store=alias_store, stability_tracker=stability_tracker,
                attention_model=attention_model,
            )
    except (SystemExit, KeyboardInterrupt) as exc:
        stopped_reason = "circuit_breaker" if isinstance(exc, SystemExit) else "interrupted"
        warn(f"Run stopped ({stopped_reason}). Saving progress...")
        if _last_processed_ts is not None:
            _save_checkpoint(
                mesh, ctx_map, annotation_store, comm_graph,
                policy_engine, _last_processed_ts, sig_idx, total_signals,
                budget, alias_store=alias_store, stability_tracker=stability_tracker,
                attention_model=attention_model,
            )
        if isinstance(exc, SystemExit):
            # Re-print the circuit breaker message but don't re-raise —
            # we want to fall through to summary/save below
            error(str(exc))
        else:
            print()  # Clean line after ^C

    # Save checkpoint for non-exception early exits (sigterm, budget_exhausted)
    if stopped_reason not in ("complete", "interrupted", "circuit_breaker") and _last_processed_ts is not None:
        _save_checkpoint(
            mesh, ctx_map, annotation_store, comm_graph,
            policy_engine, _last_processed_ts, sig_idx, total_signals,
            budget, alias_store=alias_store, stability_tracker=stability_tracker,
            attention_model=attention_model,
        )

    tracker.flush()
    insight_dedup.flush()
    print("\r" + " " * 60 + "\r", end="", flush=True)  # Clear progress line

    # Prune workers that accepted nothing during backfill
    pruned = mesh.prune_idle_workers(min_workers=len(config.contexts))
    if pruned:
        info(f"  Pruned {len(pruned)} idle workers (0 signals accepted)")

    # Show mesh topology with agent-generated labels when LLM available
    # Skip LLM calls if run was interrupted
    if stopped_reason == "interrupted":
        mesh_topology(mesh.workers)
    else:
        agent_labels = {}
        for w in mesh.workers:
            try:
                al = await w.agent_label()
                if al != w.label:  # only use if agent produced something different
                    agent_labels[str(w.id)[:8]] = al
            except Exception:
                pass
        mesh_topology(mesh.workers, labels=agent_labels or None)

    # Budget
    budget_line(budget.daily_spend, budget.daily_cap_usd, budget.hourly_spend, budget.hourly_cap_usd)

    # Summary
    accepted = sum(1 for t in traces if t.accepted_workers)
    duplicates = sum(1 for t in traces if t.duplicate)
    deepest = min((sig.timestamp for sig in all_signals), default=None)
    mesh_scan_summary(
        signals_processed=len(traces),
        signals_accepted=accepted,
        duplicates=duplicates,
        windows_scanned=1,  # single-pass = 1 window
        deepest_timestamp=deepest.isoformat() if deepest else "n/a",
        stopped_reason=stopped_reason,
        worker_count=mesh.worker_count,
        avg_fullness=mesh.avg_fullness,
        total_cost=budget.total_spend,
    )

    # Policy summary
    policy_summary(
        trace_count=policy_engine.trace_count,
        intervention_count=policy_engine.intervention_count,
        budget_exceeded=policy_engine._budget.all_exceeded(),
        spectral_snapshot=policy_engine.spectral_snapshot,
    )

    # Structural coupling detection — the paper's core claim
    _print_coupling_summary(fingerprint_store)

    # Agent Intelligence Report — verify agents are using LLM, not mechanical
    agent_diags = [a.diagnostics() for a in agent_registry.all_agents()]
    intelligence_report(agent_diags)

    # S(f) scoring — compute scoring function for each insight
    from stigmergy.mesh.scoring import score_finding
    for ins in all_insights:
        components = score_finding(ins, comm_graph)
        ins.details["sf_score"] = components.score
        ins.details["sf_components"] = components.to_dict()

    # Per-finding Normalized Deviance indicators — attach compression_index
    # and action_correlation to each insight for display in Key Findings.
    nd_config = getattr(config, "normalized_deviance", None)
    _nd_enabled = nd_config.enabled if nd_config else True
    nd_action_corrs = None  # populated here if enabled; reused by ND dashboard below

    if _nd_enabled:
        from stigmergy.attention.linguistic import analyze_text as _pf_comp
        for ins in all_insights:
            ins.details["compression_index"] = _pf_comp(ins.summary).compression_index

        # Early action-correlation computation for per-finding annotation
        _nd_action_on_early = nd_config.action_correlation_enabled if nd_config else True
        if _nd_action_on_early and all_signals:
            from stigmergy.attention.action_correlation import compute_action_correlations as _cac_early
            from stigmergy.primitives.signal import extract_terms as _et_early
            _ac_sigs_early = [
                {"content": sig.content, "source": sig.source, "metadata": sig.metadata}
                for sig in all_signals
            ]
            nd_action_corrs = _cac_early(_ac_sigs_early)
            # Attach per-finding action-correlation
            for ins in all_insights:
                _f_terms = _et_early(ins.summary)
                _best_ac = -1.0
                for _ac_r in nd_action_corrs:
                    if _ac_r.topic in _f_terms:
                        _best_ac = max(_best_ac, _ac_r.ratio)
                if _best_ac >= 0:
                    ins.details["action_correlation"] = _best_ac

    # Key Findings — the system's primary business-value output.
    # Aggregates all LLM correlator insights from the entire run.
    display_unique = findings_summary(all_insights)

    # Per-person attention model — ranked findings for each target person
    if attention_model is not None and config.attention.target_persons:
        from stigmergy.attention.surfacing import rank_for_person
        from stigmergy.mesh.insight_store import _finding_hash
        from stigmergy.primitives.signal import extract_terms

        finding_dicts = []
        for ins in all_insights:
            actors = ins.details.get("actors", [])
            fh = _finding_hash(ins.type, ins.summary, actors)
            finding_dicts.append({
                "finding_hash": fh,
                "summary": ins.summary,
                "type": ins.type,
                "sf_score": ins.details.get("sf_score", 0.0),
                "terms": frozenset(extract_terms(ins.summary)),
                "signal_ids": [str(s) for s in ins.signal_ids],
            })

        for person_id in config.attention.target_persons:
            canonical = comm_graph._resolver.resolve(person_id)
            decisions = rank_for_person(finding_dicts, canonical, attention_model)
            attention_findings_summary(decisions, canonical)

    # Insight dedup funnel — visibility into collapse at each stage
    insight_funnel(raw_insight_total, len(all_insights), display_unique)

    # Phase 2: Cross-signal corroboration — findings routed back through
    # the mesh for independent assessment by multiple workers.
    from stigmergy.mesh.corroboration import run_corroboration
    quorum_findings, corr_metrics = await run_corroboration(
        all_insights,
        list(mesh.workers),
        llm=getattr(mesh, "_correlator_llm", None) or mesh._llm,
        comm_graph=comm_graph,
    )
    corroboration_summary(quorum_findings, corr_metrics)

    # Competency reinforcement — close the open loop
    mm_config = getattr(config, "meta_modeler", None)
    if mm_config is None or mm_config.competency_reinforcement:
        from stigmergy.mesh.corroboration import reinforce_competencies
        reinforce_competencies(quorum_findings, list(agent_registry.all_agents()))

    # Unity calibration: feed corroboration outcomes to field engine
    if field_engine is not None:
        for qf in quorum_findings:
            was_corr = qf.quorum_met
            field_engine.record_prediction(
                str(qf.original.agent_id)[:8],
                qf.original.confidence,
                was_corr,
            )

    # Persist insights to .stigmergy/insights.jsonl
    from stigmergy.mesh.insight_store import InsightStore, FindingRegistry
    insight_store = InsightStore()
    for ins in all_insights:
        sf = ins.details.get("sf_components", {})
        insight_store.record(ins, score=ins.details.get("sf_score", 0.0), scoring_components=sf)
    insight_count = insight_store.save()
    if insight_count > 0:
        info(f"  Persisted {insight_count} insights to .stigmergy/insights.jsonl (run {insight_store.run_id})")

    # Update finding registry — cross-run dedup and decay tracking.
    # Enrich each finding with per-finding ND metadata computed above.
    finding_registry = FindingRegistry()

    # Build per-finding compression index and action-correlation lookups
    _per_finding_compression: dict[str, float] = {}
    _per_finding_ac: dict[str, float] = {}
    if _nd_enabled and all_signals:
        from stigmergy.attention.linguistic import analyze_text as _pf_analyze
        from stigmergy.primitives.signal import extract_terms as _pf_extract
        for ins in all_insights:
            _ci = _pf_analyze(ins.summary).compression_index
            _per_finding_compression[ins.summary[:80]] = _ci
            # Match finding terms against action-correlation results
            if nd_action_corrs:
                _f_terms = _pf_extract(ins.summary)
                _best_ac = -1.0
                for ac in nd_action_corrs:
                    if ac.topic in _f_terms:
                        _best_ac = max(_best_ac, ac.ratio)
                if _best_ac >= 0:
                    _per_finding_ac[ins.summary[:80]] = _best_ac

    buffered = [
        {
            "type": ins.type,
            "summary": ins.summary,
            "confidence": ins.confidence,
            "score": ins.details.get("sf_score", 0.0),
            "details": ins.details,
            "signal_ids": [str(s) for s in ins.signal_ids],
            "compression_index": _per_finding_compression.get(ins.summary[:80], 0.0),
            "action_correlation": _per_finding_ac.get(ins.summary[:80], -1.0),
        }
        for ins in all_insights
    ]
    changed_findings = finding_registry.ingest_insights(buffered, insight_store.run_id)
    # New findings = surfaced for the first time ever (times_surfaced == 1).
    # Zero new findings across runs is the signature of a stuck pipeline
    # (the Feb–Jun freeze re-surfaced the same 20 findings every day) — used
    # below for the stale-findings guard and the end-of-run STATUS line.
    new_finding_count = sum(1 for f in changed_findings if getattr(f, "times_surfaced", 0) == 1)

    # Auto-detect state changes: check if this run's signals indicate
    # action on deferred/normalized findings (e.g., commits by actors
    # named in a finding, or signals matching finding topics).
    signal_dicts = [
        {"author": sig.author, "content": sig.content, "source": sig.source}
        for sig in all_signals
    ]
    state_changes = finding_registry.detect_state_changes(signal_dicts)
    if state_changes > 0:
        info(f"  Auto-detected {state_changes} finding state change(s)")

    # Re-emission engine — check for findings that should be retriggered
    # by context changes (new evidence, compound risk, decay escalation).
    from stigmergy.attention.reemission import evaluate_reemissions
    from stigmergy.primitives.signal import extract_terms as _extract_terms

    all_signal_terms: set[str] = set()
    signal_sources: set[str] = set()
    signal_channels: set[str] = set()
    for sig in all_signals:
        all_signal_terms.update(_extract_terms(sig.content))
        signal_sources.add(sig.source.value if hasattr(sig.source, "value") else str(sig.source))
        signal_channels.add(sig.channel)

    all_registry_findings = list(finding_registry._findings.values())
    reemission_events = evaluate_reemissions(
        all_registry_findings,
        signal_terms=frozenset(all_signal_terms) if all_signal_terms else None,
        signal_source=next(iter(signal_sources), ""),
        signal_channel=next(iter(signal_channels), ""),
    )
    if reemission_events:
        reemission_summary(reemission_events)

    # Portfolio risk — cluster findings by shared actors/channels/terms
    # and surface compound risk that exceeds threshold even when no
    # individual finding does.
    from stigmergy.attention.portfolio import build_portfolio
    from stigmergy.mesh.insight_store import _finding_hash as _pfh

    portfolio_findings = []
    for ins in all_insights:
        actors = ins.details.get("actors", [])
        portfolio_findings.append({
            "finding_hash": _pfh(ins.type, ins.summary, actors),
            "summary": ins.summary,
            "type": ins.type,
            "sf_score": ins.details.get("sf_score", 0.0),
            "actors": actors,
            "channels": [sig.channel for sig in all_signals[:10]],  # approximate
        })
    if len(portfolio_findings) >= 2:
        risk_clusters = build_portfolio(portfolio_findings)
        if risk_clusters:
            portfolio_summary(risk_clusters)

    # Normalized Deviance detection — five paper-specified indicators.
    # All detectors are pure computation, no LLM calls.
    nd_config = getattr(config, "normalized_deviance", None)
    _nd_enabled = nd_config.enabled if nd_config else True

    if _nd_enabled and all_signals:
        nd_compression = None
        nd_action_corrs = None
        nd_migration = None
        nd_solo = None
        nd_phase_lock = None

        # 1. Linguistic compression — aggregate + per-channel rolling averages
        _nd_linguistic_on = nd_config.linguistic_enabled if nd_config else True
        nd_compression_trends = None
        if _nd_linguistic_on:
            from stigmergy.attention.linguistic import (
                CompressionTracker,
                analyze_text as _analyze_compression,
            )

            # Use agent-populated tracker if available (primary path),
            # fall back to regex analysis (mechanical fallback)
            _has_agent_data = (
                '_agent_comp_tracker' in dir()
                and _agent_comp_tracker is not None
                and len(_agent_comp_tracker.all_trends(min_samples=1)) > 0
            )

            if _has_agent_data:
                # Agent path: correlator already assessed linguistic quality
                _comp_tracker = _agent_comp_tracker
                # Aggregate batch-level stats from agent trends
                _all_trends = _comp_tracker.all_trends(min_samples=1)
                if _all_trends:
                    avg_comp = sum(t.current_index for t in _all_trends) / len(_all_trends)
                    avg_ld = sum(t.current_ld for t in _all_trends) / len(_all_trends)
                    avg_se = sum(t.current_se for t in _all_trends) / len(_all_trends)
                else:
                    avg_comp = avg_ld = avg_se = 0.0
                nd_compression = {
                    "compression_index": round(avg_comp, 4),
                    "hedging_density": 0.0,  # agent doesn't decompose
                    "passive_voice_ratio": 0.0,
                    "nominalization_rate": 0.0,
                    "specificity_score": round(1.0 - avg_comp, 4),
                    "lexical_diversity": round(avg_ld, 4),
                    "shannon_entropy": round(avg_se, 4),
                    "source": "agent",
                }
            else:
                # Regex fallback: no agent data, compute mechanically
                combined_text = ". ".join(sig.content[:200] for sig in all_signals[:500])
                _comp_result = _analyze_compression(combined_text)
                nd_compression = {
                    "compression_index": _comp_result.compression_index,
                    "hedging_density": _comp_result.hedging_density,
                    "passive_voice_ratio": _comp_result.passive_voice_ratio,
                    "nominalization_rate": _comp_result.nominalization_rate,
                    "specificity_score": _comp_result.specificity_score,
                    "lexical_diversity": _comp_result.lexical_diversity,
                    "shannon_entropy": _comp_result.shannon_entropy,
                    "temporal_orientation": _comp_result.temporal_orientation,
                    "bureaucratic_density": _comp_result.bureaucratic_density,
                    "source": "regex",
                }
                # Per-channel rolling averages (regex path)
                _comp_tracker = CompressionTracker()
                for sig in all_signals:
                    _sig_comp = _analyze_compression(sig.content)
                    _comp_tracker.update(
                        sig.channel, _sig_comp.compression_index,
                        ld=_sig_comp.lexical_diversity, se=_sig_comp.shannon_entropy,
                    )

            nd_compression_trends = _comp_tracker.increasing_trends(min_samples=3)
            if nd_compression_trends:
                nd_compression["increasing_channels"] = [
                    {"channel": t.key, "avg": t.rolling_avg, "peak": t.peak_index,
                     "samples": t.sample_count,
                     "ld": t.current_ld, "born_caged": t.is_born_caged}
                    for t in nd_compression_trends[:5]
                ]
            # Born Caged channels
            _born_caged = _comp_tracker.born_caged_channels(min_samples=3)
            if _born_caged:
                nd_compression["born_caged_channels"] = [
                    {"channel": t.key, "initial": t.initial_index,
                     "avg": t.rolling_avg}
                    for t in _born_caged[:5]
                ]

        # 2. Action-correlation ratio — reuse early computation if available
        _nd_action_on = nd_config.action_correlation_enabled if nd_config else True
        if _nd_action_on and nd_action_corrs is None:
            from stigmergy.attention.action_correlation import compute_action_correlations
            _ac_signals = [
                {"content": sig.content, "source": sig.source, "metadata": sig.metadata}
                for sig in all_signals
            ]
            nd_action_corrs = compute_action_correlations(_ac_signals)

        # 3. Solo ownership detection
        _nd_solo_on = nd_config.solo_ownership_enabled if nd_config else False
        if _nd_solo_on:
            from stigmergy.attention.solo_ownership import detect_solo_ownership
            _solo_signals = [
                {"content": sig.content, "source": sig.source,
                 "metadata": sig.metadata, "author": sig.author,
                 "channel": sig.channel, "timestamp": sig.timestamp}
                for sig in all_signals
            ]
            nd_solo = detect_solo_ownership(_solo_signals)

        # 4. Channel migration detection
        _nd_migration_on = nd_config.channel_migration_enabled if nd_config else False
        if _nd_migration_on:
            from stigmergy.attention.channel_migration import detect_channel_migration
            _mig_signals = [
                {"content": sig.content, "source": sig.source,
                 "metadata": sig.metadata, "channel": sig.channel}
                for sig in all_signals
            ]
            # Get channel metadata from Slack adapter if available
            _chan_meta = {}
            for _src_name, _src_adapter, _ in sources:
                if _src_name == "slack" and hasattr(_src_adapter, "_channel_meta"):
                    _chan_meta = _src_adapter._channel_meta
            nd_migration = detect_channel_migration(_mig_signals, channel_meta=_chan_meta)

        # 5. Phase-lock detection
        _nd_phase_on = nd_config.phase_lock_enabled if nd_config else False
        if _nd_phase_on:
            from stigmergy.attention.phase_lock import detect_phase_lock
            _ac_map = {}
            if nd_action_corrs:
                _ac_map = {ac.topic: ac.ratio for ac in nd_action_corrs}
            nd_phase_lock = detect_phase_lock(
                portfolio_findings, action_correlations=_ac_map,
            )

        normalized_deviance_summary(
            compression_signals=nd_compression,
            action_correlations=nd_action_corrs,
            migration_alerts=nd_migration,
            solo_alerts=nd_solo,
            phase_lock_alerts=nd_phase_lock,
        )

    finding_registry.save()
    decay_summary(finding_registry)
    identity_summary(comm_graph._resolver)

    # Persist corroboration results alongside findings
    if quorum_findings:
        import json
        corr_path = Path(".stigmergy") / "corroboration.json"
        corr_data = {
            "run_id": insight_store.run_id if insight_count > 0 else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": corr_metrics,
            "quorum_findings": [
                {
                    "summary": qf.original.summary[:200],
                    "type": qf.original.type,
                    "quorum_met": qf.quorum_met,
                    "quorum_confidence": round(qf.quorum_confidence, 4),
                    "corroboration_count": qf.corroboration_count,
                    "contradiction_count": qf.contradiction_count,
                    "total_consulted": qf.total_consulted,
                    "verdicts": [
                        {
                            "worker_id": str(v.worker_id)[:8],
                            "verdict": v.verdict,
                            "confidence": round(v.confidence, 4),
                            "reasoning": v.reasoning[:200],
                        }
                        for v in qf.verdicts
                    ],
                }
                for qf in quorum_findings
            ],
        }
        with open(corr_path, "w") as f:
            json.dump(corr_data, f, indent=2)
        info(f"  Persisted corroboration results to {corr_path}")

    # Save state — context learning, annotations, traces, budgets persist
    contexts = [w.context for w in mesh.workers]
    _save_context_state(contexts, {k: UUID(v) for k, v in ctx_map.items()})
    annotation_store.save()
    discovery_store.save()
    comm_graph.save(Path(".stigmergy/graph.json"))
    if alias_store is not None:
        alias_store.save()
    if stability_tracker is not None:
        stability_tracker.flush()
    # Mesh self-organized state — dice back in the cube
    from stigmergy.mesh.state_store import save_mesh_state
    from stigmergy.mesh.caucus import RunStats as _RunStats
    _run_stats = _RunStats(
        signals=len(traces),
        duration_seconds=time.monotonic() - _mesh_start_time,
        workers=mesh.worker_count,
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
    )
    save_mesh_state(
        workers=mesh.workers,
        agents=list(agent_registry.all_agents()),
        compression_tracker=_agent_comp_tracker,
        run_stats=_run_stats,
    )
    # Unity field engine state — PID, calibration, eigenmonitor history
    if field_engine is not None:
        field_engine.save_state()
    if attention_model is not None:
        attention_model.save(Path(".stigmergy/attention.json"))
    # Persist dedup index for cross-run duplicate detection
    if hasattr(mesh, '_dedup_index'):
        mesh._dedup_index.save(DEDUP_INDEX_PATH)
    # Save Slack user cache for next run
    for _src_name, _src_adapter, _src_live in sources:
        if _src_name == "slack" and hasattr(_src_adapter, "save_user_cache"):
            _src_adapter.save_user_cache()
    policy_engine._traces.save()
    policy_engine._budget.save()
    state = _load_state()
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["last_mode"] = "mesh"
    state["last_sources"] = [name for name, _, _ in sources]
    state["last_summary"] = {
        "signals_processed": len(traces),
        "accepted": accepted,
        "duplicates": duplicates,
        "workers": mesh.worker_count,
        "avg_fullness": round(mesh.avg_fullness, 3),
        "traces": policy_engine.trace_count,
        "interventions": policy_engine.intervention_count,
        "fingerprints": fingerprint_store.count,
        "insights": len(all_insights),
        "insights_raw": raw_insight_total,
        "insights_display_unique": display_unique,
        "stopped_reason": stopped_reason,
        "new_findings": new_finding_count,
        "spend_usd": round(budget.daily_spend, 4),
        "findings_registry": finding_registry.summary(),
    }
    # Stale-findings guard. A run can report "complete" yet surface nothing new
    # — that is exactly how the pipeline sat frozen for months without anyone
    # noticing. Track consecutive zero-new-finding runs and escalate the volume.
    if new_finding_count == 0:
        stale_streak = int(state.get("consecutive_no_new_findings", 0)) + 1
    else:
        stale_streak = 0
    state["consecutive_no_new_findings"] = stale_streak

    # Clear checkpoint on complete run; preserve on interrupted/budget runs
    # so the next run can resume from where we left off
    if stopped_reason == "complete":
        state.pop("checkpoint", None)
    _save_state(state)

    # Archive all outputs to timestamped directory
    archive_dir = _archive_run_outputs()
    if archive_dir:
        info(f"  Run archived to {archive_dir}")

    # Close LLM client so asyncio.run() can exit cleanly
    if hasattr(mesh, '_llm') and mesh._llm is not None and hasattr(mesh._llm, 'close'):
        await mesh._llm.close()
    _corr_llm = getattr(mesh, "_correlator_llm", None)
    if _corr_llm is not None and _corr_llm is not mesh._llm and hasattr(_corr_llm, "close"):
        await _corr_llm.close()

    # ── Machine-readable STATUS line + status-based exit code ──────────────
    # Make the run's health unmissable instead of buried at the end of a long
    # log, and propagate a non-zero exit so launchd/monitoring actually trips
    # when a run does not finish (root incident: exit stayed 0 while dead).
    exit_code = {
        "complete": 0,
        "sigterm": 2,          # killed by timeout(1) before draining the window
        "budget_exhausted": 3, # hit the daily/hourly cap mid-run
        "circuit_breaker": 4,
        "interrupted": 5,
    }.get(stopped_reason, 6)

    status_word = "OK" if exit_code == 0 else "INCOMPLETE"
    status_line = (
        f"STATUS: {status_word} | stopped_reason={stopped_reason} "
        f"| signals={len(traces)}/{total_signals} | new_findings={new_finding_count} "
        f"| spend=${budget.daily_spend:.2f} | exit={exit_code}"
    )
    (info if exit_code == 0 else error)(status_line)

    if stale_streak >= 1:
        warn(
            f"STALE FINDINGS: {stale_streak} consecutive run(s) produced 0 new findings. "
            f"If this persists, the pipeline is likely stuck (window/checkpoint/source access) "
            f"— investigate rather than trusting the unchanged output."
        )

    return exit_code


def _process_mesh_signal(
    sig: Signal,
    mesh,
    budget: DollarBudgetTracker,
    config: StigmergyConfig,
    counters: dict,
    verbose: bool = True,
) -> None:
    """Process a single signal through the mesh with rich output."""
    import asyncio

    trace = asyncio.get_event_loop().run_until_complete(mesh.ingest(sig))
    counters["total"] += 1

    budget.sync_from_llm()

    if trace.duplicate:
        counters["duplicates"] += 1
        if verbose:
            info(f"    [dup] {sig.content[:60]}")
        return

    if trace.accepted_workers:
        counters["accepted"] += 1

    # Build worker label map for display
    worker_labels = {}
    for w in mesh.workers:
        worker_labels[str(w.id)[:8]] = w.label[:20]

    consensus_action = None
    consensus_weight = 0.0
    if trace.consensus:
        consensus_action = trace.consensus.action
        consensus_weight = trace.consensus.total_weight
        if consensus_action == "surface" and trace.output_delivered:
            counters["surfaced"] += 1

    if verbose:
        mesh_signal_detail(
            source=sig.source,
            channel=sig.channel,
            author=sig.author,
            content_preview=sig.content,
            hops=trace.hops,
            consensus_action=consensus_action,
            consensus_weight=consensus_weight,
            output_delivered=trace.output_delivered,
            worker_labels=worker_labels,
        )
    else:
        if trace.hops:
            accepted_hops = [h for h in trace.hops if h.accepted]
            mode = accepted_hops[0].learning_mode if accepted_hops else trace.hops[0].learning_mode
        else:
            mode = "none"
        mesh_signal_line(
            source=sig.source,
            channel=sig.channel,
            author=sig.author,
            hops=trace.total_hops,
            workers_accepted=len(trace.accepted_workers),
            learning_mode=mode,
            content_preview=sig.content,
        )


def _stream_insight(stream, ins, seen_hashes: set[str] | None = None) -> bool:
    """Append a single insight to the JSONL stream file.

    If seen_hashes is provided, computes a finding hash and skips the write
    if already seen. Returns True if written, False if deduplicated.
    """
    import json as _json
    from stigmergy.mesh.insight_store import _finding_hash

    actors = ins.details.get("actors", []) if ins.details else []
    fh = _finding_hash(ins.type, ins.summary, actors)

    if seen_hashes is not None:
        if fh in seen_hashes:
            return False
        seen_hashes.add(fh)

    record = {
        "type": ins.type,
        "summary": ins.summary,
        "confidence": ins.confidence,
        "signal_ids": [str(s) for s in ins.signal_ids],
        "details": ins.details,
        "agent_id": str(ins.agent_id),
        "finding_hash": fh,
    }
    stream.write(_json.dumps(record) + "\n")
    stream.flush()
    return True


async def _run_loop_mesh(config: StigmergyConfig, live: bool, since_days: int | None = None) -> int:
    """Continuous mesh mode: backfill, then poll, showing detailed routing."""
    heading("stigmergy mesh (continuous)")
    info("Press Ctrl+C to stop")
    print()

    _refresh_repos_if_available(config)

    mesh, agent_registry, budget, ctx_map, signal_cache, annotation_store, policy_engine, fingerprint_store, comm_graph, alias_store, stability_tracker, attention_model, discovery_store, health_pulse_collector, meta_modeler, field_engine = bootstrap_mesh(config)
    # Extract compression tracker from agents (created in bootstrap, attached to each agent)
    _all_agents = list(agent_registry.all_agents())
    _agent_comp_tracker = getattr(_all_agents[0], "_compression_tracker", None) if _all_agents else None
    sources = _build_sources(config, live)
    prev_worker_count = mesh.worker_count

    # Preflight: verify LLM is reachable before committing to a full run.
    if mesh._llm is not None:
        await _preflight_llm(mesh._llm)

    graph_info = policy_engine._graph.summary() if policy_engine._graph.node_count > 0 else "no structure graph"
    info(f"Mesh initialized: {mesh.worker_count} workers, {len(config.agents)} agents, {graph_info}")

    counters = {"total": 0, "accepted": 0, "surfaced": 0, "duplicates": 0}
    all_insights: list = []  # Collect all insights for Key Findings summary
    raw_insight_total = 0  # Track pre-dedup count for funnel metrics
    _MAX_INSIGHTS = 5000  # Safety cap for multi-day runs
    _MAX_SIGNAL_META = 10000  # Safety cap for signal metadata
    # Lightweight signal metadata for state change auto-detection.
    # Only stores author/content/source — not full Signal objects.
    all_signal_metadata: list[dict] = []
    # Stream insights to JSONL for persistence beyond the in-memory cap
    _insights_stream_path = Path(".stigmergy") / "insights_stream.jsonl"
    _insights_stream_path.parent.mkdir(parents=True, exist_ok=True)
    _insights_stream = open(_insights_stream_path, "a")

    # Stream-level dedup: simple set of finding hashes, pre-populated from
    # existing stream file so cross-run and within-run duplicates are caught.
    from stigmergy.mesh.insight_store import _finding_hash
    _stream_seen_hashes: set[str] = set()
    if _insights_stream_path.exists():
        try:
            import json as _json_preload
            with open(_insights_stream_path) as _pf:
                for _pl in _pf:
                    _pl = _pl.strip()
                    if _pl:
                        _pd = _json_preload.loads(_pl)
                        _pa = _pd.get("details", {}).get("actors", []) if _pd.get("details") else []
                        _ph = _finding_hash(_pd.get("type", ""), _pd.get("summary", ""), _pa)
                        _stream_seen_hashes.add(_ph)
            if _stream_seen_hashes:
                info(f"  Stream dedup: {len(_stream_seen_hashes)} known findings loaded from prior entries")
        except Exception:
            pass  # Start fresh if stream file is corrupted

    # Cadence monitor — detects worker silence and injects synthetic signals
    from stigmergy.mesh.stability import CadenceMonitor
    cadence_monitor = CadenceMonitor(mesh)
    stop = asyncio.Event()

    def handle_sigint():
        if stop.is_set():
            # Second Ctrl+C — force exit
            raise KeyboardInterrupt
        stop.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal_mod.SIGINT, handle_sigint)

    # --- Phase 1: Initial backfill ---
    state = _load_state()
    prev_sources = set(state.get("last_sources", []))
    checkpoint = state.get("checkpoint")

    if since_days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=since_days)
        # If there's a checkpoint from an interrupted run with the same --since
        # window, resume from where we left off instead of reprocessing.
        if checkpoint:
            resume_ts = datetime.fromisoformat(checkpoint["timestamp"])
            if resume_ts > since:
                info(f"Resuming from checkpoint: {checkpoint['signals_processed']} signals processed "
                     f"(${checkpoint.get('spend_usd', 0):.2f} spent). Skipping signals before "
                     f"{resume_ts.strftime('%Y-%m-%d %H:%M')} UTC.")
                since = resume_ts
    else:
        last_run = state.get("last_run")
        since = datetime.fromisoformat(last_run) if last_run else datetime.now(timezone.utc) - timedelta(days=30)

    heading("Phase 1: Backfill")
    tracker = NoveltyTracker()
    insight_dedup = InsightDeduplicator()
    for source_name, adapter, _is_live in sources:
        info(f"Connecting {source_name}{'  (live)' if _is_live else '  (mock)'}...")
        try:
            await adapter.connect()
        except (ConnectionError, OSError) as e:
            warn(f"  {source_name}: connection failed ({e}), skipping")
            continue
        # New sources get a wider initial backfill window
        if source_name not in prev_sources and since_days is None and prev_sources:
            source_since = datetime.now(timezone.utc) - timedelta(days=7)
            info(f"Backfilling {source_name} since {source_since.strftime('%Y-%m-%d %H:%M')} UTC (first run for this source)...")
        else:
            source_since = since
            info(f"Backfilling {source_name} since {source_since.strftime('%Y-%m-%d %H:%M')} UTC...")

        source_signals: list[Signal] = []
        async for sig in adapter.backfill(source_since):
            if stop.is_set():
                break
            source_signals.append(sig)

        if stop.is_set():
            info(f"  {source_name}: interrupted during collection ({len(source_signals)} signals collected)")
            break

        if not source_signals:
            extras = []
            if hasattr(adapter, "bot_skipped") and adapter.bot_skipped > 0:
                extras.append(f"{adapter.bot_skipped} bot filtered")
            if hasattr(adapter, "_stats") and adapter._stats and adapter._stats.get("filtered"):
                extras.append(f"{adapter._stats['filtered']} empty/system filtered")
            extra_note = f" ({', '.join(extras)})" if extras else ""
            info(f"  {source_name}: no signals{extra_note}")
            continue

        # Sort chronologically (oldest first) during backfill — this keeps
        # temporal_proximity high as workers learn progressively forward in time.
        # Newest-first causes older signals to score near-zero (exponential decay).
        source_signals.sort(key=lambda s: s.timestamp)
        mesh_poll_header(source_name, len(source_signals), source_since.strftime("%Y-%m-%d %H:%M"))
        extras = []
        if hasattr(adapter, "bot_skipped") and adapter.bot_skipped > 0:
            extras.append(f"{adapter.bot_skipped} bot filtered")
        if hasattr(adapter, "_stats") and adapter._stats:
            s = adapter._stats
            if s.get("thread_replies"):
                extras.append(f"{s['thread_replies']} from threads")
            if s.get("filtered"):
                extras.append(f"{s['filtered']} empty/system filtered")
        if extras:
            info(f"  ({', '.join(extras)})")

        _src_total = len(source_signals)
        _src_start = time.monotonic()
        _src_idx = 0
        _last_heartbeat_spend = budget.daily_spend
        _HEARTBEAT_INTERVAL = 50  # signals between heartbeats
        _HEARTBEAT_SPEND = 1.0    # dollars between heartbeats

        for sig in source_signals:
            if stop.is_set():
                break
            if not budget.can_spend():
                warn("Budget exhausted.")
                break
            for w in budget.check_warnings():
                warn(w)

            trace = await mesh.ingest(sig)
            counters["total"] += 1
            _src_idx += 1

            # Periodic heartbeat + checkpoint so humans know the process
            # is alive and interrupted runs can resume without reprocessing.
            spend_delta = budget.daily_spend - _last_heartbeat_spend
            if _src_idx % _HEARTBEAT_INTERVAL == 0 or spend_delta >= _HEARTBEAT_SPEND:
                _el = _fmt_duration(time.monotonic() - _src_start)
                info(f"  ... {_src_idx}/{_src_total} {source_name}  ${budget.daily_spend:.2f} spent  {counters.get('surfaced', 0)} surfaced  {_el} elapsed")
                _last_heartbeat_spend = budget.daily_spend
                # Checkpoint: save progress so restart can skip processed signals
                _save_checkpoint(
                    mesh, ctx_map, annotation_store, comm_graph, policy_engine,
                    last_signal_ts=sig.timestamp, signals_processed=counters["total"],
                    total_signals=_src_total, budget=budget,
                    alias_store=alias_store, stability_tracker=stability_tracker,
                    attention_model=attention_model,
                )
            all_signal_metadata.append({
                "author": sig.author, "content": sig.content,
                "source": sig.source, "channel": sig.channel,
                "metadata": sig.metadata, "timestamp": sig.timestamp,
            })
            if len(all_signal_metadata) > _MAX_SIGNAL_META:
                all_signal_metadata = all_signal_metadata[-_MAX_SIGNAL_META:]

            # Record spectral fingerprint
            if not trace.duplicate and trace.familiarity_scores:
                from stigmergy.mesh.fingerprints import fingerprint_from_trace
                fingerprint_store.record(fingerprint_from_trace(trace, sig))

            budget.sync_from_llm()

            # Agent reflection (suppress re-discovery of known findings)
            insights, _raw = await _reflect_and_cache(sig, trace, mesh, agent_registry, signal_cache, comm_graph, known_finding_hashes=_stream_seen_hashes)
            raw_insight_total += _raw

            # Policy evaluation
            from stigmergy.policy.traces import TraceEvent as PolicyTraceEvent
            policy_interventions = []
            try:
                trace_event = PolicyTraceEvent.from_signal(sig)
                policy_interventions = await policy_engine.evaluate(trace_event)
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise

            if trace.duplicate:
                counters["duplicates"] += 1
                continue

            if trace.accepted_workers:
                counters["accepted"] += 1

            consensus_action = None
            consensus_weight = 0.0
            if trace.consensus:
                consensus_action = trace.consensus.action
                consensus_weight = trace.consensus.total_weight
                if consensus_action == "surface" and trace.output_delivered:
                    counters["surfaced"] += 1

            # Determine learning mode for novelty category
            if trace.hops:
                accepted_hops = [h for h in trace.hops if h.accepted]
                mode = accepted_hops[0].learning_mode if accepted_hops else trace.hops[0].learning_mode
            else:
                mode = "none"

            category = f"{sig.source}:{mode}"
            novelty = tracker.check(category)

            if novelty == "detail":
                worker_labels = {str(w.id)[:8]: w.label[:20] for w in mesh.workers}
                mesh_signal_detail(
                    source=sig.source,
                    channel=sig.channel,
                    author=sig.author,
                    content_preview=sig.content,
                    hops=trace.hops,
                    consensus_action=consensus_action,
                    consensus_weight=consensus_weight,
                    output_delivered=trace.output_delivered,
                    worker_labels=worker_labels,
                )
            elif novelty == "compact":
                mesh_signal_line(
                    source=sig.source,
                    channel=sig.channel,
                    author=sig.author,
                    hops=trace.total_hops,
                    workers_accepted=len(trace.accepted_workers),
                    learning_mode=mode,
                    content_preview=sig.content,
                )

            if novelty != "suppress":
                for intv in policy_interventions:
                    intervention_line(intv)

            # Insights: display each unique insight inline once (dedup),
            # but always collect all instances for the end-of-run report.
            for ins in insights:
                if insight_dedup.should_display(ins):
                    insight_line(ins)
                _stream_insight(_insights_stream, ins, seen_hashes=_stream_seen_hashes)
            all_insights.extend(insights)
            if len(all_insights) > _MAX_INSIGHTS:
                all_insights = all_insights[-_MAX_INSIGHTS:]

        # Source backfill complete — print summary so you can see the transition
        _src_elapsed = _fmt_duration(time.monotonic() - _src_start)
        info(f"  {source_name} backfill complete: {_src_idx}/{_src_total} signals in {_src_elapsed}  ${budget.daily_spend:.2f} spent  {counters.get('surfaced', 0)} surfaced")

        # Cadence monitor: check for worker silence at end of each source's backfill
        cadence_signals = cadence_monitor.check(counters["total"])
        for csig in cadence_signals:
            ctrace = await mesh.ingest(csig)
            counters["total"] += 1
            # Run agent reflection so the correlator can reason about silence
            c_insights, c_raw = await _reflect_and_cache(
                csig, ctrace, mesh, agent_registry, signal_cache, comm_graph, known_finding_hashes=_stream_seen_hashes,
            )
            raw_insight_total += c_raw
            for cins in c_insights:
                if insight_dedup.should_display(cins):
                    insight_line(cins)
                _stream_insight(_insights_stream, cins, seen_hashes=_stream_seen_hashes)
            all_insights.extend(c_insights)
            info(f"  [cadence] {csig.content[:80]}...")

        tracker.flush()
        insight_dedup.flush()

        # Report topology changes
        if mesh.worker_count != prev_worker_count:
            mesh_lifecycle_event("fork" if mesh.worker_count > prev_worker_count else "decay",
                                 f"{prev_worker_count} -> {mesh.worker_count} workers")
            prev_worker_count = mesh.worker_count

    # Prune workers that accepted nothing during backfill
    pruned = mesh.prune_idle_workers(min_workers=len(config.contexts))
    if pruned:
        info(f"  Pruned {len(pruned)} idle workers (0 signals accepted)")
    prev_worker_count = mesh.worker_count

    # Health pulse + unity field engine + meta-modeler (post-backfill checkpoint)
    if not stop.is_set():
        from stigmergy.mesh.insight_store import FindingRegistry as _FReg
        _finding_reg = _FReg()
        pulse = health_pulse_collector.collect(
            counters["total"], mesh,
            stability_tracker=stability_tracker,
            agents=list(agent_registry.all_agents()),
            finding_registry=_finding_reg,
            compression_tracker=_agent_comp_tracker,
        )
        # Unity field engine: mechanical self-regulation (when enabled)
        if field_engine is not None:
            _field_state = field_engine.tick(
                pulse=pulse, mesh=mesh,
                agents=list(agent_registry.all_agents()),
                stability_tracker=stability_tracker,
                spectral=stability_tracker,
                compression_tracker=_agent_comp_tracker,
            )
            if meta_modeler is not None and hasattr(meta_modeler, "set_field_state"):
                meta_modeler.set_field_state(_field_state)
        # Meta-modeler: evaluate mesh health (when enabled)
        if meta_modeler is not None:
            assessment = await meta_modeler.evaluate(pulse)
            if assessment is not None:
                meta_modeler.apply_recommendations(
                    assessment, agents=list(agent_registry.all_agents()),
                )

    # Show topology after backfill with agent-generated labels (skip LLM calls if stopping)
    if not stop.is_set():
        agent_labels = {}
        for w in mesh.workers:
            try:
                al = await w.agent_label()
                if al != w.label:
                    agent_labels[str(w.id)[:8]] = al
            except Exception:
                pass
        mesh_topology(mesh.workers, labels=agent_labels or None)
        budget_line(budget.daily_spend, budget.daily_cap_usd, budget.hourly_spend, budget.hourly_cap_usd)
    else:
        mesh_topology(mesh.workers)
        budget_line(budget.daily_spend, budget.daily_cap_usd, budget.hourly_spend, budget.hourly_cap_usd)

    if stop.is_set():
        pass  # skip to Phase 2, fall through to save
    else:
        # Backfill complete — clear checkpoint so restarts don't re-enter backfill.
        # Update last_run to now so polling starts from this point forward.
        _bf_state = _load_state()
        _bf_state.pop("checkpoint", None)
        _bf_state["last_run"] = datetime.now(timezone.utc).isoformat()
        _save_state(_bf_state)

        # --- Phase 2: Continuous polling ---
        heading("Phase 2: Polling")

        has_live = any(is_live for _, _, is_live in sources)
        if not has_live:
            info("All sources are mocks. Backfill complete.")
            info("Use --live for continuous polling with real sources. Ctrl+C to exit.")

        # Adaptive poll schedule: [name, adapter, next_poll_mono, base_interval, current_interval, is_live, polled_once]
        # Dampen: empty polls widen interval (up to max_interval)
        # Reverse-dampen: signals found snap interval back to base
        dampen_factor = 1.5
        max_interval = 3600  # 1 hour ceiling

        poll_schedule: list[list] = []
        for source_name, adapter, is_live in sources:
            base_interval = 300
            if source_name == "github":
                base_interval = config.sources.github.poll_interval
            elif source_name == "linear":
                base_interval = config.sources.linear.poll_interval
            poll_schedule.append([source_name, adapter, time.monotonic() + base_interval,
                                  base_interval, base_interval, is_live, True])

        last_backfill = datetime.now(timezone.utc)

        while not stop.is_set():
            now_mono = time.monotonic()

            # Reset agent batch caches each poll cycle so the correlator
            # re-evaluates patterns from scratch (epistemic independence).
            for agent in agent_registry.all_agents():
                agent.reset_batch_state()

            for entry in poll_schedule:
                source_name, adapter, next_poll, base_interval, current_interval, is_live, polled_once = entry

                if now_mono < next_poll:
                    continue

                # Mocks already polled during backfill
                if not is_live and polled_once:
                    continue

                if not budget.can_spend():
                    warn("Budget exhausted. Waiting for reset...")
                    break

                poll_signals: list[Signal] = []
                async for sig in adapter.backfill(last_backfill):
                    if stop.is_set():
                        break
                    poll_signals.append(sig)

                if stop.is_set():
                    break

                if not poll_signals:
                    # Dampen: widen interval since nothing is happening
                    old_interval = current_interval
                    current_interval = min(current_interval * dampen_factor, max_interval)
                    entry[4] = current_interval
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    info(f"  {ts}  {source_name}: no new signals  (interval {_fmt_duration(old_interval)} -> {_fmt_duration(current_interval)})")
                else:
                    # Reverse-dampen: snap back to base, signals are flowing
                    if current_interval > base_interval:
                        info(f"  {source_name}: signals found, interval reset to {_fmt_duration(base_interval)}")
                    current_interval = base_interval
                    entry[4] = current_interval

                    poll_signals.sort(key=lambda s: s.timestamp, reverse=True)
                    mesh_poll_header(source_name, len(poll_signals),
                                     datetime.now(timezone.utc).strftime("%H:%M:%S"))

                    _poll_total = len(poll_signals)
                    _poll_start = time.monotonic()
                    _poll_idx = 0
                    _poll_last_heartbeat_spend = budget.daily_spend

                    for sig in poll_signals:
                        if stop.is_set():
                            break
                        if not budget.can_spend():
                            warn("Budget exhausted.")
                            break

                        trace = await mesh.ingest(sig)
                        counters["total"] += 1
                        _poll_idx += 1

                        # Periodic heartbeat
                        _poll_spend_delta = budget.daily_spend - _poll_last_heartbeat_spend
                        if _poll_idx % 50 == 0 or _poll_spend_delta >= 1.0:
                            _el = _fmt_duration(time.monotonic() - _poll_start)
                            info(f"  ... {_poll_idx}/{_poll_total} {source_name}  ${budget.daily_spend:.2f} spent  {_el} elapsed")
                            _poll_last_heartbeat_spend = budget.daily_spend
                        all_signal_metadata.append({
                            "author": sig.author, "content": sig.content,
                            "source": sig.source, "channel": sig.channel,
                            "metadata": sig.metadata, "timestamp": sig.timestamp,
                        })
                        if len(all_signal_metadata) > _MAX_SIGNAL_META:
                            all_signal_metadata = all_signal_metadata[-_MAX_SIGNAL_META:]

                        # Record spectral fingerprint
                        if not trace.duplicate and trace.familiarity_scores:
                            from stigmergy.mesh.fingerprints import fingerprint_from_trace
                            fingerprint_store.record(fingerprint_from_trace(trace, sig))

                        budget.sync_from_llm()

                        if trace.duplicate:
                            counters["duplicates"] += 1
                            continue

                        if trace.accepted_workers:
                            counters["accepted"] += 1

                        worker_labels = {str(w.id)[:8]: w.label[:20] for w in mesh.workers}
                        consensus_action = None
                        consensus_weight = 0.0
                        if trace.consensus:
                            consensus_action = trace.consensus.action
                            consensus_weight = trace.consensus.total_weight
                            if consensus_action == "surface" and trace.output_delivered:
                                counters["surfaced"] += 1

                        mesh_signal_detail(
                            source=sig.source,
                            channel=sig.channel,
                            author=sig.author,
                            content_preview=sig.content,
                            hops=trace.hops,
                            consensus_action=consensus_action,
                            consensus_weight=consensus_weight,
                            output_delivered=trace.output_delivered,
                            worker_labels=worker_labels,
                        )

                        # Agent reflection (suppress re-discovery of known findings)
                        insights, _raw = await _reflect_and_cache(sig, trace, mesh, agent_registry, signal_cache, comm_graph, known_finding_hashes=_stream_seen_hashes)
                        raw_insight_total += _raw
                        # Insights: display each unique insight inline once (dedup),
                        # but always collect all instances for the end-of-run report.
                        for ins in insights:
                            if insight_dedup.should_display(ins):
                                insight_line(ins)
                            _stream_insight(_insights_stream, ins, seen_hashes=_stream_seen_hashes)
                        all_insights.extend(insights)
                        if len(all_insights) > _MAX_INSIGHTS:
                            all_insights = all_insights[-_MAX_INSIGHTS:]

                        # Policy evaluation
                        from stigmergy.policy.traces import TraceEvent as PolicyTraceEvent
                        try:
                            trace_event = PolicyTraceEvent.from_signal(sig)
                            poll_interventions = await policy_engine.evaluate(trace_event)
                            for intv in poll_interventions:
                                intervention_line(intv)
                        except BaseException as exc:
                            if not isinstance(exc, Exception):
                                raise

                    # Show topology if it changed
                    if mesh.worker_count != prev_worker_count:
                        mesh_lifecycle_event(
                            "fork" if mesh.worker_count > prev_worker_count else "decay",
                            f"{prev_worker_count} -> {mesh.worker_count} workers")
                        prev_worker_count = mesh.worker_count
                        mesh_topology(mesh.workers)

                # Cadence monitor: check for worker silence after each poll batch
                cadence_signals = cadence_monitor.check(counters["total"])
                for csig in cadence_signals:
                    ctrace = await mesh.ingest(csig)
                    counters["total"] += 1
                    # Run agent reflection so the correlator can reason about silence
                    c_insights, c_raw = await _reflect_and_cache(
                        csig, ctrace, mesh, agent_registry, signal_cache, comm_graph, known_finding_hashes=_stream_seen_hashes,
                    )
                    raw_insight_total += c_raw
                    for cins in c_insights:
                        if insight_dedup.should_display(cins):
                            insight_line(cins)
                        _stream_insight(_insights_stream, cins, seen_hashes=_stream_seen_hashes)
                    all_insights.extend(c_insights)
                    if len(all_insights) > _MAX_INSIGHTS:
                        all_insights = all_insights[-_MAX_INSIGHTS:]
                    info(f"  [cadence] {csig.content[:80]}...")

                # Unity field engine + meta-modeler: run after each poll batch
                # (only when signals were processed this cycle)
                if poll_signals and not stop.is_set():
                    from stigmergy.mesh.insight_store import FindingRegistry as _PollFReg
                    _poll_finding_reg = _PollFReg()
                    _poll_pulse = health_pulse_collector.collect(
                        counters["total"], mesh,
                        stability_tracker=stability_tracker,
                        agents=list(agent_registry.all_agents()),
                        finding_registry=_poll_finding_reg,
                        compression_tracker=_agent_comp_tracker,
                    )
                    if field_engine is not None:
                        _poll_field_state = field_engine.tick(
                            pulse=_poll_pulse, mesh=mesh,
                            agents=list(agent_registry.all_agents()),
                            stability_tracker=stability_tracker,
                            spectral=stability_tracker,
                            compression_tracker=_agent_comp_tracker,
                        )
                        if meta_modeler is not None and hasattr(meta_modeler, "set_field_state"):
                            meta_modeler.set_field_state(_poll_field_state)
                    if meta_modeler is not None:
                        _poll_assessment = await meta_modeler.evaluate(_poll_pulse)
                        if _poll_assessment is not None:
                            meta_modeler.apply_recommendations(
                                _poll_assessment, agents=list(agent_registry.all_agents()),
                            )

                entry[2] = now_mono + current_interval
                entry[6] = True
                last_backfill = datetime.now(timezone.utc)

            # All mocks done and no live sources? We're done.
            all_done = all(not e[5] and e[6] for e in poll_schedule)
            if all_done and not has_live:
                info("\nAll mock sources processed. Nothing left to poll.")
                break

            # Sleep until next poll, with periodic heartbeat
            active_polls = [e[2] for e in poll_schedule if e[5] or not e[6]]
            if active_polls:
                sleep_until = min(active_polls)
                sleep_for = max(1.0, sleep_until - time.monotonic())
            else:
                sleep_for = 10

            next_sources = [e[0] for e in poll_schedule if e[5] and e[2] <= (min(active_polls) + 1 if active_polls else 0)]
            info(f"\n  Next poll: {', '.join(next_sources) if next_sources else 'sources'} in {_fmt_duration(sleep_for)}  (Ctrl+C to stop)")

            heartbeat_interval = 60
            remaining = sleep_for
            while remaining > 0 and not stop.is_set():
                chunk = min(remaining, heartbeat_interval)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=chunk)
                except asyncio.TimeoutError:
                    pass
                remaining -= chunk
                if remaining > 0 and not stop.is_set():
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    info(f"  {ts}  waiting... {_fmt_duration(remaining)} until next poll")

    # Close insight stream
    _insights_stream.close()

    stopping = stop.is_set()
    if stopping:
        warn("Shutting down gracefully... saving state.")

    # --- Essential saves FIRST (no LLM calls, fast) ---
    contexts = [w.context for w in mesh.workers]
    _save_context_state(contexts, {k: UUID(v) for k, v in ctx_map.items()})
    annotation_store.save()
    comm_graph.save(Path(".stigmergy/graph.json"))
    if alias_store is not None:
        alias_store.save()
    if stability_tracker is not None:
        stability_tracker.flush()
    # Mesh self-organized state — dice back in the cube
    from stigmergy.mesh.state_store import save_mesh_state
    from stigmergy.mesh.caucus import RunStats as _RunStats
    _run_stats = _RunStats(
        signals=len(traces),
        duration_seconds=time.monotonic() - _mesh_start_time,
        workers=mesh.worker_count,
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
    )
    save_mesh_state(
        workers=mesh.workers,
        agents=list(agent_registry.all_agents()),
        compression_tracker=_agent_comp_tracker,
        run_stats=_run_stats,
    )
    # Unity field engine state — PID, calibration, eigenmonitor history
    if field_engine is not None:
        field_engine.save_state()
    if attention_model is not None:
        attention_model.save(Path(".stigmergy/attention.json"))
    # Save Slack user cache for next run
    for _src_name, _src_adapter, _src_live in sources:
        if _src_name == "slack" and hasattr(_src_adapter, "save_user_cache"):
            _src_adapter.save_user_cache()
    policy_engine._traces.save()
    policy_engine._budget.save()

    # S(f) scoring — local computation, always runs
    from stigmergy.mesh.scoring import score_finding
    for ins in all_insights:
        components = score_finding(ins, comm_graph)
        ins.details["sf_score"] = components.score
        ins.details["sf_components"] = components.to_dict()

    # Per-finding ND indicators for display in Key Findings (loop mode)
    _nd_cfg_l = getattr(config, "normalized_deviance", None)
    _nd_on_l = _nd_cfg_l.enabled if _nd_cfg_l else True
    if _nd_on_l and all_signal_metadata:
        from stigmergy.attention.linguistic import analyze_text as _pf_comp_l2
        for ins in all_insights:
            ins.details["compression_index"] = _pf_comp_l2(ins.summary).compression_index
        # Action-correlation per finding
        _nd_ac_on_l2 = _nd_cfg_l.action_correlation_enabled if _nd_cfg_l else True
        if _nd_ac_on_l2:
            from stigmergy.attention.action_correlation import compute_action_correlations as _cac_l2
            from stigmergy.primitives.signal import extract_terms as _et_l2
            _ac_results_l2 = _cac_l2(all_signal_metadata)
            for ins in all_insights:
                _fterms = _et_l2(ins.summary)
                _best = -1.0
                for _acr in _ac_results_l2:
                    if _acr.topic in _fterms:
                        _best = max(_best, _acr.ratio)
                if _best >= 0:
                    ins.details["action_correlation"] = _best

    # Key Findings
    display_unique = findings_summary(all_insights)

    # Persist insights to .stigmergy/insights.jsonl
    from stigmergy.mesh.insight_store import InsightStore, FindingRegistry
    insight_store = InsightStore()
    for ins in all_insights:
        sf = ins.details.get("sf_components", {})
        insight_store.record(ins, score=ins.details.get("sf_score", 0.0), scoring_components=sf)
    insight_count = insight_store.save()
    if insight_count > 0:
        info(f"  Persisted {insight_count} insights to .stigmergy/insights.jsonl (run {insight_store.run_id})")

    # Update finding registry — cross-run dedup and decay tracking.
    # Compute per-finding ND metadata for persistence.
    finding_registry = FindingRegistry()

    _pf_compression_l: dict[str, float] = {}
    _pf_ac_l: dict[str, float] = {}
    if all_signal_metadata:
        from stigmergy.attention.linguistic import analyze_text as _pf_analyze_l
        from stigmergy.primitives.signal import extract_terms as _pf_extract_l
        from stigmergy.attention.action_correlation import compute_action_correlations as _pf_cac_l
        _pf_ac_results = _pf_cac_l(all_signal_metadata)
        for ins in all_insights:
            _pf_compression_l[ins.summary[:80]] = _pf_analyze_l(ins.summary).compression_index
            if _pf_ac_results:
                _f_terms_l = _pf_extract_l(ins.summary)
                _best_l = -1.0
                for _ac_r in _pf_ac_results:
                    if _ac_r.topic in _f_terms_l:
                        _best_l = max(_best_l, _ac_r.ratio)
                if _best_l >= 0:
                    _pf_ac_l[ins.summary[:80]] = _best_l

    buffered = [
        {
            "type": ins.type,
            "summary": ins.summary,
            "confidence": ins.confidence,
            "score": ins.details.get("sf_score", 0.0),
            "details": ins.details,
            "signal_ids": [str(s) for s in ins.signal_ids],
            "compression_index": _pf_compression_l.get(ins.summary[:80], 0.0),
            "action_correlation": _pf_ac_l.get(ins.summary[:80], -1.0),
        }
        for ins in all_insights
    ]
    changed_findings = finding_registry.ingest_insights(buffered, insight_store.run_id)

    # Auto-detect state changes
    state_changes = finding_registry.detect_state_changes(all_signal_metadata)
    if state_changes > 0:
        info(f"  Auto-detected {state_changes} finding state change(s)")

    finding_registry.save()

    # Save state with summary
    state = _load_state()
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["last_mode"] = "mesh"
    state["last_sources"] = [name for name, _, _ in sources]
    state["last_summary"] = {
        "signals_processed": counters["total"],
        "accepted": counters["accepted"],
        "surfaced": counters["surfaced"],
        "duplicates": counters["duplicates"],
        "workers": mesh.worker_count,
        "avg_fullness": round(mesh.avg_fullness, 3),
        "annotations": annotation_store.annotation_count,
        "traces": policy_engine.trace_count,
        "interventions": policy_engine.intervention_count,
        "fingerprints": fingerprint_store.count,
        "insights": len(all_insights),
        "insights_raw": raw_insight_total,
        "insights_display_unique": display_unique,
        "findings_registry": finding_registry.summary(),
    }
    state.pop("checkpoint", None)  # Clear stale checkpoint on clean exit
    _save_state(state)

    # Archive run outputs
    archive_dir = _archive_run_outputs()
    if archive_dir:
        info(f"  Run archived to {archive_dir}")

    if stopping:
        # Minimal summary — no LLM calls, exit fast
        mesh_topology(mesh.workers)
        budget_line(budget.daily_spend, budget.daily_cap_usd, budget.hourly_spend, budget.hourly_cap_usd)
        mesh_scan_summary(
            signals_processed=counters["total"],
            signals_accepted=counters["accepted"],
            duplicates=counters["duplicates"],
            windows_scanned=0,
            deepest_timestamp=since.isoformat(),
            stopped_reason="user_stop",
            worker_count=mesh.worker_count,
            avg_fullness=mesh.avg_fullness,
            total_cost=budget.total_spend,
        )
        info("State saved. Clean shutdown.")
    else:
        # Full summary with LLM-powered operations

        # Agent-generated topology labels
        agent_labels = {}
        for w in mesh.workers:
            try:
                al = await w.agent_label()
                if al != w.label:
                    agent_labels[str(w.id)[:8]] = al
            except Exception:
                pass
        mesh_topology(mesh.workers, labels=agent_labels or None)
        budget_line(budget.daily_spend, budget.daily_cap_usd, budget.hourly_spend, budget.hourly_cap_usd)
        mesh_scan_summary(
            signals_processed=counters["total"],
            signals_accepted=counters["accepted"],
            duplicates=counters["duplicates"],
            windows_scanned=0,
            deepest_timestamp=since.isoformat(),
            stopped_reason="user_stop",
            worker_count=mesh.worker_count,
            avg_fullness=mesh.avg_fullness,
            total_cost=budget.total_spend,
        )

        # Policy summary
        policy_summary(
            trace_count=policy_engine.trace_count,
            intervention_count=policy_engine.intervention_count,
            budget_exceeded=policy_engine._budget.all_exceeded(),
            spectral_snapshot=policy_engine.spectral_snapshot,
        )

        # Structural coupling detection
        _print_coupling_summary(fingerprint_store)

        # Agent Intelligence Report
        agent_diags = [a.diagnostics() for a in agent_registry.all_agents()]
        intelligence_report(agent_diags)

        # Insight dedup funnel
        insight_funnel(raw_insight_total, len(all_insights), display_unique)

        # Cross-signal corroboration (LLM calls)
        from stigmergy.mesh.corroboration import run_corroboration as _run_corr_loop
        quorum_findings_loop, corr_metrics_loop = await _run_corr_loop(
            all_insights,
            list(mesh.workers),
            llm=getattr(mesh, "_correlator_llm", None) or mesh._llm,
            comm_graph=comm_graph,
        )
        corroboration_summary(quorum_findings_loop, corr_metrics_loop)

        # Unity calibration: feed loop corroboration outcomes
        if field_engine is not None:
            for qf in quorum_findings_loop:
                was_corr = qf.quorum_met
                field_engine.record_prediction(
                    str(qf.original.agent_id)[:8],
                    qf.original.confidence,
                    was_corr,
                )

        # Persist corroboration results alongside findings
        if quorum_findings_loop:
            import json as _json_loop
            corr_path = Path(".stigmergy") / "corroboration.json"
            corr_data = {
                "run_id": insight_store.run_id if insight_count > 0 else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": corr_metrics_loop,
                "quorum_findings": [
                    {
                        "summary": qf.original.summary[:200],
                        "type": qf.original.type,
                        "quorum_met": qf.quorum_met,
                        "quorum_confidence": round(qf.quorum_confidence, 4),
                        "corroboration_count": qf.corroboration_count,
                        "contradiction_count": qf.contradiction_count,
                        "total_consulted": qf.total_consulted,
                        "verdicts": [
                            {
                                "worker_id": str(v.worker_id)[:8],
                                "verdict": v.verdict,
                                "confidence": round(v.confidence, 4),
                                "reasoning": v.reasoning[:200],
                            }
                            for v in qf.verdicts
                        ],
                    }
                    for qf in quorum_findings_loop
                ],
            }
            with open(corr_path, "w") as f:
                _json_loop.dump(corr_data, f, indent=2)
            info(f"  Persisted corroboration results to {corr_path}")

        # Normalized Deviance detection (loop mode) — all 5 detectors
        nd_config_loop = getattr(config, "normalized_deviance", None)
        _nd_enabled_loop = nd_config_loop.enabled if nd_config_loop else True
        if _nd_enabled_loop and all_signal_metadata:
            nd_compression_l = None
            nd_action_l = None
            nd_solo_l = None
            nd_migration_l = None
            nd_phase_l = None

            # 1. Linguistic compression + per-channel trends
            _nd_ling_on = nd_config_loop.linguistic_enabled if nd_config_loop else True
            if _nd_ling_on:
                from stigmergy.attention.linguistic import (
                    CompressionTracker as _CT_l,
                    analyze_text as _analyze_comp_l,
                )
                combined = ". ".join(s.get("content", "")[:200] for s in all_signal_metadata[:500])
                _cr = _analyze_comp_l(combined)
                nd_compression_l = {
                    "compression_index": _cr.compression_index,
                    "hedging_density": _cr.hedging_density,
                    "passive_voice_ratio": _cr.passive_voice_ratio,
                    "nominalization_rate": _cr.nominalization_rate,
                    "specificity_score": _cr.specificity_score,
                }
                _ct_l = _CT_l()
                for _sm in all_signal_metadata:
                    _ch = _sm.get("channel", "")
                    if _ch:
                        _ct_l.update(_ch, _analyze_comp_l(_sm.get("content", "")).compression_index)
                _inc_l = _ct_l.increasing_trends(min_samples=3)
                if _inc_l:
                    nd_compression_l["increasing_channels"] = [
                        {"channel": t.key, "avg": t.rolling_avg, "peak": t.peak_index,
                         "samples": t.sample_count}
                        for t in _inc_l[:5]
                    ]

            # 2. Action-correlation
            _nd_ac_on = nd_config_loop.action_correlation_enabled if nd_config_loop else True
            if _nd_ac_on:
                from stigmergy.attention.action_correlation import compute_action_correlations as _cac_l
                nd_action_l = _cac_l(all_signal_metadata)

            # 3. Solo ownership detection
            _nd_solo_on_l = nd_config_loop.solo_ownership_enabled if nd_config_loop else False
            if _nd_solo_on_l:
                from stigmergy.attention.solo_ownership import detect_solo_ownership as _dso_l
                nd_solo_l = _dso_l(all_signal_metadata)

            # 4. Channel migration detection
            _nd_mig_on_l = nd_config_loop.channel_migration_enabled if nd_config_loop else False
            if _nd_mig_on_l:
                from stigmergy.attention.channel_migration import detect_channel_migration as _dcm_l
                _chan_meta_l = {}
                for _src_name, _src_adapter, _ in sources:
                    if _src_name == "slack" and hasattr(_src_adapter, "_channel_meta"):
                        _chan_meta_l = _src_adapter._channel_meta
                nd_migration_l = _dcm_l(all_signal_metadata, channel_meta=_chan_meta_l)

            # 5. Phase-lock detection
            _nd_pl_on_l = nd_config_loop.phase_lock_enabled if nd_config_loop else False
            if _nd_pl_on_l:
                from stigmergy.attention.phase_lock import detect_phase_lock as _dpl_l
                _ac_map_l = {}
                if nd_action_l:
                    _ac_map_l = {ac.topic: ac.ratio for ac in nd_action_l}
                # Build finding dicts from all_insights for phase-lock
                _pl_findings_l = [
                    {"finding_hash": "", "summary": ins.summary, "type": ins.type,
                     "sf_score": ins.details.get("sf_score", 0.0)}
                    for ins in all_insights
                ]
                nd_phase_l = _dpl_l(_pl_findings_l, action_correlations=_ac_map_l)

            normalized_deviance_summary(
                compression_signals=nd_compression_l,
                action_correlations=nd_action_l,
                solo_alerts=nd_solo_l,
                migration_alerts=nd_migration_l,
                phase_lock_alerts=nd_phase_l,
            )

        decay_summary(finding_registry)
        identity_summary(comm_graph._resolver)

    # Close LLM client so asyncio.run() can exit cleanly
    if hasattr(mesh, '_llm') and mesh._llm is not None and hasattr(mesh._llm, 'close'):
        await mesh._llm.close()
    _corr_llm = getattr(mesh, "_correlator_llm", None)
    if _corr_llm is not None and _corr_llm is not mesh._llm and hasattr(_corr_llm, "close"):
        await _corr_llm.close()

    # Restore original SIGTERM handler
    signal_mod.signal(signal_mod.SIGTERM, _original_sigterm)

    return 0


def _clean_state() -> None:
    """Remove learned state for a fresh start. Preserves config."""
    import os
    state_dir = STATE_PATH.parent
    for name in ("state.yaml", "budgets.json", "traces.jsonl", "graph.json",
                  "run.log", "insights.jsonl", "finding_registry.json",
                  "dedup_index.json"):
        path = state_dir / name
        if path.exists():
            os.remove(path)


def run_pipeline(args: argparse.Namespace) -> int:
    if getattr(args, "clean", False):
        _clean_state()
        from stigmergy.cli.output import info
        info("Cleaned state — starting fresh")

    try:
        config = _load_config()
    except FileNotFoundError as e:
        error(str(e))
        return 1

    use_pipeline = getattr(args, "pipeline", False)

    if use_pipeline:
        # Legacy centralized pipeline
        if args.once:
            return asyncio.run(_run_once(config, args.live))
        return asyncio.run(_run_loop(config, args.live))

    # Default: mesh routing
    since_days = getattr(args, "since", None)
    try:
        if args.once:
            return asyncio.run(_run_once_mesh(config, args.live, since_days=since_days))
        return asyncio.run(_run_loop_mesh(config, args.live, since_days=since_days))
    except KeyboardInterrupt:
        print("\n\nInterrupted. State saved where possible.")
        return 0


def run_status() -> int:
    state = _load_state()
    if not state:
        info("No state found. Run `stigmergy run --once` first.")
        return 0

    heading("stigmergy status")
    separator()

    last_run = state.get("last_run", "never")
    info(f"  Last run: {last_run}")

    summary = state.get("last_summary", {})
    if summary:
        print(f"  Signals processed:  {summary.get('signals_processed', summary.get('total', 0))}")
        print(f"  Surfaced:           {summary.get('surfaced', 0)}")
        print(f"  Stored:             {summary.get('stored', 0)}")
        print(f"  Ignored:            {summary.get('ignored', 0)}")
        print(f"  Blocked:            {summary.get('blocked', 0)}")
        print(f"  Duplicates:         {summary.get('duplicates', 0)}")

    ctx_ids = state.get("context_ids", {})
    if ctx_ids:
        separator()
        info("  Context IDs:")
        for name, uid in ctx_ids.items():
            info(f"    {name}: {uid}")

    print()
    return 0
