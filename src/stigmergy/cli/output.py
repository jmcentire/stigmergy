"""Terminal formatting — no rich dependency, just ANSI codes."""

from __future__ import annotations

import sys

# ANSI codes
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"

_NO_COLOR = not sys.stdout.isatty()


def _wrap(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"{code}{text}{_RESET}"


def heading(text: str) -> None:
    print(_wrap(_BOLD + _CYAN, text))


def info(text: str) -> None:
    print(_wrap(_DIM, text))


def success(text: str) -> None:
    print(_wrap(_GREEN, text))


def warn(text: str) -> None:
    print(_wrap(_YELLOW, f"WARN: {text}"))


def error(text: str) -> None:
    print(_wrap(_RED, f"ERROR: {text}"), file=sys.stderr)


def signal_line(
    source: str,
    channel: str,
    author: str,
    action: str,
    confidence: float,
    content_preview: str,
) -> None:
    """Print a one-line signal summary."""
    src = _wrap(_BOLD, f"[{source.upper():6s}]")
    act_color = {
        "surface": _GREEN,
        "store": _DIM,
        "ignore": _DIM,
        "block": _RED,
        "replicate": _CYAN,
    }.get(action, "")
    act = _wrap(act_color, f"{action.upper():9s}")
    conf = f"{min(confidence, 1.0):.0%}"
    preview = content_preview[:80] + ("..." if len(content_preview) > 80 else "")
    print(f"  {src} {act} {conf:>4s}  {channel}  @{author}  {preview}")


def budget_line(daily_spent: float, daily_cap: float, hourly_spent: float, hourly_cap: float) -> None:
    """Print budget status."""
    daily_pct = (daily_spent / daily_cap * 100) if daily_cap > 0 else 0
    hourly_pct = (hourly_spent / hourly_cap * 100) if hourly_cap > 0 else 0
    color = _GREEN if daily_pct < 80 else (_YELLOW if daily_pct < 100 else _RED)
    line = f"  Budget: ${daily_spent:.4f}/${daily_cap:.2f} daily ({daily_pct:.0f}%) | ${hourly_spent:.4f}/${hourly_cap:.2f} hourly ({hourly_pct:.0f}%)"
    print(_wrap(color, line))


def separator() -> None:
    print(_wrap(_DIM, "  " + "-" * 70))


def session_summary(
    signals_processed: int,
    surfaced: int,
    stored: int,
    ignored: int,
    blocked: int,
    duplicates: int,
    total_cost: float,
) -> None:
    """Print end-of-session summary."""
    print()
    heading("Session Summary")
    separator()
    print(f"  Signals processed:  {signals_processed}")
    print(f"  Surfaced:           {_wrap(_GREEN, str(surfaced))}")
    print(f"  Stored:             {stored}")
    print(f"  Ignored:            {ignored}")
    print(f"  Blocked:            {_wrap(_RED, str(blocked))}")
    print(f"  Duplicates:         {duplicates}")
    separator()
    print(f"  Total cost:         ${total_cost:.4f}")
    print()


# ── Mesh-specific formatters ──────────────────────────────────


def mesh_signal_line(
    source: str,
    channel: str,
    author: str,
    hops: int,
    workers_accepted: int,
    learning_mode: str,
    content_preview: str,
) -> None:
    """Print a one-line signal summary for mesh routing."""
    src = _wrap(_BOLD, f"[{source.upper():6s}]")
    mode_color = {
        "rag_indexed": _GREEN,
        "context_summarized": _CYAN,
        "weight_shifted": _DIM,
    }.get(learning_mode, "")
    mode = _wrap(mode_color, f"{learning_mode:20s}")
    hop_info = f"{hops}hop/{workers_accepted}accept"
    preview = content_preview[:60] + ("..." if len(content_preview) > 60 else "")
    print(f"  {src} {mode} {hop_info:>12s}  {channel}  @{author}  {preview}")


def mesh_topology(workers: list, labels: dict[str, str] | None = None) -> None:
    """Print worker topology summary. Workers are WorkerNode objects.

    If labels is provided, it maps worker_id[:8] -> agent-generated label.
    Falls back to w.label (mechanical) when no agent label available.
    """
    print()
    heading("Mesh Topology")
    separator()
    print(f"  {'Worker':<12s} {'Fullness':>10s} {'Threshold':>10s} {'Signals':>8s} {'Mode':>8s} {'Label'}")
    separator()
    for w in sorted(workers, key=lambda w: w.fullness, reverse=True):
        wid = str(w.id)[:8]
        fill_pct = f"{w.fullness:.0%}"
        thresh = f"{w.adaptive_threshold:.3f}"
        sigs = str(w.signals_accepted)
        source = w.source_name or "-"
        label = (labels or {}).get(wid, w.label)
        # Color based on fullness
        color = _GREEN if w.fullness < 0.5 else (_YELLOW if w.fullness < 0.9 else _RED)
        print(f"  {_wrap(color, wid):<22s} {fill_pct:>10s} {thresh:>10s} {sigs:>8s} {source:>8s} {label}")
    separator()
    print()


def mesh_signal_detail(
    source: str,
    channel: str,
    author: str,
    content_preview: str,
    hops: list,
    consensus_action: str | None,
    consensus_weight: float,
    output_delivered: bool,
    worker_labels: dict[str, str],
) -> None:
    """Print detailed signal routing with per-worker decisions."""
    src = _wrap(_BOLD, f"[{source.upper():6s}]")
    preview = content_preview[:80] + ("..." if len(content_preview) > 80 else "")
    print(f"\n  {src}  {channel}  @{author}")
    print(f"  {_wrap(_DIM, preview)}")

    for hop in hops:
        wid = str(hop.worker_id)[:8]
        label = worker_labels.get(wid, wid)
        mode_color = {
            "rag_indexed": _GREEN,
            "context_summarized": _CYAN,
            "weight_shifted": _DIM,
        }.get(hop.learning_mode, "")
        accepted = _wrap(_GREEN, "ACCEPT") if hop.accepted else _wrap(_DIM, "reject")
        mode = _wrap(mode_color, hop.learning_mode)
        score_bar = _score_bar(hop.score)
        print(f"    hop {hop.hop}  {wid} ({label:<20s})  {score_bar} {hop.score:.3f}  {accepted}  {mode}")

    if consensus_action:
        act_color = {
            "surface": _GREEN + _BOLD,
            "store": _DIM,
            "ignore": _DIM,
            "block": _RED,
        }.get(consensus_action, "")
        act = _wrap(act_color, consensus_action.upper())
        delivered = _wrap(_GREEN, "delivered") if output_delivered else _wrap(_DIM, "held")
        print(f"    => {act} (weight={consensus_weight:.2f}) {delivered}")


def _score_bar(score: float, width: int = 10) -> str:
    """Render a tiny inline bar for a familiarity score."""
    filled = int(width * min(score, 1.0))
    bar = "#" * filled + "." * (width - filled)
    color = _GREEN if score > 0.5 else (_YELLOW if score > 0.2 else _DIM)
    return _wrap(color, f"[{bar}]")


def mesh_lifecycle_event(event_type: str, detail: str) -> None:
    """Print a mesh lifecycle event (fork, merge, decay, spawn)."""
    icon = {"fork": ">>", "merge": "<<", "decay": "~~", "spawn": "++"}.get(event_type, "--")
    color = {"fork": _CYAN, "merge": _GREEN, "decay": _YELLOW, "spawn": _CYAN}.get(event_type, _DIM)
    print(_wrap(color, f"  {icon} {event_type.upper()}: {detail}"))


def mesh_poll_header(source: str, count: int, timestamp: str) -> None:
    """Print a poll cycle header."""
    print()
    separator()
    print(f"  {_wrap(_BOLD, source.upper())} poll  {count} signals  {_wrap(_DIM, timestamp)}")
    separator()


def mesh_scan_summary(
    signals_processed: int,
    signals_accepted: int,
    duplicates: int,
    windows_scanned: int,
    deepest_timestamp: str,
    stopped_reason: str,
    worker_count: int,
    avg_fullness: float,
    total_cost: float,
) -> None:
    """Print end-of-session summary for mesh mode."""
    print()
    heading("Mesh Session Summary")
    separator()
    print(f"  Signals processed:  {signals_processed}")
    print(f"  Accepted:           {_wrap(_GREEN, str(signals_accepted))}")
    print(f"  Duplicates:         {duplicates}")
    print(f"  Windows scanned:    {windows_scanned}")
    print(f"  Deepest timestamp:  {deepest_timestamp}")
    print(f"  Stop reason:        {stopped_reason}")
    separator()
    print(f"  Workers:            {worker_count}")
    print(f"  Avg fullness:       {avg_fullness:.0%}")
    separator()
    print(f"  Total cost:         ${total_cost:.4f}")
    print()


def intervention_line(intervention) -> None:
    """Print a policy intervention. Color by severity."""
    sev = intervention.severity
    color = {
        "critical": _RED + _BOLD,
        "high": _RED,
        "medium": _YELLOW,
        "low": _DIM,
    }.get(sev, _DIM)
    icon = {"critical": "!!!", "high": "!!", "medium": "!", "low": "."}
    sev_icon = icon.get(sev, "?")
    policy = intervention.policy_name.replace("_", " ").upper()
    print(f"    {_wrap(color, f'>> [{sev_icon}] {policy}')}  {intervention.message}")


def policy_summary(trace_count: int, intervention_count: int, budget_exceeded: list,
                   spectral_snapshot=None) -> None:
    """Print policy engine summary at end of run."""
    if trace_count == 0 and intervention_count == 0 and spectral_snapshot is None:
        return
    separator()
    print(f"  Policy: {trace_count} traces, {intervention_count} interventions")
    if budget_exceeded:
        for repo, metric, current, threshold in budget_exceeded:
            metric_label = metric.replace("_", " ")
            repo_short = repo.split("/")[-1] if "/" in repo else repo
            print(f"    {_wrap(_YELLOW, f'[budget] {repo_short}: {metric_label} {current}/{threshold}')}")
    if spectral_snapshot is not None:
        _spectral_line(spectral_snapshot)


def _spectral_line(snapshot) -> None:
    """Print spectral energy summary."""
    hi = snapshot.high_freq_ratio
    lo = snapshot.low_freq_ratio
    color = _GREEN if hi < 0.55 else (_YELLOW if hi < 0.65 else _RED)
    gap = f"gap={snapshot.spectral_gap:.3f}"
    eff_dim = f"eff_dim={snapshot.effective_dimensionality:.2f}"
    shift = ""
    if snapshot.is_right_shifted:
        shift = _wrap(_RED + _BOLD, " [RIGHT-SHIFT]")
    elif snapshot.is_left_shifted:
        shift = _wrap(_MAGENTA + _BOLD, " [LEFT-SHIFT: eigenvalue collapse]")
    print(f"    {_wrap(color, f'[spectral] hi-freq={hi:.0%} lo-freq={lo:.0%} {gap} {eff_dim} nodes={snapshot.node_count}')}{shift}")


def _classification_color(classification: str) -> str:
    """Color code by finding classification type."""
    return {
        "discovery": _GREEN + _BOLD,
        "correlation": _GREEN,
        "amplification": _YELLOW,
        "retrieval": _DIM,
    }.get(classification, _DIM)


def insight_line(insight) -> None:
    """Print a single insight inline during signal processing.

    Insight types are agent-chosen strings — the formatter is generic.
    Shows the four-part structure when available.
    """
    conf = insight.confidence
    classification = insight.details.get("classification", "")
    temporal = insight.details.get("temporal_relevance", "")

    # Color by classification if available, fall back to confidence-based
    if classification:
        color = _classification_color(classification)
    else:
        color = _GREEN + _BOLD if conf > 0.7 else (_YELLOW if conf > 0.4 else _DIM)

    conf_str = f"{conf:.0%}"
    label = insight.type.replace("_", " ").upper()
    class_tag = f" {classification.upper()}" if classification else ""
    temporal_tag = f" [{temporal}]" if temporal else ""
    print(f"    {_wrap(color, f'>> {label}{class_tag}{temporal_tag}')} {conf_str:>4s}  {insight.summary}")
    # Show action-oriented fields inline (compact form)
    action = insight.details.get("recommended_action", "")
    if action:
        print(f"      {_wrap(_GREEN, 'Action:')} {action[:120]}")
    risk = insight.details.get("inaction_risk", "")
    if risk:
        print(f"      {_wrap(_RED, 'Risk:')} {risk[:120]}")


def findings_summary(insights: list, couplings: list | None = None) -> int:
    """Print end-of-run Key Findings section.

    This is the system's primary business-value output: actionable
    patterns the LLM correlator found across signals. Sorted by
    confidence, grouped by type, never suppressed.

    Returns the number of unique findings after display-level dedup.
    """
    if not insights and not couplings:
        return 0

    print()
    print(_wrap(_BOLD + _CYAN, "Key Findings"))
    print(_wrap(_DIM, "  " + "-" * 70))

    unique_count = 0
    if insights:
        # Deduplicate by summary (same insight can fire multiple times)
        seen: set[str] = set()
        unique = []
        for ins in sorted(insights, key=lambda i: i.confidence, reverse=True):
            key = ins.summary[:80]
            if key not in seen:
                seen.add(key)
                unique.append(ins)
        unique_count = len(unique)

        for ins in unique[:20]:  # Top 20
            conf = ins.confidence
            classification = ins.details.get("classification", "")
            temporal = ins.details.get("temporal_relevance", "")

            # Color by classification if available, fall back to confidence-based
            if classification:
                color = _classification_color(classification)
            else:
                color = _GREEN + _BOLD if conf > 0.7 else (_YELLOW if conf > 0.4 else _DIM)

            conf_str = f"{conf:.0%}"
            label = ins.type.replace("_", " ").upper()
            # Show classification label in header
            class_tag = f" {classification.upper()}" if classification else ""
            temporal_tag = f" [{temporal}]" if temporal else ""
            sev = ins.details.get("severity", "")
            sev_tag = f" [{sev}]" if sev and not classification else ""
            actors = ins.details.get("actors", [])
            actor_tag = f"  ({', '.join(actors)})" if actors else ""

            # S(f) score — the paper's Equation 3
            sf_score = ins.details.get("sf_score", 0.0)
            sf_tag = ""
            if sf_score > 0:
                sf_comps = ins.details.get("sf_components", {})
                parts = []
                if sf_comps:
                    for k in ("d_bridge", "c_entity", "r_coupling", "gamma"):
                        v = sf_comps.get(k, 0)
                        parts.append(f"{k}={v:.2f}")
                sf_tag = f"  S(f)={sf_score:.3f}"
                if parts:
                    sf_tag += f" [{', '.join(parts)}]"

            # Normalized Deviance per-finding indicators (Section 5.2-5.3)
            nd_tag = ""
            _nd_ci = ins.details.get("compression_index", 0.0)
            _nd_ac = ins.details.get("action_correlation", -1.0)
            _nd_ld = ins.details.get("lexical_diversity", 0.0)
            _nd_temp = ins.details.get("temporal_orientation", 0.0)
            nd_parts = []
            if _nd_ci > 0.2:
                ci_color = _RED if _nd_ci > 0.5 else _YELLOW
                nd_parts.append(_wrap(ci_color, f"compress={_nd_ci:.2f}"))
            if _nd_ld > 0 and _nd_ld < 0.15:
                ld_color = _RED if _nd_ld < 0.13 else _YELLOW
                nd_parts.append(_wrap(ld_color, f"LD={_nd_ld:.3f}"))
            if _nd_temp < -0.3:
                nd_parts.append(_wrap(_RED, f"temporal={_nd_temp:+.2f}"))
            if _nd_ac >= 0 and _nd_ac < 0.2:
                ac_color = _RED if _nd_ac < 0.05 else _YELLOW
                nd_parts.append(_wrap(ac_color, f"act-corr={_nd_ac:.2f}"))
            if nd_parts:
                nd_tag = f"  ND[{', '.join(nd_parts)}]"

            # Header line: confidence, classification, temporal, actors, score, ND
            print(f"\n  {_wrap(color, f'{conf_str}{class_tag}{temporal_tag}{sev_tag}')}{actor_tag}{sf_tag}{nd_tag}")
            # Summary (what was found)
            print(f"    {ins.summary}")

            # Unknowns (what we don't know yet)
            unknowns = ins.details.get("unknowns", "")
            if unknowns:
                print(f"    {_wrap(_CYAN, 'Unknown:')} {unknowns}")

            # Recommended action
            action = ins.details.get("recommended_action", "")
            if action:
                print(f"    {_wrap(_GREEN, 'Action:')} {action}")

            # Inaction risk (cost of doing nothing)
            risk = ins.details.get("inaction_risk", "")
            if risk:
                print(f"    {_wrap(_RED, 'If ignored:')} {risk}")

            # Artifacts (concrete copy-pasteable actions)
            artifacts = ins.details.get("artifacts", [])
            for art in artifacts:
                art_type = art.get("artifact_type", "unknown")
                art_target = art.get("target", "")
                art_content = art.get("content", "")
                if art_content:
                    target_str = f" → {art_target}" if art_target else ""
                    print(f"    {_wrap(_MAGENTA, f'Artifact: [{art_type}]{target_str}')}")
                    # Indent artifact content for readability
                    for line in art_content.split("\n"):
                        print(f"      {line}")

    if couplings:
        print()
        print(_wrap(_BOLD, "  Structural Couplings"))
        for c in couplings[:10]:
            src_info = f"{c.source_a}/{c.channel_a} <-> {c.source_b}/{c.channel_b}"
            print(f"    sim={c.activation_similarity:.2f} "
                  f"workers={len(c.shared_workers)}  {src_info}")

    print(_wrap(_DIM, "  " + "-" * 70))
    print()

    return unique_count


def insight_funnel(raw_total: int, agent_deduped: int, display_unique: int) -> None:
    """Print the insight dedup funnel: raw → agent-dedup → display-unique."""
    if raw_total == 0:
        return

    agent_dropped = raw_total - agent_deduped
    display_dropped = agent_deduped - display_unique

    print(_wrap(_BOLD, "Insight Funnel"))
    print(f"  Raw insights produced:   {raw_total}")
    print(f"  After agent dedup:       {agent_deduped}  (-{agent_dropped})")
    print(f"  After display dedup:     {display_unique}  (-{display_dropped})")
    ratio = f"{display_unique}/{raw_total}" if raw_total else "N/A"
    pct = f"{display_unique / raw_total:.0%}" if raw_total else "N/A"
    print(f"  Collapse ratio:          {ratio} ({pct} survived)")
    print()


def intelligence_report(agent_diagnostics: list[dict]) -> None:
    """Print agent intelligence monitoring report.

    Shows whether agents are exercising LLM intelligence or falling
    back to mechanical heuristics. This is the system's health check:
    if agents aren't reasoning, the architecture isn't flexing.
    """
    if not agent_diagnostics:
        return

    print()
    print(_wrap(_BOLD + _CYAN, "Agent Intelligence Report"))
    print(_wrap(_DIM, "  " + "-" * 70))

    total_agents = len(agent_diagnostics)
    with_llm = sum(1 for d in agent_diagnostics if d["has_llm"])
    without_llm = total_agents - with_llm

    # Aggregate counters
    total_llm_reflect = sum(d["reflect"]["llm_calls"] for d in agent_diagnostics)
    total_mech_reflect = sum(d["reflect"]["mechanical_calls"] for d in agent_diagnostics)
    total_llm_skipped = sum(d["reflect"]["llm_skipped_non_batch"] for d in agent_diagnostics)
    total_dedup_skips = sum(d["reflect"].get("batch_dedup_skips", 0) for d in agent_diagnostics)
    total_llm_eval = sum(d["evaluate"]["llm_calls"] for d in agent_diagnostics)
    total_mech_eval = sum(d["evaluate"]["mechanical_calls"] for d in agent_diagnostics)
    all_active = sum(1 for d in agent_diagnostics if d["intelligence_active"])

    # Header stats
    if without_llm > 0:
        llm_color = _RED + _BOLD
        llm_status = f"{with_llm}/{total_agents} agents have LLM ({without_llm} MISSING)"
    else:
        llm_color = _GREEN
        llm_status = f"{with_llm}/{total_agents} agents have LLM"
    print(f"  {_wrap(llm_color, llm_status)}")

    # Reflect path
    if total_mech_reflect > 0:
        reflect_color = _RED + _BOLD
    elif total_llm_reflect > 0:
        reflect_color = _GREEN
    else:
        reflect_color = _DIM
    dedup_part = f", {total_dedup_skips} batch-dedup skipped" if total_dedup_skips > 0 else ""
    print(f"  {_wrap(reflect_color, f'Correlator (reflect):  {total_llm_reflect} LLM calls, {total_mech_reflect} mechanical, {total_llm_skipped} non-batch skipped{dedup_part}')}")
    if total_dedup_skips > 0 and total_llm_reflect > 0:
        saved_pct = total_dedup_skips / (total_llm_reflect + total_dedup_skips) * 100
        print(f"  {_wrap(_GREEN, f'Batch dedup saved {total_dedup_skips} redundant LLM calls ({saved_pct:.0f}% reduction)')}")

    # Evaluate path
    if total_mech_eval > 0:
        eval_color = _YELLOW
    elif total_llm_eval > 0:
        eval_color = _GREEN
    else:
        eval_color = _DIM
    print(f"  {_wrap(eval_color, f'Surfacer (evaluate):   {total_llm_eval} LLM calls, {total_mech_eval} mechanical')}")

    # Intelligence status
    if all_active == total_agents and total_agents > 0:
        print(f"  {_wrap(_GREEN + _BOLD, 'All agents exercising intelligence — architecture is flexing')}")
    elif without_llm > 0:
        print(f"  {_wrap(_RED + _BOLD, f'WARNING: {without_llm} agents running without LLM — mechanical heuristics producing noise')}")
        # Show which agents are affected
        naked = [d for d in agent_diagnostics if not d["has_llm"]]
        for d in naked[:5]:
            mc = d["reflect"]["mechanical_calls"]
            me = d["evaluate"]["mechanical_calls"]
            print(f"    {_wrap(_RED, f'agent {d["agent_id"]}: {mc} mechanical reflects, {me} mechanical evals')}")
        if len(naked) > 5:
            print(f"    {_wrap(_DIM, f'... and {len(naked) - 5} more')}")

    print(_wrap(_DIM, "  " + "-" * 70))
    print()


class InsightDeduplicator:
    """Suppress repeated inline insight display within a single run.

    The correlator fires every N signals and often rediscovers the same
    patterns (especially when batch overlap >80% triggers cache return).
    This deduplicator ensures each unique insight is shown inline once,
    while still collecting all instances for the end-of-run report.
    """

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._suppressed = 0

    def should_display(self, insight) -> bool:
        """True if this insight hasn't been shown inline yet."""
        key = (insight.type, insight.summary[:80])
        if key in self._seen:
            self._suppressed += 1
            return False
        self._seen.add(key)
        return True

    def flush(self) -> None:
        """Print suppression summary and reset counter."""
        if self._suppressed > 0:
            print(_wrap(_DIM, f"  ... {self._suppressed} duplicate insight(s) suppressed"))
            self._suppressed = 0


class NoveltyTracker:
    """Suppress repetitive output with exponential decay.

    First occurrence of a category: 'detail' (always shown, verbose).
    Subsequent: 'compact' with probability 0.5^n, or 'suppress'.
    Below floor probability: always suppress.
    Call flush() after a batch to print what was hidden.
    """

    def __init__(self, decay: float = 0.5, floor: float = 0.05):
        self._counts: dict[str, int] = {}
        self._suppressed: dict[str, int] = {}
        self._decay = decay
        self._floor = floor

    def check(self, category: str) -> str:
        """Returns 'detail', 'compact', or 'suppress'."""
        import random

        n = self._counts.get(category, 0)
        self._counts[category] = n + 1
        if n == 0:
            return "detail"
        prob = self._decay ** n
        if prob < self._floor:
            self._suppressed[category] = self._suppressed.get(category, 0) + 1
            return "suppress"
        if random.random() < prob:
            return "compact"
        self._suppressed[category] = self._suppressed.get(category, 0) + 1
        return "suppress"

    def flush(self) -> None:
        """Print summary of suppressed items and reset."""
        if not self._suppressed:
            return
        parts = [f"{count} {cat}" for cat, count in sorted(self._suppressed.items())]
        total = sum(self._suppressed.values())
        self._suppressed.clear()
        print(_wrap(_DIM, f"  ... {total} suppressed ({', '.join(parts)})"))


def decay_summary(finding_registry) -> None:
    """Print finding decay tracking summary.

    Shows the trajectory from Discovery → Normalized Deviance (paper Section 7).
    Findings that resurface without state change are migrating toward
    organizational blindness.
    """
    stats = finding_registry.summary()
    if stats["total"] == 0:
        return

    print()
    print(_wrap(_BOLD + _CYAN, "Finding Registry"))
    print(_wrap(_DIM, "  " + "-" * 70))
    print(f"  Total findings: {stats['total']}  "
          f"({_wrap(_GREEN, f'{stats["active"]} active')}, "
          f"{_wrap(_YELLOW, f'{stats["deferred"]} deferred')}, "
          f"{_wrap(_RED, f'{stats["normalized"]} normalized')}, "
          f"{stats['acknowledged']} acknowledged)")

    # Show decaying findings
    decaying = finding_registry.decaying_findings()
    if decaying:
        print()
        print(f"  {_wrap(_YELLOW + _BOLD, f'Decay Alert: {len(decaying)} findings migrating toward Normalized Deviance')}")
        for f in sorted(decaying, key=lambda x: x.times_surfaced, reverse=True)[:5]:
            stage_color = _RED if f.decay_stage == "normalized" else _YELLOW
            stage = _wrap(stage_color, f.decay_stage.upper())
            print(f"    {stage}  surfaced {f.times_surfaced}x over {f.days_since_first:.0f}d  "
                  f"S(f)={f.max_score:.3f}  {f.summary[:80]}")
            if f.actors:
                print(f"      actors: {', '.join(f.actors[:5])}")

    # Show normalized findings — the ones that have fully decayed
    normalized = finding_registry.normalized_findings()
    if normalized:
        print()
        print(f"  {_wrap(_RED + _BOLD, f'Normalized Deviance: {len(normalized)} findings — known but unacted upon')}")
        for f in normalized[:3]:
            print(f"    {_wrap(_RED, f'[{f.times_surfaced}x/{f.days_since_first:.0f}d]')}  {f.summary[:90]}")

    print(_wrap(_DIM, "  " + "-" * 70))
    print()


def corroboration_summary(quorum_findings: list, metrics: dict) -> None:
    """Print Phase 2 cross-signal corroboration results.

    Shows which findings achieved quorum (corroborated by multiple workers
    from different contexts) and the overall corroboration stats.
    """
    if not quorum_findings and not metrics.get("findings_input"):
        return

    print()
    print(_wrap(_BOLD + _CYAN, "Phase 2: Cross-Signal Corroboration"))
    print(_wrap(_DIM, "  " + "-" * 70))

    # Stats line
    findings_in = metrics.get("findings_input", 0)
    workers_consulted = metrics.get("workers_consulted", 0)
    llm_calls = metrics.get("llm_calls", 0)
    mechanical = metrics.get("mechanical_calls", 0)
    quorum_met = metrics.get("quorum_met", 0)
    quorum_size = metrics.get("quorum_size", 2)

    print(f"  Findings submitted:  {findings_in}")
    print(f"  Workers consulted:   {workers_consulted} ({llm_calls} LLM, {mechanical} mechanical)")
    print(f"  Quorum threshold:    {quorum_size} workers")
    print(f"  Quorum achieved:     {_wrap(_GREEN + _BOLD if quorum_met > 0 else _DIM, str(quorum_met))}")

    # Show corroborated findings
    corroborated = [q for q in quorum_findings if q.quorum_met]
    if corroborated:
        print()
        print(f"  {_wrap(_GREEN + _BOLD, 'Corroborated Findings:')}")
        for qf in sorted(corroborated, key=lambda q: q.quorum_confidence, reverse=True):
            conf_str = f"{qf.quorum_confidence:.0%}"
            votes = f"{qf.corroboration_count}/{qf.total_consulted}"
            label = qf.original.type.replace("_", " ").upper()
            color = _GREEN + _BOLD if qf.quorum_confidence > 0.7 else _YELLOW
            print(f"\n    {_wrap(color, f'{conf_str} {label}')}  [{votes} corroborated]")
            print(f"      {qf.original.summary[:120]}")
            # Show worker verdicts (compact)
            for v in qf.verdicts:
                v_color = {
                    "corroborate": _GREEN,
                    "contradict": _RED,
                    "partial": _YELLOW,
                    "no_evidence": _DIM,
                }.get(v.verdict, _DIM)
                wid = str(v.worker_id)[:8]
                print(f"      {_wrap(v_color, f'{v.verdict:13s}')} ({v.confidence:.0%}) worker {wid}: {v.reasoning[:80]}")

    # Show contradicted findings
    contradicted = [q for q in quorum_findings if q.contradiction_count > 0]
    if contradicted:
        print()
        print(f"  {_wrap(_RED, 'Contested Findings:')}")
        for qf in contradicted[:5]:
            votes = f"{qf.contradiction_count}/{qf.total_consulted} contradicted"
            print(f"    {_wrap(_RED, votes)}  {qf.original.summary[:100]}")

    # Show findings that didn't achieve quorum
    no_quorum = [q for q in quorum_findings if not q.quorum_met and q.contradiction_count == 0]
    if no_quorum:
        count = len(no_quorum)
        print(f"\n  {_wrap(_DIM, f'{count} findings did not achieve quorum (insufficient independent corroboration)')}")

    print(_wrap(_DIM, "  " + "-" * 70))
    print()


def identity_summary(resolver) -> None:
    """Print unresolved identity summary at end of run.

    Shows how many identifiers fell through to passthrough and lists
    the most common unknowns with fuzzy-match suggestions.
    """
    unknowns = resolver.unknown_identifiers
    if not unknowns:
        return

    print()
    heading("Identity Resolution")
    separator()
    known = resolver.person_count
    total = known + len(unknowns)
    print(f"  Known identities:     {known}")
    print(f"  Unresolved:           {_wrap(_YELLOW, str(len(unknowns)))}")
    if total > 0:
        print(f"  Resolution rate:      {known / total * 100:.0f}%")

    # Show top unknowns with suggestions
    sorted_unknowns = sorted(unknowns)[:15]
    if sorted_unknowns:
        print()
        print(f"  {_wrap(_DIM, 'Top unresolved identifiers:')}")
        for unk in sorted_unknowns:
            suggestions = resolver.suggest_match(unk)
            if suggestions:
                best_id, best_score = suggestions[0]
                hint = _wrap(_DIM, f"  -> {best_id}? ({best_score:.0%})")
            else:
                hint = ""
            print(f"    {_wrap(_YELLOW, unk)}{hint}")
        if len(unknowns) > 15:
            print(f"    {_wrap(_DIM, f'... and {len(unknowns) - 15} more')}")


def attention_findings_summary(decisions: list, person_name: str) -> None:
    """Print per-person findings ranked by surfacing value.

    Shows findings annotated with P(knows), importance, and margin.
    Includes transparency layer: breakdown of why findings were surfaced
    or suppressed, so the user understands the model's reasoning.
    """
    surfaceable = [d for d in decisions if d.margin > 0]
    suppressed_known = [d for d in decisions if d.margin <= 0 and d.p_knows > 0.5]
    suppressed_low = [d for d in decisions if d.margin <= 0 and d.p_knows <= 0.5]

    total = len(decisions)
    shown = len(surfaceable)

    if not surfaceable and total == 0:
        return

    print()
    print(_wrap(_BOLD + _CYAN, f"Findings for {person_name}:"))
    print(_wrap(_DIM, "  " + "-" * 70))

    # Transparency header — always show what the model decided and why
    if total > 0:
        parts = [f"Showing {_wrap(_BOLD, str(min(shown, 15)))} of {total} findings"]
        if suppressed_known:
            parts.append(f"{len(suppressed_known)} suppressed as likely known")
        if suppressed_low:
            parts.append(f"{len(suppressed_low)} below significance threshold")
        print(f"  {'. '.join(parts)}.")

    if not surfaceable:
        print(f"\n  {_wrap(_DIM, 'All findings suppressed — model estimates you already know these.')}")
        print(_wrap(_DIM, "  " + "-" * 70))
        print()
        return

    for d in surfaceable[:15]:
        # Color by margin strength
        if d.margin > 0.5:
            color = _GREEN + _BOLD
        elif d.margin > 0.2:
            color = _GREEN
        elif d.margin > 0:
            color = _YELLOW
        else:
            color = _DIM

        label = d.finding_type.replace("_", " ").upper()
        margin_str = f"+{d.margin:.2f}"
        print(f"\n  {_wrap(color, f'[{d.margin:.2f}]')} {_wrap(_BOLD, label)}  {d.finding_summary[:80]}")
        print(f"         P(knows)={d.p_knows:.2f}  importance={d.importance:.2f}  margin={_wrap(color, margin_str)}")
        print(f"         {_wrap(_DIM, d.exposure_summary)}")

    if shown > 15:
        print(f"\n  {_wrap(_DIM, f'... {shown - 15} more surfaceable findings not shown')}")

    print(_wrap(_DIM, "  " + "-" * 70))
    print()


def reemission_summary(events: list) -> None:
    """Print re-emission events — findings retriggered by context changes.

    These are findings that were previously suppressed or decayed but
    deserve renewed attention due to new evidence, compound risk, or
    approaching normalized deviance.
    """
    if not events:
        return

    print()
    print(_wrap(_BOLD + _MAGENTA, "Re-Emission Alerts"))
    print(_wrap(_DIM, "  " + "-" * 70))
    print(f"  {len(events)} finding(s) retriggered by context change:")

    trigger_icons = {
        "decay_escalation": ("!!", _RED + _BOLD),
        "compound_risk": ("++", _YELLOW + _BOLD),
        "new_evidence": (">>", _CYAN),
    }

    for event in events[:10]:
        icon, color = trigger_icons.get(event.trigger, ("--", _DIM))
        label = event.finding_type.replace("_", " ").upper()
        stage_color = _RED if event.decay_stage == "normalized" else _YELLOW
        stage = _wrap(stage_color, event.decay_stage.upper())

        print(f"\n  {_wrap(color, f'{icon} {label}')}  {stage}  S(f)={event.current_score:.3f}")
        print(f"     {event.summary[:100]}")
        print(f"     {_wrap(_DIM, event.trigger_detail)}")
        if event.actors:
            print(f"     actors: {', '.join(event.actors[:5])}")

    print(_wrap(_DIM, "  " + "-" * 70))
    print()


def normalized_deviance_summary(
    *,
    compression_signals: dict | None = None,
    action_correlations: list | None = None,
    migration_alerts: list | None = None,
    solo_alerts: list | None = None,
    phase_lock_alerts: list | None = None,
) -> None:
    """Print unified Normalized Deviance dashboard.

    Aggregates all five ND detectors into a single output section:
    linguistic compression, action-correlation, channel migration,
    solo ownership, and phase-lock detection.
    """
    has_content = any([
        compression_signals,
        action_correlations,
        migration_alerts,
        solo_alerts,
        phase_lock_alerts,
    ])
    if not has_content:
        return

    print()
    print(_wrap(_BOLD + _MAGENTA, "Normalized Deviance Indicators"))
    print(_wrap(_DIM, "  " + "-" * 70))

    # Linguistic compression
    if compression_signals:
        idx = compression_signals.get("compression_index", 0.0)
        color = _RED if idx > 0.5 else (_YELLOW if idx > 0.3 else _DIM)
        print(f"  {_wrap(color, '[compression]')} index={idx:.2f}  "
              f"hedging={compression_signals.get('hedging_density', 0):.2f}  "
              f"passive={compression_signals.get('passive_voice_ratio', 0):.2f}  "
              f"nominalization={compression_signals.get('nominalization_rate', 0):.2f}  "
              f"specificity={compression_signals.get('specificity_score', 0):.2f}")

        # Extended metrics (variance compression research)
        ld = compression_signals.get("lexical_diversity", 0.0)
        se = compression_signals.get("shannon_entropy", 0.0)
        temporal = compression_signals.get("temporal_orientation", 0.0)
        bureaucratic = compression_signals.get("bureaucratic_density", 0.0)

        ext_parts = []
        if ld > 0:
            # LD reference: S-1 filings 0.16-0.19 (pre-cage), 10-K 0.13-0.15 (caged)
            ld_color = _GREEN if ld > 0.16 else (_YELLOW if ld > 0.13 else _RED)
            ext_parts.append(_wrap(ld_color, f"LD={ld:.3f}"))
        if se > 0:
            ext_parts.append(f"SE={se:.2f}")
        if temporal != 0:
            t_color = _GREEN if temporal > 0.2 else (_RED if temporal < -0.2 else _DIM)
            t_label = "forward" if temporal > 0 else "backward"
            ext_parts.append(_wrap(t_color, f"temporal={temporal:+.2f} ({t_label})"))
        if bureaucratic > 0:
            b_color = _RED if bureaucratic > 0.3 else (_YELLOW if bureaucratic > 0.1 else _DIM)
            ext_parts.append(_wrap(b_color, f"bureaucratic={bureaucratic:.2f}"))
        if ext_parts:
            print(f"    {' '.join(ext_parts)}")

        # Per-channel trends
        increasing = compression_signals.get("increasing_channels", [])
        if increasing:
            print(f"    {_wrap(_YELLOW, f'{len(increasing)} channel(s) with increasing compression:')}")
            for ch in increasing:
                born_tag = ""
                if ch.get("born_caged"):
                    born_tag = _wrap(_RED, " [BORN CAGED]")
                ld_tag = f" LD={ch['ld']:.3f}" if ch.get("ld") else ""
                print(f"      {ch['channel']}: avg={ch['avg']:.2f} peak={ch['peak']:.2f}"
                      f" ({ch['samples']} signals){ld_tag}{born_tag}")

        # Born Caged channels
        born_caged = compression_signals.get("born_caged_channels", [])
        if born_caged:
            print(f"    {_wrap(_RED, f'{len(born_caged)} channel(s) born caged (initial compression >= 0.3):')}")
            for ch in born_caged:
                print(f"      {ch['channel']}: initial={ch['initial']:.2f} current={ch['avg']:.2f}"
                      f" delta={ch['avg'] - ch['initial']:+.2f}")

    # Action-correlation decoupling
    if action_correlations:
        decoupled = [a for a in action_correlations if a.trend == "decoupling"]
        if decoupled:
            print(f"\n  {_wrap(_YELLOW + _BOLD, f'{len(decoupled)} topic(s) decoupling (discussion without action):')}")
            for ac in decoupled[:5]:
                explore_tag = ""
                if ac.action_count > 0 and ac.exploration_ratio < 0.1:
                    explore_tag = _wrap(_RED, " [EXPLOITATION TRAP]")
                elif ac.exploration_ratio > 0.5:
                    explore_tag = _wrap(_GREEN, f" [exploring {ac.exploration_ratio:.0%}]")
                print(f"    {_wrap(_YELLOW, ac.topic)}: "
                      f"ratio={ac.ratio:.2f} ({ac.discussion_count} discussions, "
                      f"{ac.action_count} actions){explore_tag}")

        # Exploitation trap: healthy action ratio but all actions on same systems
        exploitation_trapped = [a for a in action_correlations
                               if a.trend == "healthy" and a.exploration_ratio < 0.1
                               and a.action_count >= 3]
        if exploitation_trapped:
            print(f"\n  {_wrap(_YELLOW, f'{len(exploitation_trapped)} topic(s) in exploitation trap '
                  f'(healthy action ratio, zero exploration):')}")
            for ac in exploitation_trapped[:3]:
                print(f"    {_wrap(_YELLOW, ac.topic)}: "
                      f"ratio={ac.ratio:.2f} explore={ac.exploration_ratio:.0%} "
                      f"({ac.action_count} actions, all on same systems)")

    # Channel migration
    if migration_alerts:
        print(f"\n  {_wrap(_RED + _BOLD, f'{len(migration_alerts)} topic(s) migrating to private channels:')}")
        for ma in migration_alerts[:5]:
            print(f"    score={ma.migration_score:.2f}  "
                  f"public={ma.public_count} private={ma.private_count} work={ma.work_activity_count}  "
                  f"{ma.summary[:80]}")

    # Solo ownership
    if solo_alerts:
        print(f"\n  {_wrap(_RED + _BOLD, f'{len(solo_alerts)} solo ownership alert(s):')}")
        for sa in solo_alerts[:5]:
            sev_color = _RED + _BOLD if sa.severity == "critical" else _YELLOW
            print(f"    {_wrap(sev_color, f'[{sa.severity}]')} "
                  f"owner=@{sa.sole_owner}  "
                  f"systems={sa.system_count}  stale={sa.staleness_days:.0f}d  "
                  f"signals={sa.signal_count}")

    # Phase-lock
    if phase_lock_alerts:
        print(f"\n  {_wrap(_RED + _BOLD, f'{len(phase_lock_alerts)} phase-lock alert(s) (converging risk framing):')}")
        for pl in phase_lock_alerts[:5]:
            sev_color = _RED + _BOLD if pl.severity == "critical" else _YELLOW
            print(f"    {_wrap(sev_color, f'[{pl.severity}]')} "
                  f"similarity={pl.framing_similarity:.2f}  "
                  f"findings={pl.finding_count}  "
                  f"topics: {', '.join(pl.risk_topics[:5])}")

    print(_wrap(_DIM, "  " + "-" * 70))
    print()


def portfolio_summary(clusters: list) -> None:
    """Print portfolio risk clusters — compound risk across related findings.

    Shows clusters where multiple findings share actors, channels, or terms,
    and their compound score exceeds what any individual finding achieves.
    """
    if not clusters:
        return

    escalated = [c for c in clusters if c.escalated]

    print()
    print(_wrap(_BOLD + _CYAN, "Portfolio Risk Clusters"))
    print(_wrap(_DIM, "  " + "-" * 70))
    print(f"  {len(clusters)} cluster(s) detected, {len(escalated)} escalated")

    for cluster in clusters[:8]:
        color = _RED + _BOLD if cluster.escalated else _YELLOW
        esc_tag = " [ESCALATED]" if cluster.escalated else ""
        print(f"\n  {_wrap(color, f'Compound S(f)={cluster.compound_score:.3f}')}"
              f"  (max individual={cluster.max_individual_score:.3f})"
              f"  {len(cluster.findings)} findings{_wrap(_RED + _BOLD, esc_tag)}")

        if cluster.shared_actors:
            print(f"    shared actors: {', '.join(cluster.shared_actors[:5])}")
        if cluster.shared_terms:
            print(f"    shared terms: {', '.join(cluster.shared_terms[:8])}")
        if cluster.shared_channels:
            print(f"    shared channels: {', '.join(cluster.shared_channels[:5])}")

        for f in cluster.findings[:4]:
            score = f.get("sf_score", 0.0)
            summary = f.get("summary", "")[:80]
            print(f"    {_wrap(_DIM, f'S(f)={score:.3f}')}  {summary}")
        if len(cluster.findings) > 4:
            print(f"    {_wrap(_DIM, f'... +{len(cluster.findings) - 4} more')}")

    print(_wrap(_DIM, "  " + "-" * 70))
    print()
