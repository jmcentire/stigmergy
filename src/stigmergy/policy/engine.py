"""Policy engine: the compiler from observation to intervention.

Architecture:
  1. Mesh processes signals, producing trace events (the evidence)
  2. Mechanical detectors extract policy signals (the facts)
  3. The policy agent evaluates facts + budget context (the judgment)
  4. Interventions are produced (the action)

The engine codifies WHAT to watch for (signals, budgets, boundaries).
The agent decides WHAT TO DO about it. This is the constitution, not
the legislature.

The policy agent prompt is the program. The schemas constrain the output.
The budget context enriches the input. Everything else is plumbing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from stigmergy.policy.budgets import ActivityBudget, BudgetConfig
from stigmergy.policy.signals import PolicySignal, detect_signals
from stigmergy.policy.spectral import (
    SpectralAnalyzer,
    SpectralSnapshot,
    build_adjacency_from_graph,
    build_flow_weights_from_traces,
)
from stigmergy.policy.structure import StructureGraph
from stigmergy.policy.traces import TraceEvent, TraceStore

logger = logging.getLogger(__name__)


@dataclass
class Intervention:
    """An action recommended by the policy agent.

    Advisory interventions are informational (flag in output).
    Gate interventions would block (future: pre-merge CI checks).
    For now, everything is advisory — the system observes and reports.
    """

    intervention_type: str  # "advisory", "gate" (future)
    policy_name: str  # which policy produced this
    signal_type: str  # which signal triggered it
    repo: str
    author: str
    message: str
    severity: str = "medium"  # "low", "medium", "high", "critical"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def one_line(self) -> str:
        icon = {"low": ".", "medium": "!", "high": "!!", "critical": "!!!"}
        return f"[{icon.get(self.severity, '?')}] {self.policy_name}: {self.message}"


# ── Policy Agent Prompt ─────────────────────────────────────────

POLICY_AGENT_SYSTEM = """\
You are a policy evaluation agent for an engineering organization.

You receive structural facts about code changes (dependency crossings, \
auth surface modifications, high churn) along with budget context \
(how much activity this repo has seen this week relative to thresholds).

Your job is to decide which facts warrant human attention and what the \
appropriate response is. You are the judgment layer — the facts are \
mechanical, but the decision is yours.

Intervene when:
- A dependency crossing lacks coordination (different people, no shared \
  ticket, touching the same interface from different sides)
- Auth surface changes aren't paired with security review signals
- A repo is burning through its activity budget unusually fast, \
  suggesting scope creep or emergency firefighting
- Multiple signals compound: high churn + dependency crossing + auth \
  change = something needs attention

Do NOT intervene when:
- The activity is routine and within normal patterns
- A single person is making coordinated changes across their own PRs
- The budget utilization is elevated but not unusual for the team's \
  current sprint
- The facts are individually unremarkable and don't compound

Your interventions should be specific and actionable: "Alice's pricing \
PR crosses into availability without a shared schema contract" is useful. \
"Activity detected in multiple domains" is not.

Severity guide:
- low: worth noting in a digest, not worth interrupting for
- medium: should be flagged to the relevant team lead
- high: needs attention before merge/deploy
- critical: stop what you're doing and look at this\
"""


def policy_agent_prompt(
    signals: list[PolicySignal],
    budget_context: dict[str, dict[str, Any]],
    recent_interventions: list[str],
) -> str:
    """Build the user prompt for policy evaluation."""
    parts = ["Evaluate these structural facts and decide which warrant intervention.\n"]

    parts.append("## Signals detected\n")
    for i, sig in enumerate(signals, 1):
        parts.append(f"[{i}] {sig.signal_type} — {sig.summary}")
        if sig.details:
            for k, v in sig.details.items():
                if isinstance(v, list) and len(v) > 5:
                    parts.append(f"    {k}: {v[:5]} (+{len(v)-5} more)")
                else:
                    parts.append(f"    {k}: {v}")
        parts.append("")

    if budget_context:
        parts.append("## Budget context (this week)\n")
        for repo, metrics in budget_context.items():
            parts.append(f"  {repo}:")
            for metric, status in metrics.items():
                flag = " [EXCEEDED]" if status.get("exceeded") else ""
                parts.append(f"    {metric}: {status['current']}/{status['threshold']}{flag}")
        parts.append("")

    if recent_interventions:
        parts.append("## Recent interventions (avoid duplicates)\n")
        for ri in recent_interventions[-5:]:
            parts.append(f"  - {ri}")
        parts.append("")

    parts.append("Respond with interventions for facts that warrant action, "
                  "or an empty list if nothing needs attention.")
    return "\n".join(parts)


# ── Schemas for agent output ────────────────────────────────────

# These are imported by the engine when LLM is available.
# When no LLM, the engine falls back to mechanical rules.

from pydantic import BaseModel, Field


class InterventionDecision(BaseModel):
    """A single intervention decision from the policy agent."""

    should_intervene: bool = Field(description="Whether this warrants human attention")
    policy_name: str = Field(description="Short name: boundary_contract, auth_gate, churn_review")
    message: str = Field(description="Specific, actionable description of the concern")
    severity: str = Field(description="low, medium, high, or critical")
    signal_indices: list[int] = Field(
        default_factory=list,
        description="Which signal numbers (1-indexed) this intervention addresses",
    )


class PolicyEvaluation(BaseModel):
    """Batch output from the policy agent."""

    interventions: list[InterventionDecision] = Field(
        default_factory=list,
        description="Interventions for signals that warrant action. Empty if nothing needs attention.",
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning about why these signals do or don't warrant intervention",
    )


# ── The Engine ──────────────────────────────────────────────────

class PolicyEngine:
    """Compiles trace events into interventions via signal detection + agent judgment.

    The engine is the orchestrator:
    1. Mechanical signal detection (fast, cheap, structural facts)
    2. Budget context assembly (how stressed is this repo?)
    3. Agent evaluation (judgment call — LLM when available, heuristic fallback)
    4. Intervention output (advisory flags for the mesh output)
    """

    def __init__(
        self,
        graph: StructureGraph,
        budget: ActivityBudget,
        trace_store: TraceStore,
        llm=None,
    ) -> None:
        self._graph = graph
        self._budget = budget
        self._traces = trace_store
        self._llm = llm
        self._recent_interventions: list[str] = []
        self._spectral = SpectralAnalyzer()
        self._spectral_interval = 50  # Run spectral analysis every N events
        self._events_since_spectral = 0

    async def evaluate(self, event: TraceEvent) -> list[Intervention]:
        """Full pipeline: detect signals → check budgets → spectral check → agent judgment."""
        # 1. Record trace
        self._traces.append(event)

        # 2. Detect structural signals (mechanical)
        signals = detect_signals(event, self._graph)

        # 3. Periodic spectral analysis — the algedonic channel
        self._events_since_spectral += 1
        if self._events_since_spectral >= self._spectral_interval:
            self._events_since_spectral = 0
            algedonic = self._check_spectral()
            if algedonic:
                signals.append(PolicySignal(
                    signal_type="spectral_right_shift",
                    trace_id=event.id,
                    repo=event.repo,
                    author="system",
                    severity=algedonic["severity"],
                    details=algedonic,
                ))

        if not signals:
            return []

        # 4. Record budget activity
        self._record_budget(event, signals)

        # 5. Assemble budget context for affected repos
        repos = {event.repo} | {s.repo for s in signals}
        budget_context = {repo: self._budget.status_for(repo) for repo in repos}

        # 6. Agent evaluation or mechanical fallback
        if self._llm is not None:
            try:
                return await self._evaluate_agent(signals, budget_context)
            except Exception:
                logger.debug("Policy agent failed, falling back to mechanical", exc_info=True)

        return self._evaluate_mechanical(signals, budget_context)

    async def _evaluate_agent(
        self,
        signals: list[PolicySignal],
        budget_context: dict[str, dict],
    ) -> list[Intervention]:
        """LLM-powered policy evaluation."""
        prompt = policy_agent_prompt(signals, budget_context, self._recent_interventions)
        evaluation = await self._llm.extract(
            PolicyEvaluation,
            prompt,
            POLICY_AGENT_SYSTEM,
            max_tokens=500,
        )

        interventions = []
        for decision in evaluation.interventions:
            if not decision.should_intervene:
                continue
            # Map signal indices back to signals
            related_signals = [signals[i - 1] for i in decision.signal_indices
                               if 0 < i <= len(signals)]
            repo = related_signals[0].repo if related_signals else signals[0].repo
            author = related_signals[0].author if related_signals else signals[0].author

            intervention = Intervention(
                intervention_type="advisory",
                policy_name=decision.policy_name,
                signal_type=related_signals[0].signal_type if related_signals else "compound",
                repo=repo,
                author=author,
                message=decision.message,
                severity=decision.severity,
                details={
                    "signals": [s.signal_type for s in related_signals],
                    "reasoning": evaluation.reasoning,
                },
            )
            interventions.append(intervention)
            self._recent_interventions.append(intervention.one_line)

        return interventions

    def _evaluate_mechanical(
        self,
        signals: list[PolicySignal],
        budget_context: dict[str, dict],
    ) -> list[Intervention]:
        """Heuristic fallback when no LLM available.

        Simple rules: only flag compounding signals or exceeded budgets.
        Single signals alone are not enough for mechanical mode — the
        mechanical path should be conservative to avoid noise.
        """
        interventions = []

        # Flag auth surface changes when budget is exceeded
        auth_signals = [s for s in signals if s.signal_type == "auth_surface_touched"]
        for sig in auth_signals:
            budget_status = budget_context.get(sig.repo, {})
            auth_budget = budget_status.get("auth_touches", {})
            if auth_budget.get("exceeded", False):
                interventions.append(Intervention(
                    intervention_type="advisory",
                    policy_name="auth_gate",
                    signal_type="auth_surface_touched",
                    repo=sig.repo,
                    author=sig.author,
                    message=f"Auth surface modified in {sig.repo} — "
                            f"auth touches this week: {auth_budget['current']}/{auth_budget['threshold']}",
                    severity="high",
                    details=sig.details,
                ))

        # Flag dependency crossings when they compound with budget pressure
        crossing_signals = [s for s in signals if s.signal_type == "dependency_crossing"]
        for sig in crossing_signals:
            budget_status = budget_context.get(sig.repo, {})
            crossing_budget = budget_status.get("dependency_crossings", {})
            if crossing_budget.get("exceeded", False):
                domains = sig.details.get("domains", [])
                interventions.append(Intervention(
                    intervention_type="advisory",
                    policy_name="boundary_contract",
                    signal_type="dependency_crossing",
                    repo=sig.repo,
                    author=sig.author,
                    message=f"PR in {sig.repo} crosses {', '.join(domains)} — "
                            f"crossings this week: {crossing_budget['current']}/{crossing_budget['threshold']}",
                    severity="medium",
                    details=sig.details,
                ))

        # Flag spectral right-shift as an algedonic intervention
        spectral_signals = [s for s in signals if s.signal_type == "spectral_right_shift"]
        for sig in spectral_signals:
            interventions.append(Intervention(
                intervention_type="advisory",
                policy_name="algedonic_channel",
                signal_type="spectral_right_shift",
                repo=sig.repo,
                author="system",
                message=sig.details.get("message", "Spectral anomaly detected in dependency graph"),
                severity=sig.severity,
                details=sig.details,
            ))

        for intervention in interventions:
            self._recent_interventions.append(intervention.one_line)

        return interventions

    def _record_budget(self, event: TraceEvent, signals: list[PolicySignal]) -> None:
        """Record activity against budgets."""
        # Always record diff lines
        if event.diff_lines > 0:
            self._budget.record(event.repo, "diff_lines", event.diff_lines, event.id)

        # Record per signal type
        for sig in signals:
            if sig.signal_type == "dependency_crossing":
                self._budget.record(event.repo, "dependency_crossings", 1, event.id)
            elif sig.signal_type == "auth_surface_touched":
                self._budget.record(event.repo, "auth_touches", 1, event.id)

    def _check_spectral(self) -> dict | None:
        """Run spectral analysis on the structure graph.

        Computes graph Laplacian eigenvalues, checks for right-shift
        (algedonic signal). Returns alert dict if anomaly detected.
        """
        try:
            adjacency = build_adjacency_from_graph(self._graph)
            if len(adjacency) < 3:
                return None

            flow_weights = build_flow_weights_from_traces(
                list(self._traces._events.values())
            )
            self._spectral.analyze(adjacency, flow_weights)
            return self._spectral.algedonic_alert()
        except Exception:
            logger.debug("Spectral analysis failed", exc_info=True)
            return None

    @property
    def spectral_snapshot(self) -> SpectralSnapshot | None:
        """Latest spectral energy measurement."""
        return self._spectral.latest

    @property
    def spectral_trend(self) -> str:
        """Spectral energy trend: 'stable', 'rising', or 'falling'."""
        return self._spectral.trend

    @property
    def trace_count(self) -> int:
        return self._traces.count

    @property
    def intervention_count(self) -> int:
        return len(self._recent_interventions)
