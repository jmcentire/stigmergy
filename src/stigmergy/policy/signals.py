"""Policy signals: structural facts extracted from trace events.

Signals are cheap, mechanical observations — "this PR touches files
in two service domains" or "this PR modifies an auth path." They are
NOT judgments. The agent decides what a signal means.

Signals are the data the policy agent reasons about.
The agent is the intelligence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stigmergy.policy.structure import StructureGraph
from stigmergy.policy.traces import TraceEvent


@dataclass
class PolicySignal:
    """A structural fact detected in a trace event."""

    signal_type: str  # "dependency_crossing", "auth_surface_touched", etc.
    trace_id: str  # Which trace event triggered this
    repo: str
    author: str
    severity: str = "info"  # "info", "medium", "high" — default, agent may override
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if self.signal_type == "dependency_crossing":
            domains = self.details.get("domains", [])
            return f"PR in {self.repo} crosses domains: {', '.join(domains)}"
        if self.signal_type == "auth_surface_touched":
            files = self.details.get("auth_files", [])
            return f"Auth surface modified in {self.repo}: {len(files)} file(s)"
        if self.signal_type == "high_churn":
            return f"High churn in {self.repo}: {self.details.get('diff_lines', 0)} lines changed"
        return f"{self.signal_type} in {self.repo}"


def detect_signals(event: TraceEvent, graph: StructureGraph) -> list[PolicySignal]:
    """Extract structural signals from a trace event.

    This is mechanical extraction — no judgment. It answers:
    "What structural facts are true about this event?"
    The policy agent decides what to do about them.
    """
    signals: list[PolicySignal] = []

    files = event.files_changed

    # 1. Dependency crossing: files span multiple service domains
    if files:
        domains = graph.crossing_domains(files)
        if len(domains) > 1:
            signals.append(PolicySignal(
                signal_type="dependency_crossing",
                trace_id=event.id,
                repo=event.repo,
                author=event.author,
                details={
                    "domains": sorted(domains),
                    "files": files,
                    "diff_lines": event.diff_lines,
                },
            ))

    # 2. Auth surface touched: files modify auth/security paths
    if files:
        auth_files = graph.auth_files(files)
        if auth_files:
            signals.append(PolicySignal(
                signal_type="auth_surface_touched",
                trace_id=event.id,
                repo=event.repo,
                author=event.author,
                details={
                    "auth_files": auth_files,
                    "total_files": len(files),
                    "diff_lines": event.diff_lines,
                },
            ))

    # 3. High churn: large diffs that may need extra review
    if event.diff_lines > 500:
        signals.append(PolicySignal(
            signal_type="high_churn",
            trace_id=event.id,
            repo=event.repo,
            author=event.author,
            details={
                "diff_lines": event.diff_lines,
                "additions": event.additions,
                "deletions": event.deletions,
                "files_count": len(files),
            },
        ))

    return signals
