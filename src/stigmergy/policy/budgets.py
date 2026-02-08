"""Activity budgets: organizational capacity tracking.

Budgets are proxy measures for dysmemic pressure — when they spike,
the organization is exceeding its capacity to maintain coherence.
The agent sees budget state as context for its decisions.

Budget thresholds are declarative (codified). What to do when a
budget is exceeded is the agent's call.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    """Declarative thresholds for organizational activity budgets."""

    # Per-repo weekly limits (triggers are advisory, not hard blocks)
    diff_lines_per_week: int = 5000
    dependency_crossings_per_week: int = 5
    auth_touches_per_week: int = 3

    # Global
    window_days: int = 7


@dataclass
class BudgetEntry:
    """A single budget consumption event."""

    timestamp: str
    repo: str
    metric: str  # "diff_lines", "dependency_crossings", "auth_touches"
    value: int
    trace_id: str = ""


class ActivityBudget:
    """Tracks organizational activity metrics against thresholds.

    The budget doesn't enforce — it informs. The policy agent sees
    "dependency crossings this week: 7/5 (exceeded)" and decides
    whether to flag it, escalate it, or note it.
    """

    def __init__(self, config: BudgetConfig | None = None, path: Path | None = None) -> None:
        self._config = config or BudgetConfig()
        self._path = path or Path(".stigmergy/budgets.json")
        self._entries: list[BudgetEntry] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._entries = [BudgetEntry(**e) for e in data.get("entries", [])]
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.debug("Failed to load activity budgets: %s", exc)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"entries": [_entry_dict(e) for e in self._entries]}
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def record(self, repo: str, metric: str, value: int, trace_id: str = "") -> None:
        """Record an activity event."""
        self._entries.append(BudgetEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            repo=repo,
            metric=metric,
            value=value,
            trace_id=trace_id,
        ))

    def _window_entries(self, repo: str, metric: str) -> list[BudgetEntry]:
        """Entries within the budget window for a repo+metric."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._config.window_days)).isoformat()
        return [e for e in self._entries
                if e.repo == repo and e.metric == metric and e.timestamp >= cutoff]

    def current(self, repo: str, metric: str) -> int:
        """Current total for a repo+metric within the window."""
        return sum(e.value for e in self._window_entries(repo, metric))

    def threshold(self, metric: str) -> int:
        """The configured threshold for a metric."""
        thresholds = {
            "diff_lines": self._config.diff_lines_per_week,
            "dependency_crossings": self._config.dependency_crossings_per_week,
            "auth_touches": self._config.auth_touches_per_week,
        }
        return thresholds.get(metric, 0)

    def is_exceeded(self, repo: str, metric: str) -> bool:
        thresh = self.threshold(metric)
        return thresh > 0 and self.current(repo, metric) > thresh

    def utilization(self, repo: str, metric: str) -> float:
        """0.0 to 1.0+ (can exceed 1.0 when over budget)."""
        thresh = self.threshold(metric)
        if thresh <= 0:
            return 0.0
        return self.current(repo, metric) / thresh

    def status_for(self, repo: str) -> dict[str, dict[str, Any]]:
        """Budget status for all metrics for a repo. Fed to the policy agent."""
        metrics = ["diff_lines", "dependency_crossings", "auth_touches"]
        result = {}
        for m in metrics:
            current = self.current(repo, m)
            thresh = self.threshold(m)
            result[m] = {
                "current": current,
                "threshold": thresh,
                "utilization": round(current / thresh, 2) if thresh > 0 else 0.0,
                "exceeded": current > thresh if thresh > 0 else False,
            }
        return result

    def all_exceeded(self) -> list[tuple[str, str, int, int]]:
        """All (repo, metric, current, threshold) tuples that are exceeded."""
        exceeded = []
        repos = {e.repo for e in self._entries}
        for repo in repos:
            for metric in ("diff_lines", "dependency_crossings", "auth_touches"):
                current = self.current(repo, metric)
                thresh = self.threshold(metric)
                if thresh > 0 and current > thresh:
                    exceeded.append((repo, metric, current, thresh))
        return exceeded

    def prune_old(self) -> int:
        """Remove entries outside the budget window. Returns count removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._config.window_days * 2)).isoformat()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        return before - len(self._entries)


def _entry_dict(entry: BudgetEntry) -> dict[str, Any]:
    return {
        "timestamp": entry.timestamp,
        "repo": entry.repo,
        "metric": entry.metric,
        "value": entry.value,
        "trace_id": entry.trace_id,
    }
