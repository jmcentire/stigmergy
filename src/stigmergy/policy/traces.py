"""Trace event store: append-only evidence log.

Every signal that flows through the mesh becomes a trace event.
Trace events are the raw material for policy evaluation — they carry
the structured metadata (files changed, additions, deletions, author,
repo) that policy signals inspect.

Backed by .stigmergy/traces.jsonl (one JSON object per line).
In-memory indexes for fast lookups by repo, author, time range.
Content-addressable IDs for deduplication across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _trace_id(event_type: str, repo: str, identifier: str) -> str:
    """Content-addressable ID: same PR/issue always gets the same trace ID."""
    identity = f"{event_type}:{repo}:{identifier}"
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


@dataclass
class TraceEvent:
    """A single observed event in the engineering system."""

    id: str
    timestamp: str  # ISO 8601
    event_type: str  # "pull_request", "issue", "merge", "deploy"
    repo: str  # e.g. "acme-org/backend"
    author: str
    title: str = ""
    files_changed: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    labels: list[str] = field(default_factory=list)
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Policy engine populates these after evaluation
    signals_detected: list[str] = field(default_factory=list)
    interventions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def diff_lines(self) -> int:
        return self.additions + self.deletions

    @classmethod
    def from_signal(cls, signal) -> TraceEvent:
        """Convert a mesh Signal into a TraceEvent."""
        meta = signal.metadata or {}
        event_type = meta.get("event_type", "unknown")
        identifier = str(meta.get("pr_number", meta.get("issue_number", signal.id)))

        return cls(
            id=_trace_id(event_type, signal.channel, identifier),
            timestamp=signal.timestamp.isoformat() if hasattr(signal.timestamp, "isoformat") else str(signal.timestamp),
            event_type=event_type,
            repo=signal.channel,
            author=signal.author,
            title=signal.content.split("\n")[0][:200],
            files_changed=meta.get("files_changed", []),
            additions=meta.get("additions", 0),
            deletions=meta.get("deletions", 0),
            labels=meta.get("labels", []),
            url=meta.get("url", ""),
            metadata={k: v for k, v in meta.items()
                      if k not in ("files_changed", "additions", "deletions", "labels", "url", "event_type")},
        )


class TraceStore:
    """Append-only trace event store with in-memory indexes.

    Persists to .stigmergy/traces.jsonl. Deduplicates by trace ID
    so reruns don't create duplicate events.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(".stigmergy/traces.jsonl")
        self._events: dict[str, TraceEvent] = {}  # id -> event
        self._by_repo: dict[str, list[str]] = {}  # repo -> [event_ids]
        self._by_author: dict[str, list[str]] = {}  # author -> [event_ids]
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            for line in self._path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                data = json.loads(line)
                event = TraceEvent(**data)
                self._index(event)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.debug("Failed to load trace store: %s", exc)

    def _index(self, event: TraceEvent) -> None:
        self._events[event.id] = event
        self._by_repo.setdefault(event.repo, []).append(event.id)
        self._by_author.setdefault(event.author, []).append(event.id)

    def append(self, event: TraceEvent) -> bool:
        """Append a trace event. Returns False if duplicate (already exists)."""
        if event.id in self._events:
            return False
        self._index(event)
        return True

    def save(self) -> None:
        """Persist all events to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            for event in self._events.values():
                f.write(json.dumps(asdict(event), default=str) + "\n")

    def get(self, trace_id: str) -> TraceEvent | None:
        return self._events.get(trace_id)

    def by_repo(self, repo: str) -> list[TraceEvent]:
        ids = self._by_repo.get(repo, [])
        return [self._events[eid] for eid in ids if eid in self._events]

    def by_author(self, author: str) -> list[TraceEvent]:
        ids = self._by_author.get(author, [])
        return [self._events[eid] for eid in ids if eid in self._events]

    def since(self, cutoff: datetime) -> list[TraceEvent]:
        """All events since cutoff, sorted chronologically."""
        cutoff_str = cutoff.isoformat()
        result = [e for e in self._events.values() if e.timestamp >= cutoff_str]
        result.sort(key=lambda e: e.timestamp)
        return result

    def recent(self, limit: int = 50) -> list[TraceEvent]:
        """Most recent events."""
        events = sorted(self._events.values(), key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    @property
    def count(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        self._events.clear()
        self._by_repo.clear()
        self._by_author.clear()
