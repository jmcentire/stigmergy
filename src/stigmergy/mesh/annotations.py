"""Annotation store: the stigmergic pheromone trail.

Workers and agents can't write back to GitHub/Linear/Slack. Instead,
they deposit annotations here — keyed by resource identity, not by
signal UUID. This is the local environment modification that enables
indirect coordination: workers sense what other workers have noticed
about a resource without direct communication.

The store is content-addressable: the key is derived from the
resource's external identity (source + channel), which is stable
across reruns. Two signals from the same PR or ticket share the same
resource key, so annotations accumulate naturally.

This is pure stigmergy — modify the environment, sense the environment.
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


def resource_key(source: str, channel: str) -> str:
    """Content-addressable key for an external resource.

    Stable across reruns: the same PR, ticket, or thread always
    gets the same key regardless of which signal references it.
    """
    identity = f"{source}:{channel}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


@dataclass
class Annotation:
    """A single observation deposited about a resource.

    Annotations are append-only. They represent what an agent or worker
    noticed — a pheromone trace that other agents can sense.
    """

    agent_type: str  # "indexer", "correlator", "surfacer", "worker"
    summary: str  # what was noticed
    confidence: float  # how confident the agent was
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)


class AnnotationStore:
    """File-backed annotation store — the pheromone field.

    Keyed by resource_key (content-addressable hash of source+channel).
    Each resource accumulates annotations from multiple agents/workers.
    Workers check this store before processing to see what's already
    been noticed about a resource.

    The store persists across runs in .stigmergy/annotations.json.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(".stigmergy/annotations.json")
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._store = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.debug("Failed to load annotation store, starting fresh")
                self._store = {}

    def save(self) -> None:
        """Persist to disk if dirty."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._store, indent=2, default=str))
        self._dirty = False

    def annotate(
        self,
        source: str,
        channel: str,
        annotation: Annotation,
    ) -> str:
        """Deposit an annotation for a resource. Returns the resource key."""
        key = resource_key(source, channel)
        if key not in self._store:
            self._store[key] = []
        self._store[key].append(asdict(annotation))
        self._dirty = True
        return key

    def get(self, source: str, channel: str) -> list[dict[str, Any]]:
        """Sense: what annotations exist for this resource?"""
        key = resource_key(source, channel)
        return self._store.get(key, [])

    def get_by_key(self, key: str) -> list[dict[str, Any]]:
        """Direct key lookup."""
        return self._store.get(key, [])

    def has_annotations(self, source: str, channel: str) -> bool:
        """Quick check: has anything been noted about this resource?"""
        key = resource_key(source, channel)
        return key in self._store and len(self._store[key]) > 0

    def summary_for(self, source: str, channel: str) -> str:
        """One-line summary of annotations for a resource.

        Useful for enriching the surfacer's context: "This PR has been
        flagged by the correlator as overlapping with INT-690."
        """
        annotations = self.get(source, channel)
        if not annotations:
            return ""
        lines = []
        for ann in annotations[-3:]:  # last 3 most recent
            lines.append(f"[{ann['agent_type']}] {ann['summary']}")
        return " | ".join(lines)

    @property
    def resource_count(self) -> int:
        return len(self._store)

    @property
    def annotation_count(self) -> int:
        return sum(len(v) for v in self._store.values())

    def clear(self) -> None:
        """Reset the store."""
        self._store = {}
        self._dirty = True
