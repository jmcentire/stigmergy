"""Discovery store: what the system noticed about itself.

How it differs from AnnotationStore:
  - Annotations track what agents noticed about *external resources* (PRs, tickets).
  - Discoveries track what the system noticed about *itself* — mesh health,
    competency insights, effective patterns, anti-patterns, and "happy place"
    descriptions (what a good state looks like).

Discoveries are the stigmergic medium for meta-level coordination. The
meta-modeler deposits observations; other agents sense them as context.
No hard enforcement gates — purely advisory.

Session decay: discoveries fade across sessions. Normal observations
decay at 0.8x per session (half-life ~3 sessions). Happy-place snapshots
decay at 0.95x per session (half-life ~14 sessions) — hard-won knowledge
about what "good" looks like persists longer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Discovery:
    """A single observation about the system itself."""

    observation_type: str  # "mesh_health", "competency_insight", "happy_place", "effective_pattern", "anti_pattern"
    summary: str
    structured: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    source_agent: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signal_index: int = 0
    decay_weight: float = 1.0


class DiscoveryStore:
    """File-backed discovery store — the system's self-knowledge.

    Path: .stigmergy/discoveries.json
    Follows the same dirty-bit persistence pattern as AnnotationStore.

    On load, applies session decay:
      - Normal observations: decay_weight *= 0.8 per session
      - Happy-place snapshots: decay_weight *= 0.95 per session
      - Prune below 0.1
    """

    NORMAL_DECAY = 0.8
    HAPPY_PLACE_DECAY = 0.95
    PRUNE_THRESHOLD = 0.1

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(".stigmergy/discoveries.json")
        self._discoveries: list[dict[str, Any]] = []
        self._dirty = False
        self._load()

    def _load(self) -> None:
        """Load from disk, applying session decay."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            if not isinstance(raw, list):
                return
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to load discovery store, starting fresh")
            return

        # Apply session decay
        kept: list[dict[str, Any]] = []
        for d in raw:
            obs_type = d.get("observation_type", "")
            decay_factor = self.HAPPY_PLACE_DECAY if obs_type == "happy_place" else self.NORMAL_DECAY
            d["decay_weight"] = d.get("decay_weight", 1.0) * decay_factor

            if d["decay_weight"] >= self.PRUNE_THRESHOLD:
                kept.append(d)

        self._discoveries = kept
        if len(kept) < len(raw):
            self._dirty = True  # pruned entries need saving
            logger.debug("Discovery store: loaded %d, pruned %d", len(kept), len(raw) - len(kept))

    def save(self) -> None:
        """Persist to disk if dirty."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._discoveries, indent=2, default=str))
        self._dirty = False

    def deposit(self, discovery: Discovery) -> None:
        """Add an observation about the system."""
        self._discoveries.append(asdict(discovery))
        self._dirty = True

    def sense(
        self,
        observation_type: str | None = None,
        limit: int = 10,
        min_decay: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Query recent discoveries — most recent first.

        Optionally filter by observation_type and minimum decay weight.
        """
        filtered = self._discoveries
        if observation_type is not None:
            filtered = [d for d in filtered if d.get("observation_type") == observation_type]
        if min_decay > 0:
            filtered = [d for d in filtered if d.get("decay_weight", 0) >= min_decay]
        # Most recent first
        return list(reversed(filtered[-limit:]))

    def happy_places(self, limit: int = 3) -> list[dict[str, Any]]:
        """Strongest happy-place descriptions, sorted by confidence * decay_weight."""
        happy = [d for d in self._discoveries if d.get("observation_type") == "happy_place"]
        happy.sort(
            key=lambda d: d.get("confidence", 0) * d.get("decay_weight", 0),
            reverse=True,
        )
        return happy[:limit]

    @property
    def count(self) -> int:
        return len(self._discoveries)

    def clear(self) -> None:
        """Reset the store."""
        self._discoveries = []
        self._dirty = True
