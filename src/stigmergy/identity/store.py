"""Alias persistence — runtime-learned aliases survive across runs.

Persists to .stigmergy/learned_aliases.json. Follows the same pattern
as InsightStore, AnnotationStore, and CommunicationGraph persistence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(".stigmergy") / "learned_aliases.json"


class AliasStore:
    """Persist runtime-learned identity aliases to JSON.

    Format:
        {
            "canonical_id": ["alias1", "alias2", ...],
            ...
        }
    """

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._path = path
        self._aliases: dict[str, set[str]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        """Load persisted aliases from disk."""
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for canonical_id, aliases in data.items():
                self._aliases[canonical_id] = set(aliases)
            logger.info(
                "Loaded %d learned alias groups from %s",
                len(self._aliases),
                self._path,
            )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load alias store: %s", exc)

    def add(self, canonical_id: str, alias: str) -> None:
        """Add a runtime-learned alias."""
        alias = alias.strip().lower()
        if not alias:
            return
        if canonical_id not in self._aliases:
            self._aliases[canonical_id] = set()
        if alias not in self._aliases[canonical_id]:
            self._aliases[canonical_id].add(alias)
            self._dirty = True

    def all_aliases(self) -> dict[str, set[str]]:
        """Return all stored aliases: canonical_id -> {aliases}."""
        return dict(self._aliases)

    def save(self) -> None:
        """Flush to disk if dirty."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {cid: sorted(aliases) for cid, aliases in self._aliases.items()}
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Persisted %d alias groups to %s", len(self._aliases), self._path
        )
        self._dirty = False

    def merge(self, other: dict[str, set[str]]) -> None:
        """Merge aliases from another source (e.g. provider output)."""
        for canonical_id, aliases in other.items():
            if canonical_id not in self._aliases:
                self._aliases[canonical_id] = set()
            new = aliases - self._aliases[canonical_id]
            if new:
                self._aliases[canonical_id].update(new)
                self._dirty = True

    @property
    def group_count(self) -> int:
        return len(self._aliases)

    @property
    def total_aliases(self) -> int:
        return sum(len(a) for a in self._aliases.values())

    @property
    def dirty(self) -> bool:
        return self._dirty
