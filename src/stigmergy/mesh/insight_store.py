"""Insight persistence and cross-run finding registry.

Two responsibilities:
1. InsightStore: append-only JSONL persistence of insights from each run.
   File: .stigmergy/insights.jsonl
2. FindingRegistry: cross-run dedup and decay tracking. Detects when
   findings resurface without state change — the migration from
   Discovery to Normalized Deviance (paper Section 7).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

INSIGHTS_PATH = Path(".stigmergy") / "insights.jsonl"
REGISTRY_PATH = Path(".stigmergy") / "finding_registry.json"


# ── InsightStore ────────────────────────────────────────────


class InsightStore:
    """Append-only JSONL persistence for insights.

    Each run appends its insights to .stigmergy/insights.jsonl.
    This is the raw audit trail — every insight the system ever produced,
    with timestamps, scoring components, and signal references.
    """

    def __init__(self, path: Path = INSIGHTS_PATH) -> None:
        self._path = path
        self._buffer: list[dict] = []
        self._run_id = str(uuid4())[:8]

    @property
    def run_id(self) -> str:
        return self._run_id

    def record(self, insight, score: float = 0.0, scoring_components: dict | None = None) -> None:
        """Buffer an insight for later persistence.

        Args:
            insight: An Insight object from mesh/insights.py.
            score: The S(f) score (0 if not yet computed).
            scoring_components: The individual S(f) components (d_bridge, c_entity, etc.)
        """
        # Extract action-oriented fields to top level for easy querying
        details = insight.details or {}
        entry = {
            "insight_id": str(uuid4()),
            "run_id": self._run_id,
            "agent_id": str(insight.agent_id),
            "type": insight.type,
            "summary": insight.summary,
            "unknowns": details.get("unknowns", ""),
            "recommended_action": details.get("recommended_action", ""),
            "inaction_risk": details.get("inaction_risk", ""),
            "classification": details.get("classification", ""),
            "temporal_relevance": details.get("temporal_relevance", ""),
            "confidence": insight.confidence,
            "signal_ids": [str(sid) for sid in insight.signal_ids],
            "details": details,
            "score": round(score, 4),
            "scoring_components": scoring_components or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._buffer.append(entry)

    def save(self) -> int:
        """Flush buffered insights to JSONL file. Returns count written."""
        if not self._buffer:
            return 0

        self._path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(self._path, "a") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry, default=str) + "\n")
                count += 1

        logger.info("Persisted %d insights to %s (run %s)", count, self._path, self._run_id)
        self._buffer.clear()
        return count

    def load_all(self) -> list[dict]:
        """Load all historical insights from JSONL file."""
        if not self._path.exists():
            return []

        entries: list[dict] = []
        with open(self._path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line %d in %s", line_num, self._path)
        return entries

    def load_by_run(self, run_id: str) -> list[dict]:
        """Load insights from a specific run."""
        return [e for e in self.load_all() if e.get("run_id") == run_id]

    def run_ids(self) -> list[str]:
        """Get all unique run IDs in chronological order."""
        seen: dict[str, str] = {}  # run_id -> first created_at
        for entry in self.load_all():
            rid = entry.get("run_id", "")
            if rid and rid not in seen:
                seen[rid] = entry.get("created_at", "")
        return sorted(seen.keys(), key=lambda r: seen[r])

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)


# ── FindingRegistry ─────────────────────────────────────────


def _finding_hash(insight_type: str, summary: str, actors: list[str]) -> str:
    """Content hash for deduplication across runs.

    Two findings are "the same" if they have the same type, similar summary,
    and involve the same actors. The hash uses a normalized form of these.

    Robustness against LLM paraphrasing:
    - Type names are normalized (spaces→underscores, lowercase).
    - Summary words are sorted (bag-of-words) so word reordering
      ("coupon/discount" vs "discount/coupon") doesn't create a new hash.
    - Identifiers (PR#, SERV-NNN, @mentions) are extracted and sorted
      separately so they anchor the hash even if surrounding prose changes.
    """
    import re

    # Normalize type: "coordinated investigation" == "coordinated_investigation"
    norm_type = insight_type.lower().strip().replace(" ", "_").replace("-", "_")

    # Extract stable identifiers from summary (PR numbers, ticket IDs, @mentions)
    summary_lower = summary.lower().strip()
    prs = sorted(set(re.findall(r'pr\s*#(\d+)', summary_lower)))
    tickets = sorted(set(re.findall(r'[a-z]+-\d+', summary_lower)))
    mentions = sorted(set(re.findall(r'@([\w-]+)', summary_lower)))

    # Bag-of-words from first 120 chars (sorted, deduped — order-independent)
    # Split on any non-alphanumeric to handle "discount/coupon" vs "coupon/discount"
    words = re.split(r'[^a-z0-9]+', summary_lower[:120])
    words = sorted(set(w for w in words if len(w) >= 3))
    bow = " ".join(words)

    # Actors (from details.actors field)
    norm_actors = ",".join(sorted(a.lower().strip() for a in actors))

    content = f"{norm_type}:{bow}:{norm_actors}:{','.join(prs)}:{','.join(tickets)}:{','.join(mentions)}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Finding:
    """A deduplicated finding tracked across runs.

    Tracks the lifecycle: first_seen → last_seen → times_surfaced.
    When a finding resurfaces without state change (no commits, no ticket
    closures, no deployments referencing it), it is migrating from
    Discovery to Normalized Deviance.
    """

    finding_hash: str
    type: str
    summary: str
    actors: list[str]
    first_seen: str  # ISO timestamp
    last_seen: str  # ISO timestamp
    times_surfaced: int
    run_ids: list[str]  # which runs surfaced this
    has_state_change: bool  # did action follow?
    max_confidence: float
    max_score: float  # highest S(f) score
    decay_stage: str  # "active", "acknowledged", "deferred", "normalized"
    signal_ids: list[str] = field(default_factory=list)  # aggregated from all insights
    classification: str = "unknown"  # retrieval/amplification/correlation/discovery
    compression_index: float = 0.0   # linguistic compression [0,1] — Section 5.3
    action_correlation: float = -1.0  # action/discussion ratio — Section 5.2 (-1 = not computed)

    @property
    def is_decaying(self) -> bool:
        """Finding has been surfaced multiple times without state change."""
        return self.times_surfaced >= 2 and not self.has_state_change

    @property
    def days_since_first(self) -> float:
        """Days between first and last surfacing."""
        try:
            first = datetime.fromisoformat(self.first_seen)
            last = datetime.fromisoformat(self.last_seen)
            return (last - first).total_seconds() / 86400
        except (ValueError, TypeError):
            return 0.0


class FindingRegistry:
    """Cross-run finding dedup and decay tracking.

    Loads from .stigmergy/finding_registry.json.
    Updated after each run by ingesting new insights from InsightStore.

    Decay stages (paper Section 7):
    - "active": finding surfaced, no prior history
    - "acknowledged": finding surfaced 2+ times, state change detected
    - "deferred": finding surfaced 2+ times, no state change within 7 days
    - "normalized": finding surfaced 3+ times over 14+ days, no state change
    """

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self._path = path
        self._findings: dict[str, Finding] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for fh, fd in data.items():
                self._findings[fh] = Finding(**fd)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load finding registry: %s", exc)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for fh, finding in self._findings.items():
            data[fh] = {
                "finding_hash": finding.finding_hash,
                "type": finding.type,
                "summary": finding.summary,
                "actors": finding.actors,
                "first_seen": finding.first_seen,
                "last_seen": finding.last_seen,
                "times_surfaced": finding.times_surfaced,
                "run_ids": finding.run_ids,
                "has_state_change": finding.has_state_change,
                "max_confidence": finding.max_confidence,
                "max_score": finding.max_score,
                "decay_stage": finding.decay_stage,
                "signal_ids": finding.signal_ids,
                "classification": finding.classification,
                "compression_index": finding.compression_index,
                "action_correlation": finding.action_correlation,
            }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def ingest_insights(self, insights: list[dict], run_id: str) -> list[Finding]:
        """Ingest a batch of insight dicts from InsightStore.

        Returns list of findings that changed state (new, resurfaced, decayed).
        """
        changed: list[Finding] = []
        now = datetime.now(timezone.utc).isoformat()

        for entry in insights:
            actors = entry.get("details", {}).get("actors", [])
            fh = _finding_hash(entry.get("type", ""), entry.get("summary", ""), actors)
            confidence = entry.get("confidence", 0.0)
            score = entry.get("score", 0.0)
            entry_signal_ids = [str(s) for s in entry.get("signal_ids", [])]

            classification = entry.get("classification", "") or entry.get("details", {}).get("classification", "")
            entry_compression = entry.get("compression_index", 0.0)
            entry_action_corr = entry.get("action_correlation", -1.0)

            if fh in self._findings:
                # Resurfaced — update existing finding
                finding = self._findings[fh]
                finding.last_seen = now
                finding.times_surfaced += 1
                if run_id not in finding.run_ids:
                    finding.run_ids.append(run_id)
                finding.max_confidence = max(finding.max_confidence, confidence)
                finding.max_score = max(finding.max_score, score)
                # Update classification if provided (prefer latest)
                if classification:
                    finding.classification = classification
                # Update ND metadata (prefer latest computed values)
                if entry_compression > 0:
                    finding.compression_index = entry_compression
                if entry_action_corr >= 0:
                    finding.action_correlation = entry_action_corr
                # Aggregate signal_ids (deduplicated)
                existing = set(finding.signal_ids)
                for sid in entry_signal_ids:
                    if sid not in existing:
                        finding.signal_ids.append(sid)
                        existing.add(sid)
                self._update_decay_stage(finding)
                changed.append(finding)
            else:
                # New finding
                finding = Finding(
                    finding_hash=fh,
                    type=entry.get("type", ""),
                    summary=entry.get("summary", ""),
                    actors=actors,
                    first_seen=now,
                    last_seen=now,
                    times_surfaced=1,
                    run_ids=[run_id],
                    has_state_change=False,
                    max_confidence=confidence,
                    max_score=score,
                    decay_stage="active",
                    signal_ids=entry_signal_ids,
                    classification=classification or "unknown",
                    compression_index=entry_compression,
                    action_correlation=entry_action_corr,
                )
                self._findings[fh] = finding
                changed.append(finding)

        return changed

    def _update_decay_stage(self, finding: Finding) -> None:
        """Update decay stage based on surfacing history and state changes.

        Decay accelerators:
        - Retrieval findings (re-descriptions of known state) decay faster:
          they enter "deferred" after 1 resurfacing instead of 2, since
          repeating a known ticket adds no new information.
        - Low action-correlation (discussion without action) accelerates
          decay: findings discussed repeatedly without state changes are
          strong normalized deviance candidates (Paper Section 5.2).
        """
        if finding.has_state_change:
            finding.decay_stage = "acknowledged"
            return

        days = finding.days_since_first
        times = finding.times_surfaced
        is_retrieval = finding.classification == "retrieval"

        # Action-correlation accelerates decay: if ratio is very low
        # (much discussion, no action), the finding decays faster.
        # -1.0 means not computed — no acceleration.
        ac = finding.action_correlation
        low_action_correlation = ac >= 0 and ac < 0.1

        if is_retrieval or low_action_correlation:
            # Retrieval and low-action-correlation decay faster
            if times >= 2 and days >= 7:
                finding.decay_stage = "normalized"
            elif times >= 1:
                finding.decay_stage = "deferred"
            else:
                finding.decay_stage = "active"
        else:
            if times >= 3 and days >= 14:
                finding.decay_stage = "normalized"
            elif times >= 2 and days >= 7:
                finding.decay_stage = "deferred"
            elif times >= 2:
                finding.decay_stage = "deferred"
            else:
                finding.decay_stage = "active"

    def mark_state_change(self, finding_hash: str) -> None:
        """Mark a finding as having produced a state change.

        Called when evidence of action is detected: commits referencing
        the finding, ticket closures, deployments.
        """
        if finding_hash in self._findings:
            self._findings[finding_hash].has_state_change = True
            self._findings[finding_hash].decay_stage = "acknowledged"

    def detect_state_changes(self, signals: list[dict]) -> int:
        """Auto-detect state changes from signals matching existing findings.

        Checks if any signals reference actors or key terms from deferred/
        normalized findings. Signals that match suggest action has been
        taken — e.g., a commit by an actor named in a finding, or a ticket
        closure referencing the finding's topic.

        Args:
            signals: List of signal dicts with 'author', 'content', 'source' keys.

        Returns:
            Number of findings that transitioned to 'acknowledged'.
        """
        if not signals:
            return 0

        # Only check findings that are decaying (deferred/normalized)
        # — active findings don't need state change detection yet
        decaying = [f for f in self._findings.values()
                    if f.decay_stage in ("deferred", "normalized")]
        if not decaying:
            return 0

        # Build a lookup of signal authors and content terms
        signal_authors = set()
        signal_terms: set[str] = set()
        for sig in signals:
            author = sig.get("author", "").lower().strip()
            if author:
                signal_authors.add(author)
            content = sig.get("content", "").lower()
            # Extract simple terms from content for matching
            for word in content.split():
                word = word.strip(".,!?;:()[]{}\"'*`<>#~")
                if len(word) >= 4:
                    signal_terms.add(word)

        acknowledged = 0
        for finding in decaying:
            # Check 1: Did any finding actor produce a signal this run?
            finding_actors = {a.lower().strip() for a in finding.actors}
            actor_overlap = finding_actors & signal_authors

            # Check 2: Do signal terms overlap with finding summary terms?
            summary_terms = set()
            for word in finding.summary.lower().split():
                word = word.strip(".,!?;:()[]{}\"'*`<>#~")
                if len(word) >= 4:
                    summary_terms.add(word)
            term_overlap = summary_terms & signal_terms

            # Require actor match AND some term relevance to avoid false positives
            if actor_overlap and len(term_overlap) >= 2:
                self.mark_state_change(finding.finding_hash)
                acknowledged += 1
                logger.info(
                    "Auto-detected state change for finding %s: "
                    "actors %s active, terms overlap: %s",
                    finding.finding_hash[:8],
                    actor_overlap,
                    list(term_overlap)[:5],
                )

        return acknowledged

    def decaying_findings(self) -> list[Finding]:
        """Findings migrating toward Normalized Deviance."""
        return [f for f in self._findings.values() if f.is_decaying]

    def normalized_findings(self) -> list[Finding]:
        """Findings that have fully normalized — known but unacted upon."""
        return [f for f in self._findings.values() if f.decay_stage == "normalized"]

    @property
    def total_findings(self) -> int:
        return len(self._findings)

    @property
    def active_count(self) -> int:
        return sum(1 for f in self._findings.values() if f.decay_stage == "active")

    @property
    def deferred_count(self) -> int:
        return sum(1 for f in self._findings.values() if f.decay_stage == "deferred")

    @property
    def normalized_count(self) -> int:
        return sum(1 for f in self._findings.values() if f.decay_stage == "normalized")

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total_findings,
            "active": self.active_count,
            "deferred": self.deferred_count,
            "normalized": self.normalized_count,
            "acknowledged": sum(1 for f in self._findings.values() if f.decay_stage == "acknowledged"),
        }
