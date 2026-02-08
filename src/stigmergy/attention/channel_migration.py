"""Channel migration detection (Paper Section 5.2).

"Are discussions moving from public to private channels while the
underlying work activity continues unchanged?" and "topics whose
public-channel frequency declines while the underlying work activity
(commits, deployments, dependency changes) continues unchanged."

Tracks per-topic signal volume by channel visibility (public vs private).
Detects when public volume drops while total activity holds steady.
Pure computation — no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from stigmergy.primitives.signal import SignalSource, extract_terms


@dataclass
class MigrationAlert:
    """Alert for a topic migrating from public to private channels."""

    topic_terms: frozenset[str]
    public_count: int              # signals in public channels
    private_count: int             # signals in private channels
    work_activity_count: int       # GitHub/Linear activity for same topic
    migration_score: float         # [0, 1] confidence of migration
    summary: str                   # human-readable description


# ── Channel visibility ─────────────────────────────────────


def _is_private_channel(channel: str, channel_meta: dict[str, dict]) -> bool:
    """Check if a channel is private using metadata from Slack adapter.

    channel_meta maps channel_name -> {id, is_private, num_members, topic}.
    Falls back to name-based heuristic if metadata unavailable.
    """
    clean = channel.lstrip("#")
    if clean in channel_meta:
        return channel_meta[clean].get("is_private", False)
    # Heuristic fallback: channels starting with "priv-" or "dm-"
    return clean.startswith(("priv-", "dm-", "private-"))


def _is_work_activity(source: SignalSource, metadata: dict) -> bool:
    """Check if a signal represents work activity (not discussion)."""
    if source == SignalSource.GITHUB:
        return True
    if source == SignalSource.LINEAR:
        state = metadata.get("state", "")
        if state.lower() in ("done", "closed", "completed"):
            return True
    return False


# ── Core detection ─────────────────────────────────────────


def detect_channel_migration(
    signals: list[dict],
    *,
    channel_meta: dict[str, dict] | None = None,
    min_total_signals: int = 4,
    migration_threshold: float = 0.3,
) -> list[MigrationAlert]:
    """Detect topics migrating from public to private channels.

    Args:
        signals: list of dicts with keys: content, source, metadata, channel.
        channel_meta: channel metadata from LiveSlackAdapter._channel_meta.
        min_total_signals: minimum signals per topic to report.
        migration_threshold: minimum migration_score to trigger alert.

    Returns:
        List of MigrationAlert objects, sorted by migration_score descending.
    """
    channel_meta = channel_meta or {}

    # Group signals by topic terms
    topic_data: dict[str, dict] = {}  # term -> {public, private, work}

    for sig in signals:
        content = sig.get("content", "")
        source = sig.get("source", SignalSource.SLACK)
        metadata = sig.get("metadata", {})
        channel = sig.get("channel", "")
        terms = extract_terms(content)

        is_private = _is_private_channel(channel, channel_meta)
        is_work = _is_work_activity(source, metadata)

        for term in terms:
            if term not in topic_data:
                topic_data[term] = {
                    "public": 0, "private": 0, "work": 0,
                    "co_terms": set(),
                }
            td = topic_data[term]
            td["co_terms"].update(terms)

            if is_work:
                td["work"] += 1
            elif is_private:
                td["private"] += 1
            else:
                td["public"] += 1

    # Detect migration patterns
    alerts: list[MigrationAlert] = []
    seen_terms: set[str] = set()

    for term, td in topic_data.items():
        if term in seen_terms:
            continue
        total_slack = td["public"] + td["private"]
        total = total_slack + td["work"]
        if total < min_total_signals:
            continue

        # Migration score: how much of the discussion is private
        # while work activity continues
        if total_slack == 0:
            continue

        private_ratio = td["private"] / total_slack if total_slack > 0 else 0.0

        # Work activity factor: if work continues but discussion goes private,
        # that's a stronger migration signal
        work_factor = min(td["work"] / max(total_slack, 1), 1.0)

        migration_score = private_ratio * (0.5 + 0.5 * work_factor)
        migration_score = max(0.0, min(1.0, migration_score))

        if migration_score < migration_threshold:
            continue

        co_terms = frozenset(td["co_terms"])
        seen_terms.update(co_terms)

        summary = (
            f"Topic '{term}': {td['private']}/{total_slack} Slack signals in "
            f"private channels, {td['work']} work activity signals"
        )

        alerts.append(MigrationAlert(
            topic_terms=co_terms,
            public_count=td["public"],
            private_count=td["private"],
            work_activity_count=td["work"],
            migration_score=round(migration_score, 4),
            summary=summary,
        ))

    alerts.sort(key=lambda a: a.migration_score, reverse=True)
    return alerts
