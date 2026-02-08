"""Solo ownership detection (Paper evaluation section).

"Solo ownership of systemic problems appeared in two of five topics —
multi-system failures assigned to individual engineers without
reinforcement, where confusion or capacity limits produce stalls with
no escalation path."

Detects when a topic spanning multiple systems has only one actor
addressing it. Pure computation — no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from stigmergy.primitives.signal import SignalSource, extract_terms


@dataclass
class SoloOwnershipAlert:
    """Alert for a systemic topic owned by a single person."""

    topic_terms: frozenset[str]
    sole_owner: str                # the one person
    system_count: int              # how many systems/repos/channels this spans
    signal_count: int              # total signals in this topic
    action_count: int              # state-change signals by the sole owner
    staleness_days: float          # time since last action signal
    severity: str                  # "warning" or "critical"


# ── Action signal classification (reused from action_correlation) ──

_ACTION_EVENT_TYPES = {
    "pull_request_merged", "pull_request_closed",
    "push", "commit", "deployment", "release",
}


def _is_action(source: SignalSource, metadata: dict) -> bool:
    event_type = metadata.get("event_type", "")
    if event_type in _ACTION_EVENT_TYPES:
        return True
    if source == SignalSource.LINEAR:
        state = metadata.get("state", "")
        if state.lower() in ("done", "closed", "completed"):
            return True
    return False


# ── Core detection ─────────────────────────────────────────


def detect_solo_ownership(
    signals: list[dict],
    *,
    min_systems: int = 2,
    min_signals: int = 3,
    staleness_threshold_days: float = 7.0,
) -> list[SoloOwnershipAlert]:
    """Detect topics with solo ownership across multiple systems.

    Args:
        signals: list of dicts with keys: content, source, metadata,
                 author, channel, timestamp.
        min_systems: minimum distinct systems (repos/channels) to qualify.
        min_signals: minimum total signals to qualify.
        staleness_threshold_days: days since last action to flag staleness.

    Returns:
        List of SoloOwnershipAlert objects, sorted by severity.
    """
    # Build per-topic aggregates
    topic_data: dict[str, dict] = {}  # term -> {actors, systems, signals, actions, last_action_ts}

    for sig in signals:
        content = sig.get("content", "")
        author = sig.get("author", "unknown")
        channel = sig.get("channel", "")
        source = sig.get("source", SignalSource.SLACK)
        metadata = sig.get("metadata", {})
        timestamp = sig.get("timestamp", datetime.now(timezone.utc))
        terms = extract_terms(content)
        is_action = _is_action(source, metadata)

        # System identity: repo name from channel or metadata
        system = metadata.get("repo", channel)

        for term in terms:
            if term not in topic_data:
                topic_data[term] = {
                    "actors": {},        # actor -> action_count
                    "systems": set(),
                    "signal_count": 0,
                    "last_action_ts": None,
                    "co_terms": set(),
                }
            td = topic_data[term]
            td["signal_count"] += 1
            td["systems"].add(system)
            td["co_terms"].update(terms)

            if author not in td["actors"]:
                td["actors"][author] = 0
            if is_action:
                td["actors"][author] += 1
                if td["last_action_ts"] is None or timestamp > td["last_action_ts"]:
                    td["last_action_ts"] = timestamp

    # Find solo ownership patterns
    now = datetime.now(timezone.utc)
    alerts: list[SoloOwnershipAlert] = []
    seen_terms: set[str] = set()

    for term, td in topic_data.items():
        if term in seen_terms:
            continue
        if td["signal_count"] < min_signals:
            continue
        if len(td["systems"]) < min_systems:
            continue

        # Filter to actors who have produced at least one action
        action_actors = {a: c for a, c in td["actors"].items() if c > 0}

        # Solo ownership: exactly one actor producing actions
        if len(action_actors) != 1:
            continue

        sole_owner = next(iter(action_actors))
        action_count = action_actors[sole_owner]

        # Staleness
        if td["last_action_ts"] is not None:
            staleness = (now - td["last_action_ts"]).total_seconds() / 86400
        else:
            staleness = float("inf")

        # Severity
        if staleness > staleness_threshold_days and len(td["systems"]) >= 3:
            severity = "critical"
        elif staleness > staleness_threshold_days or len(td["systems"]) >= 3:
            severity = "warning"
        else:
            severity = "warning"

        co_terms = frozenset(td["co_terms"])
        seen_terms.update(co_terms)

        alerts.append(SoloOwnershipAlert(
            topic_terms=co_terms,
            sole_owner=sole_owner,
            system_count=len(td["systems"]),
            signal_count=td["signal_count"],
            action_count=action_count,
            staleness_days=round(staleness, 1),
            severity=severity,
        ))

    # Sort by severity (critical first), then by system_count descending
    severity_order = {"critical": 0, "warning": 1}
    alerts.sort(key=lambda a: (severity_order.get(a.severity, 2), -a.system_count))
    return alerts
