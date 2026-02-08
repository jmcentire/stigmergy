"""Action-correlation ratio (Paper Section 5.2).

"Are discussions producing state changes — commits, ticket closures,
deployments — or has mention frequency decoupled from action frequency?"

Compares discussion volume to state-change evidence per topic/actor over
time. Pure computation — no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from stigmergy.primitives.signal import SignalSource, extract_terms


@dataclass
class ActionCorrelation:
    """Discussion-to-action ratio for a topic cluster.

    Extended with exploration/exploitation classification (March, 1991):
    - Exploitation: repeated actions on the same systems/repos/files
    - Exploration: actions touching new systems, new topics, new territory

    A team with high action-correlation but all exploitation is in the
    March trap — refining what they already know how to do while the
    environment shifts beneath them. The Slootman paper calls this
    "sustained, unrelieved focus creating structural vulnerabilities."
    """

    topic: str                    # representative term or cluster label
    topic_terms: frozenset[str]   # terms defining this topic
    discussion_count: int         # signals that are discussion
    action_count: int             # signals that are state changes
    ratio: float                  # action_count / discussion_count (higher = healthy)
    trend: str                    # "improving", "stable", "decoupling"
    window_days: int
    # Exploration/exploitation (March 1991)
    exploration_ratio: float = 0.0  # fraction of action signals on novel systems [0, 1]


# ── Signal classification ──────────────────────────────────

# Sources / event types that represent state changes (actions)
_ACTION_SOURCES = {SignalSource.GITHUB}
_ACTION_EVENT_TYPES = {
    "pull_request_merged", "pull_request_closed",
    "push", "commit", "deployment", "release",
}

# Sources / event types that represent discussion
_DISCUSSION_SOURCES = {SignalSource.SLACK}
_DISCUSSION_EVENT_TYPES = {
    "message", "comment", "issue_comment", "pr_comment",
}


def _is_action_signal(source: SignalSource, metadata: dict) -> bool:
    """Classify a signal as an action (state change)."""
    event_type = metadata.get("event_type", "")
    if event_type in _ACTION_EVENT_TYPES:
        return True
    if source in _ACTION_SOURCES and event_type in ("pull_request_merged", "push", "release"):
        return True
    # Linear state transitions to Done/Closed are actions
    if source == SignalSource.LINEAR:
        state = metadata.get("state", "")
        if state.lower() in ("done", "closed", "completed", "cancelled"):
            return True
    return False


def _is_discussion_signal(source: SignalSource, metadata: dict) -> bool:
    """Classify a signal as discussion."""
    event_type = metadata.get("event_type", "")
    if event_type in _DISCUSSION_EVENT_TYPES:
        return True
    if source in _DISCUSSION_SOURCES:
        return True
    # Linear comments are discussion
    if source == SignalSource.LINEAR and event_type in ("comment", "issue_comment"):
        return True
    return False


# ── Core computation ───────────────────────────────────────


def compute_action_correlations(
    signals: list[dict],
    *,
    min_discussion_count: int = 3,
    window_days: int = 7,
    decoupling_threshold: float = 0.1,
) -> list[ActionCorrelation]:
    """Compute action-correlation ratios per topic from a signal stream.

    Args:
        signals: list of dicts with keys: content, source (SignalSource),
                 metadata (dict), timestamp (datetime).
        min_discussion_count: minimum discussion signals to report a topic.
        window_days: sliding window for trend calculation.
        decoupling_threshold: ratio below which a topic is "decoupling".

    Returns:
        List of ActionCorrelation objects, sorted by ratio ascending
        (worst decoupling first).
    """
    # Group signals by extracted terms → topic clusters
    topic_discussions: dict[str, int] = {}   # term -> discussion count
    topic_actions: dict[str, int] = {}       # term -> action count
    topic_terms_map: dict[str, set[str]] = {}  # term -> all co-occurring terms

    # Exploration/exploitation tracking (March 1991)
    # Track which systems (repos/channels) each term's actions touch.
    # A system seen for the first time across all signals = exploration.
    global_action_systems: set[str] = set()  # all systems seen so far
    topic_action_systems: dict[str, set[str]] = {}  # term -> systems with actions
    topic_novel_actions: dict[str, int] = {}   # term -> actions on novel systems

    for sig in signals:
        content = sig.get("content", "")
        source = sig.get("source", SignalSource.SLACK)
        metadata = sig.get("metadata", {})
        terms = extract_terms(content)

        is_action = _is_action_signal(source, metadata)
        is_discussion = _is_discussion_signal(source, metadata)

        # System identity for exploration/exploitation
        system = metadata.get("repo", "") or sig.get("channel", "")

        for term in terms:
            if is_action:
                topic_actions[term] = topic_actions.get(term, 0) + 1
                # Track exploration: is this action on a system we haven't seen?
                if term not in topic_action_systems:
                    topic_action_systems[term] = set()
                    topic_novel_actions[term] = 0
                if system and system not in topic_action_systems[term]:
                    topic_action_systems[term].add(system)
                    # Novel if this system hasn't appeared in ANY topic's actions yet
                    if system not in global_action_systems:
                        topic_novel_actions[term] = topic_novel_actions.get(term, 0) + 1
                        global_action_systems.add(system)
            if is_discussion:
                topic_discussions[term] = topic_discussions.get(term, 0) + 1
            if term not in topic_terms_map:
                topic_terms_map[term] = set()
            topic_terms_map[term].update(terms)

    # Build correlations for topics with enough discussion
    results: list[ActionCorrelation] = []
    seen_terms: set[str] = set()

    for term, disc_count in sorted(
        topic_discussions.items(),
        key=lambda x: (x[1], topic_actions.get(x[0], 0)),
        reverse=True,
    ):
        if disc_count < min_discussion_count:
            continue
        if term in seen_terms:
            continue

        act_count = topic_actions.get(term, 0)
        ratio = act_count / disc_count if disc_count > 0 else 0.0

        # Trend determination (simplified without historical data)
        if ratio >= 0.5:
            trend = "healthy"
        elif ratio >= decoupling_threshold:
            trend = "stable"
        else:
            trend = "decoupling"

        # Exploration ratio: fraction of actions on novel systems
        novel = topic_novel_actions.get(term, 0)
        exploration_ratio = novel / act_count if act_count > 0 else 0.0

        co_terms = frozenset(topic_terms_map.get(term, {term}))
        seen_terms.update(co_terms)

        results.append(ActionCorrelation(
            topic=term,
            topic_terms=co_terms,
            discussion_count=disc_count,
            action_count=act_count,
            ratio=round(ratio, 4),
            trend=trend,
            window_days=window_days,
            exploration_ratio=round(exploration_ratio, 4),
        ))

    # Sort by ratio ascending — worst decoupling first
    results.sort(key=lambda x: x.ratio)
    return results
