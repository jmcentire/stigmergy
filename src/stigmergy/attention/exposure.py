"""Exposure events and extraction from signals.

Mirrors the EdgeExtractor pattern in graph/extractor.py — same hook point
in mesh.ingest(), same signal metadata, parallel extraction.

Each exposure event records that a person was exposed to a piece of
information through a specific channel at a specific time. The exposure
type determines the base attention weight (authored > assigned > reviewed >
commented > mentioned > reacted > channel_present).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from stigmergy.primitives.signal import Signal, extract_terms

logger = logging.getLogger(__name__)

# Matches @username patterns in message content (Slack-style mentions).
# Handles both raw @handles and Slack's <@U12345> format.
_AT_MENTION_RE = re.compile(r"@(\w[\w.-]*)")


@dataclass(frozen=True)
class ExposureEvent:
    """A single exposure: person saw something at a point in time."""

    person_id: str  # canonical_id from IdentityResolver
    source: str  # "github", "linear", "slack"
    channel: str  # repo name, ticket ID, slack channel
    exposure_type: str  # "authored", "assigned", "reviewed", etc.
    timestamp: datetime
    topic_terms: frozenset[str]  # domain terms from the signal
    signal_id: str  # link back to source signal


class ExposureExtractor:
    """Extract exposure events from signals.

    Mirrors EdgeExtractor: holds internal caches so that when a
    review/comment arrives, it can look up who authored the PR/issue.
    """

    def __init__(self, resolver=None) -> None:
        self._resolver = resolver
        # Cache PR authors: "repo#pr_number" -> author
        self._pr_authors: dict[str, str] = {}
        # Cache Linear issue authors: "identifier" -> author
        self._issue_authors: dict[str, str] = {}

    def _resolve(self, identifier: str) -> str:
        """Resolve an identifier to canonical ID if resolver available."""
        if self._resolver is not None:
            return self._resolver.resolve(identifier)
        return identifier

    def extract(self, signal: Signal) -> list[ExposureEvent]:
        """Extract exposure events from a signal.

        Returns list of ExposureEvent (may be empty for unknown sources).
        """
        source = signal.source
        if source == "github":
            return self._extract_github(signal)
        elif source == "linear":
            return self._extract_linear(signal)
        elif source == "slack":
            return self._extract_slack(signal)
        return []

    def _make_event(
        self,
        person_id: str,
        source: str,
        channel: str,
        exposure_type: str,
        timestamp: datetime,
        terms: frozenset[str],
        signal_id: str,
    ) -> ExposureEvent:
        return ExposureEvent(
            person_id=self._resolve(person_id),
            source=source,
            channel=channel,
            exposure_type=exposure_type,
            timestamp=timestamp,
            topic_terms=terms,
            signal_id=signal_id,
        )

    def _extract_github(self, signal: Signal) -> list[ExposureEvent]:
        """GitHub-specific exposure extraction."""
        events: list[ExposureEvent] = []
        meta = signal.metadata
        event_type = meta.get("event_type", "")
        channel = signal.channel  # repo name
        pr_number = meta.get("pr_number")
        timestamp = signal.timestamp
        terms = frozenset(signal.terms)
        sig_id = str(signal.id)

        if event_type == "pull_request":
            if pr_number is not None:
                pr_key = f"{channel}#{pr_number}"
                action = meta.get("action", "")
                if action == "opened":
                    self._pr_authors[pr_key] = signal.author

                # Author is always exposed (authored)
                events.append(self._make_event(
                    signal.author, "github", channel, "authored",
                    timestamp, terms, sig_id,
                ))

                # Each assignee gets assigned exposure
                assignees = meta.get("assignees", [])
                for assignee in assignees:
                    if assignee != signal.author:
                        events.append(self._make_event(
                            assignee, "github", channel, "assigned",
                            timestamp, terms, sig_id,
                        ))

        elif event_type == "review":
            if pr_number is not None:
                pr_key = f"{channel}#{pr_number}"
                # Reviewer gets reviewed exposure
                events.append(self._make_event(
                    signal.author, "github", channel, "reviewed",
                    timestamp, terms, sig_id,
                ))
                # PR author gets re-exposed (authored)
                pr_author = self._pr_authors.get(pr_key)
                if pr_author and pr_author != signal.author:
                    events.append(self._make_event(
                        pr_author, "github", channel, "authored",
                        timestamp, terms, sig_id,
                    ))

        elif event_type == "pr_comment":
            if pr_number is not None:
                pr_key = f"{channel}#{pr_number}"
                # Commenter gets commented exposure
                events.append(self._make_event(
                    signal.author, "github", channel, "commented",
                    timestamp, terms, sig_id,
                ))
                # PR author gets re-exposed (authored)
                pr_author = self._pr_authors.get(pr_key)
                if pr_author and pr_author != signal.author:
                    events.append(self._make_event(
                        pr_author, "github", channel, "authored",
                        timestamp, terms, sig_id,
                    ))

        elif event_type == "issue":
            # Issue author
            events.append(self._make_event(
                signal.author, "github", channel, "authored",
                timestamp, terms, sig_id,
            ))
            assignees = meta.get("assignees", [])
            for assignee in assignees:
                if assignee != signal.author:
                    events.append(self._make_event(
                        assignee, "github", channel, "assigned",
                        timestamp, terms, sig_id,
                    ))

        elif event_type == "issue_comment":
            events.append(self._make_event(
                signal.author, "github", channel, "commented",
                timestamp, terms, sig_id,
            ))
            # Look up issue author from cache if available
            issue_number = meta.get("issue_number")
            if issue_number is not None:
                issue_key = f"{channel}#issue{issue_number}"
                issue_author = self._pr_authors.get(issue_key)
                if issue_author and issue_author != signal.author:
                    events.append(self._make_event(
                        issue_author, "github", channel, "authored",
                        timestamp, terms, sig_id,
                    ))

        return events

    def _extract_linear(self, signal: Signal) -> list[ExposureEvent]:
        """Linear-specific exposure extraction."""
        events: list[ExposureEvent] = []
        meta = signal.metadata
        event_type = meta.get("event_type", "")
        identifier = meta.get("identifier", signal.channel)
        channel = meta.get("team", identifier)
        timestamp = signal.timestamp
        terms = frozenset(signal.terms)
        sig_id = str(signal.id)

        if event_type == "linear_comment":
            # Commenter gets commented exposure
            events.append(self._make_event(
                signal.author, "linear", channel, "commented",
                timestamp, terms, sig_id,
            ))
            # Issue author gets re-exposed (authored)
            issue_author = self._issue_authors.get(identifier)
            if issue_author and issue_author != signal.author:
                events.append(self._make_event(
                    issue_author, "linear", channel, "authored",
                    timestamp, terms, sig_id,
                ))
        else:
            # Issue created/updated — store author
            self._issue_authors[identifier] = signal.author

            # Author gets authored exposure
            events.append(self._make_event(
                signal.author, "linear", channel, "authored",
                timestamp, terms, sig_id,
            ))

            # Assignee gets assigned exposure
            assignee = meta.get("assignee", "")
            if assignee and assignee != signal.author:
                events.append(self._make_event(
                    assignee, "linear", channel, "assigned",
                    timestamp, terms, sig_id,
                ))

        return events

    def _extract_slack(self, signal: Signal) -> list[ExposureEvent]:
        """Slack-specific exposure extraction.

        Extracts:
        - authored: message author
        - mentioned: anyone @mentioned in the message content or metadata
        - reacted: anyone who reacted (emoji) to the message
        - commented: thread reply authors (the replier engaged with the thread)
        - authored: thread parent author re-exposed on replies
        """
        events: list[ExposureEvent] = []
        meta = signal.metadata
        channel = signal.channel
        timestamp = signal.timestamp
        terms = frozenset(signal.terms)
        sig_id = str(signal.id)
        is_thread_reply = meta.get("is_thread_reply", False)

        if is_thread_reply:
            # Thread reply: replier gets commented exposure
            events.append(self._make_event(
                signal.author, "slack", channel, "commented",
                timestamp, terms, sig_id,
            ))
            # Thread parent author gets re-exposed via the thread
            thread_author = meta.get("thread_author")
            if thread_author and thread_author != signal.author:
                events.append(self._make_event(
                    thread_author, "slack", channel, "authored",
                    timestamp, terms, sig_id,
                ))
        else:
            # Top-level message: author gets authored exposure
            events.append(self._make_event(
                signal.author, "slack", channel, "authored",
                timestamp, terms, sig_id,
            ))

        # @mentions in message content — parse @username patterns
        seen_mentioned: set[str] = set()
        for match in _AT_MENTION_RE.finditer(signal.content):
            mentioned_id = match.group(1)
            if mentioned_id != signal.author and mentioned_id not in seen_mentioned:
                seen_mentioned.add(mentioned_id)
                events.append(self._make_event(
                    mentioned_id, "slack", channel, "mentioned",
                    timestamp, terms, sig_id,
                ))

        # @mentions from metadata (structured mentions from adapter)
        meta_mentions = meta.get("mentions", [])
        for mentioned_id in meta_mentions:
            if mentioned_id != signal.author and mentioned_id not in seen_mentioned:
                seen_mentioned.add(mentioned_id)
                events.append(self._make_event(
                    mentioned_id, "slack", channel, "mentioned",
                    timestamp, terms, sig_id,
                ))

        # Reactions — people who emoji-reacted have acknowledged the message.
        # Each reactor gets a "reacted" exposure (weaker than commented,
        # stronger than channel_present — they actively engaged).
        reactors = meta.get("reactors", [])
        seen_reacted: set[str] = set()
        for reactor_id in reactors:
            if reactor_id != signal.author and reactor_id not in seen_reacted:
                seen_reacted.add(reactor_id)
                events.append(self._make_event(
                    reactor_id, "slack", channel, "reacted",
                    timestamp, terms, sig_id,
                ))

        return events
