"""Edge extraction from signals — hooks into the mesh's signal flow.

Every signal type maps to specific edge types:

| Signal              | Event Type       | Edge Extracted                           |
|---------------------|------------------|------------------------------------------|
| GitHub PR opened    | pull_request     | (stores author for later review edges)   |
| GitHub PR review    | review           | reviewer → PR author (pr_review)         |
| GitHub PR comment   | pr_comment       | commenter → PR author (pr_comment)       |
| GitHub PR merged    | pull_request     | merger → PR author (pr_merge)            |
| Linear issue        | (no event_type)  | author → assignee (linear_assign)        |
| Linear comment      | linear_comment   | commenter → issue author (linear_comment)|
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from stigmergy.graph.graph import CommunicationGraph, Edge
from stigmergy.primitives.signal import Signal

logger = logging.getLogger(__name__)


class EdgeExtractor:
    """Extract communication graph edges from signals."""

    def __init__(self, graph: CommunicationGraph) -> None:
        self._graph = graph
        # Cache PR authors: "repo#pr_number" → author
        # When a review or comment arrives, we look up who opened the PR
        self._pr_authors: dict[str, str] = {}
        # Cache Linear issue authors: "identifier" → author
        self._issue_authors: dict[str, str] = {}

    def extract(self, signal: Signal) -> list[Edge]:
        """Extract edges from a signal and add them to the graph.

        Returns the list of edges extracted (may be empty).
        """
        source = signal.source
        if source == "github":
            return self._extract_github(signal)
        elif source == "linear":
            return self._extract_linear(signal)
        return []

    def _extract_github(self, signal: Signal) -> list[Edge]:
        """GitHub-specific edge extraction."""
        edges: list[Edge] = []
        meta = signal.metadata
        event_type = meta.get("event_type", "")
        channel = signal.channel  # repo name
        pr_number = meta.get("pr_number")
        timestamp = signal.timestamp

        if event_type == "pull_request":
            # PR opened or merged — store the author for later attribution
            if pr_number is not None:
                pr_key = f"{channel}#{pr_number}"
                action = meta.get("action", "")

                if action == "opened":
                    self._pr_authors[pr_key] = signal.author

                if action == "closed" and meta.get("merged"):
                    # Merged PR: merger → PR author edge
                    merged_by = meta.get("merged_by", "")
                    pr_author = self._pr_authors.get(pr_key, signal.author)
                    if merged_by and merged_by != pr_author:
                        edge = self._graph.add_edge(
                            source=merged_by,
                            target=pr_author,
                            edge_type="pr_merge",
                            channel=channel,
                            timestamp=timestamp,
                        )
                        edges.append(edge)

        elif event_type == "review":
            # Review: reviewer → PR author
            if pr_number is not None:
                pr_key = f"{channel}#{pr_number}"
                pr_author = self._pr_authors.get(pr_key)
                if pr_author and signal.author != pr_author:
                    edge = self._graph.add_edge(
                        source=signal.author,
                        target=pr_author,
                        edge_type="pr_review",
                        channel=channel,
                        timestamp=timestamp,
                    )
                    edges.append(edge)

        elif event_type == "pr_comment":
            # PR comment: commenter → PR author
            if pr_number is not None:
                pr_key = f"{channel}#{pr_number}"
                pr_author = self._pr_authors.get(pr_key)
                if pr_author and signal.author != pr_author:
                    edge = self._graph.add_edge(
                        source=signal.author,
                        target=pr_author,
                        edge_type="pr_comment",
                        channel=channel,
                        timestamp=timestamp,
                    )
                    edges.append(edge)

        return edges

    def _extract_linear(self, signal: Signal) -> list[Edge]:
        """Linear-specific edge extraction."""
        edges: list[Edge] = []
        meta = signal.metadata
        event_type = meta.get("event_type", "")
        identifier = meta.get("identifier", signal.channel)
        channel = meta.get("team", identifier)
        timestamp = signal.timestamp

        if event_type == "linear_comment":
            # Comment on an issue: commenter → issue author
            issue_author = self._issue_authors.get(identifier)
            if issue_author and signal.author != issue_author:
                edge = self._graph.add_edge(
                    source=signal.author,
                    target=issue_author,
                    edge_type="linear_comment",
                    channel=channel,
                    timestamp=timestamp,
                )
                edges.append(edge)

        else:
            # Issue created/updated — store author and extract assign edge
            self._issue_authors[identifier] = signal.author

            assignee = meta.get("assignee", "")
            if assignee and assignee != signal.author:
                edge = self._graph.add_edge(
                    source=signal.author,
                    target=assignee,
                    edge_type="linear_assign",
                    channel=channel,
                    timestamp=timestamp,
                )
                edges.append(edge)

            # Related issues: author of related → author of this
            relations = meta.get("relations", [])
            for rel in relations:
                related_id = rel.get("issue", "")
                if related_id:
                    related_author = self._issue_authors.get(related_id)
                    if related_author and related_author != signal.author:
                        edge = self._graph.add_edge(
                            source=related_author,
                            target=signal.author,
                            edge_type="linear_related",
                            channel=channel,
                            timestamp=timestamp,
                        )
                        edges.append(edge)

        return edges
