"""Communication graph — weighted directed multigraph of org communication.

Nodes are canonical person IDs (from IdentityResolver).
Edges are weighted, typed, and temporally decaying.

Persists to .stigmergy/graph.json alongside annotations.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stigmergy.graph.identity import IdentityResolver

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """A directed, typed communication edge between two people."""

    source: str  # canonical person ID
    target: str  # canonical person ID
    edge_type: str  # "pr_review", "pr_comment", "linear_comment", etc.
    weight: float = 1.0  # accumulates with each interaction
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel: str = ""  # repo or linear team — where the interaction happened
    count: int = 1  # number of interactions of this type


@dataclass
class NodeState:
    """State for a node (person) in the communication graph."""

    canonical_id: str
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal_count: int = 0
    primary_channels: set[str] = field(default_factory=set)


class CommunicationGraph:
    """Weighted directed multigraph of organizational communication.

    Edges are keyed by (source, target, edge_type). Multiple interaction
    types between the same pair create separate edges. Temporal decay
    ensures stale connections fade.
    """

    def __init__(
        self,
        resolver: IdentityResolver,
        decay_half_life_days: float = 30.0,
    ) -> None:
        self._nodes: dict[str, NodeState] = {}
        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._resolver = resolver
        self._decay_half_life = decay_half_life_days

    @property
    def resolver(self) -> IdentityResolver:
        return self._resolver

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        channel: str,
        timestamp: datetime | None = None,
    ) -> Edge:
        """Add or strengthen an edge. Resolves identities first.

        If the edge already exists, increments count and weight.
        Returns the edge.
        """
        ts = timestamp or datetime.now(timezone.utc)
        src = self._resolver.resolve(source)
        tgt = self._resolver.resolve(target)

        # Don't create self-loops
        if src == tgt:
            return Edge(source=src, target=tgt, edge_type=edge_type)

        # Ensure nodes exist
        self._ensure_node(src, ts, channel)
        self._ensure_node(tgt, ts, channel)

        key = (src, tgt, edge_type)
        if key in self._edges:
            edge = self._edges[key]
            edge.count += 1
            edge.weight += 1.0
            edge.last_seen = max(edge.last_seen, ts)
            if channel:
                edge.channel = channel
        else:
            edge = Edge(
                source=src,
                target=tgt,
                edge_type=edge_type,
                weight=1.0,
                last_seen=ts,
                channel=channel,
                count=1,
            )
            self._edges[key] = edge

        # Update node activity
        self._nodes[src].signal_count += 1
        self._nodes[src].last_active = max(self._nodes[src].last_active, ts)
        if channel:
            self._nodes[src].primary_channels.add(channel)

        return edge

    def effective_weight(self, source: str, target: str) -> float:
        """Sum of all edge types between two people, with temporal decay applied."""
        src = self._resolver.resolve(source)
        tgt = self._resolver.resolve(target)
        now = datetime.now(timezone.utc)
        total = 0.0
        for (s, t, _), edge in self._edges.items():
            if s == src and t == tgt:
                total += self._decayed_weight(edge, now)
        return total

    def neighbors(self, person: str) -> dict[str, float]:
        """All people connected to person (outgoing + incoming), with effective weights."""
        pid = self._resolver.resolve(person)
        now = datetime.now(timezone.utc)
        result: dict[str, float] = {}
        for (s, t, _), edge in self._edges.items():
            if s == pid:
                other = t
            elif t == pid:
                other = s
            else:
                continue
            result[other] = result.get(other, 0.0) + self._decayed_weight(edge, now)
        return result

    def outgoing(self, person: str) -> dict[str, float]:
        """People this person communicates TO, with effective weights."""
        pid = self._resolver.resolve(person)
        now = datetime.now(timezone.utc)
        result: dict[str, float] = {}
        for (s, t, _), edge in self._edges.items():
            if s == pid:
                result[t] = result.get(t, 0.0) + self._decayed_weight(edge, now)
        return result

    def incoming(self, person: str) -> dict[str, float]:
        """People communicating TO this person, with effective weights."""
        pid = self._resolver.resolve(person)
        now = datetime.now(timezone.utc)
        result: dict[str, float] = {}
        for (s, t, _), edge in self._edges.items():
            if t == pid:
                result[s] = result.get(s, 0.0) + self._decayed_weight(edge, now)
        return result

    def all_nodes(self) -> list[str]:
        """All canonical person IDs in the graph."""
        return list(self._nodes.keys())

    def all_edges(self) -> list[Edge]:
        """All edges in the graph."""
        return list(self._edges.values())

    def get_node(self, person: str) -> NodeState | None:
        pid = self._resolver.resolve(person)
        return self._nodes.get(pid)

    def save(self, path: Path) -> None:
        """Persist to .stigmergy/graph.json."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": {
                nid: {
                    "canonical_id": n.canonical_id,
                    "first_seen": n.first_seen.isoformat(),
                    "last_active": n.last_active.isoformat(),
                    "signal_count": n.signal_count,
                    "primary_channels": sorted(n.primary_channels),
                }
                for nid, n in self._nodes.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                    "last_seen": e.last_seen.isoformat(),
                    "channel": e.channel,
                    "count": e.count,
                }
                for e in self._edges.values()
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved communication graph: %d nodes, %d edges to %s",
                     len(self._nodes), len(self._edges), path)

    def load(self, path: Path) -> None:
        """Load from .stigmergy/graph.json."""
        if not path.exists():
            logger.debug("No graph file at %s, starting fresh", path)
            return

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load graph from %s, starting fresh", path)
            return

        for nid, ndata in data.get("nodes", {}).items():
            self._nodes[nid] = NodeState(
                canonical_id=ndata["canonical_id"],
                first_seen=datetime.fromisoformat(ndata["first_seen"]),
                last_active=datetime.fromisoformat(ndata["last_active"]),
                signal_count=ndata.get("signal_count", 0),
                primary_channels=set(ndata.get("primary_channels", [])),
            )

        for edata in data.get("edges", []):
            key = (edata["source"], edata["target"], edata["edge_type"])
            self._edges[key] = Edge(
                source=edata["source"],
                target=edata["target"],
                edge_type=edata["edge_type"],
                weight=edata.get("weight", 1.0),
                last_seen=datetime.fromisoformat(edata["last_seen"]),
                channel=edata.get("channel", ""),
                count=edata.get("count", 1),
            )

        logger.info("Loaded communication graph: %d nodes, %d edges from %s",
                     len(self._nodes), len(self._edges), path)

    def _ensure_node(self, canonical_id: str, timestamp: datetime, channel: str) -> None:
        """Create a node if it doesn't exist."""
        if canonical_id not in self._nodes:
            self._nodes[canonical_id] = NodeState(
                canonical_id=canonical_id,
                first_seen=timestamp,
                last_active=timestamp,
                primary_channels={channel} if channel else set(),
            )

    def _decayed_weight(self, edge: Edge, now: datetime) -> float:
        """Apply exponential temporal decay to an edge's weight."""
        age_days = (now - edge.last_seen).total_seconds() / 86400.0
        if age_days <= 0:
            return edge.weight
        decay = math.exp(-math.log(2) * age_days / self._decay_half_life)
        return edge.weight * decay
