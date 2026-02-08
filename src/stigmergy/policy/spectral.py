"""Spectral energy analysis on the structure graph.

Theoretical foundation: BWGNN (Band-Width Graph Neural Networks) and
Viable System Model (VSM) algedonic channel.

The graph Laplacian L = D - A decomposes signal flow into frequency
components via its eigenvalues λ_1 ≤ λ_2 ≤ ... ≤ λ_n. In a healthy
organization:
  - Low-frequency energy dominates: smooth flow across the graph
  - Information propagates along natural dependency paths

An anomaly produces spectral right-shift:
  - High-frequency energy increases: discordant flow, signals crossing
    unexpected boundaries
  - This is the VSM algedonic signal: "pain" that bypasses normal
    hierarchy to reach System 5 (human leadership)

System 5 is deliberately human. The mesh is a prosthetic for
organizational perception — it detects the right-shift, but a human
decides what to do about it.

Implementation:
  - Build adjacency matrix from structure graph + trace event flow
  - Compute graph Laplacian eigenvalues (O(n²) but n is small: ~100 nodes)
  - Track spectral energy distribution over time windows
  - Detect right-shift: high-frequency energy ratio exceeds baseline
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpectralSnapshot:
    """A single spectral energy measurement."""

    timestamp: datetime
    total_energy: float  # sum of all eigenvalues (trace of Laplacian)
    low_freq_ratio: float  # energy in bottom 50% of spectrum
    high_freq_ratio: float  # energy in top 50% of spectrum
    spectral_gap: float  # λ_2 - λ_1 (algebraic connectivity)
    max_eigenvalue: float  # λ_max
    node_count: int
    edge_count: int

    @property
    def is_right_shifted(self) -> bool:
        """Is high-frequency energy abnormally dominant?

        A right-shift means discordant flow — the algedonic signal.
        Threshold: high_freq > 0.65 of total energy (normal is ~0.5).
        """
        return self.high_freq_ratio > 0.65

    @property
    def is_left_shifted(self) -> bool:
        """Is low-frequency energy abnormally dominant?

        A left-shift means eigenvalue collapse — the spectral signature
        of Normalized Deviance (paper Section 6.2). Independent evaluation
        modes are merging as framing converges across teams.

        Threshold: low_freq > 0.75 of total energy AND spectral_gap < 0.5.
        The gap condition prevents false positives from well-connected
        graphs that naturally have high low-frequency energy.
        """
        return self.low_freq_ratio > 0.75 and self.spectral_gap < 0.5

    @property
    def effective_dimensionality(self) -> float:
        """Fraction of eigenvalues above noise threshold.

        Under healthy conditions, multiple distinct eigenvalues exist
        (full rank). As Normalized Deviance progresses, eigenvalues
        merge — the correlation matrix loses rank. Low effective
        dimensionality = loss of independent evaluation modes.

        Returns ratio in [0, 1]. Below 0.3 suggests eigenvalue collapse.
        """
        if self.node_count < 2 or self.total_energy < 1e-10:
            return 1.0
        # Approximate: if low_freq dominates heavily, few independent modes
        # remain. Use the ratio of spectral_gap to max_eigenvalue as proxy
        # for how many eigenvalues are "distinct" vs collapsed.
        if self.max_eigenvalue < 1e-10:
            return 1.0
        gap_ratio = self.spectral_gap / self.max_eigenvalue
        # High gap_ratio = well-separated eigenvalues = high dimensionality
        # Low gap_ratio = collapsed eigenvalues = low dimensionality
        return min(1.0, gap_ratio * 2.0)

    @property
    def severity(self) -> str:
        """How severe is the spectral anomaly?"""
        if self.high_freq_ratio > 0.8:
            return "critical"
        if self.high_freq_ratio > 0.7:
            return "high"
        if self.high_freq_ratio > 0.65:
            return "medium"
        if self.is_left_shifted:
            return "medium"
        return "low"


@dataclass
class SpectralAnalyzer:
    """Tracks spectral energy of the structure graph over time.

    Maintains a history of snapshots and detects right-shift trends.
    The graph is enriched with trace event flow: edges get additional
    weight from recent cross-boundary activity.
    """

    _history: list[SpectralSnapshot] = field(default_factory=list)
    _max_history: int = 100  # Keep last 100 snapshots

    def analyze(
        self,
        adjacency: dict[str, list[str]],
        flow_weights: dict[tuple[str, str], float] | None = None,
    ) -> SpectralSnapshot:
        """Compute spectral energy from an adjacency structure.

        Args:
            adjacency: node_id -> [neighbor_ids]. Undirected graph.
            flow_weights: optional (source, target) -> weight for edges
                         that have seen recent trace event flow. Amplifies
                         the corresponding adjacency entries.

        Returns:
            SpectralSnapshot with energy distribution.
        """
        nodes = sorted(adjacency.keys())
        n = len(nodes)

        if n < 2:
            snapshot = SpectralSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_energy=0.0,
                low_freq_ratio=0.5,
                high_freq_ratio=0.5,
                spectral_gap=0.0,
                max_eigenvalue=0.0,
                node_count=n,
                edge_count=0,
            )
            self._record(snapshot)
            return snapshot

        # Build adjacency matrix
        node_idx = {node: i for i, node in enumerate(nodes)}
        A = np.zeros((n, n), dtype=np.float64)

        edge_count = 0
        for src, neighbors in adjacency.items():
            i = node_idx.get(src)
            if i is None:
                continue
            for tgt in neighbors:
                j = node_idx.get(tgt)
                if j is None:
                    continue
                weight = 1.0
                # Amplify edges with recent flow
                if flow_weights:
                    flow = flow_weights.get((src, tgt), 0.0)
                    flow += flow_weights.get((tgt, src), 0.0)
                    if flow > 0:
                        weight += math.log1p(flow)  # Logarithmic amplification
                A[i, j] = weight
                A[j, i] = weight  # Ensure symmetry
                edge_count += 1

        edge_count //= 2  # Each edge counted twice

        # Graph Laplacian: L = D - A
        D = np.diag(A.sum(axis=1))
        L = D - A

        # Eigenvalues (sorted ascending)
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.maximum(eigenvalues, 0.0)  # Numerical stability

        total_energy = float(eigenvalues.sum())
        if total_energy == 0:
            snapshot = SpectralSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_energy=0.0,
                low_freq_ratio=0.5,
                high_freq_ratio=0.5,
                spectral_gap=0.0,
                max_eigenvalue=0.0,
                node_count=n,
                edge_count=edge_count,
            )
            self._record(snapshot)
            return snapshot

        # Split spectrum into low and high frequency halves
        mid = n // 2
        low_energy = float(eigenvalues[:mid].sum())
        high_energy = float(eigenvalues[mid:].sum())

        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy

        # Spectral gap: λ_2 - λ_1 (algebraic connectivity)
        # λ_1 is always 0 for connected components
        spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if n > 1 else 0.0
        max_eigenvalue = float(eigenvalues[-1])

        snapshot = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=total_energy,
            low_freq_ratio=low_ratio,
            high_freq_ratio=high_ratio,
            spectral_gap=spectral_gap,
            max_eigenvalue=max_eigenvalue,
            node_count=n,
            edge_count=edge_count,
        )
        self._record(snapshot)
        return snapshot

    def _record(self, snapshot: SpectralSnapshot) -> None:
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def history(self) -> list[SpectralSnapshot]:
        return list(self._history)

    @property
    def trend(self) -> str:
        """Trend direction of high-frequency energy ratio.

        Returns: "stable", "rising" (right-shifting), or "falling" (normalizing).
        """
        if len(self._history) < 3:
            return "stable"

        recent = self._history[-3:]
        ratios = [s.high_freq_ratio for s in recent]

        # Simple linear trend
        if ratios[-1] > ratios[0] + 0.05:
            return "rising"
        if ratios[-1] < ratios[0] - 0.05:
            return "falling"
        return "stable"

    @property
    def eigenvalue_trend(self) -> str:
        """Trend direction of effective dimensionality.

        Returns: "stable", "collapsing" (losing independent modes),
        or "recovering" (regaining diversity).
        """
        if len(self._history) < 3:
            return "stable"

        recent = self._history[-3:]
        dims = [s.effective_dimensionality for s in recent]

        if dims[-1] < dims[0] - 0.1:
            return "collapsing"
        if dims[-1] > dims[0] + 0.1:
            return "recovering"
        return "stable"

    @property
    def latest(self) -> SpectralSnapshot | None:
        return self._history[-1] if self._history else None

    def algedonic_alert(self) -> dict[str, Any] | None:
        """Check if the algedonic channel should fire.

        Returns alert dict if spectral anomaly detected, None otherwise.
        The algedonic channel bypasses normal hierarchy — this is a
        System 5 signal that should reach human leadership directly.

        Detects both right-shift (Discovery) and left-shift (Normalized
        Deviance). Both are organizational pathologies, but with opposite
        spectral signatures.
        """
        if not self._history:
            return None

        latest = self._history[-1]

        # Right-shift: discordant cross-boundary activity (Discovery)
        if latest.is_right_shifted:
            return {
                "type": "algedonic",
                "subtype": "right_shift",
                "severity": latest.severity,
                "high_freq_ratio": latest.high_freq_ratio,
                "spectral_gap": latest.spectral_gap,
                "effective_dimensionality": latest.effective_dimensionality,
                "trend": self.trend,
                "message": (
                    f"Spectral right-shift detected: {latest.high_freq_ratio:.0%} "
                    f"high-frequency energy (normal: ~50%). "
                    f"This indicates discordant cross-boundary activity — "
                    f"organizational signal flow is not following expected structure. "
                    f"Trend: {self.trend}."
                ),
            }

        # Left-shift: eigenvalue collapse (Normalized Deviance)
        if latest.is_left_shifted:
            return {
                "type": "algedonic",
                "subtype": "left_shift",
                "severity": latest.severity,
                "low_freq_ratio": latest.low_freq_ratio,
                "spectral_gap": latest.spectral_gap,
                "effective_dimensionality": latest.effective_dimensionality,
                "trend": self.eigenvalue_trend,
                "message": (
                    f"Eigenvalue collapse detected: {latest.low_freq_ratio:.0%} "
                    f"low-frequency energy, effective dimensionality "
                    f"{latest.effective_dimensionality:.2f}. "
                    f"This indicates loss of independent evaluation modes — "
                    f"framing is converging across teams (Normalized Deviance). "
                    f"Trend: {self.eigenvalue_trend}."
                ),
            }

        return None


def build_adjacency_from_graph(graph) -> dict[str, list[str]]:
    """Extract adjacency dict from a StructureGraph.

    Only includes repo nodes that have at least one connection —
    isolated nodes (repos with no shared packages) are excluded because
    they add noise to the spectral analysis without information. A
    disconnected graph with many isolates will always show high-frequency
    energy, which is sparsity, not anomaly.
    """
    # Build repo-to-repo edges via shared packages
    raw_adjacency: dict[str, list[str]] = {}
    repo_nodes = [nid for nid, n in graph._nodes.items() if n.node_type == "repo"]
    for i, repo_a in enumerate(repo_nodes):
        for repo_b in repo_nodes[i + 1:]:
            shared = graph.shared_packages(repo_a, repo_b)
            if shared:
                raw_adjacency.setdefault(repo_a, []).append(repo_b)
                raw_adjacency.setdefault(repo_b, []).append(repo_a)

    # Only include connected nodes (exclude isolates)
    return {node: neighbors for node, neighbors in raw_adjacency.items() if neighbors}


def build_flow_weights_from_traces(traces, cutoff_hours: float = 168.0) -> dict[tuple[str, str], float]:
    """Build flow weights from recent trace events.

    Each trace event adds weight to edges between repos. When the same
    author submits events to multiple repos within a time window, those
    repos get cross-connected (the author is the flow conduit). This
    creates spectral signatures that reflect actual collaboration patterns
    rather than just package dependency structure.

    Args:
        traces: list of TraceEvent objects
        cutoff_hours: only consider traces within this many hours (default: 1 week)
    """
    from collections import defaultdict
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=cutoff_hours)

    # Track per-author repo activity to build cross-repo flow
    author_repos: dict[str, list[str]] = defaultdict(list)
    flow: dict[tuple[str, str], float] = {}

    for event in traces:
        # Parse timestamp if string
        ts = event.timestamp
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                continue

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        if ts < cutoff:
            continue

        repo = event.repo
        if not repo:
            continue

        # Per-repo activity weight
        weight = 1.0 + math.log1p(event.diff_lines) if event.diff_lines > 0 else 1.0
        key = (repo, repo)
        flow[key] = flow.get(key, 0.0) + weight

        # Track author's repo activity for cross-repo flow
        if event.author:
            author_repos[event.author].append(repo)

    # Build cross-repo flow from shared authors:
    # If an author touches repos A and B, those repos have information flow
    for author, repos in author_repos.items():
        unique_repos = sorted(set(repos))
        if len(unique_repos) < 2:
            continue
        for i, repo_a in enumerate(unique_repos):
            for repo_b in unique_repos[i + 1:]:
                key = (repo_a, repo_b)
                flow[key] = flow.get(key, 0.0) + 1.0

    return flow
