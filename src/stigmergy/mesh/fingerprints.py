"""Spectral fingerprints: vocabulary-independent structural coupling detection.

Each signal that routes through the mesh produces an activation vector —
the familiarity scores it received from every worker. Two signals with
similar activation vectors are structurally related regardless of whether
they share vocabulary, because they activate the same regions of the
organizational knowledge topology.

This is the paper's Section 4.5 core claim: "A booking service and a
reservation service that both touch the same database views will produce
traces that route through overlapping mesh regions, making their coupling
visible in the mesh geometry even if the word 'booking' never appears in
the reservation service's codebase."

The FingerprintStore accumulates activation vectors and detects:
1. Structural couplings: high activation similarity, low term overlap
2. Cross-source corroboration: same structure activated from different sources
3. Cross-source divergence: correlated signals with inconsistent activations

Uses cosine similarity on activation vectors, which is O(d) per pair
where d = number of workers (typically 3-20). The expensive operation is
pairwise comparison, which we limit to recent signals and candidate pairs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpectralFingerprint:
    """A signal's activation pattern across the mesh.

    The activation vector is the signal's position in the mesh's
    knowledge space. Two signals with similar vectors are structurally
    related — they activate the same organizational knowledge regions.
    """

    signal_id: UUID
    timestamp: datetime
    source: str
    channel: str
    author: str
    terms: frozenset[str]  # For term overlap computation
    activation: dict[str, float]  # worker_id_short -> score
    accepted_workers: frozenset[str]  # worker_ids that accepted
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def activation_vector(self) -> np.ndarray:
        """Sorted activation values as a numpy vector."""
        if not self.activation:
            return np.array([])
        return np.array([self.activation[k] for k in sorted(self.activation)])

    @property
    def worker_keys(self) -> list[str]:
        """Sorted worker keys for vector alignment."""
        return sorted(self.activation)


@dataclass
class CouplingDetection:
    """A detected structural coupling between two signals."""

    signal_a_id: UUID
    signal_b_id: UUID
    activation_similarity: float  # Cosine similarity of activation vectors
    term_overlap: float  # Jaccard similarity of terms
    coupling_strength: float  # activation_sim - term_overlap (high = hidden coupling)
    source_a: str
    source_b: str
    channel_a: str
    channel_b: str
    shared_workers: list[str]  # Workers that accepted both signals
    description: str = ""

    @property
    def is_cross_source(self) -> bool:
        return self.source_a != self.source_b

    @property
    def is_hidden_coupling(self) -> bool:
        """High activation similarity with low vocabulary overlap.

        This is the detection that no keyword search could find:
        structurally related signals that don't share words.
        """
        return self.activation_similarity > 0.6 and self.term_overlap < 0.15

    @property
    def severity(self) -> str:
        if self.coupling_strength > 0.7:
            return "high"
        if self.coupling_strength > 0.4:
            return "medium"
        return "low"


class FingerprintStore:
    """Accumulates spectral fingerprints and detects structural couplings.

    Fingerprints are activation vectors — each signal's familiarity scores
    across all mesh workers. Similar vectors mean structurally related
    signals, even with zero vocabulary overlap.
    """

    def __init__(self, max_fingerprints: int = 2000) -> None:
        self._fingerprints: list[SpectralFingerprint] = []
        self._by_source: dict[str, list[int]] = {}  # source -> indices
        self._by_channel: dict[str, list[int]] = {}  # channel -> indices
        self._max = max_fingerprints
        self._worker_universe: list[str] = []  # Stable ordering of all workers seen

    def record(self, fingerprint: SpectralFingerprint) -> None:
        """Add a fingerprint to the store."""
        idx = len(self._fingerprints)
        self._fingerprints.append(fingerprint)
        self._by_source.setdefault(fingerprint.source, []).append(idx)
        self._by_channel.setdefault(fingerprint.channel, []).append(idx)

        # Update worker universe
        for wk in fingerprint.activation:
            if wk not in self._worker_universe:
                self._worker_universe.append(wk)
        self._worker_universe.sort()

        # Evict oldest if over limit
        if len(self._fingerprints) > self._max:
            self._evict()

    def _evict(self) -> None:
        """Remove oldest fingerprints to stay under the limit."""
        keep = self._max * 3 // 4  # Keep 75%
        self._fingerprints = self._fingerprints[-keep:]
        # Rebuild indices
        self._by_source.clear()
        self._by_channel.clear()
        for i, fp in enumerate(self._fingerprints):
            self._by_source.setdefault(fp.source, []).append(i)
            self._by_channel.setdefault(fp.channel, []).append(i)

    @property
    def count(self) -> int:
        return len(self._fingerprints)

    def _aligned_vector(self, fp: SpectralFingerprint) -> np.ndarray:
        """Get activation vector aligned to the global worker universe."""
        vec = np.zeros(len(self._worker_universe))
        for wk, score in fp.activation.items():
            try:
                idx = self._worker_universe.index(wk)
                vec[idx] = score
            except ValueError:
                pass
        return vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Centered cosine similarity (Pearson correlation).

        Raw cosine treats [0.4, 0.4, 0.4] and [0.41, 0.41, 0.41] as
        nearly identical (sim ≈ 1.0), which floods coupling detection
        with noise when all workers score all signals at similar levels.

        Centering (subtracting the mean) measures whether the *pattern*
        of activation differs — which workers scored above/below their
        average — not whether the absolute magnitudes are similar.
        """
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        norm_a = np.linalg.norm(a_centered)
        norm_b = np.linalg.norm(b_centered)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a_centered, b_centered) / (norm_a * norm_b))

    def _term_overlap(self, a: frozenset[str], b: frozenset[str]) -> float:
        """Jaccard similarity of term sets."""
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def detect_couplings(
        self,
        min_activation_similarity: float = 0.5,
        max_term_overlap: float = 0.3,
        recent_n: int = 200,
    ) -> list[CouplingDetection]:
        """Detect structural couplings in recent fingerprints.

        Returns pairs with high activation similarity and low term overlap —
        the hidden couplings that vocabulary-based search would miss.

        Requires at least 5 workers for meaningful detection. With fewer
        workers, activation vectors are in too-low-dimensional space and
        all signals appear coupled (3D cosine similarity is meaningless).

        Limits comparison to recent_n fingerprints to keep O(n²) manageable.
        """
        # Need sufficient dimensional diversity for meaningful comparison.
        # With 3 workers, every signal accepted by the same worker has
        # cosine similarity ~1.0 — that's not a hidden coupling, it's
        # just low-dimensional space.
        if len(self._worker_universe) < 5:
            return []

        recent = self._fingerprints[-recent_n:]
        if len(recent) < 2:
            return []

        # Check effective dimensionality when we have enough data.
        # If one worker dominates (>50% of fingerprints), activation vectors
        # are too concentrated for meaningful coupling detection. Only applies
        # to larger datasets — with <20 fingerprints, the worker universe
        # check above is sufficient.
        if len(recent) >= 20:
            dominant_counts: dict[str, int] = {}
            for fp in recent:
                if fp.activation:
                    dom = max(fp.activation, key=fp.activation.get)
                    dominant_counts[dom] = dominant_counts.get(dom, 0) + 1
            min_count = max(1, len(recent) // 20)  # 5% of fingerprints
            effective_dims = sum(1 for c in dominant_counts.values() if c >= min_count)
            if effective_dims < 3:
                return []

        # Pre-compute aligned vectors
        vectors = [self._aligned_vector(fp) for fp in recent]

        couplings: list[CouplingDetection] = []
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                # Skip pairs from the same channel — coupling within a
                # channel is expected, not hidden. The paper's insight is
                # coupling ACROSS channels (booking ↔ reservation service).
                if recent[i].channel == recent[j].channel:
                    continue

                # Sparse vector guard: require at least 3 non-zero dimensions
                # in each vector. With fewer, cosine similarity is meaningless —
                # two signals visiting the same 1-2 workers get sim ~1.0 trivially.
                nz_i = np.nonzero(vectors[i])[0]
                nz_j = np.nonzero(vectors[j])[0]
                if len(nz_i) < 3 or len(nz_j) < 3:
                    continue

                # Variance guard: skip flat vectors where all workers scored
                # similarly. Without variance, centered cosine is meaningless —
                # the activation pattern carries no structural information.
                if np.std(vectors[i]) < 0.05 or np.std(vectors[j]) < 0.05:
                    continue

                act_sim = self._cosine_similarity(vectors[i], vectors[j])
                if act_sim < min_activation_similarity:
                    continue

                term_sim = self._term_overlap(recent[i].terms, recent[j].terms)
                if term_sim > max_term_overlap:
                    continue  # High vocab overlap = expected coupling, not hidden

                # Require 2+ shared accepted workers. One shared worker just
                # means both signals happened to route to the same place —
                # that's not structural coupling, it's routing coincidence.
                shared = recent[i].accepted_workers & recent[j].accepted_workers
                if len(shared) < 2:
                    continue

                coupling_strength = act_sim - term_sim

                fp_a, fp_b = recent[i], recent[j]
                couplings.append(CouplingDetection(
                    signal_a_id=fp_a.signal_id,
                    signal_b_id=fp_b.signal_id,
                    activation_similarity=act_sim,
                    term_overlap=term_sim,
                    coupling_strength=coupling_strength,
                    source_a=fp_a.source,
                    source_b=fp_b.source,
                    channel_a=fp_a.channel,
                    channel_b=fp_b.channel,
                    shared_workers=sorted(shared),
                ))

        # Sort by coupling strength descending, return top 100
        couplings.sort(key=lambda c: c.coupling_strength, reverse=True)
        return couplings[:100]

    def detect_cross_source_consistency(
        self,
        recent_n: int = 200,
        min_similarity: float = 0.4,
    ) -> list[CouplingDetection]:
        """Detect cross-source corroborations and divergences.

        When signals from different sources (GitHub PR + Linear issue) activate
        the same mesh regions, they corroborate each other. When correlated
        signals (same channel/author) from different sources activate different
        regions, something is inconsistent — a potential credibility signal.

        Returns cross-source pairs sorted by activation similarity.
        """
        if len(self._worker_universe) < 5:
            return []

        recent = self._fingerprints[-recent_n:]
        if len(recent) < 2:
            return []

        # Pre-compute
        vectors = [self._aligned_vector(fp) for fp in recent]

        # Group by source
        by_source: dict[str, list[int]] = {}
        for i, fp in enumerate(recent):
            by_source.setdefault(fp.source, []).append(i)

        sources = list(by_source.keys())
        if len(sources) < 2:
            return []

        results: list[CouplingDetection] = []
        for si in range(len(sources)):
            for sj in range(si + 1, len(sources)):
                for i in by_source[sources[si]]:
                    for j in by_source[sources[sj]]:
                        # Only compare signals with some shared context
                        # (same channel pattern or same author)
                        fp_a, fp_b = recent[i], recent[j]
                        shared_context = (
                            fp_a.author == fp_b.author
                            or _channel_related(fp_a.channel, fp_b.channel)
                        )
                        if not shared_context:
                            continue

                        # Variance guard — same as detect_couplings
                        if np.std(vectors[i]) < 0.05 or np.std(vectors[j]) < 0.05:
                            continue

                        act_sim = self._cosine_similarity(vectors[i], vectors[j])
                        if act_sim < min_similarity:
                            continue

                        term_sim = self._term_overlap(fp_a.terms, fp_b.terms)
                        shared = fp_a.accepted_workers & fp_b.accepted_workers

                        results.append(CouplingDetection(
                            signal_a_id=fp_a.signal_id,
                            signal_b_id=fp_b.signal_id,
                            activation_similarity=act_sim,
                            term_overlap=term_sim,
                            coupling_strength=act_sim,
                            source_a=fp_a.source,
                            source_b=fp_b.source,
                            channel_a=fp_a.channel,
                            channel_b=fp_b.channel,
                            shared_workers=sorted(shared),
                            description=f"Cross-source: {fp_a.source}/{fp_b.source} "
                                        f"via {fp_a.author}",
                        ))

        results.sort(key=lambda c: c.coupling_strength, reverse=True)
        return results

    def summary(self) -> dict[str, Any]:
        """Summary stats for output."""
        return {
            "fingerprints": len(self._fingerprints),
            "workers_tracked": len(self._worker_universe),
            "sources": sorted(self._by_source.keys()),
        }


def _channel_related(a: str, b: str) -> bool:
    """Heuristic: are two channels likely about the same thing?

    Matches on repo name overlap (acme-org/backend ↔ PLAT-xxx in backend).
    """
    a_parts = set(a.lower().replace("/", " ").replace("-", " ").split())
    b_parts = set(b.lower().replace("/", " ").replace("-", " ").split())
    # Remove very common words
    common = {"the", "and", "for", "com"}
    a_parts -= common
    b_parts -= common
    if not a_parts or not b_parts:
        return False
    overlap = a_parts & b_parts
    return len(overlap) > 0


def fingerprint_from_trace(trace, signal) -> SpectralFingerprint:
    """Create a SpectralFingerprint from a MeshTrace and its originating Signal."""
    activation = {
        str(wid)[:8]: score
        for wid, score in trace.familiarity_scores.items()
    }
    accepted = frozenset(str(wid)[:8] for wid in trace.accepted_workers)

    return SpectralFingerprint(
        signal_id=signal.id,
        timestamp=signal.timestamp,
        source=signal.source,
        channel=signal.channel,
        author=signal.author,
        terms=frozenset(signal.terms),
        activation=activation,
        accepted_workers=accepted,
        metadata={
            "entry_worker": str(trace.entry_worker_id)[:8] if trace.entry_worker_id else None,
            "hops": trace.total_hops,
        },
    )
