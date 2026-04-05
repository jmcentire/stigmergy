"""Structural mesh experiment — complement-coded continuous vectors through Fuzzy ART.

Uses the embedding channel (cosine similarity on 14-dim complement-coded vectors)
as the primary routing signal, with terms as backup. Familiarity reweighted to
prioritize structural similarity over keyword overlap.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.mesh.mesh import Mesh, MeshTrace
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.signal import Signal

logger = logging.getLogger(__name__)


# Reweighted for structural signals: embedding dominates
STRUCTURAL_WEIGHTS = FamiliarityWeights(
    embedding_similarity=0.60,  # Primary: cosine on 14-dim structural vector
    keyword_overlap=0.10,       # Backup: discretized term overlap
    source_affinity=0.05,       # Minimal: all from same source
    temporal_proximity=0.10,    # Temporal clustering still matters
    author_affinity=0.05,       # Minimal: author = tx_id (unique)
    signal_credibility=0.10,    # All transactions equally costly
    sequence_similarity=0.0,
)


@dataclass
class StructuralMeshResult:
    """Results from running structural signals through the mesh."""
    signals_processed: int = 0
    accepted: int = 0
    rejected: int = 0
    anomalies: list[dict] = field(default_factory=list)
    workers_final: int = 0
    tx_max_scores: dict[str, float] = field(default_factory=dict)
    tx_routing: dict[str, list[dict]] = field(default_factory=dict)


async def run_structural_mesh(
    signals: list[Signal],
    num_workers: int = 12,
    anomaly_threshold: float = 0.12,
) -> StructuralMeshResult:
    """Feed complement-coded structural signals through the mesh.

    Key difference from text-based run: familiarity is dominated by
    embedding_similarity (0.60 weight), which computes cosine distance
    on the 14-dim complement-coded structural vectors. Workers specialize
    by structural shape, not vocabulary.
    """
    mesh = Mesh(
        AgentRegistry(),
        max_workers=num_workers * 5,
        dedup_enabled=False,
        worker_capacity=1000,
        base_threshold=0.12,
        max_threshold=0.85,
        threshold_curve=2.0,
        gap_threshold=0.06,
        familiarity_weights=STRUCTURAL_WEIGHTS,
    )
    for i in range(num_workers):
        mesh.spawn_worker(source_name=f"structural_seed_{i}")

    result = StructuralMeshResult()
    total = len(signals)
    logger.info(f"Starting structural mesh: {total:,} signals, {num_workers} workers")
    logger.info(f"Familiarity weights: {STRUCTURAL_WEIGHTS.as_dict()}")

    for i, signal in enumerate(signals):
        if i > 0 and i % 5000 == 0:
            n_workers = len(mesh.workers)
            logger.info(
                f"  [{i:,}/{total:,}] workers={n_workers}, "
                f"accepted={result.accepted}, anomalies={len(result.anomalies)}"
            )

        trace = await mesh.ingest(signal)
        result.signals_processed += 1

        tx_id = signal.metadata.get("tx_id", "") or signal.metadata.get("address", "") or str(signal.id)
        max_score = max(trace.familiarity_scores.values()) if trace.familiarity_scores else 0
        result.tx_max_scores[tx_id] = max_score

        if trace.accepted_workers:
            result.accepted += 1
            for wid in trace.accepted_workers:
                if tx_id not in result.tx_routing:
                    result.tx_routing[tx_id] = []
                result.tx_routing[tx_id].append({
                    "worker_id": str(wid)[:8],
                    "score": trace.familiarity_scores.get(wid, 0),
                })
        else:
            result.rejected += 1

        if max_score < anomaly_threshold:
            result.anomalies.append({
                "tx_id": tx_id,
                "max_score": max_score,
                "content": signal.content,
                "time_step": signal.metadata.get("time_step"),
                "coefficients": signal.metadata.get("coefficients"),
            })

    result.workers_final = len(mesh.workers)

    logger.info(f"\n=== Structural Mesh Results ===")
    logger.info(f"Processed: {result.signals_processed:,}")
    logger.info(f"Accepted: {result.accepted:,}, Rejected: {result.rejected:,}")
    logger.info(f"Anomalies (max_score < {anomaly_threshold}): {len(result.anomalies):,}")
    logger.info(f"Workers: {num_workers} → {result.workers_final}")

    # Worker specializations — show structural profile
    logger.info(f"\nWorker structural profiles:")
    coeff_names = ["frag", "relay", "variance", "divergence", "density", "fan", "complexity"]
    for worker in sorted(mesh.workers, key=lambda w: -w.context.signal_count)[:15]:
        ctx = worker.context
        # Get centroid from vector store if available
        centroid = None
        try:
            centroid = asyncio.get_event_loop().run_until_complete(
                ctx.vector_store.get_centroid("structural")
            )
        except Exception:
            pass

        if centroid and len(centroid) >= 7:
            # First 7 dims are the structural coefficients (before complement)
            profile = {n: f"{centroid[i]:.2f}" for i, n in enumerate(coeff_names)}
            logger.info(
                f"  {str(worker.id)[:8]}: signals={ctx.signal_count}, "
                f"profile={profile}"
            )
        else:
            term_sample = sorted(list(ctx.terms))[:8]
            logger.info(
                f"  {str(worker.id)[:8]}: signals={ctx.signal_count}, "
                f"terms=[{', '.join(term_sample)}]"
            )

    return result


def evaluate_structural_results(
    result: StructuralMeshResult,
    classes: pl.DataFrame,
) -> dict:
    """Evaluate structural mesh results against known labels."""
    from stigmergy.ownership.bitcoin_bridge import evaluate_results as _eval

    # Wrap in a compatible object
    class _Compat:
        def __init__(self, r):
            self.tx_max_scores = r.tx_max_scores
            self.anomalies = r.anomalies

    return _eval(_Compat(result), classes)
