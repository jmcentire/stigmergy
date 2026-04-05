"""Bridge between Elliptic++ Bitcoin data and the stigmergic mesh.

Converts Bitcoin transactions into Signals and feeds them through the
mesh. The signal content encodes STRUCTURAL properties of each transaction
(input/output fan, timing, feature patterns) rather than identity.

The mesh has never seen financial data. It will self-organize workers
that specialize in transaction patterns. Transactions that fail resonance
with all workers are structural anomalies — potentially illicit patterns
the mesh discovered without being told what to look for.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import UUID

import numpy as np
import polars as pl

from stigmergy.mesh.mesh import Mesh, MeshTrace
from stigmergy.mesh.fingerprints import FingerprintStore, SpectralFingerprint
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.signal import Signal

logger = logging.getLogger(__name__)

# Base timestamp for time steps (Elliptic uses abstract time steps 1-49)
_BASE_TS = datetime(2018, 1, 1, tzinfo=timezone.utc)
_STEP_DELTA = timedelta(weeks=2)  # ~2 weeks per time step


def load_elliptic(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load Elliptic++ transaction data.

    Returns (features, classes, edges) DataFrames.
    """
    features = pl.read_csv(data_dir / "txs_features.csv")
    classes = pl.read_csv(data_dir / "txs_classes.csv")
    edges = pl.read_csv(data_dir / "txs_edgelist.csv")

    logger.info(f"Loaded: {len(features):,} transactions, {len(edges):,} edges")
    logger.info(f"Classes: illicit={classes.filter(pl.col('class')==1).height}, "
                f"licit={classes.filter(pl.col('class')==2).height}, "
                f"unknown={classes.filter(pl.col('class')==3).height}")

    return features, classes, edges


def transactions_to_signals(
    features: pl.DataFrame,
    edges: pl.DataFrame,
    limit: int | None = None,
) -> list[Signal]:
    """Convert Elliptic++ transactions into Signals for mesh ingestion.

    Signal content encodes structural transaction properties as descriptive
    text that the mesh's term extraction can tokenize. The mesh will cluster
    by vocabulary overlap — transactions with similar structural descriptions
    will route to the same workers.

    Key structural properties encoded:
    - Time step (temporal ordering)
    - Local features (transaction-specific: input/output counts, values, fees)
    - Aggregate neighbor features (1-hop neighborhood statistics)
    """
    # Build adjacency for computing fan-in/fan-out
    tx_ids = set(features["txId"].to_list())
    edge_list = edges.filter(
        pl.col("txId1").is_in(tx_ids) & pl.col("txId2").is_in(tx_ids)
    )

    # Compute in-degree and out-degree per transaction
    out_degree = edge_list.group_by("txId1").agg(pl.len().alias("out_deg"))
    in_degree = edge_list.group_by("txId2").agg(pl.len().alias("in_deg"))

    # Join degrees to features
    enriched = features.join(out_degree, left_on="txId", right_on="txId1", how="left")
    enriched = enriched.join(in_degree, left_on="txId", right_on="txId2", how="left")
    enriched = enriched.with_columns([
        pl.col("out_deg").fill_null(0),
        pl.col("in_deg").fill_null(0),
    ])

    # Sort by time step (chronological ingestion)
    enriched = enriched.sort("Time step")

    if limit:
        enriched = enriched.head(limit)

    signals = []
    rows = enriched.to_dicts()

    for row in rows:
        tx_id = str(row["txId"])
        time_step = row["Time step"]
        in_deg = row["in_deg"]
        out_deg = row["out_deg"]

        # Extract key local features (first 94 are local per Elliptic paper)
        # Features are anonymized but we can describe their statistical properties
        local_feats = [row.get(f"Local_feature_{i}", 0) or 0 for i in range(1, 95)]
        agg_feats = [row.get(f"Aggregate_feature_{i}", 0) or 0 for i in range(1, 73)]

        # Compute structural descriptors from features
        local_arr = np.array(local_feats, dtype=np.float64)
        agg_arr = np.array(agg_feats, dtype=np.float64)

        local_nonzero = int(np.count_nonzero(local_arr))
        local_mean = float(np.mean(local_arr))
        local_std = float(np.std(local_arr))
        local_max = float(np.max(np.abs(local_arr)))

        agg_nonzero = int(np.count_nonzero(agg_arr))
        agg_mean = float(np.mean(agg_arr))
        agg_std = float(np.std(agg_arr))

        # Categorize structural properties into descriptive terms
        # These become the vocabulary the mesh clusters on
        parts = [f"timestep_{time_step}"]

        # Fan structure
        if in_deg == 0:
            parts.append("source_transaction")
        elif in_deg == 1:
            parts.append("single_input")
        elif in_deg <= 3:
            parts.append("few_inputs")
        else:
            parts.append("many_inputs")
            parts.append("aggregation_pattern")

        if out_deg == 0:
            parts.append("terminal_transaction")
        elif out_deg == 1:
            parts.append("single_output")
        elif out_deg <= 3:
            parts.append("few_outputs")
        else:
            parts.append("many_outputs")
            parts.append("distribution_pattern")

        # Structural patterns from fan combinations
        if in_deg > 3 and out_deg <= 1:
            parts.append("consolidation")  # many-to-one: mixing, collection
        elif in_deg <= 1 and out_deg > 3:
            parts.append("dispersal")  # one-to-many: distribution, peel chain start
        elif in_deg > 3 and out_deg > 3:
            parts.append("throughput")  # many-to-many: exchange, mixer
        elif in_deg == 1 and out_deg == 1:
            parts.append("relay")  # pass-through: layering

        # Feature magnitude categories
        if local_std > 2.0:
            parts.append("high_variance_local")
        elif local_std < 0.3:
            parts.append("low_variance_local")
        else:
            parts.append("moderate_variance_local")

        if local_max > 5.0:
            parts.append("extreme_local_value")

        if agg_std > 2.0:
            parts.append("high_variance_neighborhood")
        elif agg_std < 0.3:
            parts.append("low_variance_neighborhood")

        # Feature density (how many features are active)
        if local_nonzero < 20:
            parts.append("sparse_features")
        elif local_nonzero > 70:
            parts.append("dense_features")

        if agg_nonzero < 10:
            parts.append("sparse_neighborhood")
        elif agg_nonzero > 50:
            parts.append("dense_neighborhood")

        # Neighborhood divergence (do neighbors look different from this tx?)
        if agg_std > 0 and local_std > 0:
            divergence = abs(agg_mean - local_mean) / max(local_std, 0.01)
            if divergence > 2.0:
                parts.append("neighborhood_divergent")
            elif divergence < 0.5:
                parts.append("neighborhood_consistent")

        content = " ".join(parts)
        timestamp = _BASE_TS + _STEP_DELTA * time_step

        signals.append(Signal(
            content=content,
            source="bitcoin",
            channel=f"ts_{time_step}",
            author=tx_id,
            timestamp=timestamp,
            metadata={
                "tx_id": tx_id,
                "time_step": time_step,
                "in_degree": in_deg,
                "out_degree": out_deg,
            },
        ))

    logger.info(f"Built {len(signals):,} signals from transactions")
    return signals


@dataclass
class BitcoinMeshResult:
    """Results from running Bitcoin transactions through the mesh."""
    signals_processed: int = 0
    accepted: int = 0
    rejected: int = 0
    anomalies: list[dict] = field(default_factory=list)
    workers_final: int = 0
    # Per-transaction routing: tx_id -> (worker_id, score)
    tx_routing: dict[str, list[dict]] = field(default_factory=dict)
    # Max familiarity score per transaction
    tx_max_scores: dict[str, float] = field(default_factory=dict)


async def run_bitcoin_mesh(
    signals: list[Signal],
    num_workers: int = 12,
    anomaly_threshold: float = 0.12,
) -> BitcoinMeshResult:
    """Feed Bitcoin transaction signals through the stigmergic mesh.

    The mesh processes transactions chronologically. Workers self-organize
    into transaction pattern categories. Anomalies are transactions that
    fail resonance with ALL workers — novel patterns the mesh hasn't seen.
    """
    mesh = Mesh(
        AgentRegistry(),
        max_workers=num_workers * 5,
        dedup_enabled=False,  # each transaction is unique
        worker_capacity=1000,
        base_threshold=0.12,
        max_threshold=0.85,
        threshold_curve=2.0,
        gap_threshold=0.06,
    )
    for i in range(num_workers):
        mesh.spawn_worker(source_name=f"btc_seed_{i}")

    result = BitcoinMeshResult()
    total = len(signals)
    logger.info(f"Starting Bitcoin mesh: {total:,} signals, {num_workers} workers")

    for i, signal in enumerate(signals):
        if i > 0 and i % 5000 == 0:
            n_workers = len(mesh.workers)
            logger.info(
                f"  [{i:,}/{total:,}] workers={n_workers}, "
                f"accepted={result.accepted}, anomalies={len(result.anomalies)}"
            )

        trace = await mesh.ingest(signal)
        result.signals_processed += 1

        tx_id = signal.metadata.get("tx_id", "")
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
            })

    result.workers_final = len(mesh.workers)

    # Log results
    logger.info(f"\n=== Bitcoin Mesh Results ===")
    logger.info(f"Processed: {result.signals_processed:,}")
    logger.info(f"Accepted: {result.accepted:,}, Rejected: {result.rejected:,}")
    logger.info(f"Anomalies: {len(result.anomalies):,}")
    logger.info(f"Workers: {num_workers} → {result.workers_final}")

    # Worker specializations
    logger.info(f"\nWorker specializations:")
    for worker in sorted(mesh.workers, key=lambda w: -w.context.signal_count)[:15]:
        ctx = worker.context
        term_sample = sorted(list(ctx.terms))[:12]
        logger.info(
            f"  {str(worker.id)[:8]}: signals={ctx.signal_count}, "
            f"terms=[{', '.join(term_sample)}]"
        )

    return result


def evaluate_results(
    result: BitcoinMeshResult,
    classes: pl.DataFrame,
) -> dict:
    """Evaluate mesh results against known labels.

    Key question: do the mesh's anomalies and low-scoring transactions
    correlate with known illicit transactions?
    """
    # Build label lookup: tx_id -> class (1=illicit, 2=licit, 3=unknown)
    labels = {}
    for row in classes.iter_rows(named=True):
        labels[str(row["txId"])] = row["class"]

    # Score distribution by class
    illicit_scores = []
    licit_scores = []
    unknown_scores = []

    for tx_id, max_score in result.tx_max_scores.items():
        label = labels.get(tx_id, 3)
        if label == 1:
            illicit_scores.append(max_score)
        elif label == 2:
            licit_scores.append(max_score)
        else:
            unknown_scores.append(max_score)

    illicit_arr = np.array(illicit_scores) if illicit_scores else np.array([0])
    licit_arr = np.array(licit_scores) if licit_scores else np.array([0])

    # Effect size: Cohen's d
    pooled_std = np.sqrt((np.var(illicit_arr) + np.var(licit_arr)) / 2)
    cohens_d = (np.mean(licit_arr) - np.mean(illicit_arr)) / max(pooled_std, 1e-10)

    # Anomaly enrichment: are anomalies disproportionately illicit?
    anomaly_tx_ids = {a["tx_id"] for a in result.anomalies}
    anomaly_illicit = sum(1 for t in anomaly_tx_ids if labels.get(t) == 1)
    anomaly_licit = sum(1 for t in anomaly_tx_ids if labels.get(t) == 2)
    anomaly_unknown = sum(1 for t in anomaly_tx_ids if labels.get(t, 3) == 3)

    # Base rate
    total_illicit = sum(1 for v in labels.values() if v == 1)
    total_licit = sum(1 for v in labels.values() if v == 2)
    total_labeled = total_illicit + total_licit
    base_rate = total_illicit / max(total_labeled, 1)

    # Anomaly illicit rate
    anomaly_labeled = anomaly_illicit + anomaly_licit
    anomaly_rate = anomaly_illicit / max(anomaly_labeled, 1)
    lift = anomaly_rate / max(base_rate, 1e-10)

    # Bottom decile enrichment: among the 10% lowest-scoring labeled transactions,
    # what fraction are illicit?
    labeled_scores = []
    for tx_id, score in result.tx_max_scores.items():
        label = labels.get(tx_id, 3)
        if label in (1, 2):
            labeled_scores.append((score, label))

    labeled_scores.sort(key=lambda x: x[0])
    bottom_10pct = labeled_scores[:max(1, len(labeled_scores) // 10)]
    bottom_illicit = sum(1 for _, l in bottom_10pct if l == 1)
    bottom_rate = bottom_illicit / max(len(bottom_10pct), 1)
    bottom_lift = bottom_rate / max(base_rate, 1e-10)

    results = {
        "illicit_mean_score": float(np.mean(illicit_arr)),
        "licit_mean_score": float(np.mean(licit_arr)),
        "cohens_d": float(cohens_d),
        "anomaly_count": len(result.anomalies),
        "anomaly_illicit": anomaly_illicit,
        "anomaly_licit": anomaly_licit,
        "anomaly_unknown": anomaly_unknown,
        "anomaly_illicit_rate": anomaly_rate,
        "base_illicit_rate": base_rate,
        "anomaly_lift": lift,
        "bottom_10pct_illicit_rate": bottom_rate,
        "bottom_10pct_lift": bottom_lift,
        "total_illicit": total_illicit,
        "total_licit": total_licit,
    }

    logger.info(f"\n=== Evaluation Against Known Labels ===")
    logger.info(f"Illicit mean score: {results['illicit_mean_score']:.4f}")
    logger.info(f"Licit mean score: {results['licit_mean_score']:.4f}")
    logger.info(f"Cohen's d (licit - illicit): {results['cohens_d']:.4f}")
    logger.info(f"Base illicit rate: {results['base_illicit_rate']:.4f}")
    logger.info(f"Anomaly illicit rate: {results['anomaly_illicit_rate']:.4f} "
                f"(lift: {results['anomaly_lift']:.2f}x)")
    logger.info(f"Bottom 10% illicit rate: {results['bottom_10pct_illicit_rate']:.4f} "
                f"(lift: {results['bottom_10pct_lift']:.2f}x)")

    return results
