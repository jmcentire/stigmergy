"""Complement-coded continuous structural signal construction for Fuzzy ART.

Replaces text-based signal content with continuous-valued structural
coefficient vectors bounded in [0, 1], complement-coded to prevent
category proliferation.

The seven structural coefficients:
  D_L — Layering Depth Index (chain depth normalized)
  F_o — Ownership Fragmentation Ratio (Gini-Simpson of shareholdings)
  N_P — Nominee Probability Score (directorship fanout)
  J_S — Jurisdictional Secrecy Coefficient (Financial Secrecy Index)
  C_B — Cross-Border Complexity (jurisdiction diversity in chain)
  T_F — Temporal Fan-Out/Fan-In Ratio (structural motif)
  V_S — Feature Variance Score (neighborhood heterogeneity)

Complement-coded: I = (a, 1-a), giving 14-dim input vectors.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from stigmergy.primitives.signal import Signal

logger = logging.getLogger(__name__)

# Base timestamp for Elliptic time steps
_BASE_TS = datetime(2018, 1, 1, tzinfo=timezone.utc)
_STEP_DELTA = timedelta(weeks=2)

# ── Normalization helpers ──

def _sigmoid(x: float, midpoint: float = 5.0, steepness: float = 1.0) -> float:
    """Sigmoid normalization to [0, 1]. Midpoint controls the inflection."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def _tanh_norm(x: float, scale: float = 1.0) -> float:
    """Tanh normalization to [0, 1]. Maps (-inf, inf) to (0, 1)."""
    return (math.tanh(x / scale) + 1.0) / 2.0


def _clip(x: float) -> float:
    """Clip to [0, 1]."""
    return max(0.0, min(1.0, x))


def complement_code(a: list[float]) -> list[float]:
    """Complement code a vector: I = (a, 1-a). Doubles dimensionality."""
    return a + [1.0 - x for x in a]


# ── Elliptic Bitcoin structural coefficients ──

def elliptic_transaction_to_structural_vector(
    row: dict,
    in_deg: int,
    out_deg: int,
) -> list[float]:
    """Compute 7 continuous structural coefficients from an Elliptic++ transaction.

    Maps the Elliptic feature space to the ASD structural coefficient framework:
      D_L → not applicable for transactions (use chain_depth proxy from graph)
      F_o → output fragmentation (how many outputs, how distributed)
      N_P → intermediary score (is this transaction a relay/pass-through)
      J_S → not applicable (no jurisdiction in Bitcoin, use feature variance proxy)
      C_B → neighborhood complexity
      T_F → fan-out / fan-in ratio (structuring vs integration motif)
      V_S → feature variance (heterogeneity of local features)
    """
    # Extract local features (first 94 in Elliptic)
    local_feats = np.array([row.get(f"Local_feature_{i}", 0) or 0 for i in range(1, 95)], dtype=np.float64)
    agg_feats = np.array([row.get(f"Aggregate_feature_{i}", 0) or 0 for i in range(1, 73)], dtype=np.float64)

    local_std = float(np.std(local_feats))
    local_mean = float(np.mean(local_feats))
    local_max = float(np.max(np.abs(local_feats)))
    local_nonzero = int(np.count_nonzero(local_feats))

    agg_std = float(np.std(agg_feats))
    agg_mean = float(np.mean(agg_feats))
    agg_nonzero = int(np.count_nonzero(agg_feats))

    # C1: Fragmentation Ratio (F_o) — output dispersal
    # High when transaction splits to many outputs (structuring signature)
    # Normalized: 0 = single output, 1 = highly fragmented
    f_o = _sigmoid(out_deg, midpoint=3.0, steepness=0.8)

    # C2: Intermediary Score (N_P) — relay probability
    # High when both in-degree and out-degree are present (pass-through)
    # Low for source (in=0) or sink (out=0) nodes
    total_deg = in_deg + out_deg
    if total_deg > 0:
        balance = 1.0 - abs(in_deg - out_deg) / total_deg  # 1 = perfectly balanced relay
        n_p = balance * _sigmoid(total_deg, midpoint=4.0, steepness=0.5)
    else:
        n_p = 0.0

    # C3: Feature Variance Score (V_S) — local heterogeneity
    # High variance = structurally unusual transaction
    v_s = _clip(local_std / 3.0)  # normalize: std of 3.0 → 1.0

    # C4: Neighborhood Divergence (J_S proxy) — how different is this tx from neighbors
    # High divergence = transaction doesn't fit its local neighborhood
    if agg_std > 0 and local_std > 0:
        divergence = abs(agg_mean - local_mean) / max(local_std, 0.01)
        j_s = _clip(divergence / 3.0)
    else:
        j_s = 0.0

    # C5: Feature Extremity (C_B) — how extreme are this tx's features?
    # Measures the tail behavior: fraction of features > 2 std from mean
    if local_std > 0:
        z_scores = np.abs((local_feats - local_mean) / local_std)
        c_b = float(np.mean(z_scores > 2.0))  # fraction of extreme features
    else:
        c_b = 0.0

    # C6: Fan ratio (T_F) — structuring vs integration motif
    # Maps fan-out/fan-in asymmetry to [0, 1] via tanh
    # 0.5 = balanced, 0 = pure fan-in (integration), 1 = pure fan-out (structuring)
    if in_deg + out_deg > 0:
        fan_ratio = (out_deg - in_deg) / (out_deg + in_deg)
        t_f = (fan_ratio + 1.0) / 2.0  # map [-1, 1] to [0, 1]
    else:
        t_f = 0.5

    # C7: Neighborhood Heterogeneity — how diverse is the local neighborhood
    # High when aggregate features have high coefficient of variation
    if agg_mean != 0:
        c_n = _clip(agg_std / abs(agg_mean) / 3.0)  # CV normalized to [0,1]
    else:
        c_n = 0.0

    # Assemble the 7-coefficient structural vector
    structural_vector = [
        _clip(f_o),   # Fragmentation / fan-out
        _clip(n_p),   # Intermediary / relay score
        _clip(v_s),   # Feature variance
        _clip(j_s),   # Neighborhood divergence
        _clip(c_b),   # Feature density
        _clip(t_f),   # Fan ratio (structuring vs integration)
        _clip(c_n),   # Neighborhood complexity
    ]

    return structural_vector


def build_structural_signals(
    features: pl.DataFrame,
    edges: pl.DataFrame,
    limit: int | None = None,
) -> list[Signal]:
    """Build complement-coded structural signals from Elliptic++ data.

    Each transaction becomes a Signal with:
    - embeddings["structural"]: 14-dim complement-coded vector
    - content: minimal structural description (for terms/bloom backup)
    - metadata: tx_id, time_step, coefficients
    """
    tx_ids = set(features["txId"].to_list())
    edge_list = edges.filter(
        pl.col("txId1").is_in(tx_ids) & pl.col("txId2").is_in(tx_ids)
    )

    out_degree = dict(edge_list.group_by("txId1").agg(pl.len().alias("c")).iter_rows())
    in_degree = dict(edge_list.group_by("txId2").agg(pl.len().alias("c")).iter_rows())

    sorted_features = features.sort("Time step")
    if limit:
        sorted_features = sorted_features.head(limit)

    signals = []
    rows = sorted_features.to_dicts()

    for row in rows:
        tx_id = str(row["txId"])
        time_step = row["Time step"]
        in_deg = in_degree.get(int(row["txId"]), 0)
        out_deg = out_degree.get(int(row["txId"]), 0)

        # Compute structural coefficients
        sv = elliptic_transaction_to_structural_vector(row, in_deg, out_deg)
        cc = complement_code(sv)

        # Minimal content for terms channel (backup routing)
        # Use coefficient magnitudes as discretized terms
        term_parts = []
        coeff_names = ["frag", "relay", "variance", "divergence", "density", "fan", "complexity"]
        for name, val in zip(coeff_names, sv):
            if val > 0.7:
                term_parts.append(f"high_{name}")
            elif val > 0.3:
                term_parts.append(f"mid_{name}")
            else:
                term_parts.append(f"low_{name}")

        content = " ".join(term_parts)
        timestamp = _BASE_TS + _STEP_DELTA * time_step

        signals.append(Signal(
            content=content,
            source="bitcoin_structural",
            channel=f"ts_{time_step}",
            author=tx_id,
            timestamp=timestamp,
            embeddings={"structural": cc},
            metadata={
                "tx_id": tx_id,
                "time_step": time_step,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "coefficients": {n: v for n, v in zip(coeff_names, sv)},
            },
        ))

    logger.info(f"Built {len(signals):,} structural signals (14-dim complement-coded)")

    # Log coefficient distributions
    if signals:
        all_coeffs = np.array([s.metadata["coefficients"][n] for s in signals for n in coeff_names]).reshape(-1, 7)
        for i, name in enumerate(coeff_names):
            col = all_coeffs[:, i]
            logger.info(f"  {name}: mean={col.mean():.3f}, std={col.std():.3f}, "
                        f"min={col.min():.3f}, max={col.max():.3f}")

    return signals
