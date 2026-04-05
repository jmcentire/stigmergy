"""Wallet-level structural signals for Bitcoin ASD experiment.

Converts Elliptic++ wallet address features into complement-coded
structural coefficient vectors. Unlike per-transaction signals, wallets
are persistent entities with accumulated behavioral profiles — the right
substrate for stigmergic pheromone trail accumulation.

The 7 structural coefficients map to wallet behavioral topology:
  C1: Activity Asymmetry — send/receive imbalance (pass-through vs endpoint)
  C2: Volume Concentration — Gini of transaction values (uniform vs bursty)
  C3: Temporal Regularity — how periodic is the wallet's activity
  C4: Counterparty Diversity — unique addresses transacted with
  C5: Fee Behavior — fee patterns relative to transaction value
  C6: Lifetime Intensity — activity density over wallet lifetime
  C7: Velocity Profile — transaction rate acceleration/deceleration
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

_BASE_TS = datetime(2018, 1, 1, tzinfo=timezone.utc)
_STEP_DELTA = timedelta(weeks=2)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sigmoid(x: float, mid: float, steep: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-steep * (x - mid)))


def complement_code(a: list[float]) -> list[float]:
    return a + [1.0 - x for x in a]


def wallet_to_structural_vector(row: dict) -> list[float]:
    """Compute 7 structural coefficients from wallet features."""

    num_sent = row.get("num_txs_as_sender", 0) or 0
    num_recv = row.get("num_txs_as receiver", 0) or 0
    total_txs = row.get("total_txs", 0) or 0
    lifetime = row.get("lifetime_in_blocks", 0) or 0
    timesteps = row.get("num_timesteps_appeared_in", 1) or 1

    btc_total = row.get("btc_transacted_total", 0) or 0
    btc_min = row.get("btc_transacted_min", 0) or 0
    btc_max = row.get("btc_transacted_max", 0) or 0
    btc_mean = row.get("btc_transacted_mean", 0) or 0
    btc_median = row.get("btc_transacted_median", 0) or 0

    sent_total = row.get("btc_sent_total", 0) or 0
    recv_total = row.get("btc_received_total", 0) or 0

    fees_total = row.get("fees_total", 0) or 0
    fees_mean = row.get("fees_mean", 0) or 0
    fees_share_mean = row.get("fees_as_share_mean", 0) or 0

    blocks_btwn_mean = row.get("blocks_btwn_txs_mean", 0) or 0
    blocks_btwn_max = row.get("blocks_btwn_txs_max", 0) or 0
    blocks_btwn_median = row.get("blocks_btwn_txs_median", 0) or 0

    counterparties_total = row.get("transacted_w_address_total", 0) or 0
    repeat_counterparties = row.get("num_addr_transacted_multiple", 0) or 0

    # C1: Activity Asymmetry — send/receive balance
    # 0 = pure receiver (endpoint/sink), 0.5 = balanced (relay), 1 = pure sender (source)
    if total_txs > 0:
        c1 = num_sent / total_txs
    else:
        c1 = 0.5

    # C2: Volume Concentration — how uniform are transaction amounts
    # High when max >> mean (bursty/concentrated), low when uniform
    if btc_mean > 0 and btc_max > 0:
        # Coefficient of variation proxy: max/mean ratio normalized
        cv = btc_max / btc_mean
        c2 = _clip(_sigmoid(cv, mid=5.0, steep=0.5))
    else:
        c2 = 0.0

    # C3: Temporal Regularity — how periodic is activity
    # Low blocks_between variance = regular (automated), high = irregular (human)
    if blocks_btwn_mean > 0 and blocks_btwn_max > 0:
        regularity = blocks_btwn_median / blocks_btwn_max  # 1.0 = perfectly regular
        c3 = _clip(regularity)
    else:
        c3 = 0.0

    # C4: Counterparty Diversity — normalized unique counterparties
    # High = transacts with many different addresses (exchange-like)
    if total_txs > 0:
        c4 = _clip(_sigmoid(counterparties_total / total_txs, mid=2.0, steep=1.0))
    elif counterparties_total > 0:
        c4 = _clip(_sigmoid(counterparties_total, mid=10.0, steep=0.3))
    else:
        c4 = 0.0

    # C5: Fee Behavior — fees as share of value
    # High fees relative to value = possible priority/urgency or small-value structuring
    c5 = _clip(fees_share_mean * 100)  # scale up: 1% fee -> 1.0

    # C6: Lifetime Intensity — transactions per block of lifetime
    # High = dense burst of activity, low = spread out
    if lifetime > 0:
        intensity = total_txs / lifetime * 1000  # normalize: 1 tx per 1000 blocks -> 1.0
        c6 = _clip(_sigmoid(intensity, mid=1.0, steep=2.0))
    elif total_txs > 0:
        c6 = 1.0  # all activity in one block = maximum intensity
    else:
        c6 = 0.0

    # C7: Repeat Counterparty Ratio — fraction of counterparties seen multiple times
    # High = habitual relationships (legitimate business), low = one-shot (structuring)
    if counterparties_total > 0:
        c7 = _clip(repeat_counterparties / counterparties_total)
    else:
        c7 = 0.0

    return [_clip(c1), _clip(c2), _clip(c3), _clip(c4), _clip(c5), _clip(c6), _clip(c7)]


COEFF_NAMES = [
    "asymmetry", "volume_conc", "temporal_reg",
    "counterparty_div", "fee_behavior", "intensity", "repeat_ratio",
]


def build_wallet_signals(
    features: pl.DataFrame,
    classes: pl.DataFrame,
    limit: int | None = None,
) -> list[Signal]:
    """Build complement-coded structural signals from wallet data.

    Wallets are sorted by first_block_appeared_in (chronological).
    Each wallet appearance at a time step becomes one signal.
    """
    # Join classes for metadata (NOT for signal content — mesh is unsupervised)
    merged = features.join(classes, on="address", how="left")
    merged = merged.sort("first_block_appeared_in")

    if limit:
        merged = merged.head(limit)

    signals = []
    rows = merged.to_dicts()

    for row in rows:
        address = row.get("address", "")
        time_step = row.get("Time step", 0)
        label = row.get("class", 3)  # stored in metadata only, not in signal content

        sv = wallet_to_structural_vector(row)
        cc = complement_code(sv)

        # Minimal content for terms backup
        term_parts = []
        for name, val in zip(COEFF_NAMES, sv):
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
            source="bitcoin_wallet",
            channel=f"ts_{time_step}",
            author=address,  # full address as unique key
            timestamp=timestamp,
            embeddings={"structural": cc},
            metadata={
                "address": address,
                "time_step": time_step,
                "label": label,  # 1=illicit, 2=licit, 3=unknown
                "coefficients": {n: v for n, v in zip(COEFF_NAMES, sv)},
            },
        ))

    logger.info(f"Built {len(signals):,} wallet signals (14-dim complement-coded)")

    # Log coefficient distributions
    if signals:
        all_sv = np.array([[s.metadata["coefficients"][n] for n in COEFF_NAMES] for s in signals])
        for i, name in enumerate(COEFF_NAMES):
            col = all_sv[:, i]
            logger.info(f"  {name}: mean={col.mean():.3f}, std={col.std():.3f}, "
                        f"[{col.min():.3f}, {col.max():.3f}]")

    # Label distribution in signals
    labels = [s.metadata["label"] for s in signals]
    from collections import Counter
    lc = Counter(labels)
    logger.info(f"  Labels: illicit={lc.get(1,0):,}, licit={lc.get(2,0):,}, unknown={lc.get(3,0):,}")

    return signals
