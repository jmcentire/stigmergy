"""Bridge between ownership data and the stigmergic mesh.

Converts PSC ownership records into Signals and feeds them through the
mesh's ART routing. The mesh has no knowledge of ownership concepts —
it discovers structure through competitive resonance.

This is the actual ASD test: can the mesh, processing signals it has
never been trained on, surface structural anomalies that conventional
feature analysis misses?
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from stigmergy.mesh.mesh import Mesh, MeshTrace
from stigmergy.mesh.fingerprints import FingerprintStore, SpectralFingerprint
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.signal import Signal

logger = logging.getLogger(__name__)


def psc_record_to_signal(company_number: str, data: dict) -> Signal | None:
    """Convert a PSC ndjson record into a Signal for mesh ingestion.

    The content is a natural-language-ish description of the ownership
    relationship, rich enough for the mesh's term extraction to work.
    The mesh will route it based on vocabulary overlap, temporal proximity,
    source affinity, and author affinity — discovering structural categories
    without being told what to look for.
    """
    kind = data.get("kind", "")
    if "super-secure" in kind:
        return None
    if data.get("ceased_on"):
        return None

    # Build content string with ownership-relevant terms
    parts = []

    # Owner identity
    name = data.get("name", "")
    if name:
        parts.append(name)

    # Control nature
    natures = data.get("natures_of_control") or []
    natures = [n for n in natures if n]
    for n in natures:
        parts.append(n.replace("-", " "))

    # Jurisdiction signals
    nationality = data.get("nationality", "")
    country = data.get("country_of_residence", "")
    if nationality:
        parts.append(f"nationality {nationality}")
    if country:
        parts.append(f"resident {country}")

    # Corporate entity identification
    identification = data.get("identification", {})
    if identification:
        legal_auth = identification.get("legal_authority", "")
        place = identification.get("place_registered", "")
        country_reg = identification.get("country_registered", "")
        if legal_auth:
            parts.append(f"authority {legal_auth}")
        if place:
            parts.append(f"registered {place}")
        if country_reg:
            parts.append(f"jurisdiction {country_reg}")

    # Address
    address = data.get("address", {})
    if isinstance(address, dict):
        addr_country = address.get("country", "")
        locality = address.get("locality", "")
        if addr_country:
            parts.append(addr_country)
        if locality:
            parts.append(locality)

    content = f"company {company_number} controlled by {' '.join(parts)}"

    # Determine author: the declaring person/entity
    author = name or f"unknown_{company_number}"

    # Parse notification date
    notified = data.get("notified_on", "")
    if notified:
        try:
            ts_parts = notified.split("-")
            ts = datetime(int(ts_parts[0]), int(ts_parts[1]), int(ts_parts[2]),
                         tzinfo=timezone.utc)
        except (ValueError, IndexError):
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    # Determine source type for credibility scoring
    is_person = "individual" in kind
    is_corporate = "corporate" in kind or "legal-person" in kind
    event_type = "ownership_person" if is_person else "ownership_corporate"

    return Signal(
        content=content,
        source="ownership",
        channel=company_number,
        author=author,
        timestamp=ts,
        metadata={
            "event_type": event_type,
            "kind": kind,
            "natures_of_control": natures,
            "company_number": company_number,
        },
    )


def stream_psc_signals(paths: list[Path], limit: int | None = None) -> list[Signal]:
    """Parse PSC files and convert to Signals, sorted by timestamp."""
    signals = []
    count = 0
    skipped = 0

    for path in paths:
        logger.info(f"Parsing {path.name}...")
        with open(path, "rb") as f:
            for line in f:
                if limit and count >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                company_number = record.get("company_number", "")
                data = record.get("data", {})
                if not company_number or not data:
                    continue

                sig = psc_record_to_signal(company_number, data)
                if sig:
                    signals.append(sig)
                    count += 1
                else:
                    skipped += 1

                if count % 500_000 == 0 and count > 0:
                    logger.info(f"  {count:,} signals parsed ({skipped:,} skipped)")

            if limit and count >= limit:
                break

    # Sort by timestamp (chronological ingestion)
    signals.sort(key=lambda s: s.timestamp)
    logger.info(f"Total: {count:,} signals, sorted chronologically")
    return signals


@dataclass
class MeshExperimentResult:
    """Results from running ownership signals through the mesh."""
    signals_processed: int = 0
    signals_accepted: int = 0
    signals_rejected: int = 0
    gap_detections: int = 0  # new workers spawned
    workers_final: int = 0
    traces: list[MeshTrace] = field(default_factory=list)
    # Per-company: which worker accepted it, at what score
    company_routing: dict[str, list[dict]] = field(default_factory=dict)
    # Anomalies: signals with low max familiarity across all workers
    anomalies: list[dict] = field(default_factory=list)
    # Fingerprints for coupling detection
    fingerprints: list[SpectralFingerprint] = field(default_factory=list)


async def run_mesh_experiment(
    signals: list[Signal],
    num_workers: int = 10,
    anomaly_threshold: float = 0.2,
) -> MeshExperimentResult:
    """Feed ownership signals through the stigmergic mesh.

    This is the core ASD experiment. The mesh processes ownership signals
    chronologically with no prior knowledge. Workers self-organize via ART
    resonance. Signals that fail resonance with ALL workers are anomalies —
    patterns the mesh has never seen.

    Args:
        signals: Chronologically sorted ownership signals
        num_workers: Initial worker count (mesh may spawn more via gap detection)
        anomaly_threshold: Max familiarity below which a signal is flagged anomalous
    """
    mesh = Mesh(
        AgentRegistry(),
        max_workers=num_workers * 5,  # allow gap detection to spawn beyond initial
        dedup_enabled=True,
        dedup_threshold=3,
        worker_capacity=500,  # larger capacity for ownership signals
        base_threshold=0.15,
        max_threshold=0.8,
        threshold_curve=2.0,
        gap_threshold=0.08,
    )
    # Seed initial workers (mesh starts empty, gap detection spawns on first signals)
    for i in range(num_workers):
        mesh.spawn_worker(source_name=f"ownership_seed_{i}")
    fp_store = FingerprintStore()
    result = MeshExperimentResult()

    total = len(signals)
    logger.info(f"Starting mesh experiment: {total:,} signals, {num_workers} initial workers")

    for i, signal in enumerate(signals):
        if i > 0 and i % 1000 == 0:
            n_workers = len(mesh.workers)
            logger.info(
                f"  [{i:,}/{total:,}] workers={n_workers}, "
                f"accepted={result.signals_accepted}, "
                f"anomalies={len(result.anomalies)}"
            )

        trace = await mesh.ingest(signal)
        result.signals_processed += 1

        if trace.duplicate:
            continue

        # Track routing
        company = signal.metadata.get("company_number", "")
        max_score = max(trace.familiarity_scores.values()) if trace.familiarity_scores else 0
        accepted = bool(trace.accepted_workers)

        if accepted:
            result.signals_accepted += 1
        else:
            result.signals_rejected += 1

        # Track per-company routing
        if company:
            if company not in result.company_routing:
                result.company_routing[company] = []
            for wid in trace.accepted_workers:
                result.company_routing[company].append({
                    "worker_id": str(wid),
                    "score": trace.familiarity_scores.get(wid, 0),
                })

        # Detect anomalies: signals where NO worker scored above threshold
        if max_score < anomaly_threshold:
            result.anomalies.append({
                "signal_id": str(signal.id),
                "company_number": company,
                "content": signal.content[:200],
                "max_score": max_score,
                "author": signal.author,
                "timestamp": signal.timestamp.isoformat(),
                "scores": {str(k): v for k, v in trace.familiarity_scores.items()},
            })

        # Build spectral fingerprint for coupling detection
        if trace.familiarity_scores:
            fp = SpectralFingerprint(
                signal_id=signal.id,
                timestamp=signal.timestamp,
                source=signal.source,
                channel=signal.channel,
                author=signal.author,
                terms=frozenset(signal.terms),
                activation={str(k)[:8]: v for k, v in trace.familiarity_scores.items()},
                accepted_workers=frozenset(str(w)[:8] for w in trace.accepted_workers),
                metadata={"company": company, "kind": signal.metadata.get("kind", "")},
            )
            fp_store.record(fp)
            result.fingerprints.append(fp)

    result.workers_final = len(mesh.workers)
    result.gap_detections = len(mesh.workers) - num_workers  # workers spawned beyond initial

    # Log worker specializations
    logger.info(f"\n=== Mesh Experiment Results ===")
    logger.info(f"Signals processed: {result.signals_processed:,}")
    logger.info(f"Accepted: {result.signals_accepted:,}")
    logger.info(f"Rejected: {result.signals_rejected:,}")
    logger.info(f"Anomalies (max_score < {anomaly_threshold}): {len(result.anomalies):,}")
    logger.info(f"Workers: {num_workers} initial → {result.workers_final} final "
                f"({result.gap_detections} gap-spawned)")

    # Worker labels (what each worker learned to recognize)
    logger.info(f"\nWorker specializations:")
    for worker in sorted(mesh.workers, key=lambda w: -w.context.signal_count):
        ctx = worker.context
        # Show top sources and a sample of terms
        top_sources = sorted(ctx.source_counts.items(), key=lambda x: -x[1])[:3]
        src_str = ", ".join(f"{s}({c})" for s, c in top_sources)
        term_sample = sorted(list(ctx.terms))[:8]
        logger.info(
            f"  Worker {str(worker.id)[:8]}: "
            f"signals={ctx.signal_count}, "
            f"terms={len(ctx.terms)}, "
            f"sources=[{src_str}], "
            f"sample=[{', '.join(term_sample)}]"
        )

    # Coupling detection
    if len(result.fingerprints) >= 20:
        logger.info(f"\nCoupling detection ({len(result.fingerprints)} fingerprints):")
        couplings = fp_store.detect_couplings()
        if couplings:
            logger.info(f"  Found {len(couplings)} structural couplings:")
            for c in couplings[:10]:
                logger.info(f"    {c.channel_a} <-> {c.channel_b}: "
                           f"activation_sim={c.activation_similarity:.3f}, "
                           f"term_overlap={c.term_overlap:.3f}, "
                           f"hidden={c.is_hidden_coupling}")
        else:
            logger.info("  No couplings detected (may need more signals or workers)")

    return result
