#!/usr/bin/env python3
"""Simulation: feed bulk signals through the mesh and measure worker differentiation.

No polling, no adapters, no delays. Just raw signal routing through the mesh
to validate that competitive routing produces worker specialization.

Usage:
    python3 tests/simulate_differentiation.py
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from uuid import UUID

from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.mesh.mesh import Mesh
from stigmergy.mesh.topology import bloom_digest, position_distance
from stigmergy.mesh.worker import WorkerNode
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource
from stigmergy.tracing.trace import TraceLog

# ── Signal Generation ─────────────────────────────────────────

# Domain signal templates — each domain has distinct vocabulary
DOMAINS = {
    "pricing": {
        "source": SignalSource.GITHUB,
        "channel": "acme-org/pricing",
        "authors": ["bob.martinez", "eve.santos"],
        "templates": [
            "Pricing engine v3 dynamic model update with seasonal multipliers",
            "Fix nightly rate calculation for property {pid} off by {n}%",
            "Add configurable pricing floor per market segment",
            "Revenue optimization algorithm hitting diminishing returns at scale",
            "Pricing cache invalidation causing stale rates on OTA channels",
            "Implement base price adjustment for amenity-weighted scoring",
            "Pricing override audit trail for manual rate changes",
            "Competitive rate analysis integration with market data feed",
            "Pricing model training pipeline batch processing update",
            "Dynamic pricing confidence interval widening for new listings",
        ],
    },
    "availability": {
        "source": SignalSource.GITHUB,
        "channel": "acme-org/availability",
        "authors": ["alice.chen", "dave.kim"],
        "templates": [
            "Availability cache race condition causing stale data after booking",
            "Calendar sync conflict resolution for overlapping reservations",
            "Availability window calculation for minimum stay requirements",
            "Blocked dates not propagating to channel manager within SLA",
            "Inventory snapshot divergence between PMS and cache layer",
            "Availability buffer days configuration per property type",
            "Seasonal blackout date management for high-demand periods",
            "Real-time availability webhook latency exceeding threshold",
            "Availability rebuild job failing for properties with complex rules",
            "Gap night optimization for fragmented calendar availability",
        ],
    },
    "sync": {
        "source": SignalSource.GITHUB,
        "channel": "acme-org/sync-service",
        "authors": ["alice.chen", "dave.kim"],
        "templates": [
            "External PMS backpressure handling during high load ingestion",
            "PMS webhook sync failure affecting {n} properties",
            "Reservation push retry logic with exponential backoff",
            "Channel mapping inconsistency for multi-unit properties",
            "PMS sync queue depth exceeding configured threshold",
            "Bidirectional sync conflict detection for concurrent modifications",
            "Sync heartbeat monitoring for stale connection detection",
            "Booking state machine transition validation in sync pipeline",
            "Batch sync reconciliation job for overnight drift correction",
            "Event sourcing replay mechanism for sync recovery after outage",
        ],
    },
    "booking": {
        "source": SignalSource.LINEAR,
        "channel": "ENG-booking",
        "authors": ["bob.martinez", "frank.reyes"],
        "templates": [
            "Double booking prevention race condition in concurrent reservations",
            "Booking cancellation refund calculation with partial stay credits",
            "Guest communication template rendering for confirmation emails",
            "Booking modification workflow for date change requests",
            "Invoice generation failing for bookings with custom add-ons",
            "Check-in automation trigger not firing for same-day bookings",
            "Booking search performance degradation with large date ranges",
            "Group booking split payment distribution across multiple guests",
            "Booking status transition audit log for compliance tracking",
            "Reservation hold expiry timer not releasing blocked inventory",
        ],
    },
    "infrastructure": {
        "source": "deploy",
        "channel": "#deploys",
        "authors": ["grace.liu", "github-actions"],
        "templates": [
            "Deploy to production pricing service and sync service together",
            "Kubernetes pod scaling event triggered by CPU threshold breach",
            "Database migration rollback needed for schema compatibility issue",
            "Redis cluster failover completed successfully in {n} seconds",
            "CI pipeline timeout on integration test suite after dependency update",
            "Container image build optimization reducing deploy time by {n}%",
            "Infrastructure cost alert: compute spend exceeding monthly budget",
            "Load balancer health check configuration update for new endpoints",
            "Terraform state drift detected in production VPC configuration",
            "Monitoring alert threshold tuning for reduced false positive rate",
        ],
    },
    "organizational": {
        "source": SignalSource.SLACK,
        "channel": "#general",
        "authors": ["carol.park", "eve.santos", "frank.reyes"],
        "templates": [
            "Sprint planning review for Q1 roadmap prioritization",
            "Team standup notes: sync team blocked on external API changes",
            "Hiring pipeline update: 3 candidates in final round for backend role",
            "Onboarding checklist revision for new engineering hires",
            "Retrospective action items from last sprint ceremony",
            "Cross-team dependency coordination for platform migration",
            "Knowledge sharing session on Effect-TS patterns in new services",
            "Team capacity planning for upcoming quarter deliverables",
            "Process improvement proposal for code review turnaround time",
            "Meeting notes from architecture review board session",
        ],
    },
}


def generate_signals(count: int, seed: int = 42) -> list[Signal]:
    """Generate diverse signals across all domains."""
    rng = random.Random(seed)
    signals = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=count)

    domain_names = list(DOMAINS.keys())

    for i in range(count):
        domain = rng.choice(domain_names)
        cfg = DOMAINS[domain]
        template = rng.choice(cfg["templates"])
        # Add some variation
        content = template.replace("{pid}", f"P-{rng.randint(100, 999)}")
        content = content.replace("{n}", str(rng.randint(5, 95)))

        signals.append(Signal(
            content=content,
            source=cfg["source"],
            channel=cfg["channel"],
            author=rng.choice(cfg["authors"]),
            timestamp=base_time + timedelta(minutes=i * 3),
        ))

    return signals


# ── Mesh Setup ────────────────────────────────────────────────


def build_mesh(n_workers: int = 5) -> Mesh:
    """Build a fresh mesh with empty workers in a ring topology.

    No lifecycle — fork/merge/decay are disabled. Workers differentiate
    through competitive routing alone. Gap spawning handles uncovered
    signal territory.
    """
    agents = AgentRegistry()
    mesh = Mesh(
        agents,
        trace_log=TraceLog(),
        max_hops=10,
        max_workers=20,
        dedup_enabled=True,
        worker_capacity=200,
        base_threshold=0.15,
        max_threshold=0.8,
        threshold_curve=2.0,
        high_relevance_offset=0.2,
        gap_threshold=0.08,
        recenter_interval=15,
    )

    # Create workers in a ring
    workers = []
    for i in range(n_workers):
        w = mesh.spawn_worker()
        workers.append(w)

    # Ring topology: each worker connected to next
    for i in range(len(workers)):
        mesh.connect(workers[i].id, workers[(i + 1) % len(workers)])

    return mesh


# ── Metrics ───────────────────────────────────────────────────


def report_metrics(mesh: Mesh, phase: str = "") -> None:
    """Print worker differentiation metrics."""
    workers = mesh.workers
    active = [w for w in workers if w.context.signal_count > 0]

    print(f"\n{'=' * 72}")
    if phase:
        print(f"  {phase}")
    print(f"  Workers: {mesh.worker_count} total, {len(active)} active")
    print(f"  Signals ingested: {mesh.signals_ingested}")
    print(f"  Signals deduplicated: {mesh.signals_deduplicated}")
    print(f"{'=' * 72}")

    # Per-worker summary
    print(f"\n  {'Worker':<12} {'Signals':>8} {'Full%':>6} {'Thresh':>7} {'AvgFam':>7} {'Label'}")
    print(f"  {'-' * 70}")
    for w in sorted(workers, key=lambda x: x.context.signal_count, reverse=True):
        if w.context.signal_count == 0:
            continue
        label = w.label[:40]
        print(f"  {str(w.id)[:8]:<12} {w.context.signal_count:>8} {w.fullness * 100:>5.1f}% "
              f"{w.adaptive_threshold:>7.3f} {w.rolling_avg_familiarity:>7.3f} {label}")

    # Position distances between active workers
    if len(active) >= 2:
        print(f"\n  Position distances (0=identical, 1=orthogonal):")
        print(f"  {'':>12}", end="")
        for w in active[:8]:
            print(f" {str(w.id)[:6]:>8}", end="")
        print()
        for i, w1 in enumerate(active[:8]):
            print(f"  {str(w1.id)[:8]:<12}", end="")
            for j, w2 in enumerate(active[:8]):
                if j <= i:
                    print(f" {'':>8}", end="")
                else:
                    dist = position_distance(w1.position, w2.position)
                    print(f" {dist:>8.3f}", end="")
            print()

    # Source distribution per worker
    if active:
        print(f"\n  Source distribution per worker:")
        for w in sorted(active, key=lambda x: x.context.signal_count, reverse=True)[:8]:
            sources = w.context.source_counts
            total = sum(sources.values())
            if total > 0:
                parts = ", ".join(f"{s}:{c}" for s, c in sorted(sources.items(), key=lambda x: -x[1])[:3])
                print(f"  {str(w.id)[:8]}: {parts}")

    # Top terms per worker (for domain differentiation)
    if active:
        print(f"\n  Top terms per worker:")
        for w in sorted(active, key=lambda x: x.context.signal_count, reverse=True)[:8]:
            pos = w.position
            terms = pos.top_terms[:5]
            if terms:
                print(f"  {str(w.id)[:8]}: {', '.join(terms)}")

    # Topology
    topo = mesh.topology()
    active_topo = {k: v for k, v in topo.items() if any(
        mesh.get_worker(w.id) and w.context.signal_count > 0
        for w in workers if str(w.id)[:8] == k
    )}
    if active_topo:
        print(f"\n  Topology (active workers):")
        for wid, neighbors in active_topo.items():
            print(f"  {wid} -> [{', '.join(neighbors[:5])}{'...' if len(neighbors) > 5 else ''}]")

    print()


# ── Main ──────────────────────────────────────────────────────


async def run_simulation():
    print("Generating 200 signals across 6 domains...")
    signals = generate_signals(200)

    # Count domain distribution
    domain_counts = Counter()
    for s in signals:
        for domain, cfg in DOMAINS.items():
            if s.channel == cfg["channel"]:
                domain_counts[domain] += 1
                break
    print(f"  Domain distribution: {dict(domain_counts)}")

    print(f"\nBuilding mesh with 5 workers in ring topology...")
    mesh = build_mesh(n_workers=5)
    print(f"  {mesh.worker_count} workers, ring-connected")

    # Phase 1: First 50 signals (cold start / seeding)
    print(f"\n--- Phase 1: Cold start (signals 1-50) ---")
    for i, signal in enumerate(signals[:50]):
        trace = await mesh.ingest(signal)
        if i < 10:
            accepted = len(trace.accepted_workers)
            dup = trace.duplicate
            scores = [f"{s:.3f}" for s in trace.familiarity_scores.values()]
            print(f"  signal {i}: dup={dup}, accepted={accepted}, "
                  f"scores=[{', '.join(scores[:3])}], workers={mesh.worker_count}")
    report_metrics(mesh, "After 50 signals (cold start)")

    # Phase 2: Signals 51-100 (differentiation emerging)
    print(f"--- Phase 2: Differentiation (signals 51-100) ---")
    for signal in signals[50:100]:
        await mesh.ingest(signal)
    report_metrics(mesh, "After 100 signals (differentiation emerging)")

    # Phase 3: Signals 101-200 (settled state)
    print(f"--- Phase 3: Settled state (signals 101-200) ---")
    for signal in signals[100:200]:
        await mesh.ingest(signal)
    report_metrics(mesh, "After 200 signals (settled)")

    # Summary verdict
    active = [w for w in mesh.workers if w.context.signal_count > 0]
    if len(active) >= 2:
        distances = []
        for i, w1 in enumerate(active):
            for w2 in active[i + 1:]:
                distances.append(position_distance(w1.position, w2.position))
        avg_dist = sum(distances) / len(distances) if distances else 0
        max_dist = max(distances) if distances else 0
        min_dist = min(distances) if distances else 0

        print(f"{'=' * 72}")
        print(f"  DIFFERENTIATION SUMMARY")
        print(f"  Active workers: {len(active)}")
        print(f"  Avg position distance: {avg_dist:.3f}")
        print(f"  Min position distance: {min_dist:.3f}")
        print(f"  Max position distance: {max_dist:.3f}")

        labels = [w.label for w in active]
        unique_labels = len(set(labels))
        print(f"  Unique labels: {unique_labels}/{len(active)}")

        if avg_dist > 0.1 and unique_labels > 1:
            print(f"  VERDICT: Workers ARE differentiating")
        elif avg_dist > 0.05:
            print(f"  VERDICT: Partial differentiation (needs more signals)")
        else:
            print(f"  VERDICT: Workers NOT differentiating (convergence problem persists)")
        print(f"{'=' * 72}")


if __name__ == "__main__":
    asyncio.run(run_simulation())
