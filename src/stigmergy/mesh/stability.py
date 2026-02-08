"""Lyapunov stability instrumentation for the mesh.

Emits a time series to .stigmergy/stability.jsonl tracking whether
the mesh's total representation error V(x) decreases monotonically
as workers specialize. This validates or kills the Lyapunov stability
conjecture in Section 4.4 of the paper.

V(x) = sum of (1 - rolling_avg_familiarity) across all active workers.
A worker with perfect fit has familiarity ~1.0 → error ~0.0.
A worker still learning has lower familiarity → higher error.

If V(x) drops steeply then flattens, the conjecture holds.
If V(x) oscillates or trends up, cut the paragraph.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stigmergy.mesh.mesh import MeshTrace

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(".stigmergy") / "stability.jsonl"


@dataclass
class WorkerSnapshot:
    """Per-worker state at a single point in time."""

    worker_id: str
    signal_count: int
    fullness: float
    energy: float
    threshold: float
    rolling_avg_familiarity: float
    representation_error: float  # 1 - rolling_avg_familiarity
    term_count: int
    top_terms: list[str]  # top N terms for specialization comparison
    profile_delta: float  # how much the profile changed on last accept


@dataclass
class LifecycleEvent:
    """A single lifecycle event with its own timestamp."""

    timestamp: str
    event: str  # "spawn", "archive", "fork"
    worker_id: str  # primary worker affected
    detail: dict  # structured fields (not a string)


@dataclass
class StabilityEntry:
    """One row of the stability time series."""

    timestamp: str
    signal_index: int
    total_error: float  # V(x) — the Lyapunov candidate
    worker_count: int
    avg_error: float
    workers: list[WorkerSnapshot] = field(default_factory=list)
    lifecycle_events: list[LifecycleEvent] = field(default_factory=list)
    routing_target: str = ""  # worker_id that accepted
    routing_score: float = 0.0


class StabilityTracker:
    """Track mesh stability metrics for Lyapunov validation.

    Records one entry per processed signal (non-duplicate) to
    .stigmergy/stability.jsonl. Can be sampled (every Nth signal)
    for long runs.
    """

    def __init__(
        self,
        path: Path = DEFAULT_PATH,
        sample_interval: int = 1,
    ) -> None:
        self._path = path
        self._sample_interval = sample_interval
        self._signal_index = 0
        self._buffer: list[dict] = []
        self._lifecycle_events: list[LifecycleEvent] = []
        # Track routing stability: source_key -> last worker_id
        self._routing_history: dict[str, str] = {}
        self._routing_changes: int = 0

    def record_lifecycle(
        self,
        event: str,
        *,
        worker_id: str = "",
        detail: dict | None = None,
    ) -> None:
        """Record a lifecycle event (spawn, archive, fork, merge).

        Called from mesh lifecycle methods. Each event is timestamped
        immediately so topology evolution can be plotted.
        """
        self._lifecycle_events.append(LifecycleEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event=event,
            worker_id=worker_id,
            detail=detail or {},
        ))

    def record(self, mesh, trace: MeshTrace) -> None:
        """Record stability metrics after processing a signal.

        Args:
            mesh: The Mesh instance (for worker state access).
            trace: The MeshTrace from the just-processed signal.
        """
        if trace.duplicate:
            return

        self._signal_index += 1
        if self._signal_index % self._sample_interval != 0:
            return

        # Compute per-worker snapshots and V(x)
        workers: list[WorkerSnapshot] = []
        total_error = 0.0

        for worker in mesh.workers:
            avg_fam = worker.rolling_avg_familiarity
            error = 1.0 - avg_fam if worker.signals_accepted > 0 else 1.0
            total_error += error

            # Profile delta: approximate from familiarity EMA update
            # If the worker just accepted a signal, the delta is the
            # difference between the new score and the running average
            profile_delta = 0.0
            if worker.id in trace.accepted_workers:
                score = trace.familiarity_scores.get(worker.id, 0.0)
                profile_delta = abs(score - avg_fam) if worker.signals_accepted > 1 else 1.0

            # Capture top 20 terms sorted alphabetically for reproducible comparison
            all_terms = worker.context.terms
            top_terms = sorted(all_terms)[:20] if all_terms else []

            workers.append(WorkerSnapshot(
                worker_id=str(worker.id)[:8],
                signal_count=worker.context.signal_count,
                fullness=round(worker.fullness, 4),
                energy=round(worker.context.energy, 4),
                threshold=round(worker.adaptive_threshold, 4),
                rolling_avg_familiarity=round(avg_fam, 4),
                representation_error=round(error, 4),
                term_count=len(all_terms),
                top_terms=top_terms,
                profile_delta=round(profile_delta, 4),
            ))

        # Routing stability: did the same signal source go to the same worker?
        routing_target = ""
        routing_score = 0.0
        if trace.accepted_workers:
            accepted_id = str(next(iter(trace.accepted_workers)))[:8]
            routing_target = accepted_id
            routing_score = max(trace.familiarity_scores.values()) if trace.familiarity_scores else 0.0

            # Track routing consistency by source+channel
            if trace.signal_id:
                source_key = f"{trace.hops[0].worker_id}" if trace.hops else ""
                if source_key:
                    prev = self._routing_history.get(source_key)
                    if prev and prev != accepted_id:
                        self._routing_changes += 1
                    self._routing_history[source_key] = accepted_id

        # Collect ALL lifecycle events since last record
        lifecycle_events_data: list[dict] = []
        if self._lifecycle_events:
            for evt in self._lifecycle_events:
                lifecycle_events_data.append({
                    "timestamp": evt.timestamp,
                    "event": evt.event,
                    "worker_id": evt.worker_id,
                    "detail": evt.detail,
                })
            self._lifecycle_events.clear()

        worker_count = len(workers)
        avg_error = total_error / worker_count if worker_count > 0 else 0.0

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_index": self._signal_index,
            "total_error": round(total_error, 6),
            "worker_count": worker_count,
            "avg_error": round(avg_error, 6),
            "routing_target": routing_target,
            "routing_score": round(routing_score, 4),
            "routing_changes_total": self._routing_changes,
            "lifecycle_events": lifecycle_events_data,
            "workers": [
                {
                    "id": w.worker_id,
                    "signals": w.signal_count,
                    "fullness": w.fullness,
                    "energy": w.energy,
                    "threshold": w.threshold,
                    "avg_familiarity": w.rolling_avg_familiarity,
                    "error": w.representation_error,
                    "terms": w.term_count,
                    "top_terms": w.top_terms,
                    "delta": w.profile_delta,
                }
                for w in workers
            ],
        }
        self._buffer.append(entry)

    def flush(self) -> int:
        """Write buffered entries to disk. Returns count written."""
        if not self._buffer:
            return 0

        self._path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(self._path, "a") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry, default=str) + "\n")
                count += 1

        # Persist routing history alongside stability data
        if self._routing_history:
            routing_path = self._path.parent / "routing_history.json"
            with open(routing_path, "w") as f:
                json.dump({
                    "signal_index": self._signal_index,
                    "routing_changes_total": self._routing_changes,
                    "mappings": self._routing_history,
                }, f, indent=2)

        logger.info("Wrote %d stability entries to %s", count, self._path)
        self._buffer.clear()
        return count

    @property
    def signal_index(self) -> int:
        return self._signal_index

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)


class CadenceMonitor:
    """Detect unusual silence in the mesh and inject synthetic signals.

    Rather than bolting a separate monitoring system onto the mesh, the
    cadence monitor reads what the mesh already knows about itself:
    worker energy differentials, source distribution entropy, and
    acceptance rate history. The mesh's own spectral dynamics ARE the
    coherence signal.

    When a source goes quiet, the workers specialized in that source
    lose energy while others don't. That asymmetric energy compression
    is a louder signal than the silence itself — it's the eigenvalue
    collapse detector from the paper applied to the mesh's own input
    channel.

    Emits synthetic signals with source="mesh:cadence" that flow through
    normal mesh processing, letting the correlator reason about absence
    alongside presence.
    """

    import math as _math

    # Minimum signals a worker must have processed before we consider
    # it established enough to detect silence (avoids false positives
    # on workers that just spawned).
    MIN_BASELINE_SIGNALS = 20

    # If actual_rate / expected_rate falls below this, flag a cadence break.
    SILENCE_THRESHOLD = 0.3

    def __init__(
        self,
        mesh,
        cooldown_signals: int = 100,
    ) -> None:
        self._mesh = mesh
        self._cooldown_signals = cooldown_signals
        # worker_id (str[:8]) -> signal_index when last silence signal was emitted
        self._cooldown_tracker: dict[str, int] = {}
        # worker_id (str[:8]) -> list of snapshot tuples.
        # Format: (signal_index, signal_count, source_counts, timestamp)
        # The 4th element (timestamp) is optional for backward compatibility
        # with manually constructed snapshots in tests.
        self._snapshots: dict[str, list[tuple]] = {}

    _DOW_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

    @staticmethod
    def _period_bucket(ts: datetime | None) -> str | None:
        """Classify a timestamp into a day-of-week bucket.

        Seven bins — one per day. Monday's baseline comes from previous
        Mondays, Saturday from previous Saturdays. This is the discretized
        first harmonic of the organizational rhythm: the DFT of 7 sample
        points gives the fundamental weekly cycle plus two overtones.

        Callers use hierarchical fallback when a specific day lacks data:
        day-of-week → weekday/weekend → all-time.
        """
        if ts is None:
            return None
        return CadenceMonitor._DOW_NAMES[ts.weekday()]

    @staticmethod
    def _is_weekend_bucket(bucket: str | None) -> bool | None:
        """Classify a day-of-week bucket as weekend or weekday."""
        if bucket is None:
            return None
        return bucket in ("sat", "sun")

    def _record_snapshot(self, signal_index: int, timestamp: datetime | None = None) -> None:
        """Record current worker signal counts and source distribution."""
        ts = timestamp or datetime.now(timezone.utc)
        for worker in self._mesh.workers:
            wid = str(worker.id)[:8]
            count = worker.context.signal_count
            src_counts = dict(worker.context.source_counts)
            if wid not in self._snapshots:
                self._snapshots[wid] = []
            self._snapshots[wid].append((signal_index, count, src_counts, ts))
            # Keep bounded — last 20 snapshots is enough for rate computation
            if len(self._snapshots[wid]) > 20:
                self._snapshots[wid] = self._snapshots[wid][-20:]

    def check(self, signal_index: int, timestamp: datetime | None = None) -> list:
        """Periodic check for cadence breaks. Returns synthetic Signal list.

        Called at checkpoint intervals (e.g. every 50 signals). Records a
        snapshot and checks each worker for silence.
        """
        from stigmergy.primitives.signal import Signal, SignalSource

        self._record_snapshot(signal_index, timestamp=timestamp)

        results: list[Signal] = []
        for worker in self._mesh.workers:
            wid = str(worker.id)[:8]

            # Need enough history for a meaningful baseline
            if worker.context.signal_count < self.MIN_BASELINE_SIGNALS:
                continue

            baseline = self._compute_baseline(wid)
            if baseline <= 0:
                continue

            if not self._detect_silence(wid, baseline):
                continue

            # Cooldown check — don't re-emit for same worker too soon
            last_emitted = self._cooldown_tracker.get(wid, -self._cooldown_signals - 1)
            if signal_index - last_emitted < self._cooldown_signals:
                continue

            coherence, coherence_score = self._check_source_coherence(wid)
            actual = self._compute_actual_rate(wid)
            sig = self._build_silence_signal(worker, baseline, actual, coherence, coherence_score)
            results.append(sig)
            self._cooldown_tracker[wid] = signal_index
            logger.info(
                "Cadence break: worker %s (baseline=%.1f sig/interval, "
                "actual=%.1f, coherence=%s, score=%.3f)",
                wid, baseline, actual, coherence, coherence_score,
            )

        return results

    def _compute_baseline(self, worker_id: str) -> float:
        """Compute expected signals per check interval from snapshot history.

        Period-aware with hierarchical fallback: tries day-of-week match
        first, then weekday/weekend, then all-time. Saturday's expected
        rate comes from historical Saturdays when possible, not from an
        all-time average diluted by weekday activity.
        """
        snapshots = self._snapshots.get(worker_id, [])
        if len(snapshots) < 3:
            return 0.0

        # Determine current period from most recent snapshot
        current_ts = snapshots[-1][3] if len(snapshots[-1]) > 3 else None
        current_bucket = self._period_bucket(current_ts)
        current_is_weekend = self._is_weekend_bucket(current_bucket)

        # Compute rate for each consecutive snapshot pair, tagged with bucket
        all_rates: list[float] = []
        day_rates: list[float] = []
        half_rates: list[float] = []
        for i in range(1, len(snapshots)):
            idx_delta = snapshots[i][0] - snapshots[i - 1][0]
            count_delta = snapshots[i][1] - snapshots[i - 1][1]
            if idx_delta > 0:
                rate = count_delta / idx_delta
                all_rates.append(rate)
                ts = snapshots[i][3] if len(snapshots[i]) > 3 else None
                bucket = self._period_bucket(ts)
                if current_bucket and bucket == current_bucket:
                    day_rates.append(rate)
                if current_is_weekend is not None and self._is_weekend_bucket(bucket) == current_is_weekend:
                    half_rates.append(rate)

        # Hierarchical: day-of-week → weekday/weekend → all-time
        if len(day_rates) >= 2:
            rates = day_rates
        elif len(half_rates) >= 2:
            rates = half_rates
        else:
            rates = all_rates

        if not rates:
            return 0.0
        return sum(rates) / len(rates)

    def _compute_actual_rate(self, worker_id: str) -> float:
        """Compute current rate from the most recent snapshot pair."""
        snapshots = self._snapshots.get(worker_id, [])
        if len(snapshots) < 2:
            return 0.0
        last = snapshots[-1]
        prev = snapshots[-2]
        idx_delta = last[0] - prev[0]
        if idx_delta <= 0:
            return 0.0
        count_delta = last[1] - prev[1]
        return count_delta / idx_delta

    def _detect_silence(self, worker_id: str, baseline: float) -> bool:
        """Check if current rate indicates silence relative to baseline."""
        actual = self._compute_actual_rate(worker_id)
        if baseline <= 0:
            return False
        return actual / baseline < self.SILENCE_THRESHOLD

    @staticmethod
    def _source_entropy(source_counts: dict[str, int]) -> float:
        """Shannon entropy of a source distribution.

        Measures input diversity. High entropy = signals from many sources
        (healthy heterogeneous flow). Low entropy = dominated by one source
        or only one source active. Zero = single source or no signals.
        """
        import math
        total = sum(source_counts.values())
        if total <= 0:
            return 0.0
        h = 0.0
        for cnt in source_counts.values():
            if cnt > 0:
                p = cnt / total
                h -= p * math.log2(p)
        return h

    @staticmethod
    def _asymmetry_score(entropy_ratio: float, volume_ratio: float) -> float:
        """Continuous asymmetric compression score (0.0–1.0).

        Replaces hard thresholds with a smooth gradient. Uses sqrt to
        open up the space near zero — subtle divergences between entropy
        and volume compression become detectable where linear comparison
        would miss them.

        The product formulation (sqrt of gap * sqrt of volume presence)
        requires BOTH conditions: entropy must have compressed more than
        volume AND volume must still be present. Symmetric collapse
        (both near zero) naturally scores near zero because the volume
        presence term vanishes.

        This is the "exponentiating or square-rooting makes subtle things
        less subtle" principle: the sqrt amplifies small asymmetries in
        the compression ratio, making the interesting quiet louder than
        the background noise floor.
        """
        import math

        # How much each compressed (0 = no change, 1 = fully compressed)
        entropy_compression = max(0.0, 1.0 - entropy_ratio)
        volume_compression = max(0.0, 1.0 - volume_ratio)

        # The asymmetry gap: entropy compressed more than volume
        gap = entropy_compression - volume_compression
        if gap <= 0:
            return 0.0

        # Volume presence: how much signal flow remains.
        # Symmetric compression → volume_ratio near 0 → low presence.
        # This is why sqrt works: sqrt(0.5)*sqrt(0.5)=0.5 (clear signal)
        # but sqrt(0.05)*sqrt(0.05)=0.05 (noise floor).
        presence = min(volume_ratio, 1.0)

        # sqrt amplification on both terms — opens up the discrimination
        # space near the origin where linear would compress everything
        # to indistinguishable near-zero values.
        score = math.sqrt(gap) * math.sqrt(presence)

        return min(score, 1.0)

    # Score above which asymmetric compression is classified as incoherent.
    # Not a cliff — the score itself is a gradient. This just maps the
    # continuous score to the discrete classification the API returns.
    _INCOHERENCE_THRESHOLD = 0.35

    def _check_source_coherence(self, worker_id: str) -> tuple[str, float]:
        """Determine if silence is coherent or incoherent.

        Returns (classification, score) where score is a continuous 0.0–1.0
        asymmetry measure. The classification is derived from the score but
        the score itself is the primary signal — it's what goes into the
        synthetic signal metadata for the correlator to reason about.

        Period-aware with hierarchical fallback:
        day-of-week → weekday/weekend → all-time.

        This prevents false positives on learned rhythms. If weekends
        normally have asymmetric source activity (one engineer working
        GitHub while Slack and Linear are quiet), the detector compares
        against previous weekends, not against Tuesday's baseline.
        The anomaly is an arrhythmia, not a slow pulse.
        """
        import math

        snapshots = self._snapshots.get(worker_id, [])
        if len(snapshots) < 4:
            return "unknown", 0.0

        # Compute per-window metrics: (entropy, volume, day_bucket, is_weekend)
        windows: list[tuple[float, float, str | None, bool | None]] = []
        for i in range(1, len(snapshots)):
            prev_src = snapshots[i - 1][2]
            curr_src = snapshots[i][2]
            idx_delta = snapshots[i][0] - snapshots[i - 1][0]
            ts = snapshots[i][3] if len(snapshots[i]) > 3 else None
            bucket = self._period_bucket(ts)
            is_wknd = self._is_weekend_bucket(bucket)

            if idx_delta <= 0:
                continue

            all_sources = set(prev_src) | set(curr_src)
            if not all_sources:
                windows.append((0.0, 0.0, bucket, is_wknd))
                continue

            deltas = {}
            for src in all_sources:
                deltas[src] = curr_src.get(src, 0) - prev_src.get(src, 0)

            active_deltas = {s: d for s, d in deltas.items() if d > 0}
            total_delta = sum(active_deltas.values())
            volume = total_delta / idx_delta

            if total_delta <= 0 or len(active_deltas) < 1:
                windows.append((0.0, volume, bucket, is_wknd))
                continue

            h = 0.0
            for d in active_deltas.values():
                p = d / total_delta
                if p > 0:
                    h -= p * math.log2(p)
            windows.append((h, volume, bucket, is_wknd))

        if len(windows) < 3:
            return "unknown", 0.0

        # Recent windows (last 2) and historical (rest)
        recent = windows[-2:]
        historical = windows[:-2]

        # Hierarchical period matching: day → weekday/weekend → all-time
        recent_buckets = [w[2] for w in recent if w[2] is not None]
        recent_weekend = [w[3] for w in recent if w[3] is not None]

        baseline_windows = historical  # default: all-time
        if recent_buckets:
            target_day = recent_buckets[-1]
            day_matched = [w for w in historical if w[2] == target_day]
            if len(day_matched) >= 2:
                baseline_windows = day_matched
            elif recent_weekend:
                target_wknd = recent_weekend[-1]
                half_matched = [w for w in historical if w[3] == target_wknd]
                if len(half_matched) >= 2:
                    baseline_windows = half_matched

        if not baseline_windows:
            return "unknown", 0.0

        avg_baseline_h = sum(w[0] for w in baseline_windows) / len(baseline_windows)
        avg_recent_h = sum(w[0] for w in recent) / len(recent)
        avg_baseline_v = sum(w[1] for w in baseline_windows) / len(baseline_windows)
        avg_recent_v = sum(w[1] for w in recent) / len(recent)

        if avg_baseline_h <= 0:
            # Baseline period has zero entropy (learned single-source pattern).
            # If recent also has near-zero entropy, the rhythm matches.
            if avg_recent_h <= avg_baseline_h + 0.1:
                return "healthy", 0.0
            return "unknown", 0.0

        entropy_ratio = avg_recent_h / avg_baseline_h
        volume_ratio = avg_recent_v / avg_baseline_v if avg_baseline_v > 0 else 0

        score = self._asymmetry_score(entropy_ratio, volume_ratio)

        if score > self._INCOHERENCE_THRESHOLD:
            return "incoherent", score
        return "healthy", score

    def _build_silence_signal(
        self,
        worker,
        baseline: float,
        actual: float,
        coherence: str,
        coherence_score: float = 0.0,
    ):
        """Build a synthetic signal describing the detected silence."""
        from stigmergy.primitives.signal import Signal, SignalSource

        wid = str(worker.id)[:8]
        label = worker.label if hasattr(worker, "label") else wid
        source_dist = ", ".join(
            f"{src}={cnt}" for src, cnt in sorted(
                worker.context.source_counts.items(),
                key=lambda x: -x[1],
            )[:5]
        ) if worker.context.source_counts else "no sources"

        # Include energy for the correlator to reason about decay asymmetry
        energy = worker.context.energy if hasattr(worker.context, "energy") else 0.0

        coherence_desc = {
            "incoherent": "asymmetric compression — some sources active, others muted",
            "healthy": "symmetric compression — all sources quiet together",
            "unknown": "insufficient history to determine compression character",
        }.get(coherence, coherence)

        content = (
            f"Cadence break detected: worker {wid} ({label}) has gone quiet. "
            f"Expected ~{baseline:.2f} signals/interval but actual is {actual:.2f} "
            f"(ratio {actual/baseline:.2f} < {self.SILENCE_THRESHOLD}). "
            f"Coherence: {coherence_desc} (score={coherence_score:.3f}). "
            f"Worker energy: {energy:.4f}. "
            f"Source distribution: {source_dist}. "
            f"Total signals processed: {worker.context.signal_count}."
        )

        return Signal(
            content=content,
            source=SignalSource.CADENCE,
            channel=f"worker:{wid}",
            author="cadence-monitor",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "event_type": "cadence_break",
                "worker_id": wid,
                "worker_label": label,
                "baseline_rate": round(baseline, 4),
                "actual_rate": round(actual, 4),
                "silence_ratio": round(actual / baseline if baseline > 0 else 0, 4),
                "coherence": coherence,
                "coherence_score": round(coherence_score, 4),
                "worker_energy": round(energy, 4),
                "source_counts": dict(worker.context.source_counts),
            },
        )
