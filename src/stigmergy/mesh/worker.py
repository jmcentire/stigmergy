"""WorkerNode: a Context operating as a node in the worker mesh.

Theoretical foundation: Adaptive Resonance Theory (ART).

Each WorkerNode is an ART category. The adaptive threshold is the
vigilance criterion ρ — the minimum match quality for resonance.
The three learning modes map to ART resonance states:

  - rag_indexed     → full resonance:  |I ∩ w_J|/|I| ≥ ρ_high
                      Input matches category strongly. Full weight
                      update (RAG indexing). The ART stability proof
                      guarantees convergence: match-based learning
                      (update only on accept) prevents catastrophic
                      forgetting on non-stationary streams.

  - context_summarized → partial resonance:  ρ ≤ |I ∩ w_J|/|I| < ρ_high
                      Input accepted but below high threshold.
                      Lightweight update (terms/counts, no vector store).
                      Category drifts slowly toward this input.

  - weight_shifted  → mismatch/reset:  |I ∩ w_J|/|I| < ρ
                      Input rejected — does not resonate. Only the
                      cheapest learning occurs (source affinity nudge).
                      Signal propagates to the next category (BFS hop).

The fullness-based threshold creates complement coding: as a worker
fills, its vigilance rises, preventing category collapse (a single
worker absorbing everything). This is the ART analog of orienting
subsystem response — full workers become more discriminating.

Wraps an existing Context with:
- Vigilance criterion (adaptive threshold): rises with fullness
- receive(): local resonance check — accept or propagate
- Two learning tiers:
    Agent tier (budget permitting): LLM extracts structured facts via schema
    Mechanical tier (fallback): term extraction, bloom filters, counts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from stigmergy.core.familiarity import FamiliarityWeights, extract_step_sequence, familiarity
from stigmergy.mesh.topology import WorkerPosition, bloom_digest
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal
from stigmergy.structures.bloom import CountingBloomFilter

if TYPE_CHECKING:
    from stigmergy.services.llm import LLMService

logger = logging.getLogger(__name__)


@dataclass
class ReceiveResult:
    """Outcome of a worker's local resonance check on a signal.

    Maps to ART resonance states:

    learning_mode is a self-describing string — a fresh pair of eyes should
    understand what happened from the mode name alone. Well-known modes:
    - "rag_indexed": full resonance — signal indexed into vector store.
      ART: |I ∩ w_J|/|I| ≥ ρ_high. Full weight update.
    - "context_summarized": partial resonance — terms/counts updated,
      no vector storage. ART: ρ ≤ match < ρ_high. Slow category drift.
    - "weight_shifted": mismatch/reset — source affinity nudged only.
      ART: match < ρ. Signal propagates to next category via BFS.
    Workers may develop new modes as the system evolves.
    """

    accepted: bool
    worker_id: UUID
    score: float
    learning_mode: str
    learning_detail: str = ""  # human-readable explanation of what was learned


class WorkerNode:
    """A Context operating as an ART category node in the worker mesh.

    The vigilance criterion (adaptive_threshold) creates the "fullness as
    natural throttle" behavior: empty workers accept almost anything, full
    workers only accept highly relevant signals. This is the ART complement
    coding analog — rising vigilance prevents category collapse.

    Match-based learning provides the formal stability guarantee: weights
    are only updated when a signal resonates (is accepted). This prevents
    catastrophic forgetting on infinite non-stationary input streams —
    the core ART theorem (Carpenter & Grossberg, 1987).
    """

    def __init__(
        self,
        context: Context,
        *,
        base_threshold: float = 0.15,
        max_threshold: float = 0.8,
        threshold_curve: float = 2.0,
        high_relevance_offset: float = 0.2,
        source_name: str | None = None,
        llm: LLMService | None = None,
    ) -> None:
        self.context = context
        self.base_threshold = base_threshold
        self.max_threshold = max_threshold
        self.threshold_curve = threshold_curve
        self.high_relevance_offset = high_relevance_offset
        self.source_name = source_name  # if this is a source entry worker
        self._llm = llm  # agent intelligence — None means mechanical fallback

        # Mesh routing stats
        self.signals_received: int = 0
        self.signals_accepted: int = 0
        self.signals_forwarded: int = 0

        # Position tracking — cached, invalidated on accept
        self._position_dirty: bool = True
        self._cached_position: WorkerPosition | None = None

        # Rolling average familiarity of accepted signals (EMA)
        self.rolling_avg_familiarity: float = 0.0
        self._familiarity_ema_alpha: float = 0.1  # smoothing factor

        # Recentering counter — counts accepts since last recenter
        self.recenter_counter: int = 0

        # Fork cooldown — prevents re-triggering fork immediately after forking.
        # Decrements once per lifecycle check; worker cannot fork while > 0.
        self._fork_cooldown: int = 0

    @property
    def id(self) -> UUID:
        return self.context.id

    @property
    def fullness(self) -> float:
        return self.context.fullness

    @property
    def is_full(self) -> bool:
        return self.fullness >= 1.0

    @property
    def position(self) -> WorkerPosition:
        """Worker's position in signal space, derived from its bloom filter.

        Cached and invalidated when the worker accepts a signal.
        """
        if self._position_dirty or self._cached_position is None:
            ctx = self.context
            # Top terms by bloom count (proxy for frequency)
            if ctx.terms:
                scored = sorted(
                    ((t, ctx.term_bloom._counters[ctx.term_bloom._indices(t)[0]])
                     for t in ctx.terms if len(t) > 3),
                    key=lambda x: x[1],
                    reverse=True,
                )
                top_terms = tuple(t for t, _ in scored[:10])
            else:
                top_terms = ()

            self._cached_position = WorkerPosition(
                worker_id=self.id,
                bloom_digest=bloom_digest(ctx.term_bloom),
                top_terms=top_terms,
                signal_count=ctx.signal_count,
                fullness=self.fullness,
            )
            self._position_dirty = False
        return self._cached_position

    def _update_familiarity_ema(self, score: float) -> None:
        """Update rolling average familiarity using exponential moving average."""
        alpha = self._familiarity_ema_alpha
        if self.signals_accepted <= 1:
            self.rolling_avg_familiarity = score
        else:
            self.rolling_avg_familiarity = alpha * score + (1 - alpha) * self.rolling_avg_familiarity

    # Workers need at least this many signals before applying full vigilance.
    # Below this, vigilance is reduced to encourage learning during cold start.
    # ART analog: initial learning rate — new categories are permissive.
    _COLD_START_MIN_SIGNALS = 20

    @property
    def adaptive_threshold(self) -> float:
        """Vigilance criterion ρ — rises as worker fills.

        ART correspondence: this IS the vigilance parameter ρ from
        Adaptive Resonance Theory. The match function |I ∩ w_J|/|I|
        (implemented in familiarity.py) must exceed ρ for resonance.

        At 0% full: base_threshold (low vigilance — accepts most things)
        At 100% full: max_threshold (high vigilance — only highly relevant)
        The curve exponent controls how aggressively vigilance rises.

        Cold start: workers with < 20 signals use reduced vigilance
        (down to 25% of base) to learn more aggressively before becoming
        selective. This is the ART learning rate analog.
        """
        fill = self.fullness
        base = self.base_threshold + (self.max_threshold - self.base_threshold) * (fill ** self.threshold_curve)

        # Cold start discount: gradually ramp up from 25% to 100% of threshold
        sig_count = self.context.signal_count
        if sig_count < self._COLD_START_MIN_SIGNALS:
            cold_factor = 0.25 + 0.75 * (sig_count / self._COLD_START_MIN_SIGNALS)
            return base * cold_factor

        return base

    @property
    def vigilance(self) -> float:
        """Alias for adaptive_threshold — the ART vigilance criterion ρ."""
        return self.adaptive_threshold

    @property
    def high_relevance_threshold(self) -> float:
        """ρ_high — above this score, signals achieve full resonance (RAG indexed)."""
        return min(1.0, self.adaptive_threshold + self.high_relevance_offset)

    @property
    def label(self) -> str:
        """Human-readable label based on top terms."""
        if not self.context.terms:
            return f"worker-{str(self.id)[:8]}"
        # Filter out short/common words
        meaningful = sorted(
            (t for t in self.context.terms if len(t) > 3),
            key=lambda t: self.context.term_bloom.query(t) if hasattr(self.context.term_bloom, 'query') else 0,
            reverse=True,
        )
        top = meaningful[:3] if meaningful else sorted(self.context.terms)[:3]
        return "/".join(top)

    async def receive(self, signal: Signal, score: float) -> ReceiveResult:
        """ART resonance check: does this signal match this category?

        Three resonance states based on familiarity score vs vigilance ρ:
        - score ≥ ρ_high → full resonance (rag_indexed): full weight update
        - ρ ≤ score < ρ_high → partial resonance (context_summarized): slow drift
        - score < ρ → mismatch (weight_shifted): signal propagates to next category
        """
        self.signals_received += 1

        if score < self.adaptive_threshold:
            # Below threshold: weight-shifted only (cheapest learning)
            self._weight_shift(signal)
            self.signals_forwarded += 1
            return ReceiveResult(
                accepted=False,
                worker_id=self.id,
                score=score,
                learning_mode="weight_shifted",
                learning_detail=f"score {score:.3f} below threshold {self.adaptive_threshold:.3f}, nudged source affinity for {signal.source}",
            )

        self.signals_accepted += 1
        self._position_dirty = True
        self._update_familiarity_ema(score)
        self.recenter_counter += 1

        if score >= self.high_relevance_threshold:
            # High relevance: agent-extracted facts when LLM available,
            # mechanical fallback when budget exhausted or no LLM configured.
            terms, learning_mode = await self._extract_facts(signal)
            await self.context.ingest_signal(
                signal_id=str(signal.id),
                embeddings=dict(signal.embeddings),
                terms=terms,
                source=signal.source,
                author=signal.author,
                score=score,
                metadata=dict(signal.metadata),
                signal_timestamp=signal.timestamp,
            )
            # Update sequence representation for story signals
            steps = extract_step_sequence(signal)
            if steps is not None:
                self.context.update_sequences(steps)
            return ReceiveResult(
                accepted=True,
                worker_id=self.id,
                score=score,
                learning_mode=learning_mode,
                learning_detail=f"score {score:.3f} above high threshold {self.high_relevance_threshold:.3f}",
            )

        # Medium relevance: context-summarized (update state, skip vector store)
        self._context_summarize(signal, score)
        return ReceiveResult(
            accepted=True,
            worker_id=self.id,
            score=score,
            learning_mode="context_summarized",
            learning_detail=f"score {score:.3f} in mid-range, updated terms and counts without vector storage",
        )

    def _context_summarize(self, signal: Signal, score: float) -> None:
        """Medium learning: update terms, counts, energy. No vector storage."""
        ctx = self.context
        ctx.terms.update(signal.terms)
        ctx.term_bloom.add_many(signal.terms)
        ctx.signal_history.append(str(signal.id))
        ctx.source_counts[signal.source] = ctx.source_counts.get(signal.source, 0) + 1
        ctx.author_counts[signal.author] = ctx.author_counts.get(signal.author, 0) + 1
        ctx.signal_count += 1
        # Use the later of wall-clock and signal timestamp for last_signal.
        # During live processing, these are nearly identical. During backfill,
        # signal timestamps reflect the actual signal-time gaps (days between
        # signals), which energy decay in _check_decay uses to properly decay
        # workers that haven't received signals in a while — even when
        # wall-clock time is compressed to minutes.
        ctx.last_signal = max(datetime.now(timezone.utc), signal.timestamp)
        ctx.energy += score * 0.5

    def _weight_shift(self, signal: Signal) -> None:
        """Cheapest learning: just nudge source/author affinity slightly.

        Important: does NOT increment source_counts or author_counts.
        Weight-shift fires on mismatch (ART reset state) — the signal was
        rejected. Counting rejected signals as accepted would bias source
        affinity toward sources that produce many low-relevance signals.
        """
        # No-op: weight_shift is the minimum ART response to mismatch.
        # The signal is rejected; we don't update any counts.
        pass

    async def _extract_facts(self, signal: Signal) -> tuple[set[str], str]:
        """Extract domain concepts from a signal.

        Agent tier: LLM reads the signal and extracts structured facts
        via schema-constrained tool use. The schema shapes the output.
        Mechanical tier: regex + heuristic fallback when no LLM or budget exhausted.

        Returns (terms, learning_mode).
        """
        if self._llm is not None:
            try:
                from stigmergy.primitives.schemas import SignalFacts
                from stigmergy.services.agent_prompts import (
                    INDEXER_SYSTEM,
                    indexer_prompt,
                )

                facts = await self._llm.extract(
                    SignalFacts,
                    indexer_prompt(signal.content, signal.source, signal.channel, signal.author),
                    INDEXER_SYSTEM,
                    max_tokens=300,
                )
                # Agent-extracted concepts become the terms
                terms = set()
                for concept in facts.concepts:
                    # Keep multi-word concepts as-is (they're more meaningful)
                    terms.add(concept.lower())
                return terms, "rag_indexed"
            except Exception:
                logger.debug("Agent extraction failed, falling back to mechanical", exc_info=True)

        # Mechanical fallback
        return signal.terms, "rag_indexed"

    async def agent_label(self) -> str:
        """Ask the agent to describe this worker's specialization.

        Falls back to mechanical label if no LLM available.
        """
        if self._llm is None or not self.context.terms:
            return self.label

        try:
            from stigmergy.primitives.schemas import WorkerProfile
            from stigmergy.services.agent_prompts import (
                PROFILER_SYSTEM,
                profiler_prompt,
            )

            # Get top concepts by frequency
            ctx = self.context
            scored = sorted(
                ((t, ctx.term_bloom._counters[ctx.term_bloom._indices(t)[0]])
                 for t in ctx.terms if len(t) > 2),
                key=lambda x: x[1],
                reverse=True,
            )
            top_concepts = [t for t, _ in scored[:20]]

            profile = await self._llm.extract(
                WorkerProfile,
                profiler_prompt(top_concepts, ctx.signal_count),
                PROFILER_SYSTEM,
                max_tokens=200,
            )
            return profile.specialization
        except Exception:
            logger.debug("Agent label failed, falling back to mechanical", exc_info=True)
            return self.label

    def score_signal(
        self,
        signal: Signal,
        weights: FamiliarityWeights,
        centroid: list[float] | None = None,
        signal_bloom: CountingBloomFilter | None = None,
    ) -> float:
        """Score a signal against this worker using familiarity()."""
        return familiarity(
            signal,
            self.context,
            weights=weights,
            centroid=centroid,
            signal_bloom=signal_bloom,
        )
