"""Attention model — P(knows) computation and per-person state.

The core equation:

    P(knows) = 1 - prod(1 - p_i)   for each relevant exposure event i

Each p_i = base_weight * temporal_decay * channel_size_discount
         * volume_discount * topic_overlap

This is the noisy-OR model: each exposure independently contributes
to the probability of knowing. Multiple exposures compound.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from stigmergy.attention.exposure import ExposureEvent

logger = logging.getLogger(__name__)

# Base weights by exposure type — how deeply engaged was the person?
BASE_WEIGHTS: dict[str, float] = {
    "authored": 0.95,
    "assigned": 0.90,
    "reviewed": 0.85,
    "commented": 0.75,
    "mentioned": 0.50,
    "reacted": 0.35,
    "channel_present": 0.20,
}

# Maximum exposures to retain per person (bounded memory)
MAX_EXPOSURES = 500


@dataclass
class AttentionParameters:
    """Tunable parameters for the attention model."""

    decay_half_life_days: float = 14.0
    channel_size_reference: int = 5
    volume_discount_half: int = 50
    topic_overlap_threshold: float = 0.1
    importance_weight: float = 1.0
    annoyance_weight: float = 0.3
    # Asymmetric learning rates — the cost of under-informing is invisible;
    # the cost of over-informing is visible and self-correcting.
    calibration_lr_up: float = 0.05    # "already_knew" — small upward nudge
    calibration_lr_down: float = 0.20  # "had_no_idea" — aggressive downward correction


@dataclass
class PersonAttentionState:
    """Per-person attention tracking state."""

    person_id: str
    exposures: list[ExposureEvent] = field(default_factory=list)
    channels_seen: dict[str, int] = field(default_factory=dict)  # channel -> unique author count
    signal_volume_7d: int = 0  # rolling 7-day signal count
    attention_base_rate: float = 0.5  # calibration parameter
    feedback_knew: int = 0
    feedback_novel: int = 0

    def add_exposure(self, event: ExposureEvent) -> None:
        """Add an exposure event, maintaining bounded size."""
        self.exposures.append(event)
        if len(self.exposures) > MAX_EXPOSURES:
            self.exposures = self.exposures[-MAX_EXPOSURES:]

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "exposures": [
                {
                    "person_id": e.person_id,
                    "source": e.source,
                    "channel": e.channel,
                    "exposure_type": e.exposure_type,
                    "timestamp": e.timestamp.isoformat(),
                    "topic_terms": sorted(e.topic_terms),
                    "signal_id": e.signal_id,
                }
                for e in self.exposures
            ],
            "channels_seen": self.channels_seen,
            "signal_volume_7d": self.signal_volume_7d,
            "attention_base_rate": self.attention_base_rate,
            "feedback_knew": self.feedback_knew,
            "feedback_novel": self.feedback_novel,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PersonAttentionState:
        exposures = []
        for e in data.get("exposures", []):
            exposures.append(ExposureEvent(
                person_id=e["person_id"],
                source=e["source"],
                channel=e["channel"],
                exposure_type=e["exposure_type"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                topic_terms=frozenset(e.get("topic_terms", [])),
                signal_id=e["signal_id"],
            ))
        return cls(
            person_id=data["person_id"],
            exposures=exposures,
            channels_seen=data.get("channels_seen", {}),
            signal_volume_7d=data.get("signal_volume_7d", 0),
            attention_base_rate=data.get("attention_base_rate", 0.5),
            feedback_knew=data.get("feedback_knew", 0),
            feedback_novel=data.get("feedback_novel", 0),
        )


class AttentionModel:
    """Core attention model: accumulates exposures, computes P(knows)."""

    def __init__(
        self,
        resolver=None,
        params: AttentionParameters | None = None,
    ) -> None:
        self._resolver = resolver
        self._params = params or AttentionParameters()
        self._persons: dict[str, PersonAttentionState] = {}

    @property
    def params(self) -> AttentionParameters:
        return self._params

    def _get_or_create(self, person_id: str) -> PersonAttentionState:
        if person_id not in self._persons:
            self._persons[person_id] = PersonAttentionState(person_id=person_id)
        return self._persons[person_id]

    def record_exposure(self, event: ExposureEvent) -> None:
        """Record a single exposure event."""
        state = self._get_or_create(event.person_id)
        state.add_exposure(event)

        # Track channel author counts for channel size discount
        channel_key = f"{event.source}:{event.channel}"
        if channel_key not in state.channels_seen:
            state.channels_seen[channel_key] = 0
        # Use a set-like counter: increment only represents unique authors
        # seen. The real unique count is maintained by the mesh; here we
        # just track a running count per channel for the person.
        state.channels_seen[channel_key] = max(
            state.channels_seen[channel_key],
            state.channels_seen.get(channel_key, 0),
        )

    def record_exposures(self, events: list[ExposureEvent]) -> None:
        """Record multiple exposure events (batch)."""
        for event in events:
            self.record_exposure(event)

    def update_channel_authors(self, channel_key: str, unique_authors: int) -> None:
        """Update the unique author count for a channel across all persons."""
        for state in self._persons.values():
            if channel_key in state.channels_seen:
                state.channels_seen[channel_key] = unique_authors

    def update_signal_volume(self, person_id: str, volume_7d: int) -> None:
        """Update the 7-day signal volume for a person."""
        state = self._get_or_create(person_id)
        state.signal_volume_7d = volume_7d

    def p_knows(
        self,
        person_id: str,
        finding_terms: frozenset[str],
        now: datetime | None = None,
    ) -> float:
        """Compute P(already_knows) for a person regarding a finding.

        Uses noisy-OR over relevant exposure events:
            P(knows) = 1 - prod(1 - p_i)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        state = self._persons.get(person_id)
        if state is None:
            return 0.0

        params = self._params
        survival_product = 1.0  # prod(1 - p_i)

        for exposure in state.exposures:
            p_i = self._single_exposure_probability(
                exposure, finding_terms, now, state, params,
            )
            if p_i <= 0.0:
                continue
            survival_product *= (1.0 - p_i)

        return 1.0 - survival_product

    def _single_exposure_probability(
        self,
        exposure: ExposureEvent,
        finding_terms: frozenset[str],
        now: datetime,
        state: PersonAttentionState,
        params: AttentionParameters,
    ) -> float:
        """Compute p_i for a single exposure event."""
        # Base weight by exposure type
        base = BASE_WEIGHTS.get(exposure.exposure_type, 0.10)

        # Temporal decay: exp(-ln(2) * age_days / half_life)
        age = now - exposure.timestamp
        age_days = max(age.total_seconds() / 86400.0, 0.0)
        decay = math.exp(-math.log(2) * age_days / params.decay_half_life_days)

        # Channel size discount: 0.8^(log2(authors / reference))
        channel_key = f"{exposure.source}:{exposure.channel}"
        channel_authors = state.channels_seen.get(channel_key, params.channel_size_reference)
        if channel_authors <= 0:
            channel_authors = params.channel_size_reference
        ratio = channel_authors / params.channel_size_reference
        if ratio > 1.0:
            channel_discount = 0.8 ** math.log2(ratio)
        else:
            channel_discount = 1.0

        # Volume discount: half_volume / (half_volume + signals_per_day)
        signals_per_day = state.signal_volume_7d / 7.0 if state.signal_volume_7d > 0 else 0.0
        volume_discount = params.volume_discount_half / (params.volume_discount_half + signals_per_day)

        # Topic overlap: Jaccard similarity, thresholded
        if finding_terms and exposure.topic_terms:
            intersection = len(finding_terms & exposure.topic_terms)
            union = len(finding_terms | exposure.topic_terms)
            overlap = intersection / union if union > 0 else 0.0
        elif not finding_terms and not exposure.topic_terms:
            overlap = 1.0  # both empty = full overlap
        else:
            overlap = 0.0

        if overlap < params.topic_overlap_threshold:
            return 0.0  # Below threshold — exposure not relevant

        p_i = base * decay * channel_discount * volume_discount * overlap

        # Apply per-person calibration as a multiplier
        # attention_base_rate shifts the overall sensitivity
        calibration = state.attention_base_rate / 0.5  # normalize around default
        p_i = min(1.0, max(0.0, p_i * calibration))

        return p_i

    def record_feedback(
        self,
        person_id: str,
        finding_hash: str,
        response: str,
    ) -> None:
        """Record calibration feedback from a user.

        response: "already_knew" or "had_no_idea"
        Update rule: rate <- rate + lr * (target - rate)

        Asymmetric learning rates: lr_up=0.05 for "already_knew" (small
        upward nudge) vs lr_down=0.20 for "had_no_idea" (aggressive
        downward correction). The cost of under-informing is invisible;
        the cost of over-informing is visible and self-correcting.
        """
        state = self._get_or_create(person_id)

        if response == "already_knew":
            lr = self._params.calibration_lr_up
            target = 0.95
            state.feedback_knew += 1
        elif response == "had_no_idea":
            lr = self._params.calibration_lr_down
            target = 0.05
            state.feedback_novel += 1
        else:
            logger.warning("Unknown feedback response: %s", response)
            return

        state.attention_base_rate += lr * (target - state.attention_base_rate)
        # Bounded [0.05, 0.95]
        state.attention_base_rate = max(0.05, min(0.95, state.attention_base_rate))

    def effective_annoyance_weight(self, person_id: str) -> float:
        """C_redundant adapted to the person's recent false-positive rate.

        Instead of a static 0.3, scales annoyance by how often this person
        reports "already_knew". Someone who consistently already knows what
        we surface should have a higher bar (more annoyance cost).

        C_redundant = base_annoyance * (0.5 + fp_rate)

        fp_rate=0.0 (everything novel) → C = base * 0.5 (low bar, surface more)
        fp_rate=0.5 (balanced)         → C = base * 1.0 (default behavior)
        fp_rate=1.0 (everything known) → C = base * 1.5 (high bar, surface less)

        With no feedback yet, fp_rate defaults to 0.5 → no adaptation.
        """
        state = self._persons.get(person_id)
        if state is None:
            return self._params.annoyance_weight

        total_feedback = state.feedback_knew + state.feedback_novel
        if total_feedback == 0:
            fp_rate = 0.5  # prior: balanced
        else:
            fp_rate = state.feedback_knew / total_feedback

        return self._params.annoyance_weight * (0.5 + fp_rate)

    def get_state(self, person_id: str) -> PersonAttentionState | None:
        return self._persons.get(person_id)

    def all_person_ids(self) -> list[str]:
        return list(self._persons.keys())

    def save(self, path: Path) -> None:
        """Persist attention state to JSON."""
        data = {
            "params": {
                "decay_half_life_days": self._params.decay_half_life_days,
                "channel_size_reference": self._params.channel_size_reference,
                "volume_discount_half": self._params.volume_discount_half,
                "topic_overlap_threshold": self._params.topic_overlap_threshold,
                "importance_weight": self._params.importance_weight,
                "annoyance_weight": self._params.annoyance_weight,
                "calibration_lr_up": self._params.calibration_lr_up,
                "calibration_lr_down": self._params.calibration_lr_down,
            },
            "persons": {
                pid: state.to_dict()
                for pid, state in self._persons.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load attention state from JSON."""
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load attention state: %s", e)
            return

        # Restore params (learning rates, etc. may have been tuned)
        params_data = data.get("params", {})
        if params_data:
            for key in (
                "decay_half_life_days", "channel_size_reference",
                "volume_discount_half", "topic_overlap_threshold",
                "importance_weight", "annoyance_weight",
                "calibration_lr_up", "calibration_lr_down",
            ):
                if key in params_data:
                    setattr(self._params, key, params_data[key])
            # Backward compat: old symmetric "calibration_lr" → apply to both
            if "calibration_lr" in params_data and "calibration_lr_up" not in params_data:
                old_lr = params_data["calibration_lr"]
                self._params.calibration_lr_up = old_lr
                self._params.calibration_lr_down = old_lr

        # Restore per-person state
        for pid, pdata in data.get("persons", {}).items():
            self._persons[pid] = PersonAttentionState.from_dict(pdata)
