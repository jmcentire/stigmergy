"""Meta-modeler: LLM agent that observes the mesh itself.

Not the signals the mesh processes, but HOW it processes them.
Receives health pulses, produces assessments, deposits discoveries.

Advisory only — observations go into the DiscoveryStore where other
agents sense them as context. No hard enforcement gates.

Budget: ~800 input + ~400 output tokens per fire. At 50-signal
intervals processing 700 signals = ~14 calls = ~$0.014/run with Haiku.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stigmergy.mesh.discovery_store import Discovery, DiscoveryStore
from stigmergy.mesh.health_pulse import MeshHealthPulse

if TYPE_CHECKING:
    from stigmergy.services.llm import LLMService

logger = logging.getLogger(__name__)


class MetaModeler:
    """LLM-powered mesh self-assessment agent.

    Receives health pulses at checkpoint intervals, calls the LLM to
    assess mesh state, deposits observations to the discovery store.
    """

    def __init__(
        self,
        llm: LLMService,
        discovery_store: DiscoveryStore,
        *,
        fire_interval: int = 1,
        max_output_tokens: int = 400,
        perturbation_enabled: bool = False,
    ) -> None:
        self._llm = llm
        self._discovery_store = discovery_store
        self._fire_interval = max(1, fire_interval)
        self._max_output_tokens = max_output_tokens
        self._perturbation_enabled = perturbation_enabled
        self._pulse_count = 0
        self._history: list[dict[str, Any]] = []  # last 20 assessments
        self._max_history = 20
        self._field_state: Any | None = None  # Injected by FieldEngine when available

    def set_field_state(self, field_state: Any) -> None:
        """Inject latest CERTX field state for prompt enrichment."""
        self._field_state = field_state

    async def evaluate(self, pulse: MeshHealthPulse) -> Any | None:
        """Evaluate a health pulse. Returns MetaModelAssessment or None.

        Fires every Nth pulse (fire_interval). Returns None on off-cycles
        or if the LLM call fails.
        """
        self._pulse_count += 1
        if self._pulse_count % self._fire_interval != 0:
            return None

        try:
            from stigmergy.primitives.schemas import MetaModelAssessment
            from stigmergy.services.agent_prompts import META_MODELER_SYSTEM

            prompt = self._build_prompt(pulse)
            assessment = await self._llm.extract(
                MetaModelAssessment,
                prompt,
                META_MODELER_SYSTEM,
                max_tokens=self._max_output_tokens,
            )

            # Deposit observations to discovery store
            self._deposit_observations(assessment, pulse)

            # Keep history for trend
            self._history.append({
                "signal_index": pulse.signal_index,
                "mesh_state": assessment.mesh_state,
                "health_score": assessment.health_score,
                "diagnosis": assessment.diagnosis[:200],
            })
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            logger.info(
                "Meta-modeler: state=%s score=%.2f obs=%d recs=%d",
                assessment.mesh_state, assessment.health_score,
                len(assessment.observations), len(assessment.recommendations),
            )

            return assessment

        except Exception as exc:
            logger.warning("Meta-modeler evaluation failed: %s", exc)
            return None

    def _build_prompt(self, pulse: MeshHealthPulse) -> str:
        """Format pulse data + previous assessments + happy-place targets."""
        parts = ["Assess the current mesh health from this pulse:\n"]

        # Current pulse data
        pd = pulse.to_dict()
        parts.append(f"Signal index: {pd['signal_index']}")
        parts.append(f"Workers: {pd['worker_count']}")
        parts.append(f"Total error V(x): {pd['total_error']}")
        parts.append(f"Avg familiarity: {pd['avg_familiarity']}")
        parts.append(f"Routing changes: {pd['routing_changes']}")
        parts.append(f"Spectral trend: {pd['spectral_trend']}")
        parts.append(f"Spectral high-freq: {pd['spectral_high_freq']}")
        parts.append(f"Effective dimensionality: {pd['effective_dimensionality']}")

        parts.append("\nChange rates (EMA-smoothed):")
        parts.append(f"  d_error/dt: {pd['d_error_dt']}")
        parts.append(f"  d_familiarity/dt: {pd['d_familiarity_dt']}")
        parts.append(f"  d_worker_count/dt: {pd['d_worker_count_dt']}")
        parts.append(f"  d_routing_changes/dt: {pd['d_routing_changes_dt']}")
        parts.append(f"  d_confidence/dt: {pd['d_confidence_dt']}")
        parts.append(f"  d_spectral/dt: {pd['d_spectral_dt']}")

        parts.append(f"\nFindings: active={pd['active_findings']}, "
                      f"deferred={pd['deferred_findings']}, "
                      f"normalized={pd['normalized_findings']}")

        parts.append(f"\nQuorum: {pd['last_quorum_met']} met / {pd['last_quorum_total']} total")

        # Worker summaries (compact)
        if pd["workers"]:
            parts.append("\nWorker summaries:")
            for w in pd["workers"]:
                parts.append(
                    f"  {w['id']} ({w['label']}): fam={w['familiarity']}, "
                    f"energy={w['energy']}, thresh={w['threshold']}, "
                    f"signals={w['signal_count']}"
                )

        # Agent summaries (compact)
        if pd["agents"]:
            parts.append("\nAgent summaries:")
            for a in pd["agents"]:
                parts.append(
                    f"  {a['id']}: conf={a['confidence']}, "
                    f"llm={a['llm_calls']}, mech={a['mechanical_calls']}, "
                    f"top={a['top_competency']}"
                )

        # Previous assessment trend
        if self._history:
            parts.append("\nRecent assessment history:")
            for h in self._history[-3:]:
                parts.append(
                    f"  [sig {h['signal_index']}] state={h['mesh_state']}, "
                    f"score={h['health_score']:.2f}"
                )

        # Happy-place targets from prior sessions
        happy = self._discovery_store.happy_places(limit=2)
        if happy:
            parts.append("\nHappy-place targets (from prior sessions):")
            for hp in happy:
                parts.append(f"  (strength={hp.get('decay_weight', 0):.2f}): {hp.get('summary', '')[:200]}")

        # Unity CERTX state (zero extra LLM cost — mechanical measurement)
        if self._field_state is not None and hasattr(self._field_state, "certx_summary"):
            parts.append(f"\n{self._field_state.certx_summary()}")

        return "\n".join(parts)

    def _deposit_observations(self, assessment: Any, pulse: MeshHealthPulse) -> None:
        """Write assessment results to discovery store."""
        # Main health assessment
        self._discovery_store.deposit(Discovery(
            observation_type="mesh_health",
            summary=assessment.diagnosis[:300],
            structured={
                "mesh_state": assessment.mesh_state,
                "health_score": assessment.health_score,
                "worker_count": pulse.worker_count,
                "total_error": pulse.total_error,
            },
            confidence=assessment.health_score,
            source_agent="meta_modeler",
            signal_index=pulse.signal_index,
        ))

        # Individual observations
        for obs_text in assessment.observations:
            self._discovery_store.deposit(Discovery(
                observation_type="effective_pattern" if assessment.health_score > 0.7 else "anti_pattern",
                summary=obs_text[:300],
                confidence=assessment.health_score,
                source_agent="meta_modeler",
                signal_index=pulse.signal_index,
            ))

        # Happy-place snapshot (only when healthy)
        if assessment.happy_place and assessment.mesh_state == "healthy":
            self._discovery_store.deposit(Discovery(
                observation_type="happy_place",
                summary=assessment.happy_place[:300],
                structured={
                    "health_score": assessment.health_score,
                    "worker_count": pulse.worker_count,
                    "avg_familiarity": pulse.avg_familiarity,
                    "total_error": pulse.total_error,
                },
                confidence=assessment.health_score,
                source_agent="meta_modeler",
                signal_index=pulse.signal_index,
            ))

        # Recommendations as competency insights
        for rec in assessment.recommendations:
            if rec.action in ("boost_competency", "dampen_competency"):
                self._discovery_store.deposit(Discovery(
                    observation_type="competency_insight",
                    summary=f"{rec.action}: {rec.target} — {rec.detail}"[:300],
                    structured={
                        "action": rec.action,
                        "target": rec.target,
                        "intensity": rec.intensity,
                    },
                    confidence=assessment.health_score * rec.intensity,
                    source_agent="meta_modeler",
                    signal_index=pulse.signal_index,
                ))

    def apply_recommendations(
        self,
        assessment: Any,
        agents: list | None = None,
    ) -> list[str]:
        """Apply mechanical recommendations (competency boosts/dampens).

        Returns list of actions taken. Only acts on boost_competency and
        dampen_competency when perturbation is enabled.
        """
        actions_taken: list[str] = []
        if not self._perturbation_enabled or agents is None:
            return actions_taken

        for rec in assessment.recommendations:
            if rec.action == "boost_competency":
                delta = 0.02 * rec.intensity
                for agent in agents:
                    agent.competencies.reinforce(rec.target, delta)
                actions_taken.append(f"boost {rec.target} by {delta:.4f}")
            elif rec.action == "dampen_competency":
                delta = -0.02 * rec.intensity
                for agent in agents:
                    agent.competencies.reinforce(rec.target, delta)
                actions_taken.append(f"dampen {rec.target} by {delta:.4f}")

        if actions_taken:
            logger.info("Meta-modeler applied: %s", ", ".join(actions_taken))

        return actions_taken

    @property
    def assessment_count(self) -> int:
        return len(self._history)

    @property
    def latest_state(self) -> str:
        """Most recent mesh state assessment, or 'unknown'."""
        if self._history:
            return self._history[-1].get("mesh_state", "unknown")
        return "unknown"
