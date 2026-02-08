"""Tests for meta-modeler agent with mock LLM."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from stigmergy.mesh.discovery_store import DiscoveryStore
from stigmergy.mesh.health_pulse import MeshHealthPulse, WorkerSummary, AgentSummary
from stigmergy.mesh.meta_modeler import MetaModeler
from stigmergy.primitives.schemas import MetaModelAssessment, MetaModelRecommendation


# ── Helpers ──────────────────────────────────────────────────


def _make_pulse(
    signal_index: int = 100,
    worker_count: int = 5,
    total_error: float = 1.5,
    avg_familiarity: float = 0.7,
) -> MeshHealthPulse:
    return MeshHealthPulse(
        signal_index=signal_index,
        worker_count=worker_count,
        total_error=total_error,
        avg_familiarity=avg_familiarity,
        spectral_trend="stable",
        workers=[
            WorkerSummary(
                id="w1", label="PMS sync", familiarity=0.8,
                energy=0.5, threshold=0.3, signal_count=42,
            ),
        ],
        agents=[
            AgentSummary(
                id="a1", confidence=0.85, llm_calls=5,
                mechanical_calls=0, top_competency="temporal_analysis",
            ),
        ],
    )


def _make_assessment(
    mesh_state: str = "healthy",
    health_score: float = 0.85,
    happy_place: str = "Mesh well-differentiated with 5 workers",
) -> MetaModelAssessment:
    return MetaModelAssessment(
        mesh_state=mesh_state,
        diagnosis="V(x) is decreasing steadily. Workers are well-differentiated.",
        observations=[
            "V(x) decreased 12% over last 3 pulses.",
            "Worker familiarity averaging 0.7 across all workers.",
        ],
        recommendations=[],
        happy_place=happy_place,
        health_score=health_score,
    )


def _make_mock_llm(assessment: MetaModelAssessment | None = None) -> MagicMock:
    """Create a mock LLM service that returns a MetaModelAssessment."""
    mock = MagicMock()
    mock.extract = AsyncMock(return_value=assessment or _make_assessment())
    return mock


# ── Tests ────────────────────────────────────────────────────


class TestMetaModeler:
    @pytest.fixture
    def store(self, tmp_path):
        return DiscoveryStore(path=tmp_path / "discoveries.json")

    @pytest.mark.asyncio
    async def test_evaluate_deposits_to_store(self, store):
        llm = _make_mock_llm()
        modeler = MetaModeler(llm, store)
        pulse = _make_pulse()

        result = await modeler.evaluate(pulse)

        assert result is not None
        assert result.mesh_state == "healthy"
        # Should have deposited: 1 health + 2 observations + 1 happy_place
        assert store.count >= 3

    @pytest.mark.asyncio
    async def test_fire_interval_skips_off_cycles(self, store):
        llm = _make_mock_llm()
        modeler = MetaModeler(llm, store, fire_interval=3)

        pulse = _make_pulse()
        r1 = await modeler.evaluate(pulse)
        r2 = await modeler.evaluate(pulse)
        r3 = await modeler.evaluate(pulse)

        # Should fire on pulses 1 (skip), 2 (skip), 3 (fire)
        assert r1 is None
        assert r2 is None
        assert r3 is not None

    @pytest.mark.asyncio
    async def test_assessment_history_tracked(self, store):
        llm = _make_mock_llm()
        modeler = MetaModeler(llm, store)

        for i in range(5):
            pulse = _make_pulse(signal_index=i * 50)
            await modeler.evaluate(pulse)

        assert modeler.assessment_count == 5
        assert modeler.latest_state == "healthy"

    @pytest.mark.asyncio
    async def test_happy_place_deposited_only_when_healthy(self, store):
        # Thrashing state — no happy place
        assessment = _make_assessment(mesh_state="thrashing", health_score=0.3, happy_place="")
        llm = _make_mock_llm(assessment)
        modeler = MetaModeler(llm, store)

        await modeler.evaluate(_make_pulse())

        happy = store.happy_places()
        assert len(happy) == 0

    @pytest.mark.asyncio
    async def test_happy_place_deposited_when_healthy(self, store):
        assessment = _make_assessment(mesh_state="healthy", health_score=0.9,
                                       happy_place="5 workers well-differentiated")
        llm = _make_mock_llm(assessment)
        modeler = MetaModeler(llm, store)

        await modeler.evaluate(_make_pulse())

        happy = store.happy_places()
        assert len(happy) == 1
        assert "5 workers" in happy[0]["summary"]

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self, store):
        llm = MagicMock()
        llm.extract = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        modeler = MetaModeler(llm, store)

        result = await modeler.evaluate(_make_pulse())

        assert result is None
        assert store.count == 0

    @pytest.mark.asyncio
    async def test_competency_recommendations_deposited(self, store):
        assessment = MetaModelAssessment(
            mesh_state="stuck",
            diagnosis="Mesh is not learning.",
            observations=["No improvement in 5 pulses."],
            recommendations=[
                MetaModelRecommendation(
                    action="boost_competency",
                    target="temporal_analysis",
                    detail="Temporal patterns underweighted",
                    intensity=0.7,
                ),
            ],
            happy_place="",
            health_score=0.3,
        )
        llm = _make_mock_llm(assessment)
        modeler = MetaModeler(llm, store)

        await modeler.evaluate(_make_pulse())

        # Should deposit competency insight
        insights = store.sense(observation_type="competency_insight")
        assert len(insights) >= 1
        assert "boost_competency" in insights[0]["summary"]

    def test_apply_recommendations_perturbation_disabled(self, store):
        assessment = _make_assessment()
        assessment.recommendations = [
            MetaModelRecommendation(
                action="boost_competency",
                target="temporal_analysis",
                detail="Boost",
                intensity=0.5,
            ),
        ]
        modeler = MetaModeler(_make_mock_llm(), store, perturbation_enabled=False)

        actions = modeler.apply_recommendations(assessment, agents=[])
        assert actions == []  # perturbation disabled

    def test_apply_recommendations_perturbation_enabled(self, store):
        from stigmergy.primitives.agent import Agent, CompetencyModel

        assessment = _make_assessment()
        assessment.recommendations = [
            MetaModelRecommendation(
                action="boost_competency",
                target="temporal_analysis",
                detail="Boost",
                intensity=1.0,
            ),
        ]
        agent = Agent(competencies=CompetencyModel({"temporal_analysis": 0.5}))
        modeler = MetaModeler(_make_mock_llm(), store, perturbation_enabled=True)

        actions = modeler.apply_recommendations(assessment, agents=[agent])

        assert len(actions) == 1
        assert agent.competencies.strength("temporal_analysis") > 0.5

    def test_build_prompt_includes_pulse_data(self, store):
        modeler = MetaModeler(_make_mock_llm(), store)
        pulse = _make_pulse(signal_index=200, worker_count=7)

        prompt = modeler._build_prompt(pulse)

        assert "200" in prompt
        assert "Workers: 7" in prompt
        assert "d_error/dt" in prompt

    def test_build_prompt_includes_happy_place_targets(self, store):
        from stigmergy.mesh.discovery_store import Discovery
        store.deposit(Discovery(
            observation_type="happy_place",
            summary="5 workers, V(x) stable at 0.3",
            confidence=0.9,
            decay_weight=0.95,
        ))

        modeler = MetaModeler(_make_mock_llm(), store)
        prompt = modeler._build_prompt(_make_pulse())

        assert "Happy-place targets" in prompt
        assert "V(x) stable" in prompt


class TestMetaModelSchemas:
    def test_assessment_schema_valid(self):
        a = MetaModelAssessment(
            mesh_state="healthy",
            diagnosis="All good",
            observations=["V(x) stable"],
            recommendations=[],
            happy_place="Well differentiated",
            health_score=0.9,
        )
        assert a.mesh_state == "healthy"
        assert a.health_score == 0.9

    def test_recommendation_schema_valid(self):
        r = MetaModelRecommendation(
            action="boost_competency",
            target="temporal_analysis",
            detail="Underweighted",
            intensity=0.7,
        )
        assert r.action == "boost_competency"
        assert r.intensity == 0.7

    def test_health_score_clamped(self):
        with pytest.raises(Exception):
            MetaModelAssessment(
                mesh_state="healthy",
                diagnosis="",
                health_score=1.5,  # > 1.0
            )
