"""Shared fixtures for tests."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from stigmergy.primitives.agent import Agent, CompetencyModel, RuleSet
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource
from stigmergy.services.embedding import StubEmbeddingService
from stigmergy.services.llm import StubLLMService


@pytest.fixture
def embedding_service():
    return StubEmbeddingService()


@pytest.fixture
def llm_service():
    return StubLLMService()


@pytest.fixture
def make_signal():
    """Factory for creating test signals."""
    def _make(
        content: str = "test signal",
        source: SignalSource = SignalSource.SLACK,
        channel: str = "#test",
        author: str = "test.user",
        embeddings: dict | None = None,
        **kwargs,
    ) -> Signal:
        return Signal(
            content=content,
            source=source,
            channel=channel,
            author=author,
            timestamp=datetime.now(timezone.utc),
            embeddings=embeddings or {},
            **kwargs,
        )
    return _make


@pytest.fixture
def make_context():
    """Factory for creating test contexts."""
    def _make(**kwargs) -> Context:
        return Context(**kwargs)
    return _make


@pytest.fixture
def make_agent():
    """Factory for creating test agents."""
    def _make(
        contexts: dict | None = None,
        confidence: float = 0.8,
        weights: dict | None = None,
        **kwargs,
    ) -> Agent:
        return Agent(
            contexts=contexts or {},
            confidence=confidence,
            weights=weights or {"technical": 0.8, "business": 0.3},
            **kwargs,
        )
    return _make
