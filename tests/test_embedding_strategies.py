"""Tests for concrete embedding strategy implementations."""

import math

import numpy as np
import pytest

from stigmergy.services.embedding_strategies import (
    EMBEDDING_DIM,
    SemanticStrategy,
    SocialStrategy,
    TechnicalStrategy,
    l2_normalize,
    signed_hash,
    zero_vector,
)


# ── Shared Utilities ─────────────────────────────────────────────────


class TestZeroVector:
    def test_shape(self):
        v = zero_vector()
        assert len(v) == EMBEDDING_DIM

    def test_all_zeros(self):
        v = zero_vector()
        assert all(x == 0.0 for x in v)


class TestL2Normalize:
    def test_unit_vector(self):
        v = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        v[0] = 3.0
        v[1] = 4.0
        result = l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_zero_vector_unchanged(self):
        v = zero_vector()
        result = l2_normalize(v)
        assert np.allclose(result, zero_vector())


class TestSignedHash:
    def test_deterministic(self):
        b1, s1 = signed_hash("hello", 64)
        b2, s2 = signed_hash("hello", 64)
        assert b1 == b2
        assert s1 == s2

    def test_bucket_in_range(self):
        for token in ["a", "bb", "ccc", "dddd"]:
            bucket, _ = signed_hash(token, 32)
            assert 0 <= bucket < 32

    def test_sign_is_plus_or_minus_one(self):
        for token in ["x", "y", "z"]:
            _, sign = signed_hash(token, 10)
            assert sign in (-1, 1)


# ── SemanticStrategy ─────────────────────────────────────────────────


class TestSemanticStrategy:
    async def test_name(self):
        s = SemanticStrategy()
        assert s.name == "semantic"

    async def test_output_dimension(self):
        s = SemanticStrategy()
        result = await s.embed("Hello world, this is a test.")
        assert len(result) == EMBEDDING_DIM

    async def test_empty_content_zero_vector(self):
        s = SemanticStrategy()
        result = await s.embed("")
        assert all(v == 0.0 for v in result)

    async def test_normalized_or_zero(self):
        s = SemanticStrategy()
        result = await s.embed("some meaningful content here")
        norm = math.sqrt(sum(v * v for v in result))
        assert norm < 1e-6 or abs(norm - 1.0) < 1e-6

    async def test_no_nan_or_inf(self):
        s = SemanticStrategy()
        result = await s.embed("test content")
        assert all(math.isfinite(v) for v in result)

    async def test_deterministic_fallback(self):
        s = SemanticStrategy()
        r1 = await s.embed("deterministic test")
        r2 = await s.embed("deterministic test")
        assert r1 == r2

    async def test_different_content_different_vectors(self):
        s = SemanticStrategy()
        r1 = await s.embed("python machine learning")
        r2 = await s.embed("javascript web development")
        assert r1 != r2


# ── TechnicalStrategy ────────────────────────────────────────────────


class TestTechnicalStrategy:
    async def test_name(self):
        s = TechnicalStrategy()
        assert s.name == "technical"

    async def test_output_dimension(self):
        s = TechnicalStrategy()
        result = await s.embed("def hello_world():\n    pass")
        assert len(result) == EMBEDDING_DIM

    async def test_empty_content_zero_vector(self):
        s = TechnicalStrategy()
        result = await s.embed("")
        assert all(v == 0.0 for v in result)

    async def test_normalized_or_zero(self):
        s = TechnicalStrategy()
        result = await s.embed("import os\nclass Foo:\n    def bar(self):\n        pass")
        norm = math.sqrt(sum(v * v for v in result))
        assert norm < 1e-6 or abs(norm - 1.0) < 1e-6

    async def test_deterministic(self):
        s = TechnicalStrategy()
        r1 = await s.embed("def foo(): return bar.baz()")
        r2 = await s.embed("def foo(): return bar.baz()")
        assert r1 == r2

    async def test_code_aware_tokenization(self):
        tokens = TechnicalStrategy._tokenize_code_aware("getUserName")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    async def test_snake_case_tokenization(self):
        tokens = TechnicalStrategy._tokenize_code_aware("get_user_name")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    async def test_detect_patterns_function(self):
        patterns = TechnicalStrategy._detect_patterns("def my_func():\n    pass")
        assert "function_definition" in patterns

    async def test_detect_patterns_import(self):
        patterns = TechnicalStrategy._detect_patterns("import os.path")
        assert "import_statement" in patterns

    async def test_detect_patterns_class(self):
        patterns = TechnicalStrategy._detect_patterns("class MyClass:")
        assert "class_definition" in patterns


# ── SocialStrategy ───────────────────────────────────────────────────


class TestSocialStrategy:
    async def test_name(self):
        s = SocialStrategy()
        assert s.name == "social"

    async def test_output_dimension(self):
        s = SocialStrategy()
        result = await s.embed("", metadata={"author": "alice"})
        assert len(result) == EMBEDDING_DIM

    async def test_no_metadata_zero_vector(self):
        s = SocialStrategy()
        result = await s.embed("")
        assert all(v == 0.0 for v in result)

    async def test_empty_metadata_zero_vector(self):
        s = SocialStrategy()
        result = await s.embed("", metadata={})
        assert all(v == 0.0 for v in result)

    async def test_author_produces_nonzero(self):
        s = SocialStrategy()
        result = await s.embed("", metadata={"author": "bob"})
        assert any(v != 0.0 for v in result)

    async def test_different_authors_different_vectors(self):
        s = SocialStrategy()
        r1 = await s.embed("", metadata={"author": "alice"})
        r2 = await s.embed("", metadata={"author": "bob"})
        assert r1 != r2

    async def test_mentions_contribute(self):
        s = SocialStrategy()
        r1 = await s.embed("", metadata={"author": "alice", "mentions": []})
        r2 = await s.embed("", metadata={"author": "alice", "mentions": ["bob"]})
        assert r1 != r2

    async def test_channel_contributes(self):
        s = SocialStrategy()
        r1 = await s.embed("", metadata={"channel": "general"})
        r2 = await s.embed("", metadata={"channel": "engineering"})
        assert r1 != r2

    async def test_deterministic(self):
        s = SocialStrategy()
        meta = {"author": "alice", "mentions": ["bob"], "channel": "dev"}
        r1 = await s.embed("", metadata=meta)
        r2 = await s.embed("", metadata=meta)
        assert r1 == r2

    async def test_normalized_or_zero(self):
        s = SocialStrategy()
        result = await s.embed("", metadata={"author": "test", "mentions": ["a", "b"]})
        norm = math.sqrt(sum(v * v for v in result))
        assert norm < 1e-6 or abs(norm - 1.0) < 1e-6
