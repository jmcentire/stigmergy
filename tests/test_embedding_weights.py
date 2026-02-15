"""Tests for per-context embedding dimension weights."""

import math

import pytest

from stigmergy.core.embedding_weights import (
    EmbeddingDimensionWeights,
    compute_weighted_embedding_similarity,
    deserialize_dimension_weights,
    ensure_strategies_present,
    get_normalized_weights,
    initialize_dimension_weights,
    record_acceptance,
    serialize_dimension_weights,
)


# ── EmbeddingDimensionWeights ────────────────────────────────────────


class TestEmbeddingDimensionWeights:
    def test_defaults(self):
        w = EmbeddingDimensionWeights()
        assert w.logits == {}
        assert w.learning_rate == 0.05
        assert w.update_count == 0

    def test_non_finite_logit_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            EmbeddingDimensionWeights(logits={"x": float("nan")})

    def test_inf_logit_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            EmbeddingDimensionWeights(logits={"x": float("inf")})


# ── get_normalized_weights ───────────────────────────────────────────


class TestGetNormalizedWeights:
    def test_equal_logits_equal_weights(self):
        w = EmbeddingDimensionWeights(
            logits={"a": 0.0, "b": 0.0, "c": 0.0}
        )
        nw = get_normalized_weights(w)
        for val in nw.values():
            assert abs(val - 1 / 3) < 1e-9

    def test_sum_to_one(self):
        w = EmbeddingDimensionWeights(
            logits={"a": 1.0, "b": 2.0, "c": 0.5}
        )
        nw = get_normalized_weights(w)
        assert abs(sum(nw.values()) - 1.0) < 1e-9

    def test_higher_logit_higher_weight(self):
        w = EmbeddingDimensionWeights(logits={"a": 1.0, "b": 0.0})
        nw = get_normalized_weights(w)
        assert nw["a"] > nw["b"]

    def test_empty_logits_empty_result(self):
        w = EmbeddingDimensionWeights()
        nw = get_normalized_weights(w)
        assert nw == {}

    def test_strategies_filter(self):
        w = EmbeddingDimensionWeights(logits={"a": 1.0, "b": 2.0})
        nw = get_normalized_weights(w, strategies=["a"])
        assert "b" not in nw
        assert abs(nw["a"] - 1.0) < 1e-9

    def test_missing_strategy_gets_default(self):
        w = EmbeddingDimensionWeights(logits={"a": 0.0})
        nw = get_normalized_weights(w, strategies=["a", "b"])
        # Both have logit 0.0, so equal
        assert abs(nw["a"] - 0.5) < 1e-9
        assert abs(nw["b"] - 0.5) < 1e-9


# ── compute_weighted_embedding_similarity ────────────────────────────


class TestWeightedSimilarity:
    def test_identical_vectors(self):
        sig = {"semantic": [1.0, 0.0, 0.0]}
        ctx = {"semantic": [1.0, 0.0, 0.0]}
        result = compute_weighted_embedding_similarity(sig, ctx)
        assert abs(result.similarity - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        sig = {"semantic": [1.0, 0.0]}
        ctx = {"semantic": [0.0, 1.0]}
        result = compute_weighted_embedding_similarity(sig, ctx)
        assert abs(result.similarity) < 1e-6

    def test_empty_intersection(self):
        sig = {"semantic": [1.0]}
        ctx = {"technical": [1.0]}
        result = compute_weighted_embedding_similarity(sig, ctx)
        assert result.similarity == 0.0
        assert result.strategies_used == []

    def test_dimension_mismatch_raises(self):
        sig = {"semantic": [1.0, 0.0]}
        ctx = {"semantic": [1.0]}
        with pytest.raises(ValueError, match="dimension mismatch"):
            compute_weighted_embedding_similarity(sig, ctx)

    def test_multiple_strategies(self):
        sig = {
            "semantic": [1.0, 0.0],
            "technical": [0.0, 1.0],
        }
        ctx = {
            "semantic": [1.0, 0.0],
            "technical": [0.0, 1.0],
        }
        result = compute_weighted_embedding_similarity(sig, ctx)
        assert abs(result.similarity - 1.0) < 1e-6
        assert len(result.strategies_used) == 2

    def test_weighted_emphasis(self):
        w = EmbeddingDimensionWeights(
            logits={"semantic": 10.0, "technical": -10.0}
        )
        sig = {
            "semantic": [1.0, 0.0],
            "technical": [1.0, 0.0],
        }
        ctx = {
            "semantic": [1.0, 0.0],
            "technical": [0.0, 1.0],
        }
        result = compute_weighted_embedding_similarity(sig, ctx, w)
        # Semantic is weighted very highly (~1.0), technical barely counts
        assert result.similarity > 0.9

    def test_zero_norm_returns_zero_similarity(self):
        sig = {"semantic": [0.0, 0.0]}
        ctx = {"semantic": [1.0, 0.0]}
        result = compute_weighted_embedding_similarity(sig, ctx)
        assert abs(result.per_strategy_similarities["semantic"]) < 1e-6


# ── record_acceptance ────────────────────────────────────────────────


class TestRecordAcceptance:
    def test_updates_logits(self):
        w = initialize_dimension_weights(["a", "b"])
        record_acceptance(w, {"a": 1.0, "b": 0.0})
        assert w.logits["a"] > 0.0
        assert w.logits["b"] < 0.0

    def test_increments_update_count(self):
        w = initialize_dimension_weights(["a"])
        assert w.update_count == 0
        record_acceptance(w, {"a": 0.5})
        assert w.update_count == 1

    def test_zero_sum_deltas(self):
        w = initialize_dimension_weights(["a", "b", "c"])
        result = record_acceptance(w, {"a": 0.8, "b": 0.3, "c": 0.5})
        delta_sum = sum(result.logit_deltas.values())
        assert abs(delta_sum) < 1e-9

    def test_learning_rate_decays(self):
        w = initialize_dimension_weights(["a"])
        r1 = record_acceptance(w, {"a": 1.0})
        r2 = record_acceptance(w, {"a": 1.0})
        assert r2.effective_learning_rate < r1.effective_learning_rate

    def test_empty_contributions_raises(self):
        w = initialize_dimension_weights(["a"])
        with pytest.raises(ValueError, match="non-empty"):
            record_acceptance(w, {})

    def test_non_finite_contribution_raises(self):
        w = initialize_dimension_weights(["a"])
        with pytest.raises(ValueError, match="Non-finite"):
            record_acceptance(w, {"a": float("nan")})

    def test_new_strategy_inserted(self):
        w = initialize_dimension_weights(["a"])
        record_acceptance(w, {"a": 0.5, "b": 0.5})
        assert "b" in w.logits


# ── initialize_dimension_weights ─────────────────────────────────────


class TestInitializeDimensionWeights:
    def test_equal_logits(self):
        w = initialize_dimension_weights(["a", "b", "c"])
        for val in w.logits.values():
            assert val == 0.0

    def test_empty_strategies(self):
        w = initialize_dimension_weights()
        assert w.logits == {}

    def test_custom_learning_rate(self):
        w = initialize_dimension_weights(["a"], learning_rate=0.1)
        assert w.learning_rate == 0.1

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="Learning rate"):
            initialize_dimension_weights(learning_rate=0.0)

    def test_duplicate_strategy_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            initialize_dimension_weights(["a", "a"])

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            initialize_dimension_weights(["a", ""])


# ── ensure_strategies_present ────────────────────────────────────────


class TestEnsureStrategiesPresent:
    def test_adds_missing(self):
        w = initialize_dimension_weights(["a"])
        ensure_strategies_present(w, ["b"])
        assert "b" in w.logits
        assert w.logits["b"] == 0.0

    def test_preserves_existing(self):
        w = EmbeddingDimensionWeights(logits={"a": 1.5})
        ensure_strategies_present(w, ["a", "b"])
        assert w.logits["a"] == 1.5

    def test_empty_name_raises(self):
        w = initialize_dimension_weights()
        with pytest.raises(ValueError, match="non-empty"):
            ensure_strategies_present(w, [""])


# ── Serialization Roundtrip ──────────────────────────────────────────


class TestSerialization:
    def test_roundtrip(self):
        w = EmbeddingDimensionWeights(
            logits={"a": 1.0, "b": -0.5},
            learning_rate=0.1,
            update_count=42,
        )
        data = serialize_dimension_weights(w)
        w2 = deserialize_dimension_weights(data)
        assert w2.logits == w.logits
        assert w2.learning_rate == w.learning_rate
        assert w2.update_count == w.update_count

    def test_missing_logits_raises(self):
        with pytest.raises(ValueError, match="logits"):
            deserialize_dimension_weights({})

    def test_invalid_logits_type_raises(self):
        with pytest.raises(ValueError, match="dict"):
            deserialize_dimension_weights({"logits": "not_a_dict"})

    def test_non_finite_logit_raises(self):
        with pytest.raises(ValueError, match="Non-finite"):
            deserialize_dimension_weights({"logits": {"a": float("nan")}})

    def test_defaults_for_missing_fields(self):
        w = deserialize_dimension_weights({"logits": {"a": 0.0}})
        assert w.learning_rate == 0.05
        assert w.update_count == 0
