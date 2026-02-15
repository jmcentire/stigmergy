"""Per-context embedding dimension weights.

Each context learns which embedding strategies (semantic, technical, social)
are most predictive for its domain via match-based learning.

Weights are stored as logits with softmax normalization.
Learning rate decays as: lr_eff = lr / (1 + update_count * 0.01).
Updates are zero-sum across strategies (relative importance only).
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator


class EmbeddingDimensionWeights(BaseModel):
    """Per-strategy logits with softmax-normalized access.

    Mutable — updates logits in-place on signal acceptance.
    """

    logits: dict[str, float] = Field(default_factory=dict)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    update_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_logits_finite(self) -> "EmbeddingDimensionWeights":
        for strategy, val in self.logits.items():
            if not math.isfinite(val):
                raise ValueError(
                    f"Logit values must be finite. "
                    f"Found non-finite value for strategy '{strategy}': {val}"
                )
        return self


class DimensionWeightsUpdateResult(BaseModel):
    """Diagnostic info from a weight update."""

    effective_learning_rate: float
    logit_deltas: dict[str, float]


class WeightedSimilarityResult(BaseModel):
    """Result of weighted embedding similarity computation."""

    similarity: float
    per_strategy_similarities: dict[str, float]
    strategies_used: list[str]
    effective_weights: dict[str, float]


# ── Pure Functions ───────────────────────────────────────────────────


def get_normalized_weights(
    dimension_weights: EmbeddingDimensionWeights,
    strategies: list[str] | None = None,
) -> dict[str, float]:
    """Compute softmax-normalized weights from logits.

    Numerically stable: subtract max before exp.
    Missing strategies get logit 0.0.
    """
    if strategies is not None:
        effective = {
            s: dimension_weights.logits.get(s, 0.0)
            for s in strategies
        }
    else:
        effective = dict(dimension_weights.logits)

    if not effective:
        return {}

    # Validate all finite
    for strategy, val in effective.items():
        if not math.isfinite(val):
            raise ValueError(
                f"Logit values must be finite. "
                f"Found non-finite value for strategy '{strategy}'."
            )

    max_logit = max(effective.values())
    exps = {s: math.exp(v - max_logit) for s, v in effective.items()}
    total = sum(exps.values())

    return {s: e / total for s, e in exps.items()}


def compute_weighted_embedding_similarity(
    signal_embeddings: dict[str, list[float]],
    context_prototypes: dict[str, list[float]],
    dimension_weights: EmbeddingDimensionWeights | None = None,
) -> WeightedSimilarityResult:
    """Compute weighted cosine similarity across strategies.

    Only the intersection of strategies present in both signal and context
    (and optionally weights) is used. Weights are re-normalized over the
    intersection. Empty intersection returns similarity 0.0.
    """
    # Find strategy intersection
    signal_strats = set(signal_embeddings.keys())
    context_strats = set(context_prototypes.keys())
    intersection = signal_strats & context_strats

    if not intersection:
        return WeightedSimilarityResult(
            similarity=0.0,
            per_strategy_similarities={},
            strategies_used=[],
            effective_weights={},
        )

    # Get weights for intersection
    if dimension_weights is not None:
        raw_weights = {
            s: dimension_weights.logits.get(s, 0.0)
            for s in intersection
        }
    else:
        raw_weights = {s: 0.0 for s in intersection}

    # Softmax normalize over intersection
    max_logit = max(raw_weights.values())
    exps = {s: math.exp(v - max_logit) for s, v in raw_weights.items()}
    total_exp = sum(exps.values())
    eff_weights = {s: e / total_exp for s, e in exps.items()}

    # Compute per-strategy cosine similarity
    per_strategy = {}
    for strategy in intersection:
        sig_vec = signal_embeddings[strategy]
        ctx_vec = context_prototypes[strategy]

        if len(sig_vec) != len(ctx_vec):
            raise ValueError(
                f"Embedding dimension mismatch for strategy '{strategy}': "
                f"signal has {len(sig_vec)} dimensions, "
                f"prototype has {len(ctx_vec)} dimensions."
            )

        # Validate finite
        for source, vec in [("signal", sig_vec), ("context", ctx_vec)]:
            if any(not math.isfinite(v) for v in vec):
                raise ValueError(
                    f"Non-finite values found in embedding vector "
                    f"for strategy '{strategy}' in '{source}'."
                )

        dot = sum(a * b for a, b in zip(sig_vec, ctx_vec))
        norm_a = math.sqrt(sum(a * a for a in sig_vec))
        norm_b = math.sqrt(sum(b * b for b in ctx_vec))

        if norm_a < 1e-12 or norm_b < 1e-12:
            per_strategy[strategy] = 0.0
        else:
            per_strategy[strategy] = dot / (norm_a * norm_b)

    # Weighted sum
    similarity = sum(
        eff_weights[s] * per_strategy[s] for s in intersection
    )

    return WeightedSimilarityResult(
        similarity=similarity,
        per_strategy_similarities=per_strategy,
        strategies_used=sorted(intersection),
        effective_weights=eff_weights,
    )


def record_acceptance(
    dimension_weights: EmbeddingDimensionWeights,
    strategy_contributions: dict[str, float],
) -> DimensionWeightsUpdateResult:
    """Update logits on signal acceptance (match-based learning).

    Logit update: logit_i += lr_eff * (contribution_i - mean_contribution).
    Zero-sum across strategies. Mutates weights in-place.
    """
    if not strategy_contributions:
        raise ValueError(
            "strategy_contributions must be non-empty. "
            "At least one strategy contribution is required."
        )

    for strategy, val in strategy_contributions.items():
        if not math.isfinite(val):
            raise ValueError(
                f"Non-finite contribution value for strategy "
                f"'{strategy}'. All contributions must be finite."
            )

    if dimension_weights.learning_rate <= 0:
        raise ValueError(
            f"Learning rate must be positive. "
            f"Got {dimension_weights.learning_rate}."
        )

    # Ensure all strategies are present in logits
    for strategy in strategy_contributions:
        if strategy not in dimension_weights.logits:
            dimension_weights.logits[strategy] = 0.0

    # Effective learning rate with decay
    lr_eff = dimension_weights.learning_rate / (
        1 + dimension_weights.update_count * 0.01
    )

    # Zero-sum update: logit_i += lr * (contrib_i - mean_contrib)
    mean_contrib = sum(strategy_contributions.values()) / len(
        strategy_contributions
    )

    deltas = {}
    for strategy, contrib in strategy_contributions.items():
        delta = lr_eff * (contrib - mean_contrib)
        dimension_weights.logits[strategy] += delta
        deltas[strategy] = delta

    dimension_weights.update_count += 1

    return DimensionWeightsUpdateResult(
        effective_learning_rate=lr_eff,
        logit_deltas=deltas,
    )


def initialize_dimension_weights(
    strategy_names: list[str] | None = None,
    learning_rate: float = 0.05,
) -> EmbeddingDimensionWeights:
    """Factory: create weights with equal logits (0.0) for given strategies."""
    if learning_rate <= 0 or learning_rate > 1.0:
        raise ValueError(
            f"Learning rate must be in (0.0, 1.0]. Got {learning_rate}."
        )

    logits: dict[str, float] = {}
    if strategy_names:
        seen: set[str] = set()
        for i, name in enumerate(strategy_names):
            if not name:
                raise ValueError(
                    f"Strategy names must be non-empty strings. "
                    f"Found empty string at index {i}."
                )
            if name in seen:
                raise ValueError(
                    f"Duplicate strategy name '{name}' found. "
                    f"Strategy names must be unique."
                )
            seen.add(name)
            logits[name] = 0.0

    return EmbeddingDimensionWeights(
        logits=logits,
        learning_rate=learning_rate,
        update_count=0,
    )


def ensure_strategies_present(
    dimension_weights: EmbeddingDimensionWeights,
    strategy_names: list[str],
) -> EmbeddingDimensionWeights:
    """Insert missing strategies with logit 0.0. Mutates in-place."""
    for name in strategy_names:
        if not name:
            raise ValueError("Strategy names must be non-empty strings.")
        if name not in dimension_weights.logits:
            dimension_weights.logits[name] = 0.0
    return dimension_weights


def serialize_dimension_weights(
    dimension_weights: EmbeddingDimensionWeights,
) -> dict:
    """Serialize to a plain dict for YAML/JSON."""
    return {
        "logits": dict(dimension_weights.logits),
        "learning_rate": dimension_weights.learning_rate,
        "update_count": dimension_weights.update_count,
    }


def deserialize_dimension_weights(data: dict) -> EmbeddingDimensionWeights:
    """Deserialize from a plain dict. Validates all values."""
    if "logits" not in data:
        raise ValueError(
            "Missing required key 'logits' in serialized dimension weights."
        )

    logits = data["logits"]
    if not isinstance(logits, dict):
        raise ValueError(
            f"Expected 'logits' to be a dict, got {type(logits).__name__}."
        )

    for strategy, val in logits.items():
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(
                f"Non-finite or non-float logit value for "
                f"strategy '{strategy}': {val}."
            )

    return EmbeddingDimensionWeights(
        logits={k: float(v) for k, v in logits.items()},
        learning_rate=data.get("learning_rate", 0.05),
        update_count=data.get("update_count", 0),
    )
