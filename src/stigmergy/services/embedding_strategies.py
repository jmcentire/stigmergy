"""Concrete embedding strategy implementations.

Three strategies, all producing L2-normalized 64-dim vectors:
  1. SemanticStrategy — LLM embedding with keyword-hash fallback
  2. TechnicalStrategy — code-aware tokenization + hash-bucketed features
  3. SocialStrategy — author/mention/channel graph embedding

All use hashlib.md5 for deterministic cross-process hashing.
Zero vector is the "no information" sentinel.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 64


# ── Shared Utilities ─────────────────────────────────────────────────


def zero_vector() -> np.ndarray:
    """Return the zero vector sentinel."""
    return np.zeros(EMBEDDING_DIM, dtype=np.float64)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Returns zero vector unchanged."""
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def signed_hash(token: str, num_buckets: int) -> tuple[int, int]:
    """Deterministic (bucket, sign) from MD5 digest.

    bytes[0:8] -> bucket index, bytes[8:16] -> sign.
    """
    digest = hashlib.md5(token.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[0:8], "little") % num_buckets
    sign = 1 if int.from_bytes(digest[8:16], "little") % 2 == 0 else -1
    return bucket, sign


# ── LLM Service Protocol ────────────────────────────────────────────


@runtime_checkable
class EmbeddingLLMService(Protocol):
    """Optional LLM service for semantic embedding."""

    async def embed_text(self, text: str) -> list[float]: ...


# ── SemanticStrategy ─────────────────────────────────────────────────


class SemanticStrategy:
    """General semantic embedding via LLM or keyword-hash fallback."""

    def __init__(self, llm_service: EmbeddingLLMService | None = None) -> None:
        self._llm = llm_service

    @property
    def name(self) -> str:
        return "semantic"

    async def embed(self, content: str, metadata: dict | None = None) -> list[float]:
        """Produce semantic embedding. Never raises."""
        try:
            # Try LLM first
            if self._llm is not None:
                try:
                    raw = await self._llm.embed_text(content)
                    if raw and all(math.isfinite(v) for v in raw):
                        vec = np.array(raw[:EMBEDDING_DIM], dtype=np.float64)
                        if len(vec) < EMBEDDING_DIM:
                            vec = np.pad(vec, (0, EMBEDDING_DIM - len(vec)))
                        return l2_normalize(vec).tolist()
                except Exception as exc:
                    logger.warning("LLM embedding failed, using fallback: %s", exc)

            # Fallback: deterministic feature hashing
            return self._hash_fallback(content)
        except Exception as exc:
            logger.warning("SemanticStrategy.embed failed: %s", exc)
            return zero_vector().tolist()

    def _hash_fallback(self, content: str) -> list[float]:
        """Keyword-based feature hashing."""
        if not content or not content.strip():
            return zero_vector().tolist()

        tokens = content.lower().split()
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float64)

        for token in tokens:
            token = re.sub(r"[^\w]", "", token)
            if len(token) < 2:
                continue
            bucket, sign = signed_hash(token, EMBEDDING_DIM)
            vec[bucket] += sign

        return l2_normalize(vec).tolist()


# ── TechnicalStrategy ────────────────────────────────────────────────

# Dimension ranges:
#   [0:32]  - general token features
#   [32:48] - pattern type indicators
#   [48:64] - pattern-specific token features

_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_PATTERNS = {
    "function_definition": re.compile(
        r"\b(?:def|function|func|fn)\s+(\w+)"
    ),
    "import_statement": re.compile(
        r"\b(?:import|from|require|include)\s+[\w.]+"
    ),
    "api_path": re.compile(r'["\']/(api|v\d)/[\w/]+["\']'),
    "error_pattern": re.compile(
        r"\b(?:Error|Exception|raise|throw|panic)\b"
    ),
    "class_definition": re.compile(r"\b(?:class|struct|interface)\s+(\w+)"),
    "decorator": re.compile(r"@\w+"),
    "dotted_path": re.compile(r"\b\w+(?:\.\w+){2,}\b"),
}

# Map pattern types to indicator dimensions [32:48]
_PATTERN_DIM_MAP = {
    "function_definition": 32,
    "import_statement": 34,
    "api_path": 36,
    "error_pattern": 38,
    "class_definition": 40,
    "decorator": 42,
    "dotted_path": 44,
}


class TechnicalStrategy:
    """Code-aware embedding via tokenization + hash-bucketed features."""

    @property
    def name(self) -> str:
        return "technical"

    async def embed(self, content: str, metadata: dict | None = None) -> list[float]:
        """Produce technical embedding. Never raises."""
        try:
            if not content or not content.strip():
                return zero_vector().tolist()

            vec = np.zeros(EMBEDDING_DIM, dtype=np.float64)

            # General token features [0:32]
            tokens = self._tokenize_code_aware(content)
            for token in tokens:
                bucket, sign = signed_hash(token, 32)  # dims 0-31
                vec[bucket] += sign

            # Pattern type indicators [32:48]
            patterns = self._detect_patterns(content)
            for ptype, matches in patterns.items():
                dim = _PATTERN_DIM_MAP.get(ptype)
                if dim is not None:
                    vec[dim] += len(matches)
                    vec[dim + 1] += 1.0  # binary indicator

                # Pattern-specific token features [48:64]
                for match in matches:
                    sub_tokens = self._tokenize_code_aware(match)
                    for st in sub_tokens:
                        bucket, sign = signed_hash(
                            f"pat:{ptype}:{st}", 16
                        )
                        vec[48 + bucket] += sign

            return l2_normalize(vec).tolist()
        except Exception as exc:
            logger.warning("TechnicalStrategy.embed failed: %s", exc)
            return zero_vector().tolist()

    @staticmethod
    def _tokenize_code_aware(text: str) -> list[str]:
        """Split text into sub-tokens handling camelCase, snake_case, dots."""
        # Replace dots and underscores with spaces for splitting
        text = text.replace(".", " ").replace("_", " ")
        # Split camelCase
        text = _CAMEL_SPLIT.sub(" ", text)
        tokens = text.lower().split()
        return [t for t in tokens if len(t) >= 2]

    @staticmethod
    def _detect_patterns(text: str) -> dict[str, list[str]]:
        """Run regex pattern detectors. Returns {pattern_type: [matches]}."""
        result = {}
        for ptype, pattern in _PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                result[ptype] = matches
        return result


# ── SocialStrategy ───────────────────────────────────────────────────


class SocialStrategy:
    """Author/mention/channel graph embedding via hash projection."""

    @property
    def name(self) -> str:
        return "social"

    async def embed(self, content: str, metadata: dict | None = None) -> list[float]:
        """Produce social embedding from metadata. Never raises."""
        try:
            if metadata is None:
                metadata = {}

            author = metadata.get("author", "") or ""
            mentions = metadata.get("mentions", []) or []
            channel = metadata.get("channel", "") or ""

            if not author and not mentions and not channel:
                return zero_vector().tolist()

            vec = np.zeros(EMBEDDING_DIM, dtype=np.float64)

            # Author [0:32]
            if author:
                bucket, sign = signed_hash(f"author:{author}", 32)
                vec[bucket] += sign * 2.0  # author gets higher weight

            # Mentions [0:32]
            for mention in mentions:
                if mention:
                    bucket, sign = signed_hash(f"mention:{mention}", 32)
                    vec[bucket] += sign

            # Author+mention pairs [32:48]
            if author:
                for mention in mentions:
                    if mention:
                        bucket, sign = signed_hash(
                            f"pair:{author}+{mention}", 16
                        )
                        vec[32 + bucket] += sign

            # Channel [48:64]
            if channel:
                bucket, sign = signed_hash(f"channel:{channel}", 16)
                vec[48 + bucket] += sign * 1.5

            return l2_normalize(vec).tolist()
        except Exception as exc:
            logger.warning("SocialStrategy.embed failed: %s", exc)
            return zero_vector().tolist()
