# Stigmergy: Sequence-Aware Familiarity

## What This Is

A targeted modification to Stigmergy's ART mesh routing to add sequence-aware familiarity scoring for multi-step story signals from Chronicler.

## Current State

Stigmergy's familiarity function uses six weighted components (embedding_similarity 35%, keyword_overlap 20%, source_affinity 15%, temporal_proximity 10%, author_affinity 10%, signal_credibility 10%). All treat signals atomically — no understanding of sequential structure.

Signal.metadata is a dict[str, Any] that can carry arbitrary data. Story signals from Chronicler will include metadata["steps"] as a list of step_type strings.

Context accumulates terms, source_counts, author_counts, and embeddings. No sequence representation exists.

## What Changes

1. Add sequence_similarity weight to FamiliarityWeights (default 0.0 for backward compat)
2. Add sequence representation to Context (step_sequences list, sequence_ngrams counter)
3. Implement sequence similarity metric (n-gram overlap or normalized edit distance)
4. Update worker learning to capture step sequences on rag_indexed acceptance
5. Return 0.0 for non-story signals (no metadata["steps"])

## Why

When Chronicler emits stories (multi-step event narratives), the ART mesh needs to cluster them by journey structure, not just text content. Two checkout flows with similar step sequences should route to the same worker even if their prose descriptions differ. This enables pattern discovery like "users usually do X→Y→Z; we can shorten to X→Z."

## Constraints

- Backward compatible: default weight 0.0 means existing behavior unchanged
- Pure function: deterministic, no randomness, no LLM
- Performance: O(n) or better where n = sequence length
- Immutable signals: never modify Signal after creation
- All existing 1200+ tests must pass
