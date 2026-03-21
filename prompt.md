# Sequence-Aware Familiarity Enhancement

## Context

Stigmergy's ART mesh routing system uses familiarity-based signal clustering to route signals to appropriate workers. The current system treats all signals atomically without understanding sequential structure, which limits its ability to cluster multi-step story signals by journey structure and blocks pattern discovery for sequence optimization.

## Problem

Currently, two checkout flows with similar step sequences (e.g., "login→cart→payment" vs "login→cart→shipping→payment") may route to different workers despite having similar journey structure. This prevents:

- Effective clustering of story signals by journey patterns
- Pattern discovery for sequence optimization (identifying sequences like X→Y→Z that could be shortened to X→Z)
- Cross-journey analysis for user experience improvements

## Requirements

### Core Functionality
- Add `sequence_similarity` weight to `FamiliarityWeights` configuration
- Add sequence representation to `Context` for storing learned patterns
- Implement sequence similarity metric that operates in O(n) or better time complexity
- Update worker learning to accumulate step sequence knowledge
- Return 0.0 sequence similarity for non-story signals

### Compatibility
- Maintain backward compatibility - default weight 0.0 preserves existing behavior
- All existing 1200+ tests must continue to pass
- Never modify `Signal` objects after creation (immutable)

### Technical Constraints
- Pure function implementation - no randomness, no LLM calls
- Performance constraint O(n) or better where n = sequence length
- Respect data privacy constraints for cross-journey pattern analysis

## Input Structure

Story signals from Chronicler include:
```python
signal.metadata['steps'] = ["step_type1", "step_type2", "step_type3"]
```

The existing familiarity function uses six weighted components and should be extended with the seventh sequence-based component.

## Expected Behavior

- Story signals with similar step sequences should have higher familiarity scores
- Pattern accumulation should enable discovery of optimizable sequences
- Non-story signals should return 0.0 for sequence similarity
- Default configuration (weight=0.0) should preserve all existing routing behavior