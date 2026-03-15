# Stigmergy — System Context

## What It Is
Stigmergic signal processing for organizational awareness. ART network processes multi-source signals to detect coupling, deviance, and communication patterns.

## How It Works
Signal flow: Dedup (SimHash) -> Embed (optional) -> Entry selection -> BFS routing -> ART resonance -> Gap detection -> Agent evaluation -> Consensus -> Constraints -> Lifecycle

## Key Constraints
- Signals immutable after ingestion (C001)
- No global coordinator — ART competition only (C002)
- Caucus phase required for cold-start (C003)
- Mechanical fallback when budget exhausted (C004)
- PII filtering on output (C005)
- Frozen primitives (C007)

## Architecture
~129 Python files. Core: mesh (ART topology), core (consensus/familiarity/energy), pipeline (signal flow), adapters (GitHub/Linear/Slack), unity (CERTX field equations), attention (P(knows)), identity (multi-source resolution).

## Done Checklist
- [ ] Caucus prevents mesh collapse
- [ ] SimHash dedup prevents duplicate LLM calls
- [ ] Budget exhaustion falls back to mechanical mode
- [ ] PII filtered from all outputs
- [ ] State persists across restarts
