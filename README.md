# Stigmergy

Organizational awareness through stigmergic signal processing.

Stigmergy ingests work artifacts from GitHub, Linear, and Slack, routes them through an [ART](https://en.wikipedia.org/wiki/Adaptive_resonance_theory)-based mesh of self-organizing agents, and surfaces structural patterns — coordination gaps, knowledge silos, dependency risks — without anyone having to ask.

## How it works

1. **Ingest** signals from your tools (PRs, issues, commits, Slack threads)
2. **Route** each signal through a competitive mesh (stop-on-first-accept, like biological stigmergy)
3. **Correlate** across sources to find patterns no single tool reveals
4. **Surface** findings with PII/credential filtering and configurable delivery

The mesh uses Adaptive Resonance Theory for stable category formation in non-stationary environments: new patterns create new workers, familiar patterns reinforce existing ones, stale patterns decay. No retraining required.

## Quick start

```bash
# Clone and install
git clone https://github.com/jmcentire/stigmergy.git
cd stigmergy
make install

# Run with mock data (no API keys needed)
stigmergy run --once

# Run with live GitHub data
gh auth login
stigmergy run --once --live
```

## Requirements

- Python 3.12+
- [gh CLI](https://cli.github.com/) (for live GitHub data)

## Installation

### Development install (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### With LLM support

```bash
pip install -e ".[cli]"
export ANTHROPIC_API_KEY=your-key
stigmergy config set llm.provider anthropic
```

Without the Anthropic integration, stigmergy uses deterministic heuristics for assessments. The LLM adds richer correlation but is entirely optional.

## Usage

```bash
# Interactive setup — configures sources, identity, constraints
stigmergy init

# Single pass over all configured sources
stigmergy run --once

# Single pass with live API sources
stigmergy run --once --live

# Continuous monitoring
stigmergy run --live

# Check source connectivity (no tokens spent)
stigmergy check
stigmergy check --slack

# View/modify configuration
stigmergy config show
stigmergy config set budget.daily_cap_usd 10.00

# View current mesh state
stigmergy status
```

## Live sources

| Source | Auth | What it ingests |
|--------|------|----------------|
| GitHub | `gh auth login` | PRs, issues, commits, reviews, comments |
| Linear | `LINEAR_API_KEY` env var | Issues, projects, cycles, comments |
| Slack | `SLACK_BOT_TOKEN` env var | Channel messages, threads, reactions |

## Identity resolution

Stigmergy unifies people across sources — the same person may appear as a GitHub handle, Slack display name, Linear UUID, or email address. Resolution runs automatically from configured providers and learns new aliases at runtime.

Identity data lives in `config/` (gitignored). To set up:

```bash
stigmergy init  # walks through team setup interactively
```

Or manually create `config/team_roster.csv`:

```csv
Alice Wang,Alice Wang <alice@example.com>,@Alice Wang,alice212
Bob Kim,Bob Kim <bob.kim@example.com>,@Bob,bobkim
```

## Configuration

Config lives in `.stigmergy/config.yaml` (created by `stigmergy init`):

```yaml
sources:
  github:
    enabled: true
    repos: ["org/repo1", "org/repo2"]
  linear:
    enabled: false
  slack:
    enabled: false

llm:
  provider: stub          # stub (free) or anthropic

budget:
  daily_cap_usd: 5.00
  hourly_cap_usd: 1.00

constraints:
  path: config/constraints.yaml   # PII/credential kill + redaction rules
```

## Constraint filtering

All output passes through a constraint engine before delivery. By default:

- **Kill** (null-route): SSNs, credit cards, emails, credentials, API keys, compensation data, HR actions
- **Redact** (mask): phone numbers, physical addresses

Rules are configurable in `config/constraints.yaml`.

## Architecture

```
signals (GitHub, Linear, Slack)
    |
    v
[ Ingestion ] --> normalized Signal objects
    |
    v
[ Mesh Router ] --> BFS competitive routing, stop-on-first-accept
    |
    v
[ Workers ] --> ART categories: familiarity match, weight update, fork/merge/decay
    |
    v
[ Correlator ] --> cross-signal pattern detection
    |
    v
[ Constraint Filter ] --> PII kill / redact
    |
    v
[ Output ] --> findings, insights, structural metrics
```

Key design principles:

- **One pattern**: Workers, supervisors, and control layers are the same agent-context-signal pattern at different scales
- **Stop-on-first-accept**: BFS routing with Simon's satisficing — first worker above threshold takes the signal
- **Complement coding**: Full workers raise vigilance thresholds to prevent category monopoly
- **Match-based learning**: Weights update only on acceptance (ART stability guarantee)
- **Immutable signals**: Signals are frozen; derived state lives in contexts

## Testing

```bash
make test          # run all tests
make test-v        # verbose output
pytest -k "mesh"   # run by keyword
```

## Budget

Default caps: $5/day, $1/hour. When the LLM budget is exhausted, stigmergy falls back to heuristic-only mode — it never stops running, just reduces assessment depth. Adjust caps:

```bash
stigmergy config set budget.daily_cap_usd 10.00
stigmergy config set budget.hourly_cap_usd 2.00
```

## Project structure

```
src/stigmergy/
  adapters/         Source connectors (GitHub, Linear, Slack — mock + live)
  attention/        Attention model, portfolio scoring, surfacing
  cli/              CLI entry point, config, budget, live adapters
  constraints/      Output filtering (PII/credential kill and redaction)
  core/             Algorithms: familiarity, consensus, energy, lifecycle
  delivery/         Output delivery framework
  identity/         Person identity resolution across sources
  mesh/             ART mesh: routing, workers, topology, insights, stability
  pipeline/         Signal ingestion pipeline
  policy/           Policy engine, spectral analysis, budget enforcement
  primitives/       Data types: Signal, Context, Agent, Assessment
  services/         LLM, embedding, vector store, token budget
  structures/       Bloom filters, LSH, SimHash, ring buffers, tries
  unity/            Field equations, eigenmonitor, PID control
  tracing/          Execution tracing
```

## Theoretical foundations

Stigmergy implements ideas from:

- **Adaptive Resonance Theory** (Grossberg/Carpenter) — stable category formation with vigilance-gated plasticity
- **Stigmergy** (Grass&eacute;, Theraulaz) — coordination through shared environment rather than direct communication
- **Crawford-Sobel** (1982) — information degradation under strategic communication; babbling equilibrium at bias >= 1/4
- **Beer's Viable System Model** — System 4 intelligence function, algedonic signals
- **Spectral graph analysis** — anomaly detection via Laplacian eigenvalue distribution

For the full theoretical treatment, see: [Ambient Structure Discovery via Stigmergic Mesh](https://github.com/jmcentire) (paper forthcoming).

## License

[MIT](LICENSE)
