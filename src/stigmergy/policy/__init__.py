"""Policy engine: declarative organizational policies with enforceable interventions.

Turns mesh observations into actionable guardrails. The mesh tells you
what's happening; the policy engine tells you what should happen about it.

Architecture:
  traces.py       - Append-only trace event store (the evidence)
  structure.py    - Dependency graph between repos/services (the map)
  signals.py      - Pattern detectors: dependency crossings, auth surface changes
  interventions.py - Actions: advisory flags, gate blocks
  budgets.py      - Activity counters that trigger escalation
  engine.py       - The compiler: trace → signal → budget check → intervention
"""
