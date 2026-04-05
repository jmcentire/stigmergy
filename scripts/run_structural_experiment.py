"""Run the complement-coded structural mesh experiment on Elliptic++.

Saves results to data/structural_mesh_results.json for comparison
with the text-based baseline.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)

from stigmergy.ownership.structural_signals import build_structural_signals
from stigmergy.ownership.structural_mesh import run_structural_mesh, evaluate_structural_results

import polars as pl

data_dir = Path("data/elliptic/Elliptic++ Dataset")

print("Loading data...", file=sys.stderr)
features = pl.read_csv(data_dir / "txs_features.csv")
classes = pl.read_csv(data_dir / "txs_classes.csv")
edges = pl.read_csv(data_dir / "txs_edgelist.csv")

print("Building structural signals...", file=sys.stderr)
signals = build_structural_signals(features, edges)

print("Running structural mesh...", file=sys.stderr)
result = asyncio.run(run_structural_mesh(signals, num_workers=12))

print("Evaluating...", file=sys.stderr)
eval_results = evaluate_structural_results(result, classes)

# Save results
output = {
    "experiment": "structural_complement_coded",
    "signals": result.signals_processed,
    "workers_final": result.workers_final,
    "accepted": result.accepted,
    "rejected": result.rejected,
    "anomalies": len(result.anomalies),
    "evaluation": eval_results,
}
with open("data/structural_mesh_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to data/structural_mesh_results.json", file=sys.stderr)
print(json.dumps(output, indent=2), file=sys.stderr)
