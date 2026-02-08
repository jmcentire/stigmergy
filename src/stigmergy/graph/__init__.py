"""Communication graph — person-to-person edges extracted from signals.

The graph captures who communicates with whom, through which channels,
and how recently. It enables:
- Bridge distance (Discovery detection)
- Constraint analysis (Normalized Deviance detection)
- Structural hole identification (Channel migration detection)
"""

from stigmergy.graph.extractor import EdgeExtractor
from stigmergy.graph.graph import CommunicationGraph, Edge, NodeState
from stigmergy.graph.identity import IdentityResolver, PersonProfile
from stigmergy.graph.metrics import (
    bridge_distance,
    burt_constraint,
    clustering_coefficient,
    connected_components,
    effective_size,
    structural_holes,
)

__all__ = [
    "CommunicationGraph",
    "Edge",
    "EdgeExtractor",
    "IdentityResolver",
    "NodeState",
    "PersonProfile",
    "bridge_distance",
    "burt_constraint",
    "clustering_coefficient",
    "connected_components",
    "effective_size",
    "structural_holes",
]
