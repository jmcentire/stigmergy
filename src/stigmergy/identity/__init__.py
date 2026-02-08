"""Identity resolution — multi-source person identity unification.

Cross-cutting service consumed by graph, mesh, scoring, and CLI.
Supports pluggable providers (static files, live APIs) and persists
runtime-learned aliases across runs.
"""

from stigmergy.identity.provider import IdentityProvider, PersonRecord
from stigmergy.identity.resolver import IdentityResolver, PersonProfile
from stigmergy.identity.sensitivity import DataClassification
from stigmergy.identity.store import AliasStore

__all__ = [
    "AliasStore",
    "DataClassification",
    "IdentityProvider",
    "IdentityResolver",
    "PersonProfile",
    "PersonRecord",
]
