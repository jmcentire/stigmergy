"""Re-export shim — identity resolution moved to stigmergy.identity.

All imports from stigmergy.graph.identity continue to work.
New code should import from stigmergy.identity directly.
"""

from stigmergy.identity.resolver import IdentityResolver, PersonProfile

__all__ = ["IdentityResolver", "PersonProfile"]
