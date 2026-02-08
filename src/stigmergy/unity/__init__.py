"""Unity system: field equations for the stigmergic mesh.

Defines the phase space agents operate in. Programs the physics,
not the behavior. Every theoretical concept gets a concrete home.
Every parameter is a knob.
"""

from stigmergy.unity.field_config import FieldConfig
from stigmergy.unity.field_state import FieldState, compute_mesh_field_state

__all__ = ["FieldConfig", "FieldState", "compute_mesh_field_state"]
