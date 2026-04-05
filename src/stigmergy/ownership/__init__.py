"""Beneficial ownership detection via stigmergic mesh.

Applies the ASD (Ambient Structure Discovery) architecture to corporate
ownership data for detecting concealed beneficial ownership structures.

This module is isolated from the core stigmergy organizational awareness
pipeline. It reuses core primitives (Signal, bloom filters, SimHash, mesh
routing) but provides its own graph implementation backed by igraph for
scalability to million-node ownership networks.
"""
