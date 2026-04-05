"""Fixtures for ownership tests — isolated from core stigmergy fixtures."""

import pytest

from stigmergy.ownership.graph import (
    EntityNode,
    EntityType,
    OwnershipGraph,
)


@pytest.fixture
def empty_graph():
    """An empty ownership graph."""
    return OwnershipGraph()


@pytest.fixture
def simple_graph():
    """A simple ownership graph: 2 persons owning 3 companies."""
    from stigmergy.ownership.graph import EdgeType, OwnershipEdge

    g = OwnershipGraph()

    # Persons
    g.add_entity(EntityNode(entity_id="p1", name="Alice Smith", entity_type=EntityType.PERSON, jurisdiction="GB"))
    g.add_entity(EntityNode(entity_id="p2", name="Bob Jones", entity_type=EntityType.PERSON, jurisdiction="GB"))

    # Companies
    g.add_entity(EntityNode(entity_id="c1", name="Alpha Ltd", entity_type=EntityType.COMPANY, jurisdiction="GB"))
    g.add_entity(EntityNode(entity_id="c2", name="Beta Ltd", entity_type=EntityType.COMPANY, jurisdiction="GB"))
    g.add_entity(EntityNode(entity_id="c3", name="Gamma Ltd", entity_type=EntityType.COMPANY, jurisdiction="GB"))

    # Alice owns Alpha and Beta
    g.add_edge(OwnershipEdge(source_id="p1", target_id="c1", edge_type=EdgeType.OWNERSHIP, percentage=100.0))
    g.add_edge(OwnershipEdge(source_id="p1", target_id="c2", edge_type=EdgeType.OWNERSHIP, percentage=51.0))

    # Bob owns Beta and Gamma
    g.add_edge(OwnershipEdge(source_id="p2", target_id="c2", edge_type=EdgeType.OWNERSHIP, percentage=49.0))
    g.add_edge(OwnershipEdge(source_id="p2", target_id="c3", edge_type=EdgeType.OWNERSHIP, percentage=100.0))

    return g
