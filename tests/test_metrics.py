"""Tests for graph structural analysis metrics."""

from __future__ import annotations

import math

import pytest

from stigmergy.graph.graph import CommunicationGraph
from stigmergy.graph.identity import IdentityResolver
from stigmergy.graph.metrics import (
    bridge_distance,
    burt_constraint,
    clustering_coefficient,
    connected_components,
    effective_size,
    structural_holes,
)


@pytest.fixture
def resolver():
    r = IdentityResolver()
    for name in ["alice", "bob", "carol", "dave", "eve", "frank"]:
        r.register_alias(name, f"{name}@test.com")
    return r


@pytest.fixture
def graph(resolver):
    return CommunicationGraph(resolver)


class TestBridgeDistance:
    def test_same_person(self, graph):
        """Distance to self is 0."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        assert bridge_distance(graph, "alice", "alice") == 0.0

    def test_direct_neighbors(self, graph):
        """Distance between direct neighbors is 1."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        assert bridge_distance(graph, "alice", "bob") == 1.0

    def test_two_hop_path(self, graph):
        """Distance across two hops."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("bob", "carol", "pr_review", "repo")
        assert bridge_distance(graph, "alice", "carol") == 2.0

    def test_three_hop_path(self, graph):
        """Distance across three hops."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("bob", "carol", "pr_review", "repo")
        graph.add_edge("carol", "dave", "pr_review", "repo")
        assert bridge_distance(graph, "alice", "dave") == 3.0

    def test_disconnected_returns_inf(self, graph):
        """Disconnected people have infinite distance."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("carol", "dave", "pr_review", "repo")
        assert bridge_distance(graph, "alice", "carol") == float("inf")

    def test_unknown_person_returns_inf(self, graph):
        """Unknown person returns infinite distance."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        assert bridge_distance(graph, "alice", "unknown") == float("inf")

    def test_shortest_path_chosen(self, graph):
        """BFS finds shortest path among alternatives."""
        # Direct path: alice → carol (1 hop)
        graph.add_edge("alice", "carol", "pr_review", "repo")
        # Longer path: alice → bob → carol (2 hops)
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("bob", "carol", "pr_review", "repo")
        assert bridge_distance(graph, "alice", "carol") == 1.0

    def test_undirected_traversal(self, graph):
        """BFS uses edges as undirected (both directions)."""
        # Edge only from alice→bob, but bob should still find alice
        graph.add_edge("alice", "bob", "pr_review", "repo")
        assert bridge_distance(graph, "bob", "alice") == 1.0


class TestBurtConstraint:
    def test_no_neighbors_zero_constraint(self, graph):
        """Person with no connections has zero constraint."""
        assert burt_constraint(graph, "alice") == 0.0

    def test_single_contact_full_constraint(self, graph):
        """Person with only one contact has maximum constraint."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        # With one contact, p_ij = 1.0, constraint = 1.0^2 = 1.0
        c = burt_constraint(graph, "alice")
        assert c == pytest.approx(1.0, abs=0.01)

    def test_bridge_position_lower_constraint(self, graph):
        """Person bridging two disconnected groups has lower constraint."""
        # Alice connects to both bob and carol, who don't know each other
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("alice", "carol", "pr_review", "repo")
        c = burt_constraint(graph, "alice")
        # Two independent contacts: each p_ij = 0.5, constraint = 2 * 0.5^2 = 0.5
        assert c < 1.0
        assert c == pytest.approx(0.5, abs=0.1)

    def test_clique_high_constraint(self, graph):
        """Person in a dense clique has high constraint."""
        # Everyone knows everyone
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("alice", "carol", "pr_review", "repo")
        graph.add_edge("bob", "carol", "pr_review", "repo")
        c = burt_constraint(graph, "alice")
        # In a full triangle, constraint is higher than bridge position
        assert c > 0.5


class TestEffectiveSize:
    def test_no_neighbors(self, graph):
        """Person with no contacts has effective size 0."""
        assert effective_size(graph, "alice") == 0.0

    def test_independent_contacts(self, graph):
        """Non-redundant contacts give effective size close to n."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("alice", "carol", "pr_review", "repo")
        graph.add_edge("alice", "dave", "pr_review", "repo")
        # bob, carol, dave don't know each other → all non-redundant
        es = effective_size(graph, "alice")
        assert es == pytest.approx(3.0, abs=0.1)

    def test_redundant_contacts(self, graph):
        """Contacts who know each other reduce effective size."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("alice", "carol", "pr_review", "repo")
        graph.add_edge("bob", "carol", "pr_review", "repo")
        es = effective_size(graph, "alice")
        # bob and carol know each other → partially redundant
        assert es < 2.0


class TestStructuralHoles:
    def test_no_holes_in_complete_graph(self, graph):
        """Complete graph has no structural holes."""
        for a, b in [("alice", "bob"), ("alice", "carol"), ("bob", "carol")]:
            graph.add_edge(a, b, "pr_review", "repo")
        holes = structural_holes(graph)
        assert len(holes) == 0

    def test_detects_disconnected_clusters(self, graph):
        """Structural holes detected between disconnected groups."""
        # Cluster 1: alice, bob
        graph.add_edge("alice", "bob", "pr_review", "repo")
        # Cluster 2: carol, dave
        graph.add_edge("carol", "dave", "pr_review", "repo")
        holes = structural_holes(graph)
        # Should find holes between the two clusters
        assert len(holes) > 0
        # Distance should be infinity
        assert holes[0][2] == float("inf")


class TestClusteringCoefficient:
    def test_single_contact(self, graph):
        """Person with one contact has coefficient 0."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        assert clustering_coefficient(graph, "alice") == 0.0

    def test_full_clique(self, graph):
        """Person in a full clique has coefficient 1.0."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("alice", "carol", "pr_review", "repo")
        graph.add_edge("bob", "carol", "pr_review", "repo")
        cc = clustering_coefficient(graph, "alice")
        assert cc == pytest.approx(1.0, abs=0.01)

    def test_star_topology(self, graph):
        """Center of star has coefficient 0 (contacts don't know each other)."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("alice", "carol", "pr_review", "repo")
        graph.add_edge("alice", "dave", "pr_review", "repo")
        cc = clustering_coefficient(graph, "alice")
        assert cc == 0.0

    def test_partial_clustering(self, graph):
        """Partially connected contacts give intermediate coefficient."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("alice", "carol", "pr_review", "repo")
        graph.add_edge("alice", "dave", "pr_review", "repo")
        # Only bob-carol are connected
        graph.add_edge("bob", "carol", "pr_review", "repo")
        cc = clustering_coefficient(graph, "alice")
        # 1 link out of 3 possible = 1/3
        assert cc == pytest.approx(1.0 / 3.0, abs=0.01)


class TestConnectedComponents:
    def test_single_component(self, graph):
        """Fully connected graph has one component."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("bob", "carol", "pr_review", "repo")
        comps = connected_components(graph)
        assert len(comps) == 1
        assert comps[0] == {"alice", "bob", "carol"}

    def test_two_components(self, graph):
        """Two disconnected groups form two components."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("carol", "dave", "pr_review", "repo")
        comps = connected_components(graph)
        assert len(comps) == 2
        comp_sets = [frozenset(c) for c in comps]
        assert frozenset({"alice", "bob"}) in comp_sets
        assert frozenset({"carol", "dave"}) in comp_sets

    def test_isolated_node(self, graph):
        """Isolated node is its own component."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        # Manually add an isolated node
        graph._ensure_node("carol", graph._nodes["alice"].first_seen, "")
        comps = connected_components(graph)
        assert len(comps) == 2

    def test_sorted_by_size(self, graph):
        """Components are sorted by size (largest first)."""
        graph.add_edge("alice", "bob", "pr_review", "repo")
        graph.add_edge("alice", "carol", "pr_review", "repo")
        graph.add_edge("dave", "eve", "pr_review", "repo")
        comps = connected_components(graph)
        assert len(comps[0]) >= len(comps[1])

    def test_empty_graph(self, graph):
        """Empty graph has no components."""
        comps = connected_components(graph)
        assert len(comps) == 0
