"""Tests for the communication graph."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stigmergy.graph.graph import CommunicationGraph, Edge, NodeState
from stigmergy.graph.identity import IdentityResolver


@pytest.fixture
def resolver():
    """Simple resolver for tests."""
    r = IdentityResolver()
    # Register some test identities
    r.register_alias("alice", "alice@test.com")
    r.register_alias("alice", "alice.smith")
    r.register_alias("bob", "bob@test.com")
    r.register_alias("bob", "bob.jones")
    r.register_alias("carol", "carol@test.com")
    r.register_alias("dave", "dave@test.com")
    return r


@pytest.fixture
def graph(resolver):
    return CommunicationGraph(resolver)


class TestEdgeAccumulation:
    def test_add_single_edge(self, graph):
        """Adding an edge creates it with count=1."""
        edge = graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        assert edge.source == "alice"
        assert edge.target == "bob"
        assert edge.count == 1
        assert edge.weight == 1.0

    def test_add_duplicate_edge_accumulates(self, graph):
        """Adding the same edge type again increments count and weight."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        edge = graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        assert edge.count == 2
        assert edge.weight == 2.0

    def test_different_types_separate_edges(self, graph):
        """Different edge types between same pair create separate edges."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("alice", "bob", "pr_comment", "acme-org/backend")
        assert graph.edge_count == 2

    def test_self_loop_ignored(self, graph):
        """Self-loops are not stored."""
        graph.add_edge("alice", "alice", "pr_review", "acme-org/backend")
        assert graph.edge_count == 0

    def test_identity_resolution_in_edges(self, graph):
        """Aliases are resolved to canonical IDs."""
        graph.add_edge("alice@test.com", "bob.jones", "pr_review", "acme-org/backend")
        # Should resolve to alice → bob
        assert graph.edge_count == 1
        edges = graph.all_edges()
        assert edges[0].source == "alice"
        assert edges[0].target == "bob"

    def test_nodes_created_on_edge(self, graph):
        """Nodes are auto-created when edges are added."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        assert graph.node_count == 2
        assert graph.get_node("alice") is not None
        assert graph.get_node("bob") is not None

    def test_node_activity_tracking(self, graph):
        """Node signal_count increments on edge addition."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("alice", "carol", "pr_comment", "acme-org/backend")
        node = graph.get_node("alice")
        assert node is not None
        assert node.signal_count == 2

    def test_node_channels(self, graph):
        """Node tracks which channels it's active in."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("alice", "carol", "pr_review", "acme-org/pricing")
        node = graph.get_node("alice")
        assert node is not None
        assert "acme-org/backend" in node.primary_channels
        assert "acme-org/pricing" in node.primary_channels


class TestTemporalDecay:
    def test_fresh_edge_full_weight(self, graph):
        """Recent edges have full effective weight."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        weight = graph.effective_weight("alice", "bob")
        assert weight == pytest.approx(1.0, abs=0.01)

    def test_decayed_edge_lower_weight(self, graph):
        """Old edges have reduced effective weight."""
        ts = datetime.now(timezone.utc) - timedelta(days=60)
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend", timestamp=ts)
        weight = graph.effective_weight("alice", "bob")
        # 60 days with 30-day half-life → decay factor ~0.25
        assert weight < 0.5
        assert weight > 0.1

    def test_very_old_edge_near_zero(self, graph):
        """Very old edges approach zero weight."""
        ts = datetime.now(timezone.utc) - timedelta(days=365)
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend", timestamp=ts)
        weight = graph.effective_weight("alice", "bob")
        assert weight < 0.01


class TestNeighbors:
    def test_neighbors_includes_both_directions(self, graph):
        """Neighbors includes both incoming and outgoing connections."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("carol", "alice", "pr_comment", "acme-org/backend")

        neighbors = graph.neighbors("alice")
        assert "bob" in neighbors
        assert "carol" in neighbors

    def test_outgoing_only(self, graph):
        """Outgoing returns only edges FROM person."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("carol", "alice", "pr_comment", "acme-org/backend")

        out = graph.outgoing("alice")
        assert "bob" in out
        assert "carol" not in out

    def test_incoming_only(self, graph):
        """Incoming returns only edges TO person."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("carol", "alice", "pr_comment", "acme-org/backend")

        inc = graph.incoming("alice")
        assert "carol" in inc
        assert "bob" not in inc

    def test_no_neighbors(self, graph):
        """Person with no edges has empty neighbors."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        assert graph.neighbors("dave") == {}


class TestPersistence:
    def test_save_and_load(self, graph, resolver, tmp_path):
        """Graph survives save/load cycle."""
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("alice", "carol", "pr_comment", "acme-org/pricing")

        save_path = tmp_path / "graph.json"
        graph.save(save_path)

        # Load into new graph
        new_graph = CommunicationGraph(resolver)
        new_graph.load(save_path)

        assert new_graph.node_count == 3
        assert new_graph.edge_count == 2

    def test_load_missing_file(self, resolver, tmp_path):
        """Loading from non-existent file is a no-op."""
        g = CommunicationGraph(resolver)
        g.load(tmp_path / "nonexistent.json")
        assert g.node_count == 0

    def test_load_corrupt_file(self, resolver, tmp_path):
        """Loading from corrupt file is a no-op."""
        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("not valid json")
        g = CommunicationGraph(resolver)
        g.load(corrupt_path)
        assert g.node_count == 0

    def test_round_trip_preserves_data(self, graph, resolver, tmp_path):
        """All edge data survives round-trip."""
        ts = datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend", timestamp=ts)
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend", timestamp=ts)

        save_path = tmp_path / "graph.json"
        graph.save(save_path)

        new_graph = CommunicationGraph(resolver)
        new_graph.load(save_path)

        edges = new_graph.all_edges()
        assert len(edges) == 1
        assert edges[0].count == 2
        assert edges[0].weight == 2.0
        assert edges[0].channel == "acme-org/backend"


class TestEdgeCounts:
    def test_all_nodes(self, graph):
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("carol", "dave", "pr_comment", "acme-org/pricing")
        nodes = graph.all_nodes()
        assert set(nodes) == {"alice", "bob", "carol", "dave"}

    def test_all_edges(self, graph):
        graph.add_edge("alice", "bob", "pr_review", "acme-org/backend")
        graph.add_edge("carol", "dave", "pr_comment", "acme-org/pricing")
        edges = graph.all_edges()
        assert len(edges) == 2
