"""Tests for bulk graph construction."""

import json
from pathlib import Path

import pytest

from stigmergy.ownership.bulk_ingest import (
    _BulkBuilder,
    build_graph_from_files,
)


class TestBulkBuilder:
    def test_add_nodes(self):
        b = _BulkBuilder()
        idx1 = b.add_node("c1", "Corp A", "company", "GB", "psc")
        idx2 = b.add_node("c2", "Corp B", "company", "US", "psc")
        assert idx1 == 0
        assert idx2 == 1
        assert b.node_count == 2

    def test_add_node_idempotent(self):
        b = _BulkBuilder()
        idx1 = b.add_node("c1", "Corp A", "company")
        idx2 = b.add_node("c1", "Corp A", "company")
        assert idx1 == idx2
        assert b.node_count == 1

    def test_add_edge(self):
        b = _BulkBuilder()
        b.add_node("p1", "Alice", "person")
        b.add_node("c1", "Corp", "company")
        ok = b.add_edge("p1", "c1", "ownership")
        assert ok is True
        assert b.edge_count == 1

    def test_add_edge_missing_node(self):
        b = _BulkBuilder()
        b.add_node("p1", "Alice", "person")
        ok = b.add_edge("p1", "missing", "ownership")
        assert ok is False
        assert b.edge_count == 0

    def test_build(self):
        b = _BulkBuilder()
        b.add_node("p1", "Alice", "person", "GB")
        b.add_node("p2", "Bob", "person", "US")
        b.add_node("c1", "Corp A", "company", "GB")
        b.add_node("c2", "Corp B", "company", "BVI")
        b.add_edge("p1", "c1", "ownership", "shares", "psc")
        b.add_edge("p1", "c2", "ownership", "shares", "psc")
        b.add_edge("p2", "c2", "control", "director", "icij")

        g = b.build()
        assert g.node_count == 4
        assert g.edge_count == 3
        assert g.has_entity("p1")
        assert g.has_entity("c2")
        assert set(g.owned_by("p1")) == {"c1", "c2"}

    def test_build_large(self):
        """Test bulk construction with 10K nodes — should be fast."""
        b = _BulkBuilder()
        # 1000 persons owning 10 companies each
        for p in range(1000):
            b.add_node(f"p{p}", f"Person {p}", "person")
            for c in range(10):
                cid = f"c{p}_{c}"
                b.add_node(cid, f"Corp {p}-{c}", "company")
                b.add_edge(f"p{p}", cid, "ownership")

        assert b.node_count == 11_000  # 1000 persons + 10,000 companies
        assert b.edge_count == 10_000

        g = b.build()
        assert g.node_count == 11_000
        assert g.edge_count == 10_000


class TestBuildFromFiles:
    def _make_psc_record(self, company_number, name, **kwargs):
        data = {
            "kind": "individual-person-with-significant-control",
            "name": name,
            "name_elements": {"forename": name.split()[0], "surname": name.split()[-1]},
            "natures_of_control": ["ownership-of-shares-25-to-50-percent"],
            "notified_on": "2020-01-15",
            "nationality": "British",
            "country_of_residence": "United Kingdom",
            "date_of_birth": {"month": 3, "year": 1980},
        }
        data.update(kwargs)
        return json.dumps({"company_number": company_number, "data": data}).encode() + b"\n"

    def test_psc_only(self, tmp_path):
        psc_file = tmp_path / "psc.json"
        psc_file.write_bytes(
            self._make_psc_record("11111111", "Alice Smith")
            + self._make_psc_record("22222222", "Bob Jones")
        )

        g, stats = build_graph_from_files(psc_paths=[psc_file])
        assert g.node_count == 4  # 2 companies + 2 persons
        assert g.edge_count == 2
        assert stats["psc"]["records"] == 2

    def test_icij_only(self, tmp_path):
        icij_dir = tmp_path / "icij"
        icij_dir.mkdir()
        (icij_dir / "nodes-entities.csv").write_text(
            "node_id,name,jurisdiction,sourceID\n"
            "1001,Offshore Corp,VGB,PP\n"
        )
        (icij_dir / "nodes-officers.csv").write_text(
            "node_id,name,country_codes,sourceID\n"
            "2001,John Doe,GBR,PP\n"
        )
        (icij_dir / "relationships.csv").write_text(
            "node_id_start,node_id_end,rel_type,sourceID\n"
            "2001,1001,officer of,PP\n"
        )

        g, stats = build_graph_from_files(icij_dir=icij_dir)
        assert g.node_count == 2
        assert g.edge_count == 1
        assert stats["icij"]["entities"] == 1
        assert stats["icij"]["officers"] == 1
        assert stats["icij"]["relationships"] == 1

    def test_gleif_only(self, tmp_path):
        gleif_file = tmp_path / "gleif.csv"
        gleif_file.write_text(
            '"Relationship.StartNode.NodeID","Relationship.StartNode.NodeIDType",'
            '"Relationship.EndNode.NodeID","Relationship.EndNode.NodeIDType",'
            '"Relationship.RelationshipType","Relationship.RelationshipStatus"\n'
            '"ABC","LEI","DEF","LEI","IS_DIRECTLY_CONSOLIDATED_BY","ACTIVE"\n'
        )

        g, stats = build_graph_from_files(gleif_path=gleif_file)
        assert g.node_count == 2
        assert g.edge_count == 1
        assert stats["gleif"]["edges"] == 1

    def test_combined(self, tmp_path):
        # PSC
        psc_file = tmp_path / "psc.json"
        psc_file.write_bytes(self._make_psc_record("11111111", "Alice Smith"))

        # ICIJ
        icij_dir = tmp_path / "icij"
        icij_dir.mkdir()
        (icij_dir / "nodes-entities.csv").write_text(
            "node_id,name,jurisdiction,sourceID\n1001,Shell Co,VGB,PP\n"
        )
        (icij_dir / "nodes-officers.csv").write_text(
            "node_id,name,country_codes,sourceID\n"
        )
        (icij_dir / "relationships.csv").write_text(
            "node_id_start,node_id_end,rel_type,sourceID\n"
        )

        # GLEIF
        gleif_file = tmp_path / "gleif.csv"
        gleif_file.write_text(
            '"Relationship.StartNode.NodeID","Relationship.StartNode.NodeIDType",'
            '"Relationship.EndNode.NodeID","Relationship.EndNode.NodeIDType",'
            '"Relationship.RelationshipType","Relationship.RelationshipStatus"\n'
            '"X1","LEI","X2","LEI","IS_DIRECTLY_CONSOLIDATED_BY","ACTIVE"\n'
        )

        g, stats = build_graph_from_files(
            psc_paths=[psc_file],
            icij_dir=icij_dir,
            gleif_path=gleif_file,
        )

        # 1 company + 1 person (PSC) + 1 entity (ICIJ) + 2 LEIs (GLEIF) = 5
        assert g.node_count == 5
        # 1 PSC edge + 0 ICIJ edges + 1 GLEIF edge = 2
        assert g.edge_count == 2
