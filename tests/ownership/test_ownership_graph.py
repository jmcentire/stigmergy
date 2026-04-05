"""Tests for the ownership graph and PSC/ICIJ adapters."""

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from stigmergy.ownership.graph import (
    EdgeType,
    EntityNode,
    EntityType,
    OwnershipEdge,
    OwnershipGraph,
    person_id,
)


# ─── OwnershipGraph basics ───


class TestOwnershipGraph:
    def test_add_entity(self):
        g = OwnershipGraph()
        entity = EntityNode(
            entity_id="12345678",
            name="ACME Holdings Ltd",
            entity_type=EntityType.COMPANY,
            jurisdiction="GB",
        )
        idx = g.add_entity(entity)
        assert g.node_count == 1
        assert g.has_entity("12345678")
        assert g.get_entity("12345678") == entity

    def test_add_entity_idempotent(self):
        g = OwnershipGraph()
        entity = EntityNode(entity_id="12345678", name="ACME", entity_type=EntityType.COMPANY)
        idx1 = g.add_entity(entity)
        idx2 = g.add_entity(entity)
        assert idx1 == idx2
        assert g.node_count == 1

    def test_add_edge(self):
        g = OwnershipGraph()
        g.add_entity(EntityNode(entity_id="person_abc", name="John", entity_type=EntityType.PERSON))
        g.add_entity(EntityNode(entity_id="12345678", name="ACME", entity_type=EntityType.COMPANY))

        g.add_edge(OwnershipEdge(
            source_id="person_abc",
            target_id="12345678",
            edge_type=EdgeType.OWNERSHIP,
            percentage=75.0,
        ))
        assert g.edge_count == 1

    def test_owners_of(self):
        g = OwnershipGraph()
        g.add_entity(EntityNode(entity_id="p1", name="Alice", entity_type=EntityType.PERSON))
        g.add_entity(EntityNode(entity_id="p2", name="Bob", entity_type=EntityType.PERSON))
        g.add_entity(EntityNode(entity_id="c1", name="Corp", entity_type=EntityType.COMPANY))

        g.add_edge(OwnershipEdge(source_id="p1", target_id="c1", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="p2", target_id="c1", edge_type=EdgeType.CONTROL))

        owners = g.owners_of("c1")
        assert set(owners) == {"p1", "p2"}

    def test_owned_by(self):
        g = OwnershipGraph()
        g.add_entity(EntityNode(entity_id="p1", name="Alice", entity_type=EntityType.PERSON))
        g.add_entity(EntityNode(entity_id="c1", name="Corp1", entity_type=EntityType.COMPANY))
        g.add_entity(EntityNode(entity_id="c2", name="Corp2", entity_type=EntityType.COMPANY))

        g.add_edge(OwnershipEdge(source_id="p1", target_id="c1", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="p1", target_id="c2", edge_type=EdgeType.OWNERSHIP))

        owned = g.owned_by("p1")
        assert set(owned) == {"c1", "c2"}

    def test_chain_depth(self):
        g = OwnershipGraph()
        # p -> c1 -> c2 -> c3 (depth 3 from p)
        for eid in ["p", "c1", "c2", "c3"]:
            g.add_entity(EntityNode(entity_id=eid, name=eid, entity_type=EntityType.COMPANY))
        g.add_edge(OwnershipEdge(source_id="p", target_id="c1", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="c1", target_id="c2", edge_type=EdgeType.SUBSIDIARY))
        g.add_edge(OwnershipEdge(source_id="c2", target_id="c3", edge_type=EdgeType.SUBSIDIARY))

        assert g.chain_depth("p") == 3
        assert g.chain_depth("c2") == 1
        assert g.chain_depth("c3") == 0

    def test_detect_cycles(self):
        g = OwnershipGraph()
        for eid in ["a", "b", "c"]:
            g.add_entity(EntityNode(entity_id=eid, name=eid, entity_type=EntityType.COMPANY))
        # a -> b -> c -> a (circular)
        g.add_edge(OwnershipEdge(source_id="a", target_id="b", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="b", target_id="c", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="c", target_id="a", edge_type=EdgeType.OWNERSHIP))

        cycles = g.detect_cycles()
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b", "c"}

    def test_no_cycles_in_tree(self):
        g = OwnershipGraph()
        for eid in ["root", "left", "right"]:
            g.add_entity(EntityNode(entity_id=eid, name=eid, entity_type=EntityType.COMPANY))
        g.add_edge(OwnershipEdge(source_id="root", target_id="left", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="root", target_id="right", edge_type=EdgeType.OWNERSHIP))

        cycles = g.detect_cycles()
        assert len(cycles) == 0

    def test_connected_components(self):
        g = OwnershipGraph()
        for eid in ["a", "b", "c", "d"]:
            g.add_entity(EntityNode(entity_id=eid, name=eid, entity_type=EntityType.COMPANY))
        g.add_edge(OwnershipEdge(source_id="a", target_id="b", edge_type=EdgeType.OWNERSHIP))
        # c and d are isolated from a-b

        components = g.connected_components()
        assert len(components) == 3  # {a,b}, {c}, {d}

    def test_summary(self):
        g = OwnershipGraph()
        for eid in ["p1", "c1", "c2"]:
            g.add_entity(EntityNode(entity_id=eid, name=eid, entity_type=EntityType.COMPANY))
        g.add_edge(OwnershipEdge(source_id="p1", target_id="c1", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="p1", target_id="c2", edge_type=EdgeType.OWNERSHIP))

        s = g.summary(include_cycles=True)
        assert s["nodes"] == 3
        assert s["edges"] == 2
        assert s["components"] == 1
        assert s["cycles"] == 0

    def test_save_load_roundtrip(self, tmp_path):
        g = OwnershipGraph()
        g.add_entity(EntityNode(entity_id="c1", name="Corp", entity_type=EntityType.COMPANY, jurisdiction="GB"))
        g.add_entity(EntityNode(entity_id="p1", name="Alice", entity_type=EntityType.PERSON))
        g.add_edge(OwnershipEdge(source_id="p1", target_id="c1", edge_type=EdgeType.OWNERSHIP))

        path = tmp_path / "test_graph.pkl"
        g.save(path)

        g2 = OwnershipGraph.load(path)
        assert g2.node_count == 2
        assert g2.edge_count == 1
        assert g2.has_entity("c1")
        assert g2.has_entity("p1")

    def test_betweenness_centrality(self):
        g = OwnershipGraph()
        # Star: p -> c1, p -> c2, p -> c3
        for eid in ["p", "c1", "c2", "c3"]:
            g.add_entity(EntityNode(entity_id=eid, name=eid, entity_type=EntityType.COMPANY))
        g.add_edge(OwnershipEdge(source_id="p", target_id="c1", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="p", target_id="c2", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="p", target_id="c3", edge_type=EdgeType.OWNERSHIP))

        bc = g.betweenness_centrality()
        assert "p" in bc
        # Center of star should have highest betweenness
        assert bc["p"] >= bc["c1"]

    def test_clustering_coefficients(self):
        g = OwnershipGraph()
        for eid in ["a", "b", "c"]:
            g.add_entity(EntityNode(entity_id=eid, name=eid, entity_type=EntityType.COMPANY))
        g.add_edge(OwnershipEdge(source_id="a", target_id="b", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="b", target_id="c", edge_type=EdgeType.OWNERSHIP))
        g.add_edge(OwnershipEdge(source_id="c", target_id="a", edge_type=EdgeType.OWNERSHIP))

        cc = g.clustering_coefficients()
        assert len(cc) == 3


# ─── person_id ───


class TestPersonId:
    def test_deterministic(self):
        assert person_id("John Smith", 3, 1980) == person_id("John Smith", 3, 1980)

    def test_name_normalization(self):
        assert person_id("  John Smith  ", 3, 1980) == person_id("John Smith", 3, 1980)

    def test_different_dob_different_id(self):
        assert person_id("John Smith", 3, 1980) != person_id("John Smith", 5, 1985)

    def test_no_dob(self):
        pid = person_id("Jane Doe")
        assert pid.startswith("person_")
        assert len(pid) == 7 + 16  # "person_" + 16 hex chars


# ─── PSC adapter ───


class TestPSCAdapter:
    def _make_psc_record(
        self,
        company_number: str = "12345678",
        kind: str = "individual-person-with-significant-control",
        name: str = "John Smith",
        natures: list[str] | None = None,
        ceased: bool = False,
    ) -> bytes:
        """Create a minimal PSC ndjson record."""
        data = {
            "kind": kind,
            "name": name,
            "name_elements": {"forename": name.split()[0], "surname": name.split()[-1]},
            "natures_of_control": natures or ["ownership-of-shares-25-to-50-percent"],
            "notified_on": "2020-01-15",
            "nationality": "British",
            "country_of_residence": "United Kingdom",
            "date_of_birth": {"month": 3, "year": 1980},
        }
        if ceased:
            data["ceased_on"] = "2023-06-01"

        record = {"company_number": company_number, "data": data}
        return json.dumps(record).encode() + b"\n"

    def test_ingest_single_person(self, tmp_path):
        from stigmergy.ownership.adapters.psc import ingest_psc

        psc_file = tmp_path / "psc.json"
        psc_file.write_bytes(self._make_psc_record())

        g = OwnershipGraph()
        stats = ingest_psc(g, [psc_file])

        assert stats["records_processed"] == 1
        assert stats["companies_added"] == 1
        assert stats["persons_added"] == 1
        assert stats["edges_added"] == 1
        assert g.node_count == 2  # company + person
        assert g.edge_count == 1

    def test_skip_ceased(self, tmp_path):
        from stigmergy.ownership.adapters.psc import ingest_psc

        psc_file = tmp_path / "psc.json"
        psc_file.write_bytes(self._make_psc_record(ceased=True))

        g = OwnershipGraph()
        stats = ingest_psc(g, [psc_file])

        assert stats["skipped_ceased"] == 1
        assert stats["edges_added"] == 0

    def test_skip_super_secure(self, tmp_path):
        from stigmergy.ownership.adapters.psc import ingest_psc

        psc_file = tmp_path / "psc.json"
        psc_file.write_bytes(self._make_psc_record(
            kind="super-secure-person-with-significant-control"
        ))

        g = OwnershipGraph()
        stats = ingest_psc(g, [psc_file])

        assert stats["skipped_super_secure"] == 1
        assert stats["edges_added"] == 0

    def test_corporate_owner(self, tmp_path):
        from stigmergy.ownership.adapters.psc import ingest_psc

        data = {
            "kind": "corporate-entity-person-with-significant-control",
            "name": "Offshore Holdings BVI Ltd",
            "natures_of_control": ["ownership-of-shares-75-to-100-percent"],
            "notified_on": "2019-05-20",
            "identification": {
                "country_registered": "British Virgin Islands",
                "legal_authority": "BVI Business Companies Act",
            },
        }
        record = json.dumps({"company_number": "99887766", "data": data}).encode() + b"\n"
        psc_file = tmp_path / "psc.json"
        psc_file.write_bytes(record)

        g = OwnershipGraph()
        stats = ingest_psc(g, [psc_file])

        assert stats["corporate_owners_added"] == 1
        assert g.node_count == 2  # company + corporate owner
        assert g.edge_count == 1

    def test_multiple_records(self, tmp_path):
        from stigmergy.ownership.adapters.psc import ingest_psc

        records = b""
        records += self._make_psc_record(company_number="11111111", name="Alice Brown")
        records += self._make_psc_record(company_number="22222222", name="Bob Green")
        records += self._make_psc_record(company_number="11111111", name="Carol White")

        psc_file = tmp_path / "psc.json"
        psc_file.write_bytes(records)

        g = OwnershipGraph()
        stats = ingest_psc(g, [psc_file])

        assert stats["records_processed"] == 3
        assert stats["companies_added"] == 2  # two distinct companies
        assert stats["persons_added"] == 3  # three distinct persons
        assert stats["edges_added"] == 3

    def test_limit(self, tmp_path):
        from stigmergy.ownership.adapters.psc import ingest_psc

        records = b""
        for i in range(10):
            records += self._make_psc_record(
                company_number=f"{i:08d}",
                name=f"Person {i}",
            )

        psc_file = tmp_path / "psc.json"
        psc_file.write_bytes(records)

        g = OwnershipGraph()
        stats = ingest_psc(g, [psc_file], limit=3)

        assert stats["records_processed"] == 3


# ─── ICIJ adapter ───


class TestICIJAdapter:
    def test_ingest_entities(self, tmp_path):
        from stigmergy.ownership.adapters.icij import ingest_icij_entities

        csv_content = (
            "node_id,name,jurisdiction,sourceID\n"
            "1001,Bluewater Holdings,VGB,Panama Papers\n"
            "1002,Sunset Investments,PAN,Panama Papers\n"
        )
        csv_file = tmp_path / "entities.csv"
        csv_file.write_text(csv_content)

        g = OwnershipGraph()
        count = ingest_icij_entities(g, csv_file)

        assert count == 2
        assert g.node_count == 2

    def test_ingest_officers(self, tmp_path):
        from stigmergy.ownership.adapters.icij import ingest_icij_officers

        csv_content = (
            "node_id,name,country_codes,sourceID\n"
            "2001,John Smith,GBR,Panama Papers\n"
        )
        csv_file = tmp_path / "officers.csv"
        csv_file.write_text(csv_content)

        g = OwnershipGraph()
        count = ingest_icij_officers(g, csv_file)

        assert count == 1

    def test_ingest_relationships(self, tmp_path):
        from stigmergy.ownership.adapters.icij import (
            ingest_icij_entities,
            ingest_icij_officers,
            ingest_icij_relationships,
        )

        # Create entities and officers first
        entities_csv = "node_id,name,jurisdiction,sourceID\n1001,Corp A,VGB,PP\n"
        officers_csv = "node_id,name,country_codes,sourceID\n2001,John,GBR,PP\n"
        rels_csv = "START_ID,END_ID,TYPE,sourceID\n2001,1001,officer of,PP\n"

        (tmp_path / "entities.csv").write_text(entities_csv)
        (tmp_path / "officers.csv").write_text(officers_csv)
        (tmp_path / "rels.csv").write_text(rels_csv)

        g = OwnershipGraph()
        ingest_icij_entities(g, tmp_path / "entities.csv")
        ingest_icij_officers(g, tmp_path / "officers.csv")
        count = ingest_icij_relationships(g, tmp_path / "rels.csv")

        assert count == 1
        assert g.edge_count == 1


# ─── GLEIF adapter ───


class TestGLEIFAdapter:
    def test_ingest_gleif(self, tmp_path):
        from stigmergy.ownership.adapters.gleif import ingest_gleif

        csv_content = (
            '"Relationship.StartNode.NodeID","Relationship.StartNode.NodeIDType",'
            '"Relationship.EndNode.NodeID","Relationship.EndNode.NodeIDType",'
            '"Relationship.RelationshipType","Relationship.RelationshipStatus"\n'
            '"ABC123","LEI","DEF456","LEI","IS_DIRECTLY_CONSOLIDATED_BY","ACTIVE"\n'
            '"GHI789","LEI","DEF456","LEI","IS_ULTIMATELY_CONSOLIDATED_BY","ACTIVE"\n'
            '"JKL012","LEI","MNO345","LEI","IS_DIRECTLY_CONSOLIDATED_BY","INACTIVE"\n'
        )
        csv_file = tmp_path / "gleif.csv"
        csv_file.write_text(csv_content)

        g = OwnershipGraph()
        stats = ingest_gleif(g, csv_file)

        assert stats["records_processed"] == 3
        assert stats["edges_added"] == 2  # 1 inactive skipped
        assert stats["skipped_inactive"] == 1
        # DEF456 -> ABC123 and DEF456 -> GHI789 (parent owns child)
        assert g.node_count == 3  # ABC123, DEF456, GHI789 (inactive row skipped before entity creation)
        owned = g.owned_by("lei_DEF456")
        assert set(owned) == {"lei_ABC123", "lei_GHI789"}

    def test_gleif_limit(self, tmp_path):
        from stigmergy.ownership.adapters.gleif import ingest_gleif

        csv_content = (
            '"Relationship.StartNode.NodeID","Relationship.StartNode.NodeIDType",'
            '"Relationship.EndNode.NodeID","Relationship.EndNode.NodeIDType",'
            '"Relationship.RelationshipType","Relationship.RelationshipStatus"\n'
            '"A1","LEI","B1","LEI","IS_DIRECTLY_CONSOLIDATED_BY","ACTIVE"\n'
            '"A2","LEI","B2","LEI","IS_DIRECTLY_CONSOLIDATED_BY","ACTIVE"\n'
            '"A3","LEI","B3","LEI","IS_DIRECTLY_CONSOLIDATED_BY","ACTIVE"\n'
        )
        csv_file = tmp_path / "gleif.csv"
        csv_file.write_text(csv_content)

        g = OwnershipGraph()
        stats = ingest_gleif(g, csv_file, limit=2)

        assert stats["records_processed"] == 2
        assert stats["edges_added"] == 2
