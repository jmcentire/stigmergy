"""Bulk graph construction — 10x faster than individual add_edge() calls.

Parses all data files into Python lists first, then builds the entire
igraph Graph in a single constructor call. Eliminates 3.3M+ Python-C
boundary crossings.

Usage:
    from stigmergy.ownership.bulk_ingest import build_graph_from_files
    graph = build_graph_from_files(psc_paths=[...], icij_dir=..., gleif_path=...)
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import igraph as ig
import orjson

from stigmergy.ownership.graph import (
    EdgeType,
    EntityNode,
    EntityType,
    OwnershipGraph,
    person_id,
)

logger = logging.getLogger(__name__)


@dataclass
class _BulkBuilder:
    """Accumulates nodes and edges in Python lists, then builds graph in one shot."""

    # Node data
    entity_ids: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    entity_types: list[str] = field(default_factory=list)
    entity_jurisdictions: list[str] = field(default_factory=list)
    entity_sources: list[str] = field(default_factory=list)

    # Edge data (as index pairs)
    edge_sources: list[int] = field(default_factory=list)
    edge_targets: list[int] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    edge_natures: list[str] = field(default_factory=list)
    edge_datasets: list[str] = field(default_factory=list)
    edge_percentages: list[float | None] = field(default_factory=list)

    # Index for dedup
    _id_to_idx: dict[str, int] = field(default_factory=dict)

    def add_node(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        jurisdiction: str = "",
        source: str = "",
    ) -> int:
        """Add a node. Returns index. Idempotent."""
        if entity_id in self._id_to_idx:
            return self._id_to_idx[entity_id]

        idx = len(self.entity_ids)
        self._id_to_idx[entity_id] = idx
        self.entity_ids.append(entity_id)
        self.entity_names.append(name)
        self.entity_types.append(entity_type)
        self.entity_jurisdictions.append(jurisdiction)
        self.entity_sources.append(source)
        return idx

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        nature: str = "",
        dataset: str = "",
        percentage: float | None = None,
    ) -> bool:
        """Add an edge by entity IDs. Returns False if either endpoint missing."""
        src_idx = self._id_to_idx.get(source_id)
        tgt_idx = self._id_to_idx.get(target_id)
        if src_idx is None or tgt_idx is None:
            return False

        self.edge_sources.append(src_idx)
        self.edge_targets.append(tgt_idx)
        self.edge_types.append(edge_type)
        self.edge_natures.append(nature)
        self.edge_datasets.append(dataset)
        self.edge_percentages.append(percentage)
        return True

    def has_node(self, entity_id: str) -> bool:
        return entity_id in self._id_to_idx

    @property
    def node_count(self) -> int:
        return len(self.entity_ids)

    @property
    def edge_count(self) -> int:
        return len(self.edge_sources)

    def build(self) -> OwnershipGraph:
        """Construct the OwnershipGraph in one shot."""
        n = len(self.entity_ids)
        edges = list(zip(self.edge_sources, self.edge_targets))

        logger.info(f"Building igraph: {n:,} nodes, {len(edges):,} edges...")

        g = ig.Graph(
            n=n,
            edges=edges,
            directed=True,
            vertex_attrs={
                "name": self.entity_ids,
                "label": self.entity_names,
                "entity_type": self.entity_types,
                "jurisdiction": self.entity_jurisdictions,
                "source": self.entity_sources,
            },
            edge_attrs={
                "edge_type": self.edge_types,
                "nature_of_control": self.edge_natures,
                "source_dataset": self.edge_datasets,
            },
        )

        og = OwnershipGraph()
        og._graph = g
        og._entity_index = dict(self._id_to_idx)  # copy
        # Note: og._entities not populated for bulk (metadata in vertex attrs)

        logger.info(f"Graph built: {og.node_count:,} nodes, {og.edge_count:,} edges")
        return og


# ─── PSC parsing ───


_PSC_KIND_MAP = {
    "individual-person-with-significant-control": "person",
    "corporate-entity-person-with-significant-control": "foreign_entity",
    "legal-person-person-with-significant-control": "foreign_entity",
    "individual-beneficial-owner": "person",
    "corporate-entity-beneficial-owner": "foreign_entity",
    "legal-person-beneficial-owner": "foreign_entity",
}

_OWNERSHIP_NATURES = {"ownership-of-shares", "voting-rights"}


def _parse_psc_into_builder(builder: _BulkBuilder, paths: list[Path], limit: int | None = None) -> dict:
    """Parse PSC ndjson files into the bulk builder."""
    stats = {"records": 0, "skipped_ceased": 0, "skipped_secure": 0, "skipped_unknown": 0}

    for path in paths:
        logger.info(f"Parsing PSC: {path.name}")
        with open(path, "rb") as f:
            for line in f:
                if limit and stats["records"] >= limit:
                    return stats

                line = line.strip()
                if not line:
                    continue

                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                company_number = record.get("company_number", "")
                data = record.get("data", {})
                if not company_number or not data:
                    continue

                stats["records"] += 1
                kind = data.get("kind", "")

                if "super-secure" in kind:
                    stats["skipped_secure"] += 1
                    continue
                if data.get("ceased_on"):
                    stats["skipped_ceased"] += 1
                    continue

                entity_type_str = _PSC_KIND_MAP.get(kind)
                if not entity_type_str:
                    stats["skipped_unknown"] += 1
                    continue

                # Ensure company node
                builder.add_node(company_number, company_number, "company", "GB", "uk_psc")

                # Create owner
                if entity_type_str == "person":
                    name_el = data.get("name_elements", {})
                    name = data.get("name", "")
                    if name_el:
                        parts = [name_el.get(k, "") for k in ("forename", "middle_name", "surname") if name_el.get(k)]
                        if parts:
                            name = " ".join(parts)

                    dob = data.get("date_of_birth", {})
                    owner_id = person_id(name, dob.get("month"), dob.get("year"))
                    country = data.get("country_of_residence", "") or data.get("nationality", "")
                    builder.add_node(owner_id, name, "person", country, "uk_psc")

                elif entity_type_str == "foreign_entity":
                    corp_name = data.get("name", "Unknown")
                    owner_id = f"corp_{hashlib.sha256(corp_name.strip().lower().encode()).hexdigest()[:16]}"
                    ident = data.get("identification", {})
                    jurisdiction = ident.get("country_registered", "") or ident.get("place_registered", "")
                    builder.add_node(owner_id, corp_name, "foreign_entity", jurisdiction, "uk_psc")
                else:
                    continue

                # Edge: owner -> company
                natures = data.get("natures_of_control") or []
                natures = [n for n in natures if n]  # filter None/empty
                is_ownership = any(k in n for n in natures for k in _OWNERSHIP_NATURES)
                edge_type = "ownership" if is_ownership else "control"
                builder.add_edge(owner_id, company_number, edge_type, "; ".join(natures), "uk_psc")

                if stats["records"] % 1_000_000 == 0:
                    logger.info(f"  PSC: {stats['records']:,} records, {builder.node_count:,} nodes")

    return stats


# ─── ICIJ parsing ───


_ICIJ_REL_MAP = {
    "officer of": "directorship",
    "shareholder of": "ownership",
    "beneficial owner of": "ownership",
    "director of": "directorship",
    "secretary of": "directorship",
    "nominee director of": "nominee_for",
    "nominee shareholder of": "nominee_for",
    "intermediary of": "registered_agent",
    "registered agent of": "registered_agent",
    "related entity of": "subsidiary",
    "connected to": "control",
    "same name and registration date as": "subsidiary",
    "similar name and address as": "subsidiary",
    "registered address": "registered_agent",
}


def _parse_icij_into_builder(builder: _BulkBuilder, icij_dir: Path) -> dict:
    """Parse ICIJ CSV files into the bulk builder."""
    stats = {"entities": 0, "officers": 0, "relationships": 0, "skipped_rels": 0}

    # Entities
    entities_path = icij_dir / "nodes-entities.csv"
    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                node_id = row.get("node_id", "").strip()
                if not node_id:
                    continue
                name = row.get("name", "").strip() or row.get("original_name", "").strip()
                jurisdiction = row.get("jurisdiction", "").strip()
                source_id = row.get("sourceID", "").strip()
                eid = f"icij_{source_id}_{node_id}"
                builder.add_node(eid, name, "foreign_entity", jurisdiction, f"icij_{source_id}")
                stats["entities"] += 1

        logger.info(f"  ICIJ entities: {stats['entities']:,}")

    # Officers
    officers_path = icij_dir / "nodes-officers.csv"
    if officers_path.exists():
        with open(officers_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                node_id = row.get("node_id", "").strip()
                if not node_id:
                    continue
                name = row.get("name", "").strip()
                country = row.get("country_codes", "").strip()
                source_id = row.get("sourceID", "").strip()
                eid = f"icij_{source_id}_{node_id}"
                builder.add_node(eid, name, "person", country, f"icij_{source_id}")
                stats["officers"] += 1

        logger.info(f"  ICIJ officers: {stats['officers']:,}")

    # Intermediaries
    intermediaries_path = icij_dir / "nodes-intermediaries.csv"
    if intermediaries_path.exists():
        with open(intermediaries_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                node_id = row.get("node_id", "").strip()
                if not node_id:
                    continue
                name = row.get("name", "").strip()
                country = row.get("country_codes", "").strip()
                source_id = row.get("sourceID", "").strip()
                eid = f"icij_{source_id}_{node_id}"
                builder.add_node(eid, name, "foreign_entity", country, f"icij_{source_id}")

    # Relationships
    rels_path = icij_dir / "relationships.csv"
    if rels_path.exists():
        with open(rels_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                start_id = row.get("node_id_start", row.get("START_ID", "")).strip()
                end_id = row.get("node_id_end", row.get("END_ID", "")).strip()
                rel_type = row.get("rel_type", row.get("TYPE", "")).strip().lower()
                source_id = row.get("sourceID", "").strip()

                if not start_id or not end_id:
                    continue

                src_eid = f"icij_{source_id}_{start_id}"
                tgt_eid = f"icij_{source_id}_{end_id}"

                edge_type = _ICIJ_REL_MAP.get(rel_type, "control")
                if builder.add_edge(src_eid, tgt_eid, edge_type, rel_type, f"icij_{source_id}"):
                    stats["relationships"] += 1
                else:
                    stats["skipped_rels"] += 1

                if stats["relationships"] % 500_000 == 0 and stats["relationships"] > 0:
                    logger.info(f"  ICIJ relationships: {stats['relationships']:,}")

        logger.info(f"  ICIJ relationships: {stats['relationships']:,} ({stats['skipped_rels']:,} skipped)")

    return stats


# ─── GLEIF parsing ───


_GLEIF_REL_MAP = {
    "IS_DIRECTLY_CONSOLIDATED_BY": "subsidiary",
    "IS_ULTIMATELY_CONSOLIDATED_BY": "subsidiary",
    "IS_INTERNATIONAL_BRANCH_OF": "subsidiary",
    "IS_FUND-MANAGED_BY": "control",
    "IS_SUBFUND_OF": "subsidiary",
    "IS_FEEDER_TO": "control",
}


def _parse_gleif_into_builder(builder: _BulkBuilder, path: Path) -> dict:
    """Parse GLEIF Level 2 CSV into the bulk builder."""
    stats = {"records": 0, "edges": 0, "skipped_inactive": 0}

    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stats["records"] += 1

            start_lei = row.get("Relationship.StartNode.NodeID", "").strip().strip('"')
            end_lei = row.get("Relationship.EndNode.NodeID", "").strip().strip('"')
            rel_type = row.get("Relationship.RelationshipType", "").strip().strip('"')
            rel_status = row.get("Relationship.RelationshipStatus", "").strip().strip('"')

            if not start_lei or not end_lei:
                continue
            if rel_status != "ACTIVE":
                stats["skipped_inactive"] += 1
                continue

            # Create LEI entities
            builder.add_node(f"lei_{start_lei}", start_lei, "company", "", "gleif")
            builder.add_node(f"lei_{end_lei}", end_lei, "company", "", "gleif")

            # EndNode owns/controls StartNode
            edge_type = _GLEIF_REL_MAP.get(rel_type, "subsidiary")
            builder.add_edge(f"lei_{end_lei}", f"lei_{start_lei}", edge_type, rel_type, "gleif")
            stats["edges"] += 1

    logger.info(f"  GLEIF: {stats['records']:,} records, {stats['edges']:,} edges")
    return stats


# ─── Main entry point ───


def build_graph_from_files(
    psc_paths: list[Path] | None = None,
    icij_dir: Path | None = None,
    gleif_path: Path | None = None,
    psc_limit: int | None = None,
) -> tuple[OwnershipGraph, dict[str, Any]]:
    """Build a complete ownership graph from all available data sources.

    Parses everything into memory lists, then constructs igraph in one shot.
    ~10x faster than individual add_edge() calls.

    Returns:
        (OwnershipGraph, stats_dict)
    """
    builder = _BulkBuilder()
    all_stats: dict[str, Any] = {}

    if psc_paths:
        logger.info("=== Parsing PSC data ===")
        all_stats["psc"] = _parse_psc_into_builder(builder, psc_paths, psc_limit)

    if icij_dir and icij_dir.exists():
        logger.info("=== Parsing ICIJ data ===")
        all_stats["icij"] = _parse_icij_into_builder(builder, icij_dir)

    if gleif_path and gleif_path.exists():
        logger.info("=== Parsing GLEIF data ===")
        all_stats["gleif"] = _parse_gleif_into_builder(builder, gleif_path)

    logger.info(f"=== Building graph: {builder.node_count:,} nodes, {builder.edge_count:,} edges ===")
    graph = builder.build()

    all_stats["total"] = {
        "nodes": graph.node_count,
        "edges": graph.edge_count,
    }

    return graph, all_stats
