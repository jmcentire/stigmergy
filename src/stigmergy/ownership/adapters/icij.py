"""ICIJ Offshore Leaks adapter — parses CSV bulk download.

Reads the ICIJ Offshore Leaks database CSV files (entities, officers,
intermediaries, addresses, relationships).

Download: https://offshoreleaks.icij.org/pages/database
Format: CSV zip with separate files per node type + relationships
License: ODbL / CC-BY-SA
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

from stigmergy.ownership.graph import (
    EdgeType,
    EntityNode,
    EntityType,
    OwnershipEdge,
    OwnershipGraph,
)

logger = logging.getLogger(__name__)

# ICIJ relationship types -> EdgeType
_REL_TYPE_MAP = {
    "officer of": EdgeType.DIRECTORSHIP,
    "shareholder of": EdgeType.OWNERSHIP,
    "beneficial owner of": EdgeType.OWNERSHIP,
    "director of": EdgeType.DIRECTORSHIP,
    "secretary of": EdgeType.DIRECTORSHIP,
    "nominee director of": EdgeType.NOMINEE_FOR,
    "nominee shareholder of": EdgeType.NOMINEE_FOR,
    "intermediary of": EdgeType.REGISTERED_AGENT,
    "registered agent of": EdgeType.REGISTERED_AGENT,
    "related entity of": EdgeType.SUBSIDIARY,
    "similar name and address as": EdgeType.SUBSIDIARY,
    "connected to": EdgeType.CONTROL,
    "same name and registration date as": EdgeType.SUBSIDIARY,
}


def _icij_entity_id(node_id: str, source: str) -> str:
    """Generate a stable entity ID from ICIJ node_id."""
    return f"icij_{source}_{node_id}"


def ingest_icij_entities(graph: OwnershipGraph, path: Path) -> int:
    """Ingest ICIJ entities CSV (nodes-entities.csv).

    Returns number of entities added.
    """
    count = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row.get("node_id", "").strip()
            if not node_id:
                continue

            name = row.get("name", "").strip() or row.get("original_name", "").strip()
            jurisdiction = row.get("jurisdiction", "").strip()
            source_id = row.get("sourceID", "").strip()
            entity_id = _icij_entity_id(node_id, source_id or "unknown")

            if not graph.has_entity(entity_id):
                graph.add_entity(EntityNode(
                    entity_id=entity_id,
                    name=name,
                    entity_type=EntityType.FOREIGN_ENTITY,
                    jurisdiction=jurisdiction,
                    source=f"icij_{source_id}",
                ))
                count += 1

    logger.info(f"Ingested {count:,} ICIJ entities from {path.name}")
    return count


def ingest_icij_officers(graph: OwnershipGraph, path: Path) -> int:
    """Ingest ICIJ officers CSV (nodes-officers.csv).

    Returns number of officers added.
    """
    count = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row.get("node_id", "").strip()
            if not node_id:
                continue

            name = row.get("name", "").strip()
            country = row.get("country_codes", "").strip()
            source_id = row.get("sourceID", "").strip()
            entity_id = _icij_entity_id(node_id, source_id or "unknown")

            if not graph.has_entity(entity_id):
                graph.add_entity(EntityNode(
                    entity_id=entity_id,
                    name=name,
                    entity_type=EntityType.PERSON,
                    jurisdiction=country,
                    source=f"icij_{source_id}",
                ))
                count += 1

    logger.info(f"Ingested {count:,} ICIJ officers from {path.name}")
    return count


def ingest_icij_relationships(graph: OwnershipGraph, path: Path) -> int:
    """Ingest ICIJ relationships CSV (relationships.csv).

    Returns number of edges added.
    """
    count = 0
    skipped = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = row.get("START_ID", row.get("node_id_start", "")).strip()
            end_id = row.get("END_ID", row.get("node_id_end", "")).strip()
            rel_type = row.get("TYPE", row.get("rel_type", "")).strip().lower()
            source_id = row.get("sourceID", "").strip()

            if not start_id or not end_id:
                continue

            # Try both source-qualified and unqualified IDs
            src_entity = _icij_entity_id(start_id, source_id or "unknown")
            tgt_entity = _icij_entity_id(end_id, source_id or "unknown")

            if not graph.has_entity(src_entity) or not graph.has_entity(tgt_entity):
                skipped += 1
                continue

            edge_type = _REL_TYPE_MAP.get(rel_type, EdgeType.CONTROL)

            graph.add_edge(OwnershipEdge(
                source_id=src_entity,
                target_id=tgt_entity,
                edge_type=edge_type,
                nature_of_control=rel_type,
                source_dataset=f"icij_{source_id}",
            ))
            count += 1

    logger.info(f"Ingested {count:,} ICIJ relationships ({skipped:,} skipped) from {path.name}")
    return count
