"""GLEIF Level 2 adapter — parses Golden Copy RR-CDF CSV.

Reads the GLEIF "Who Owns Whom" relationship records. Each record
describes a corporate ownership relationship between two LEI-identified
entities.

Download: https://www.gleif.org/en/lei-data/gleif-golden-copy
Format: CSV (Golden Copy RR-CDF v2.1)
License: CC0 1.0
"""

from __future__ import annotations

import csv
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

# GLEIF relationship types -> EdgeType
_GLEIF_REL_MAP = {
    "IS_DIRECTLY_CONSOLIDATED_BY": EdgeType.SUBSIDIARY,
    "IS_ULTIMATELY_CONSOLIDATED_BY": EdgeType.SUBSIDIARY,
    "IS_INTERNATIONAL_BRANCH_OF": EdgeType.SUBSIDIARY,
    "IS_FUND-MANAGED_BY": EdgeType.CONTROL,
    "IS_SUBFUND_OF": EdgeType.SUBSIDIARY,
    "IS_FEEDER_TO": EdgeType.CONTROL,
}


def ingest_gleif(graph: OwnershipGraph, path: Path, limit: int | None = None) -> dict:
    """Ingest GLEIF Level 2 relationship records into ownership graph.

    Args:
        graph: Target OwnershipGraph
        path: Path to GLEIF RR golden copy CSV
        limit: Optional record limit for testing

    Returns:
        Ingestion statistics dict
    """
    stats = {
        "records_processed": 0,
        "entities_added": 0,
        "edges_added": 0,
        "skipped_inactive": 0,
    }

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and stats["records_processed"] >= limit:
                break

            stats["records_processed"] += 1

            start_lei = row.get("Relationship.StartNode.NodeID", "").strip().strip('"')
            end_lei = row.get("Relationship.EndNode.NodeID", "").strip().strip('"')
            rel_type = row.get("Relationship.RelationshipType", "").strip().strip('"')
            rel_status = row.get("Relationship.RelationshipStatus", "").strip().strip('"')

            if not start_lei or not end_lei:
                continue

            # Skip inactive relationships
            if rel_status != "ACTIVE":
                stats["skipped_inactive"] += 1
                continue

            # Ensure both LEI entities exist
            for lei in (start_lei, end_lei):
                lei_id = f"lei_{lei}"
                if not graph.has_entity(lei_id):
                    graph.add_entity(EntityNode(
                        entity_id=lei_id,
                        name=lei,  # GLEIF RR doesn't include names; enrich later
                        entity_type=EntityType.COMPANY,
                        source="gleif",
                    ))
                    stats["entities_added"] += 1

            # The relationship direction: StartNode IS_*_BY EndNode
            # means EndNode owns/controls StartNode
            # So edge goes: EndNode -> StartNode (parent owns child)
            edge_type = _GLEIF_REL_MAP.get(rel_type, EdgeType.SUBSIDIARY)

            graph.add_edge(OwnershipEdge(
                source_id=f"lei_{end_lei}",
                target_id=f"lei_{start_lei}",
                edge_type=edge_type,
                nature_of_control=rel_type,
                source_dataset="gleif",
            ))
            stats["edges_added"] += 1

            if stats["records_processed"] % 100_000 == 0:
                logger.info(f"GLEIF: processed {stats['records_processed']:,} records")

    logger.info(
        f"GLEIF ingestion complete: {stats['records_processed']:,} records, "
        f"{stats['entities_added']:,} entities, {stats['edges_added']:,} edges"
    )
    return stats
