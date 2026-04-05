"""UK PSC Register adapter — stream-parses Companies House bulk data.

Reads newline-delimited JSON from Companies House PSC snapshot files.
Each record becomes one or more entities + edges in the OwnershipGraph.

Uses orjson for 2-3x faster parsing. Stream-processes line-by-line
to handle 6GB+ files without OOM.

Download: https://download.companieshouse.gov.uk/en_pscdata.html
Format: ndjson, ~10M records, ~31 parts zipped
License: Open Government Licence v3.0
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Iterator

import hashlib

import orjson

from stigmergy.ownership.graph import (
    EdgeType,
    EntityNode,
    EntityType,
    OwnershipEdge,
    OwnershipGraph,
    person_id,
)

logger = logging.getLogger(__name__)

# PSC kind -> entity type mapping
_KIND_MAP = {
    "individual-person-with-significant-control": EntityType.PERSON,
    "corporate-entity-person-with-significant-control": EntityType.FOREIGN_ENTITY,
    "legal-person-person-with-significant-control": EntityType.FOREIGN_ENTITY,
    "super-secure-person-with-significant-control": EntityType.UNKNOWN,
    "individual-beneficial-owner": EntityType.PERSON,
    "corporate-entity-beneficial-owner": EntityType.FOREIGN_ENTITY,
    "legal-person-beneficial-owner": EntityType.FOREIGN_ENTITY,
    "super-secure-beneficial-owner": EntityType.UNKNOWN,
}

# Nature of control -> edge type mapping
_OWNERSHIP_NATURES = {
    "ownership-of-shares",
    "voting-rights",
}

_CONTROL_NATURES = {
    "right-to-appoint-and-remove-directors",
    "significant-influence-or-control",
    "right-to-appoint-and-remove-members",
    "right-to-share-surplus-assets",
}


def _parse_date(s: str | None) -> date | None:
    """Parse YYYY-MM-DD date string."""
    if not s:
        return None
    try:
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def _classify_natures(natures: list[str]) -> EdgeType:
    """Determine edge type from PSC natures_of_control list."""
    for n in natures:
        for ownership_key in _OWNERSHIP_NATURES:
            if ownership_key in n:
                return EdgeType.OWNERSHIP
    for n in natures:
        for control_key in _CONTROL_NATURES:
            if control_key in n:
                return EdgeType.CONTROL
    return EdgeType.OWNERSHIP  # default


def _extract_person_name(data: dict) -> str:
    """Extract person name from PSC record."""
    name_elements = data.get("name_elements", {})
    if name_elements:
        parts = []
        if name_elements.get("title"):
            parts.append(name_elements["title"])
        if name_elements.get("forename"):
            parts.append(name_elements["forename"])
        if name_elements.get("middle_name"):
            parts.append(name_elements["middle_name"])
        if name_elements.get("surname"):
            parts.append(name_elements["surname"])
        if parts:
            return " ".join(parts)
    return data.get("name", "Unknown")


def stream_psc_file(path: Path) -> Iterator[tuple[str, dict]]:
    """Stream PSC records from a single ndjson file.

    Yields (company_number, data_dict) tuples.
    """
    with open(path, "rb") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError:
                logger.warning(f"Invalid JSON at {path}:{line_num}, skipping")
                continue

            company_number = record.get("company_number", "")
            data = record.get("data", {})
            if company_number and data:
                yield company_number, data

            if line_num % 500_000 == 0:
                logger.info(f"Processed {line_num:,} records from {path.name}")


def ingest_psc(graph: OwnershipGraph, paths: list[Path], limit: int | None = None) -> dict:
    """Ingest PSC files into the ownership graph.

    Args:
        graph: Target OwnershipGraph
        paths: List of PSC ndjson file paths (can be multiple parts)
        limit: Optional record limit for testing

    Returns:
        Ingestion statistics dict
    """
    stats = {
        "records_processed": 0,
        "companies_added": 0,
        "persons_added": 0,
        "corporate_owners_added": 0,
        "edges_added": 0,
        "skipped_super_secure": 0,
        "skipped_ceased": 0,
        "errors": 0,
    }

    for path in paths:
        logger.info(f"Ingesting PSC data from {path.name}")

        for company_number, data in stream_psc_file(path):
            if limit and stats["records_processed"] >= limit:
                break

            stats["records_processed"] += 1
            kind = data.get("kind", "")

            # Skip super-secure (no useful data)
            if "super-secure" in kind:
                stats["skipped_super_secure"] += 1
                continue

            # Skip ceased PSCs (for current-state graph)
            if data.get("ceased_on"):
                stats["skipped_ceased"] += 1
                continue

            entity_type = _KIND_MAP.get(kind, EntityType.UNKNOWN)
            natures = data.get("natures_of_control", [])
            notified_on = _parse_date(data.get("notified_on"))

            # Ensure company node exists
            if not graph.has_entity(company_number):
                graph.add_entity(EntityNode(
                    entity_id=company_number,
                    name=company_number,  # will be enriched later
                    entity_type=EntityType.COMPANY,
                    jurisdiction="GB",
                    source="uk_psc",
                ))
                stats["companies_added"] += 1

            # Create owner entity
            if entity_type == EntityType.PERSON:
                name = _extract_person_name(data)
                dob = data.get("date_of_birth", {})
                owner_id = person_id(
                    name,
                    dob_month=dob.get("month"),
                    dob_year=dob.get("year"),
                )
                nationality = data.get("nationality", "")
                country = data.get("country_of_residence", "")

                if not graph.has_entity(owner_id):
                    graph.add_entity(EntityNode(
                        entity_id=owner_id,
                        name=name,
                        entity_type=EntityType.PERSON,
                        jurisdiction=country or nationality or "",
                        source="uk_psc",
                    ))
                    stats["persons_added"] += 1

            elif entity_type == EntityType.FOREIGN_ENTITY:
                corp_name = data.get("name", "Unknown Corporate Entity")
                # Use name hash as ID for corporate owners (no company number)
                owner_id = f"corp_{hashlib.sha256(corp_name.strip().lower().encode()).hexdigest()[:16]}"

                if not graph.has_entity(owner_id):
                    identification = data.get("identification", {})
                    jurisdiction = (
                        identification.get("country_registered", "")
                        or identification.get("place_registered", "")
                    )
                    graph.add_entity(EntityNode(
                        entity_id=owner_id,
                        name=corp_name,
                        entity_type=EntityType.FOREIGN_ENTITY,
                        jurisdiction=jurisdiction,
                        source="uk_psc",
                    ))
                    stats["corporate_owners_added"] += 1
            else:
                continue

            # Create ownership/control edge: owner -> company
            edge_type = _classify_natures(natures)
            nature_str = "; ".join(natures)

            graph.add_edge(OwnershipEdge(
                source_id=owner_id,
                target_id=company_number,
                edge_type=edge_type,
                nature_of_control=nature_str,
                notified_on=notified_on,
                source_dataset="uk_psc",
            ))
            stats["edges_added"] += 1

        if limit and stats["records_processed"] >= limit:
            break

    logger.info(
        f"PSC ingestion complete: {stats['records_processed']:,} records, "
        f"{stats['companies_added']:,} companies, {stats['persons_added']:,} persons, "
        f"{stats['edges_added']:,} edges"
    )
    return stats
