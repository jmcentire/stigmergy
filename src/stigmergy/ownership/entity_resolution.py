"""Entity resolution between ICIJ Offshore Leaks and UK PSC data.

Simple normalized name matching as a first pass. Does NOT use the full
IdentityResolver (designed for org-scale, O(N*P) per resolution).

Strategy:
1. Exact match on normalized company name (lowercase, strip legal suffixes)
2. Report match count and confidence before incorporating into labels
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Legal suffixes to strip for normalization
_LEGAL_SUFFIXES = re.compile(
    r"\b(ltd|limited|plc|llp|lp|inc|incorporated|corp|corporation|"
    r"sa|ag|gmbh|bv|nv|srl|sl|pty|pte|co|company)\b\.?",
    re.IGNORECASE,
)

# Noise to strip
_NOISE = re.compile(r"[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")


def _normalize_name(name: str) -> str:
    """Normalize a company name for matching."""
    if not name:
        return ""
    name = name.lower()
    name = _LEGAL_SUFFIXES.sub("", name)
    name = _NOISE.sub(" ", name)
    name = _MULTI_SPACE.sub(" ", name).strip()
    return name


def resolve_icij_to_psc(
    icij_csv_dir: Path,
    company_csv_path: Path,
) -> pl.DataFrame:
    """Match ICIJ entities to UK Companies House records by normalized name.

    Returns DataFrame with columns:
    - icij_node_id, icij_name, icij_jurisdiction, icij_source
    - company_number, company_name
    - match_type ("exact_normalized")
    - normalized_name (the key used for matching)
    """
    # Load ICIJ entities
    logger.info("Loading ICIJ entities...")
    icij_entities = pl.read_csv(
        icij_csv_dir / "nodes-entities.csv",
        encoding="utf8-lossy",
    )
    logger.info(f"  {len(icij_entities):,} ICIJ entities")

    # Load UK company names
    logger.info("Loading UK company names...")
    companies = pl.read_csv(
        company_csv_path,
        columns=[" CompanyNumber", "CompanyName"],
        schema_overrides={" CompanyNumber": pl.Utf8, "CompanyName": pl.Utf8},
        ignore_errors=True,
        truncate_ragged_lines=True,
    )
    companies = companies.rename({" CompanyNumber": "CompanyNumber"})
    companies = companies.with_columns(pl.col("CompanyNumber").str.strip_chars())
    logger.info(f"  {len(companies):,} UK companies")

    # Normalize names
    logger.info("Normalizing names...")
    icij_names = icij_entities["name"].to_list()
    icij_normalized = [_normalize_name(n) for n in icij_names]

    company_names = companies["CompanyName"].to_list()
    company_normalized = [_normalize_name(n) for n in company_names]

    # Build lookup: normalized_name -> list of company_numbers
    logger.info("Building lookup index...")
    name_to_companies: dict[str, list[tuple[str, str]]] = {}
    company_numbers = companies["CompanyNumber"].to_list()
    for i, norm in enumerate(company_normalized):
        if norm and len(norm) > 3:  # skip very short names (noise)
            if norm not in name_to_companies:
                name_to_companies[norm] = []
            name_to_companies[norm].append((company_numbers[i], company_names[i]))

    logger.info(f"  {len(name_to_companies):,} unique normalized company names")

    # Match
    logger.info("Matching...")
    matches = []
    icij_node_ids = icij_entities["node_id"].to_list()
    icij_jurisdictions = icij_entities["jurisdiction"].to_list()
    icij_sources = icij_entities["sourceID"].to_list()

    for i, norm in enumerate(icij_normalized):
        if norm and norm in name_to_companies:
            for company_number, company_name in name_to_companies[norm]:
                matches.append({
                    "icij_node_id": str(icij_node_ids[i]),
                    "icij_name": icij_names[i],
                    "icij_jurisdiction": icij_jurisdictions[i] or "",
                    "icij_source": icij_sources[i] or "",
                    "company_number": company_number,
                    "company_name": company_name,
                    "match_type": "exact_normalized",
                    "normalized_name": norm,
                })

    result = pl.DataFrame(matches) if matches else pl.DataFrame(
        schema={
            "icij_node_id": pl.Utf8, "icij_name": pl.Utf8,
            "icij_jurisdiction": pl.Utf8, "icij_source": pl.Utf8,
            "company_number": pl.Utf8, "company_name": pl.Utf8,
            "match_type": pl.Utf8, "normalized_name": pl.Utf8,
        }
    )

    # Deduplicate: keep unique (icij_node_id, company_number) pairs
    result = result.unique(subset=["icij_node_id", "company_number"])

    logger.info(f"Matches: {len(result):,} (icij_entity, uk_company) pairs")
    logger.info(f"  Unique ICIJ entities matched: {result['icij_node_id'].n_unique():,}")
    logger.info(f"  Unique UK companies matched: {result['company_number'].n_unique():,}")

    # Source breakdown
    if len(result) > 0:
        source_counts = result.group_by("icij_source").agg(pl.len().alias("count"))
        for row in source_counts.iter_rows(named=True):
            logger.info(f"  {row['icij_source']}: {row['count']:,}")

    return result
