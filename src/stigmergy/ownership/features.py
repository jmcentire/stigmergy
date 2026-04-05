"""Feature extraction for beneficial ownership detection.

Extracts per-company features from the OwnershipGraph in two tiers:
- Tier 1: Known indicators (regulatory status quo, 3 features)
- Tier 2: Novel topological (the core hypothesis, 6 independent features)

DROPPED (zero variance in PSC data — see advocate review):
- chain_depth: always 1 (PSC only records direct controllers)
- is_circular: always 0 (no SCCs in PSC graph)
- no_bo_declaration: always 0 (we only extract companies with PSC records)
- out_degree: always 0 (PSC records who controls the company, not what it controls)
- clustering_coefficient: always 0 (tree-structured ownership, no triangles)
- multi_source_count: always 1 (entity resolution not yet performed)

DROPPED (redundant — see advocate review):
- boundary_exits: identical to foreign_owner_count
- component_isolation: linear transform of component_size
- person_owner_count: in_degree - foreign_owner_count

Uses igraph bulk operations wherever possible to handle 10M+ companies.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from stigmergy.ownership.graph import OwnershipGraph

logger = logging.getLogger(__name__)

# Tax Justice Network Financial Secrecy Index — top secrecy jurisdictions
SECRECY_JURISDICTIONS = frozenset({
    "VGB", "BVI", "British Virgin Islands",
    "CYM", "KY", "Cayman Islands",
    "PAN", "Panama",
    "BMU", "BM", "Bermuda",
    "JEY", "JE", "Jersey",
    "GGY", "GG", "Guernsey",
    "IMN", "IM", "Isle of Man",
    "LIE", "LI", "Liechtenstein",
    "MCO", "MC", "Monaco",
    "LUX", "LU", "Luxembourg",
    "CHE", "CH", "Switzerland",
    "SGP", "SG", "Singapore",
    "HKG", "HK", "Hong Kong",
    "ARE", "AE", "United Arab Emirates",
    "BHS", "BS", "Bahamas",
    "MUS", "MU", "Mauritius",
    "WSM", "WS", "Samoa",
    "VUT", "VU", "Vanuatu",
    "SYC", "SC", "Seychelles",
    "BLZ", "BZ", "Belize",
    "ANT", "Anguilla", "AIA",
    "GIB", "GI", "Gibraltar",
    "MLT", "MT", "Malta",
    "CYP", "CY", "Cyprus",
    "NLD", "NL", "Netherlands",
    "IRL", "IE", "Ireland",
    "USA", "US", "United States", "Delaware",
})

NOMINEE_KEYWORDS = frozenset({
    "trust", "foundation", "anstalt", "stiftung",
    "settlement", "nominee", "fiduciary",
    "trustee", "nominees",
})

# Generic company name words (high frequency in shell/shelf companies)
GENERIC_NAME_WORDS = frozenset({
    "limited", "ltd", "plc", "llp", "lp", "inc", "corp", "corporation",
    "holdings", "holding", "group", "services", "solutions", "consulting",
    "investments", "investment", "trading", "properties", "property",
    "management", "international", "global", "capital", "ventures",
    "partners", "associates", "enterprises", "developments", "development",
    "resources", "systems", "technologies", "technology", "uk", "the",
    "and", "of", "in", "for", "co", "company",
})


def extract_features(
    graph: OwnershipGraph,
    target_source: str = "uk_psc",
    sample_size: int | None = None,
) -> pl.DataFrame:
    """Extract per-company feature vectors using bulk igraph operations.

    Returns DataFrame with 9 features (3 Tier 1 + 6 Tier 2), all with nonzero
    variance. No redundant or dead features.
    """
    g = graph._graph
    n = g.vcount()

    # ── Step 1: Identify target companies ──
    logger.info("Step 1: Identifying target companies...")
    entity_types = g.vs["entity_type"]
    sources = g.vs["source"]
    company_indices = np.array([
        i for i in range(n)
        if entity_types[i] == "company" and sources[i] == target_source
    ], dtype=np.int64)
    logger.info(f"  {len(company_indices):,} target companies")

    if sample_size and sample_size < len(company_indices):
        rng = np.random.default_rng(42)
        company_indices = rng.choice(company_indices, sample_size, replace=False)
        company_indices.sort()
        logger.info(f"  Sampled to {len(company_indices):,}")

    # ── Step 2: Bulk graph metrics (vectorized, C backend) ──
    logger.info("Step 2: Computing bulk graph metrics...")

    all_out_deg = np.array(g.outdegree(), dtype=np.int32)
    logger.info("  Degrees computed")

    wcc = g.connected_components(mode="weak")
    wcc_membership = np.array(wcc.membership, dtype=np.int32)
    wcc_sizes = np.array(wcc.sizes(), dtype=np.int64)
    logger.info(f"  WCC: {len(wcc_sizes):,} components")

    # ── Step 3: Pre-compute per-vertex metadata arrays ──
    logger.info("Step 3: Pre-computing metadata arrays...")
    jurisdictions = g.vs["jurisdiction"]
    labels = g.vs["label"]

    is_secrecy = np.array([j in SECRECY_JURISDICTIONS for j in jurisdictions], dtype=np.bool_)
    is_foreign = np.array([t == "foreign_entity" for t in entity_types], dtype=np.bool_)

    is_nominee = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        name = labels[i]
        if name:
            name_lower = name.lower()
            if any(k in name_lower for k in NOMINEE_KEYWORDS):
                is_nominee[i] = True

    logger.info("  Metadata arrays ready")

    # ── Step 4: Extract per-company features ──
    logger.info("Step 4: Extracting per-company features...")
    total = len(company_indices)

    entity_ids = []
    # Tier 1 (3 features with variance)
    t1_jurisdiction_count = np.zeros(total, dtype=np.int32)
    t1_secrecy_count = np.zeros(total, dtype=np.int32)
    t1_has_nominee = np.zeros(total, dtype=np.int8)
    # Tier 2 (6 independent features)
    t2_in_degree = np.zeros(total, dtype=np.int32)
    t2_component_size = np.zeros(total, dtype=np.int64)
    t2_max_owner_fanout = np.zeros(total, dtype=np.int32)
    t2_foreign_ratio = np.zeros(total, dtype=np.float32)
    t2_foreign_owner_count = np.zeros(total, dtype=np.int32)
    t2_secrecy_owner_ratio = np.zeros(total, dtype=np.float32)

    for idx, ci in enumerate(company_indices):
        if idx > 0 and idx % 500_000 == 0:
            logger.info(f"  Progress: {idx:,}/{total:,} ({100*idx/total:.0f}%)")

        ci = int(ci)
        entity_ids.append(g.vs[ci]["name"])

        in_nbrs = g.neighbors(ci, mode="in")
        in_deg = len(in_nbrs)

        owner_jurisdictions = set()
        secrecy = 0
        foreign_cnt = 0
        nominee_found = False
        max_fanout = 0

        for ni in in_nbrs:
            j = jurisdictions[ni]
            if j:
                owner_jurisdictions.add(j)
            if is_secrecy[ni]:
                secrecy += 1
            if is_foreign[ni]:
                foreign_cnt += 1
            if is_nominee[ni]:
                nominee_found = True
            fo = all_out_deg[ni]
            if fo > max_fanout:
                max_fanout = fo

        # Tier 1
        t1_jurisdiction_count[idx] = len(owner_jurisdictions)
        t1_secrecy_count[idx] = secrecy
        t1_has_nominee[idx] = 1 if nominee_found else 0

        # Tier 2
        t2_in_degree[idx] = in_deg
        t2_component_size[idx] = int(wcc_sizes[wcc_membership[ci]])
        t2_max_owner_fanout[idx] = max_fanout
        t2_foreign_ratio[idx] = foreign_cnt / in_deg if in_deg > 0 else 0
        t2_foreign_owner_count[idx] = foreign_cnt
        t2_secrecy_owner_ratio[idx] = secrecy / in_deg if in_deg > 0 else 0

    logger.info(f"Feature extraction complete: {total:,} companies")

    df = pl.DataFrame({
        "entity_id": entity_ids,
        # Tier 1: Known indicators
        "t1_jurisdiction_count": t1_jurisdiction_count,
        "t1_secrecy_jurisdiction_count": t1_secrecy_count,
        "t1_has_nominee_owner": t1_has_nominee,
        # Tier 2: Novel topological
        "t2_in_degree": t2_in_degree,
        "t2_component_size": t2_component_size,
        "t2_max_owner_fanout": t2_max_owner_fanout,
        "t2_foreign_ratio": t2_foreign_ratio,
        "t2_foreign_owner_count": t2_foreign_owner_count,
        "t2_secrecy_owner_ratio": t2_secrecy_owner_ratio,
    })

    return df


def enrich_with_company_data(
    features: pl.DataFrame,
    company_csv_path: Path,
) -> pl.DataFrame:
    """Enrich feature set with Companies House BasicCompanyData.

    Adds naming features and address concentration. Only matches active
    companies (~52% coverage). Naming features are null for unmatched rows.

    Args:
        features: Output of extract_features()
        company_csv_path: Path to BasicCompanyDataAsOneFile CSV

    Returns:
        Enriched DataFrame with additional columns
    """
    logger.info("Loading BasicCompanyData...")
    companies = pl.read_csv(
        company_csv_path,
        columns=[" CompanyNumber", "CompanyName", "CompanyStatus",
                 "IncorporationDate", "RegAddress.AddressLine1",
                 "RegAddress.PostCode"],
        schema_overrides={" CompanyNumber": pl.Utf8, "CompanyName": pl.Utf8},
        ignore_errors=True,
        truncate_ragged_lines=True,
    )
    companies = companies.rename({" CompanyNumber": "CompanyNumber"})
    companies = companies.with_columns(pl.col("CompanyNumber").str.strip_chars())
    logger.info(f"  {len(companies):,} companies loaded")

    # Join
    enriched = features.join(
        companies.select(["CompanyNumber", "CompanyName", "CompanyStatus",
                          "RegAddress.AddressLine1", "RegAddress.PostCode"]),
        left_on="entity_id",
        right_on="CompanyNumber",
        how="left",
    )
    matched = enriched.filter(pl.col("CompanyName").is_not_null()).height
    logger.info(f"  Matched: {matched:,} / {len(features):,} ({100*matched/len(features):.1f}%)")

    # ── Naming features ──
    logger.info("Computing naming features...")
    names = enriched["CompanyName"].to_list()
    entropies = []
    name_lengths = []
    generic_ratios = []

    for name in names:
        if not name:
            entropies.append(None)
            name_lengths.append(None)
            generic_ratios.append(None)
            continue
        tokens = [t.lower() for t in name.split() if len(t) > 1]
        n_tokens = len(tokens)
        if n_tokens == 0:
            entropies.append(0.0)
            name_lengths.append(0)
            generic_ratios.append(0.0)
            continue
        counts = Counter(tokens)
        entropy = -sum((c / n_tokens) * math.log2(c / n_tokens) for c in counts.values())
        generic_count = sum(1 for t in tokens if t in GENERIC_NAME_WORDS)
        entropies.append(entropy)
        name_lengths.append(n_tokens)
        generic_ratios.append(generic_count / n_tokens)

    # ── Address concentration (at full address level, not just postcode) ──
    # Full address distinguishes formation agents (500 companies at one address)
    # from urban density (thousands of companies in one postcode)
    logger.info("Computing address concentration...")
    addr_counts = enriched.group_by("RegAddress.AddressLine1").agg(
        pl.len().alias("addr_count")
    )
    # Also compute postcode-level for comparison
    pc_counts = enriched.group_by("RegAddress.PostCode").agg(
        pl.len().alias("pc_count")
    )

    result = enriched.with_columns([
        pl.Series("t2_naming_entropy", entropies, dtype=pl.Float32),
        pl.Series("t2_name_length", name_lengths, dtype=pl.Int32),
        pl.Series("t2_generic_name_ratio", generic_ratios, dtype=pl.Float32),
        pl.col("CompanyName").alias("company_name"),
    ]).join(
        addr_counts, on="RegAddress.AddressLine1", how="left",
    ).with_columns(
        pl.col("addr_count").fill_null(1).alias("t2_address_concentration"),
    ).join(
        pc_counts, on="RegAddress.PostCode", how="left",
    ).with_columns(
        pl.col("pc_count").fill_null(1).alias("t2_postcode_density"),
    ).drop([
        "CompanyName", "CompanyStatus", "RegAddress.AddressLine1",
        "RegAddress.PostCode", "addr_count", "pc_count",
    ])

    logger.info(f"Enrichment complete: {result.shape}")
    return result


def save_features(df: pl.DataFrame, path: Path) -> None:
    """Save feature DataFrame to Parquet."""
    df.write_parquet(path)
    logger.info(f"Saved {len(df):,} rows to {path} ({path.stat().st_size / 1e6:.1f} MB)")
