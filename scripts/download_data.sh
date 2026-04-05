#!/usr/bin/env bash
# Download public datasets for beneficial ownership research.
# All sources are freely available under CC0, OGL, or ODbL.
#
# Usage: bash scripts/download_data.sh [--psc] [--icij] [--gleif] [--sanctions] [--all]
#
# Total download: ~6-8 GB compressed

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
echo "Data directory: $DATA_DIR"

download_psc() {
    echo "=== UK PSC Register (Companies House) ==="
    echo "Source: https://download.companieshouse.gov.uk/en_pscdata.html"
    echo "License: Open Government Licence v3.0"
    local dir="$DATA_DIR/psc"
    mkdir -p "$dir"

    # Download the single-file snapshot (smaller downloads available as parts)
    local url="http://download.companieshouse.gov.uk/en_pscdata.html"
    echo "Checking for latest PSC snapshot..."

    # The snapshot URL pattern includes the date
    # We'll download the parts (31 files, ~65MB each compressed)
    for i in $(seq 1 31); do
        local part=$(printf "%d" $i)
        local file="psc-snapshot-part${part}.zip"
        if [ -f "$dir/$file" ]; then
            echo "  Already have $file, skipping"
            continue
        fi
        echo "  Downloading part $part of 31..."
        # Note: actual URL includes date, may need updating
        curl -L -o "$dir/$file" \
            "http://download.companieshouse.gov.uk/psc-snapshot-2026-04-04_${part}of31.zip" \
            2>/dev/null || echo "  Warning: part $part failed (date may differ)"
    done

    echo "Extracting PSC data..."
    cd "$dir"
    for f in psc-snapshot-part*.zip; do
        [ -f "$f" ] && unzip -o -q "$f" 2>/dev/null || true
    done
    echo "PSC download complete."
}

download_icij() {
    echo "=== ICIJ Offshore Leaks Database ==="
    echo "Source: https://offshoreleaks.icij.org/pages/database"
    echo "License: ODbL / CC-BY-SA"
    local dir="$DATA_DIR/icij"
    mkdir -p "$dir"

    if [ -f "$dir/full-oldb.csv.zip" ] || [ -d "$dir/csv" ]; then
        echo "  Already downloaded, skipping"
    else
        echo "  Downloading CSV bulk data..."
        curl -L -o "$dir/full-oldb.csv.zip" \
            "https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip"
        echo "  Extracting..."
        cd "$dir"
        unzip -o -q "full-oldb.csv.zip" -d csv
    fi
    echo "ICIJ download complete."
}

download_gleif() {
    echo "=== GLEIF Level 2 (Who Owns Whom) ==="
    echo "Source: https://www.gleif.org/en/lei-data/gleif-concatenated-file"
    echo "License: CC0 1.0"
    local dir="$DATA_DIR/gleif"
    mkdir -p "$dir"

    if [ -f "$dir/rr-cdf-v2-1.csv" ] || [ -f "$dir/rr-cdf-v2-1.csv.zip" ]; then
        echo "  Already downloaded, skipping"
    else
        echo "  Downloading Level 2 relationship data (CSV)..."
        # Golden copy CSV
        curl -L -o "$dir/rr-cdf-v2-1.csv.zip" \
            "https://leidata-preview.gleif.org/api/v2/golden-copies/csv/latest?type=rr" \
            2>/dev/null || echo "  Warning: GLEIF URL may have changed"
        [ -f "$dir/rr-cdf-v2-1.csv.zip" ] && unzip -o -q "$dir/rr-cdf-v2-1.csv.zip" -d "$dir" 2>/dev/null || true
    fi
    echo "GLEIF download complete."
}

download_sanctions() {
    echo "=== OFAC SDN List ==="
    echo "Source: https://sanctionslist.ofac.treas.gov"
    local dir="$DATA_DIR/sanctions"
    mkdir -p "$dir"

    if [ -f "$dir/sdn.csv" ]; then
        echo "  Already downloaded, skipping"
    else
        echo "  Downloading OFAC SDN CSV..."
        curl -L -o "$dir/sdn.csv" \
            "https://www.treasury.gov/ofac/downloads/sdn.csv" 2>/dev/null || true
        curl -L -o "$dir/add.csv" \
            "https://www.treasury.gov/ofac/downloads/add.csv" 2>/dev/null || true
        curl -L -o "$dir/alt.csv" \
            "https://www.treasury.gov/ofac/downloads/alt.csv" 2>/dev/null || true
    fi
    echo "Sanctions download complete."
}

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [--psc] [--icij] [--gleif] [--sanctions] [--all]"
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        --psc) download_psc ;;
        --icij) download_icij ;;
        --gleif) download_gleif ;;
        --sanctions) download_sanctions ;;
        --all)
            download_psc
            download_icij
            download_gleif
            download_sanctions
            ;;
        *) echo "Unknown argument: $arg" ;;
    esac
done

echo ""
echo "=== Download Summary ==="
echo "Data directory: $DATA_DIR"
du -sh "$DATA_DIR"/* 2>/dev/null || true
