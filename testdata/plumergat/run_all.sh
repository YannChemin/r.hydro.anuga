#!/usr/bin/env bash
# Build the whole Plumergat test dataset. Each step can also be run alone.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
if [ "$FINE_SOURCE" = "lidarhd" ]; then fine_step=04_import_lidarhd; else fine_step=04b_import_rgealti; fi
for step in 00_create_project 01_fetch_topage 02_import_dem \
            03_delineate_watersheds "$fine_step" 05_check_dem_datum; do
    echo "=== $step"
    bash "$HERE/$step.sh"
done
