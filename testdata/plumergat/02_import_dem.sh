#!/usr/bin/env bash
# Import the Copernicus GLO-30 DEM over the 51 km coarse box at 30 m with
# r.in.dem.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
in_grass "$@"

mkdir -p "$DEM_CACHE"
g.region n=$COARSE_N s=$COARSE_S e=$COARSE_E w=$COARSE_W res=$COARSE_RES
log "Importing Copernicus GLO-30 into <$DEM_COARSE>..."
run_addon r.in.dem output="$DEM_COARSE" source=copernicus_glo30 \
    resample=bilinear cache_dir="$DEM_CACHE" --overwrite
r.info "$DEM_COARSE"
r.univar -g "$DEM_COARSE"
