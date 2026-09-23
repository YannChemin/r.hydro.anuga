#!/usr/bin/env bash
# Import the IGN LiDAR HD bare-earth MNT at 0.5 m over the 2 km village
# box with r.in.lidarhd.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
in_grass "$@"

mkdir -p "$LIDARHD_CACHE"
g.region n=$FINE_N s=$FINE_S e=$FINE_E w=$FINE_W res=$LIDARHD_RES

log "LiDAR HD MNT tiles for the village box:"
run_addon r.in.lidarhd -l output="$DEM_FINE" type=mnt directory="$LIDARHD_CACHE"

log "Importing LiDAR HD MNT into <$DEM_FINE>..."
run_addon r.in.lidarhd output="$DEM_FINE" type=mnt directory="$LIDARHD_CACHE" \
    resolution=value resolution_value=$LIDARHD_RES --overwrite
r.info "$DEM_FINE"
r.univar -g "$DEM_FINE"
