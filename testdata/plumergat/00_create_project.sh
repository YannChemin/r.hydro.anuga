#!/usr/bin/env bash
# Create the plumergat GRASS project in Lambert-93 and verify it.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

if [ -d "$GISDBASE/$PROJECT" ]; then
    log "Project <$PROJECT> already exists, not recreating it."
else
    log "Creating project <$PROJECT> (EPSG:$EPSG)..."
    grass -c "EPSG:$EPSG" "$GISDBASE/$PROJECT" -e
fi

# Verify CRS and set the default region to the coarse box: a silently
# CRS-less project would only fail much later.
grass "$GISDBASE/$PROJECT/$MAPSET" --exec bash -c "
    set -e
    g.proj -g | grep -q 'srid=EPSG:$EPSG' || { echo 'ERROR: project CRS is not EPSG:$EPSG' >&2; g.proj -p; exit 1; }
    g.region n=$COARSE_N s=$COARSE_S e=$COARSE_E w=$COARSE_W res=$COARSE_RES -s
    g.region -p
"
mkdir -p "$DATA_DIR"
