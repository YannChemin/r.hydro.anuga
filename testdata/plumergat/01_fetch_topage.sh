#!/usr/bin/env bash
# Download official BD TOPAGE watersheds, rivers and land-water limit over
# the coarse box from the Sandre WFS, and import them.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
in_grass "$@"

mkdir -p "$DATA_DIR"
GPKG="$DATA_DIR/topage_plumergat.gpkg"

# Paged WFS GetFeature with a server-side BBOX filter. (ogr2ogr's WFS
# driver with -spat filtered client-side here, i.e. downloaded all of
# France.)
PAGE=500
fetch() {
    local layer="$1" out="$2" start=0 n page_file
    local bbox="$COARSE_W,$COARSE_S,$COARSE_E,$COARSE_N,urn:ogc:def:crs:EPSG::$EPSG"
    log "Fetching $layer from $TOPAGE_WFS..."
    while :; do
        page_file="$DATA_DIR/${out}_$start.gml"
        curl -sf -m 300 -o "$page_file" \
            "$TOPAGE_WFS?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES=$layer&SRSNAME=EPSG:$EPSG&BBOX=$bbox&COUNT=$PAGE&STARTINDEX=$start"
        n=$(grep -o 'numberReturned="[0-9]*"' "$page_file" | grep -o '[0-9]*')
        if [ "${n:-0}" -gt 0 ]; then
            local mode=()
            [ -f "$GPKG" ] && mode=(-update -append)
            ogr2ogr -f GPKG "${mode[@]}" "$GPKG" "$page_file" -nln "$out" \
                -nlt PROMOTE_TO_MULTI -a_srs "EPSG:$EPSG"
        fi
        rm -f "$page_file" "${page_file%.gml}.gfs"
        [ "${n:-0}" -lt "$PAGE" ] && break
        start=$((start + PAGE))
    done
    log "$layer: $(ogrinfo -so "$GPKG" "$out" | grep 'Feature Count')"
}

rm -f "$GPKG"
fetch "sa:BassinVersantTopographique_FXX" "$TOPAGE_BV"
fetch "sa:CoursEau_FXX" "$TOPAGE_RIVERS"
fetch "sa:LimiteTerreEau_FXX" "$TOPAGE_COAST"

g.region n=$COARSE_N s=$COARSE_S e=$COARSE_E w=$COARSE_W res=$COARSE_RES
for layer in "$TOPAGE_BV" "$TOPAGE_RIVERS" "$TOPAGE_COAST"; do
    v.import input="$GPKG" layer="$layer" output="$layer" --overwrite
done
v.info -t "$TOPAGE_BV"

# Report the BD TOPAGE watersheds containing the village centre.
log "BD TOPAGE watershed(s) at the village centre:"
v.what map="$TOPAGE_BV" coordinates="$VILLAGE_E,$VILLAGE_N" -a | grep -E "CdOH|TopoOH" || true
