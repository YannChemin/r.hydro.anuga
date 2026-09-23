#!/usr/bin/env bash
# Import IGN RGE ALTI 1 m over the fine box, as the fine DEM where LiDAR HD
# MNT is not yet published (the case at Plumergat as of 2026-09-23: the
# IGN LiDAR HD tile index has no MNT tile within the 51 km box, and the
# LiDAR HD MNT WMS-R layer is 100% nodata at the village).
#
# Tiles of 1 km x 1 km are requested from the IGN Geoplateforme WMS-R
# (layer ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES, the same service
# r.in.lidarhd downloads from), cached, mosaicked and imported.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
in_grass "$@"

LAYER="ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"
WMSR="https://data.geopf.fr/wms-r"
TILE=1000
PIX=$(awk -v t=$TILE -v r="$RGEALTI_RES" 'BEGIN{printf "%d", t/r}')
CACHE="$RGEALTI_CACHE"
mkdir -p "$CACHE"

tiles=()
w0=$(( (FINE_W / TILE) * TILE ))
s0=$(( (FINE_S / TILE) * TILE ))
for ((e = w0; e < FINE_E; e += TILE)); do
    for ((n = s0; n < FINE_N; n += TILE)); do
        f="$CACHE/RGEALTI_${RGEALTI_RES}m_${e}_${n}.tif"
        if [ ! -s "$f" ]; then
            log "Downloading RGE ALTI tile $e,$n..."
            curl -sf --http1.1 --retry 4 --retry-delay 5 --retry-all-errors -m 300 -o "$f.part" \
                "$WMSR?SERVICE=WMS&VERSION=1.3.0&EXCEPTIONS=text/xml&REQUEST=GetMap&LAYERS=$LAYER&FORMAT=image/geotiff&STYLES=&CRS=EPSG:$EPSG&BBOX=$e,$n,$((e + TILE)),$((n + TILE))&WIDTH=$PIX&HEIGHT=$PIX"
            # A WMS error comes back as XML, not GeoTIFF: fail loudly.
            if ! gdalinfo "$f.part" >/dev/null 2>&1; then
                echo "ERROR: tile $e,$n is not a GeoTIFF:" >&2
                head -c 500 "$f.part" >&2
                exit 1
            fi
            mv "$f.part" "$f"
        fi
        tiles+=("$f")
    done
done

# The WMS-R GeoTIFFs describe Lambert-93 on the WGS84 ellipsoid instead
# of GRS80 (EPSG code lost). The grid is exactly the requested EPSG:2154
# box, so the mosaic is stamped EPSG:2154 explicitly rather than bypassing
# r.in.gdal's CRS check.
VRT="$DATA_DIR/rgealti_fine.vrt"
gdalbuildvrt -overwrite -a_srs "EPSG:$EPSG" "$VRT" "${tiles[@]}" >/dev/null

g.region n=$FINE_N s=$FINE_S e=$FINE_E w=$FINE_W res=$RGEALTI_RES
r.in.gdal input="$VRT" output="$DEM_FINE" -r --overwrite
r.null map="$DEM_FINE" setnull=-99999
r.colors map="$DEM_FINE" color=elevation
r.support map="$DEM_FINE" units=m \
    source1="IGN RGE ALTI 1m, WMS-R $LAYER" \
    description="Fine DEM for r.hydro.anuga dual-DEM tests (NGF-IGN69 altitudes)"
r.info "$DEM_FINE"
r.univar -g "$DEM_FINE"
