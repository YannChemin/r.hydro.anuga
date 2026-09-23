# Shared settings for the Plumergat (Morbihan, Bretagne) test dataset.
#
# Every coordinate below is RGF93 v1 / Lambert-93 (EPSG:2154), the native
# CRS of both IGN LiDAR HD and Sandre BD TOPAGE, so neither needs to be
# reprojected. Edit this file (not the numbered scripts) to move or resize
# the test area.

# GRASS database and project.
GISDBASE="${GISDBASE:-$HOME/grassdata}"
PROJECT="plumergat"
MAPSET="PERMANENT"
EPSG=2154

# Plumergat village centre: 47.7422 N, 2.9167 W (Wikipedia), converted
# with `echo "47.7422 -2.9167" | cs2cs EPSG:4326 EPSG:2154`.
VILLAGE_E=257003
VILLAGE_N=6754583

# Coarse DEM box: 51 km x 51 km around the village (a whole number of
# 30 m cells and of 1 km LiDAR HD tiles).
COARSE_W=231000
COARSE_E=282000
COARSE_S=6729000
COARSE_N=6780000
COARSE_RES=30

# Fine DEM box: 2 km x 2 km over the village centre.
FINE_W=256000
FINE_E=258000
FINE_S=6753500
FINE_N=6755500

# Fine DEM source, chosen explicitly (never substituted silently):
#   lidarhd  IGN LiDAR HD MNT, 0.5 m, via r.in.lidarhd (04_import_lidarhd.sh)
#   rgealti  IGN RGE ALTI, 1 m, via WMS-R (04b_import_rgealti.sh)
# As of 2026-09-23 the IGN LiDAR HD index has no MNT tile anywhere in the
# 51 km box (nearest about 29 km east) and the LiDAR HD MNT WMS-R layer
# is 100% nodata at the village, so Plumergat uses rgealti. Switch to
# lidarhd once IGN publishes the Morbihan blocks.
FINE_SOURCE="${FINE_SOURCE:-rgealti}"
LIDARHD_RES=0.5
RGEALTI_RES=1

# Watershed delineation threshold in cells (1000 cells of 30 m = 0.9 km2).
WS_THRESHOLD=1000

# Map names.
DEM_COARSE="dem_glo30"
if [ "$FINE_SOURCE" = "lidarhd" ]; then
    DEM_FINE="lidarhd_mnt_0p5"
    FINE_RES=$LIDARHD_RES
else
    DEM_FINE="rgealti_1m"
    FINE_RES=$RGEALTI_RES
fi
TOPAGE_BV="topage_bv"
TOPAGE_RIVERS="topage_coursdeau"
TOPAGE_COAST="topage_limiteterreeau"

# Local data (downloads, reports); git-ignored.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$HERE/data"
LIDARHD_CACHE="${LIDARHD_CACHE:-$HOME/RSDATA/LidarHD}"
DEM_CACHE="${DEM_CACHE:-$HOME/RSDATA/CopernicusDEM}"
RGEALTI_CACHE="${RGEALTI_CACHE:-$HOME/RSDATA/RGEALTI}"

# Source trees of the user's addons, used when they are not installed.
ADDON_SRC="${ADDON_SRC:-$HOME/dev}"

# Sandre BD TOPAGE WFS.
TOPAGE_WFS="https://services.sandre.eaufrance.fr/geo/topage"
