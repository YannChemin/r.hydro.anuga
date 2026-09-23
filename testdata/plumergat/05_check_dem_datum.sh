#!/usr/bin/env bash
# Vertical agreement between the two DEMs over their overlap: the fine
# IGN DEM (LiDAR HD or RGE ALTI, NGF-IGN69 altitudes) aggregated to 30 m
# versus Copernicus GLO-30 (EGM2008 geoid heights; a surface model, so
# trees and buildings show up as positive differences). This is the same check r.hydro.anuga's
# dem_stack.c performs (PLAN.md section 4.2), done by hand here to set
# expectations and a test value for V11.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
in_grass "$@"

# Overlap aligned to the coarse DEM's grid.
g.region n=$FINE_N s=$FINE_S e=$FINE_E w=$FINE_W align="$DEM_COARSE"
r.resamp.stats -w input="$DEM_FINE" output=fine_dem_30m method=average --overwrite
r.mapcalc "dem_diff_coarse_minus_fine = $DEM_COARSE - fine_dem_30m" --overwrite

REPORT="$DATA_DIR/dem_datum_report.txt"
{
    echo "<$DEM_COARSE> minus <$DEM_FINE> (aggregated to 30 m) over the village box ($(date -I))"
    r.univar -e -g dem_diff_coarse_minus_fine
} | tee "$REPORT"
