#!/usr/bin/env bash
# Watershed delineation on the 30 m DEM, then comparison with BD TOPAGE,
# and a candidate r.hydro.anuga domain: the TOPAGE watersheds that touch
# the LiDAR HD box.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
in_grass "$@"

g.region raster="$DEM_COARSE"

log "Delineating watersheds (threshold=$WS_THRESHOLD cells)..."
if command -v r.watershed.opencl >/dev/null 2>&1; then
    r.watershed.opencl elevation="$DEM_COARSE" threshold="$WS_THRESHOLD" \
        drainage=ws_drainage accumulation=ws_accum stream=ws_streams \
        basin=ws_basins --overwrite
else
    r.watershed elevation="$DEM_COARSE" threshold="$WS_THRESHOLD" \
        drainage=ws_drainage accumulation=ws_accum stream=ws_streams \
        basin=ws_basins -s --overwrite
fi
r.to.vect -s input=ws_basins output=ws_basins type=area --overwrite
r.thin input=ws_streams output=ws_streams_thin --overwrite
r.to.vect -s input=ws_streams_thin output=ws_streams type=line --overwrite

# Fine box as a vector area, used to select the watersheds that matter
# for the dual-DEM tests.
g.region n=$FINE_N s=$FINE_S e=$FINE_E w=$FINE_W res=$FINE_RES
v.in.region output=fine_box --overwrite
g.region raster="$DEM_COARSE"

v.select ainput="$TOPAGE_BV" binput=fine_box output=topage_bv_village \
    operator=overlap --overwrite
v.select ainput=ws_basins binput=fine_box output=ws_basins_village \
    operator=overlap --overwrite

# Candidate model domain: dissolved union of the TOPAGE watersheds
# touching the village box.
v.db.addcolumn map=topage_bv_village columns="one integer" 2>/dev/null || true
v.db.update map=topage_bv_village column=one value=1
v.dissolve input=topage_bv_village column=one output=anuga_domain --overwrite
v.to.rast input=anuga_domain output=anuga_domain use=val value=1 --overwrite

# Agreement between delineated and official watersheds, per TOPAGE
# watershed: take its outlet as the maximum-accumulation cell inside it,
# delineate the catchment upstream of that outlet on the 30 m drainage
# map, and report intersection-over-union (IoU) with the TOPAGE polygon.
REPORT="$DATA_DIR/watershed_report.txt"
{
    echo "Plumergat watershed report ($(date -I))"
    echo "DEM <$DEM_COARSE>, threshold $WS_THRESHOLD cells"
    echo "Candidate model domain <anuga_domain>: $(r.univar -g anuga_domain | awk -F= '$1=="n"{printf "%.2f", $2*'$COARSE_RES'*'$COARSE_RES'/1e6}') km2"
    echo "cat | CdOH | TopoOH | TOPAGE km2 | delineated km2 | IoU | outlet E,N"
} > "$REPORT"

# Read the list on fd 3 so GRASS modules in the loop cannot eat it.
while IFS="|" read -r -u 3 cat cdoh name; do
    v.to.rast input=topage_bv_village cats="$cat" output=tmp_bv use=val value=1 --overwrite --quiet
    r.mapcalc "tmp_acc = if(isnull(tmp_bv), null(), abs(ws_accum))" --overwrite --quiet
    max=$(r.univar -g tmp_acc | awk -F= '$1=="max"{print $2}')
    outlet=$(r.mapcalc "tmp_out = if(tmp_acc >= $max - 0.5, 1, null())" --overwrite --quiet &&
             r.stats -g -n tmp_out | awk 'NR==1{o=$1","$2} END{print o}')
    r.water.outlet input=ws_drainage output=tmp_catch coordinates="$outlet" --overwrite --quiet
    read -r a_top a_del a_int a_uni < <(r.mapcalc "tmp_cmp = if(isnull(tmp_bv),0,1) + 2*if(isnull(tmp_catch),0,1)" --overwrite --quiet &&
        r.stats -c -n tmp_cmp | awk -v c=$((COARSE_RES * COARSE_RES)) '
            {n[$1]=$2} END{printf "%f %f %f %f\n", (n[1]+n[3])*c/1e6, (n[2]+n[3])*c/1e6, n[3]*c/1e6, (n[1]+n[2]+n[3])*c/1e6}')
    g.copy raster=tmp_catch,ws_catch_"$cat" --overwrite --quiet
    awk -v cat="$cat" -v cd="$cdoh" -v nm="$name" -v t="$a_top" -v d="$a_del" -v i="$a_int" -v u="$a_uni" -v o="$outlet" \
        'BEGIN{printf "%s | %s | %s | %.2f | %.2f | %.3f | %s\n", cat, cd, nm, t, d, (u>0?i/u:0), o}' >> "$REPORT"
done 3< <(v.db.select -c map=topage_bv_village columns=cat,CdOH,TopoOH separator="|")
g.remove -f type=raster name=tmp_bv,tmp_acc,tmp_out,tmp_catch,tmp_cmp --quiet
cat "$REPORT"
