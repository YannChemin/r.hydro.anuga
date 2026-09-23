# Plumergat test dataset

Dual-DEM, real-catchment test case for *r.hydro.anuga* (PLAN.md V12):
Plumergat, Morbihan, Bretagne (47.7422 N, 2.9167 W), on the divide
between the Loc'h and the Sal/Bono rivers, which drain to the Golfe du
Morbihan.

All settings are in `config.sh`. Each numbered script runs on its own:
it re-launches itself inside the GRASS session. `run_all.sh` runs the
whole chain. Downloads are cached under `$HOME/RSDATA/`, and logs and
reports go to `data/` (git-ignored).

| Step | Script | Output (project `~/grassdata/plumergat`, EPSG:2154) |
|---|---|---|
| 0 | `00_create_project.sh` | project; CRS and region verified |
| 1 | `01_fetch_topage.sh` | `topage_bv`, `topage_coursdeau`, `topage_limiteterreeau` (Sandre BD TOPAGE WFS, paged, server-side BBOX) |
| 2 | `02_import_dem.sh` | `dem_glo30`: Copernicus GLO-30 via *r.in.dem*, 51 km × 51 km at 30 m |
| 3 | `03_delineate_watersheds.sh` | `ws_*` (*r.watershed.opencl*), `topage_bv_village`, `anuga_domain` (vector + raster), `ws_catch_<cat>`, `data/watershed_report.txt` |
| 4 | `04_import_lidarhd.sh` (`FINE_SOURCE=lidarhd`) | `lidarhd_mnt_0p5` via *r.in.lidarhd* |
| 4b | `04b_import_rgealti.sh` (`FINE_SOURCE=rgealti`, current default) | `rgealti_1m`: IGN RGE ALTI 1 m via WMS-R, 2 km × 2 km |
| 5 | `05_check_dem_datum.sh` | `dem_diff_coarse_minus_fine`, `data/dem_datum_report.txt` |

## Results (2026-09-23)

- **Watersheds.** The village centre lies in BD TOPAGE "R DU BONO & SES
  AFFLUENTS". The 2 km box also touches "LE LOC'H DE SA SOURCE AU RAU
  DE PONT CHRIST". Their union, `anuga_domain`, covers 275 km².
  Delineation on GLO-30 (threshold 1000 cells), with each outlet taken
  as the maximum-accumulation cell inside the TOPAGE polygon, agrees
  with BD TOPAGE:

  | TOPAGE watershed | TOPAGE km² | delineated km² | IoU |
  |---|---|---|---|
  | R du Bono & ses affluents | 115.04 | 106.70 | 0.879 |
  | Le Loc'h, source to Pont Christ | 159.96 | 160.80 | 0.894 |

- **LiDAR HD is not available here yet.** The IGN LiDAR HD tile index
  (`IGNF_LIDAR-HD_METADONNEE:metadata`) has **no MNT tile anywhere in
  the 51 km box**; the nearest is about 29 km east. The LiDAR HD MNT
  WMS-R layer is 100% nodata at the village. *r.in.lidarhd* therefore
  stops with "No IGN LiDAR HD 'mnt' tiles found", which is correct
  behaviour. The fine DEM is **RGE ALTI 1 m** instead, selected
  explicitly by `FINE_SOURCE=rgealti` in `config.sh`. Switch it back
  once IGN publishes the Morbihan blocks.
- **Vertical agreement.** GLO-30 minus RGE ALTI (aggregated to 30 m)
  has a median of −0.29 m (IQR −1.04 to +0.70 m, 90th percentile
  +2.26 m, max +11.6 m). The median is within r.hydro.anuga's default
  0.5 m bias tolerance. The positive tail is expected: **GLO-30 is a
  surface model** (canopy, buildings), while RGE ALTI is bare earth.
- **GLO-30 negatives.** 4989 cells are below 0 m, mostly the coast and
  the Golfe (mean −0.7 m). Inside `anuga_domain`, 55 cells are below
  −1 m, clustered around E 262000, N 6753135 and reaching −21 m. That
  is probably a real pit (a quarry?); check it visually before
  modelling.

## Quirks handled in the scripts

- ogr2ogr's WFS driver filtered `-spat` client-side on the Sandre server
  (it started downloading all of France), so `01` uses paged GetFeature
  with a server-side BBOX.
- The IGN WMS-R GeoTIFFs label Lambert-93 with the WGS84 ellipsoid (the
  EPSG code is lost). `04b` stamps the mosaic `EPSG:2154` explicitly
  with `gdalbuildvrt -a_srs` instead of bypassing r.in.gdal's CRS check.
- The IGN WMS-R sometimes resets HTTP/2 streams (curl exit 92), so `04b`
  uses `--http1.1` with retries.

## Implications for r.hydro.anuga tests

- DEM stack: `rgealti_1m` (1 m, 2 × 2 km) over `dem_glo30` (30 m,
  51 km). Levels snap to 1, 2, 4, 8, 16 and 32 m (`res_max` 30 → 32 m,
  reported by the module).
- Size: the fine box alone is 4 M cells, i.e. 16 M triangles at 1 m.
  The 275 km² domain at 32 m adds about 1 M. That is about 14 GB with
  explicit geometry, at the WX 7100's limit (PLAN.md §4.6), so the
  first dual-DEM runs should use `res_min=2`, `coarsen=`, or a smaller
  fine box.
- A bare-earth coarse DEM would be more consistent for flood
  modelling: RGE ALTI 5 m or 25 m via the same WMS-R, or FABDEM.
  `02_import_dem.sh` is the place to add it.
