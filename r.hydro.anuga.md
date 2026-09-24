## DESCRIPTION

*r.hydro.anuga* simulates overland flow and flooding with the
shallow-water solver of [ANUGA](https://github.com/anuga-community/anuga_core),
the discontinuous-elevation (DE) finite-volume scheme on triangles. It
runs in OpenCL on a GPU (or on a CPU OpenCL device such as PoCL), with a
plain OpenMP fallback.

The mesh is built directly from one or more DEMs given in
**elevation**. At each location it uses the finest DEM available, for
example IGN LiDAR HD or RGE ALTI at 0.5-1 m inside a village and a 30 m
DEM around it. Resolution levels are powers of two between **res_min**
and **res_max**. The transition between levels is graded over **fringe**
cells per level. Every square cell is split into triangles fanned from
its centre: 4 triangles (the ANUGA `rectangular_cross` pattern), or up
to 8 next to a finer neighbour, which keeps the mesh conforming.

Forcing can be constant, a raster map, or a space-time raster dataset.
The latter covers ERA5 rainfall, evaporation, wind and pressure imported
with *t.in.era5*. Units are always given explicitly (**rain_units**,
**evap_units**, **pressure_units**); the module never guesses them.
Infiltration uses Green-Ampt with ponded head
(**infiltration**=*ga*, Rawls et al. 1983 parameters), or Green-Ampt
with redistribution between storms (**infiltration**=*gar*, Ogden and
Saghafian 1997).

Outputs are space-time raster datasets of depth and, on request, stage,
velocity, speed, direction, unit discharge, Froude number, hazard rating
and cumulative infiltration. The module can also write rasters of
maximum depth, speed, stage and hazard, arrival time, inundation
duration, the mesh level, gauge time series and a mass balance report.

**Development status:** phase 2 of the implementation plan
(`PLAN.md`). The complete option set is defined and validated. The
**-p** pre-flight report works: it selects the compute device,
describes the DEM stack at native resolution, snaps the resolution
levels, and estimates the number of triangles and the device memory.
Single-DEM meshes can be built and inspected (mesh-only mode, below).
The solver runs with **state_output** (below), on the selected OpenCL
device or with OpenMP. Its results are bitwise identical to ANUGA's own
C kernels (OpenMP; OpenCL too when there is no friction). Raster time
series (**output**), multi-DEM meshes and forcing are not implemented
yet.

## NOTES

### Compute device

With **device**=*auto*, the first GPU with double precision support
(`cl_khr_fp64`) is used, then a CPU OpenCL device, then OpenMP. Devices
are chosen by capability, not by platform order. Each candidate must
build a small double precision OpenCL C 1.1 probe kernel. For example,
with Mesa, the rusticl platform may expose a GPU without double
precision next to the Clover platform exposing the same GPU with it;
the Clover device is selected. **device**=*gpu* or *cpu* fail if no
such device exists, rather than falling back silently.

The solver runs on the selected device. OpenCL guarantees correctly
rounded double arithmetic and square roots, so the OpenCL and OpenMP
solvers give bitwise identical results without friction. Manning
friction uses a power function whose OpenCL accuracy is a few units in
the last place, so results then differ at the 1e-15 relative level.

### Resolution levels

**res_min** defaults to the finest DEM resolution. **res_max**
defaults to the coarsest, rounded up to **res_min** times a power of
two, and the rounding is reported: 1 m and 30 m DEMs give levels of 1,
2, 4, 8, 16 and 32 m. The current region must be a whole number of
**res_max** cells. Otherwise the module stops and prints an aligned
*g.region* command. DEMs must have square cells. Latitude-longitude
projects are not supported.

### Mesh-only mode

With **mesh_output** and/or **mesh_level** but no **output**, the module
builds the mesh and exits. **mesh_level** is a raster of the mesh level
of each cell of the current region (0 is the coarsest, NULL outside
the domain). **mesh_output** is a directory with `manifest.json` and one
raw binary file per array (node coordinates and elevations, triangles,
geometry, neighbours, boundary edges and tags, fp64 and scaled centroid
bed). Coordinates are relative to the origin given in the manifest.
Arrays follow ANUGA's conventions, so they can be loaded into
`anuga.Domain` directly. For a single DEM without NULL cells, the mesh
is identical, bit for bit, to `anuga.rectangular_cross` (see
`validation/compare_mesh_anuga.py`).

Leaves are active where the DEM has valid cells and, if **domain** is
given, where the leaf centre lies inside the domain. Mesh edges facing
an inactive cell carry the boundary tag *null*; edges on the region
edge are tagged *north*, *south*, *east* or *west*. Node elevations are
bilinear between DEM cell centres at leaf corners and the mean of the
DEM cells inside the leaf at leaf centres. The triangle bed is the mean
of its three vertices.

### Solver runs

With **state_output** and **duration** (seconds), the module runs the
solver and writes a directory with `manifest.json` and raw arrays: the
initial and final centroid stage and momenta, bed, friction, boundary
setup, and the time of every step. The manifest holds the mass budget:
water volume (positive depths only, as ANUGA reports it, and signed),
the integrated boundary flux, and the water added by clamping negative
depths. The change of the signed volume equals the boundary flux plus
the clamping mass.

**algorithm** selects ANUGA's DE0 (Euler), DE1 (second-order
Runge-Kutta, default) or DE2 (third-order Runge-Kutta) parameter sets.
The default CFL numbers are 0.9 (DE0), 0.5 (DE1; ANUGA uses 1.0) and 1.0
(DE2); **cfl** overrides them. **friction_method** is *flat*
(default, as ANUGA) or *sloped*. **boundary** sets each side (*north*,
*south*, *east*, *west*, or *null* for edges facing NULL or
masked-out cells) to *reflective* (default), *transmissive* or
*dirichlet:stage[:xmom:ymom]*. **initial_depth** or **initial_stage**
give the initial water; **manning** or **manning_value** give
Manning's n.

For thin water films on coarse, steep terrain (typical of
rain-on-grid), DE1 and DE2 at a CFL of 1 can drive depths negative
often; clamping them adds water. This is why DE1 defaults to a CFL of
0.5. The added amount is reported, with a warning above 1% of the
initial volume; a lower **cfl** or DE0 reduces it further. ANUGA
behaves identically at the same settings.

### Elevation precision

On the compute device, elevation is stored as unsigned 32-bit integers
in units of 0.1 mm above an integer datum, and converted back exactly.
The CPU and GPU paths therefore see bitwise identical terrain. Water
levels and velocities remain in double precision.

## EXAMPLES

Pre-flight report for a 1 m RGE ALTI tile over a 30 m Copernicus DEM,
restricted to a watershed mask:

```sh
g.region n=6780000 s=6728992 e=282016 w=230976 res=32
r.hydro.anuga -p elevation=rgealti_1m,dem_glo30 domain=anuga_domain
```

Dam break in a closed channel, 60 s, with the solver state written
out:

```sh
g.region n=200 s=0 e=1000 w=0 res=10
r.mapcalc "dem = 0"
r.mapcalc "h0 = if(x() < 500, 2.0, 0.5)"
r.hydro.anuga elevation=dem initial_depth=h0 duration=60 output_step=30 \
    state_output=/tmp/dam_break
```

The same report as `key=value` pairs, for scripts:

```sh
r.hydro.anuga -p elevation=rgealti_1m,dem_glo30 domain=anuga_domain format=shell
```

Build the mesh of a watershed from a 30 m DEM and map it:

```sh
g.region raster=dem_glo30
r.hydro.anuga elevation=dem_glo30 domain=anuga_domain \
    mesh_level=mesh_level mesh_output=/tmp/plumergat_mesh
```

## REFERENCES

- Green, W. H., and Ampt, G. A. (1911). Studies on soil physics. *The
  Journal of Agricultural Science*, 4(1), 1-24.
- Rawls, W. J., Brakensiek, D. L., and Miller, N. (1983). Green-Ampt
  infiltration parameters from soils data. *Journal of Hydraulic
  Engineering*, 109(1), 62-70.
- Ogden, F. L., and Saghafian, B. (1997). Green and Ampt infiltration
  with redistribution. *Journal of Irrigation and Drainage
  Engineering*, 123(5), 386-393.
- La Follette, P., Ogden, F. L., and Jan, A. (2023). Layered Green and
  Ampt infiltration with redistribution. *Water Resources Research*,
  59, e2022WR033742.
- Fernández-Pato, J., Caviedes-Voullième, D., and García-Navarro, P.
  (2016). Rainfall/runoff simulation with 2D full shallow water
  equations: sensitivity analysis and calibration of infiltration
  parameters. *Journal of Hydrology*, 536, 496-513.

## SEE ALSO

*[r.hydro.hbv](r.hydro.hbv.md),
[r.hydro.rri](r.hydro.rri.md),
[r.sim.water](r.sim.water.md),
[r.watershed.opencl](r.watershed.opencl.md),
[t.in.era5](t.in.era5.md),
[t.rast.list](t.rast.list.md)*

## AUTHORS

Yann Chemin. The shallow-water numerics are derived from ANUGA
(Copyright 2004-2015 Australian National University and Geoscience
Australia, Apache License 2.0). This addon was developed with AI
assistance (Claude); see `PLAN.md`.
