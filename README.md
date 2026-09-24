# r.hydro.anuga

GRASS GIS addon: the shallow-water flood solver of
[ANUGA](https://github.com/anuga-community/anuga_core) (discontinuous
elevation finite volumes on triangles) running in OpenCL on a GPU, with
an OpenMP fallback, on a multi-resolution mesh built directly from one
or more DEMs.

**Status:** the five development phases of [PLAN.md](PLAN.md) are done
(phases 0–5).

## What it does

- **Meshes from DEMs.**
  - One or more DEMs, each read at its native resolution. For example,
    IGN RGE ALTI or LiDAR HD at 1 m in a village, inside a 30 m DEM over
    the whole watershed.
  - The mesh uses the finest DEM available at each location. Resolution
    levels step down by powers of two, with a graded fringe of
    `fringe=` cells per level and 2:1 balance.
  - Cells are fanned into 4 triangles (ANUGA's `rectangular_cross`), or
    5–8 next to a finer neighbour.
  - The vertical bias between DEMs is checked, and fails loudly unless
    `dem_offset=` is given. Seams are blended.
  - `refine=` adds resolution over any other area. `domain=` restricts
    the mesh, e.g. to a watershed.
- **Solver.**
  - ANUGA's DE0/DE1/DE2 algorithms (DE1 by default, at CFL 0.5), with
    Manning friction (flat or sloped).
  - Reflective, transmissive and Dirichlet boundaries per side, and for
    edges facing masked-out cells.
  - Initial depth or stage from rasters.
  - Elevation is stored on the device as scaled 32-bit integers (0.1 mm
    above a datum), converted back exactly.
- **Outputs.**
  - Time series of depth, stage, velocity, speed, direction, unit
    discharge, momentum, Froude number and hazard. They are registered
    as space-time raster datasets, in absolute time with `start=` or
    relative otherwise.
  - Maximum depth, speed, stage and hazard, arrival time and inundation
    duration, from statistics updated every time step. Final state for
    hot starts.
  - A mass-balance table.
  - With `-f`, every output is also written at full resolution over each
    finer DEM's footprint.
- **Devices.** `device=auto` picks the first GPU with double precision
  that builds the kernels (so Mesa rusticl without fp64 is skipped in
  favour of Clover), then a CPU OpenCL device such as PoCL, then OpenMP.
  `-p` prints the device, the resolution levels, a triangle count
  estimate and memory needs without running anything.

## Verified results

| Check | Result |
|---|---|
| Single-level mesh vs `anuga.rectangular_cross` | every geometry and connectivity array bitwise identical |
| Solver vs ANUGA's own C kernels (unified mode), DE0/DE1/DE2, friction, all boundary types, up to 505 steps | final state and step count **bitwise identical** |
| Same, on multi-level meshes (ANUGA built from the exported mesh), dam break through all levels, 4738 steps | **bitwise identical** |
| OpenCL (PoCL, Radeon Pro WX 7100) vs OpenMP, no friction | bitwise identical (up to 7208 steps) |
| Same, with friction | a few ulp per step (OpenCL `pow()`); up to about 1e-3 relative after long wetting transients |
| Lake at rest, single level and across 4 levels with hanging nodes | velocities ≤ 1.5e-14 |
| Dam break vs Stoker (wet) and Ritter (dry) | relative L1 error ≤ 0.6% (ANUGA's own tolerances: 1–25%) |
| Wave crossing an 8 → 1 m fringe | 0.01% reflection |
| Mass balance (signed volume vs boundary flux + clamping) | exact to about 1e-12 |
| Raster output volume at 1×, 2× and 4× the mesh cell size | equal to the solver volume to 1e-12 |
| Speed, 4 M triangles, 394 steps | WX 7100: 39 s; 32 CPU threads (Ryzen 9 3950X): 205 s |
| Plumergat: 30 m GLO-30 watershed (275 km²) with RGE ALTI in the village at 2 m | 5.12 M triangles, 1737 steps, mass conserved, main and 2 m detail time series registered |

The test suite has TESTS_TOTAL pytest tests.

## Not implemented

These features are in the plan (phases 6–9) but not implemented. Their
options exist so that `-p` can estimate a complete run, but a
simulation given any of them stops with "not implemented yet" instead
of ignoring it:

- rainfall and ERA5/STRDS forcing (`rain*=`, `evap*=`, wind,
  pressure), inflow hydrographs, stage and Flather boundaries;
- Green–Ampt / GAR infiltration (design and parameters in PLAN.md §6);
- land cover and building inputs, initial momentum, gauges, `end=`,
  `min_timestep=`, `coarsen=` and `relief_tolerance=`;
- streamed (banded) reading of very large DEMs, local time stepping,
  compact per-leaf geometry and other performance work, riverwalls,
  buildings as holes.

A global time step is set by the smallest cells. That is why the
Plumergat run at 2 m takes about an hour on a laptop CPU, while its
mesh size alone would suggest minutes.

## Build

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

This needs the OpenCL headers and an ICD loader (`-lOpenCL`). The
OpenCL kernels are embedded in the executable at build time.

## Try it

```sh
bash testdata/plumergat/run_all.sh    # builds the Plumergat test project
grass ~/grassdata/plumergat/PERMANENT --exec bash -c '
  g.region n=6780000 s=6728992 e=282016 w=230976 res=32
  r.hydro.anuga -p elevation=rgealti_1m,dem_glo30 domain=anuga_domain res_min=2
  r.mapcalc "h0 = 0.05"
  r.hydro.anuga elevation=rgealti_1m,dem_glo30 domain=anuga_domain res_min=2 \
      initial_depth=h0 duration=60 output_step=30 output=plm max_depth=plm_max -f'
```

See the manual (`r.hydro.anuga.md`) for all options and more examples.

## Tests

```sh
grass --tmp-project XY --exec python3 -m pytest tests
```

The tests find the module on `PATH`, in `R_HYDRO_ANUGA_BIN_DIR`, or in
`$HOME/dev/grass/dist.*/bin`. OpenCL tests are skipped without a double
precision OpenCL device. The ANUGA comparison tests need ANUGA in
`.venv-anuga` (or `ANUGA_PYTHON`) and are skipped otherwise:

```sh
python3 -m venv .venv-anuga
PATH=$PWD/.venv-anuga/bin:$PATH .venv-anuga/bin/pip install \
    numpy scipy matplotlib netCDF4 meson-python meson ninja cython pybind11
PATH=$PWD/.venv-anuga/bin:$PATH .venv-anuga/bin/pip install \
    --no-build-isolation ~/dev/anuga_core
```

## Layout

| Path | Content |
|---|---|
| `PLAN.md` | design, decisions, validation plan, phase records |
| `main.c` | options, validation, orchestration |
| `ocl_backend.c/.h` | capability-based OpenCL device selection |
| `preflight.c/.h` | `-p` report: DEM scan, level snapping, estimates |
| `dem_stack.c/.h` | DEMs at native resolution, bias check, seam blending |
| `sampler.c/.h` | rasters at native resolution: nearest, bilinear, box mean |
| `quadtree.c/.h` | uniform and graded quadtrees, hanging-node masks |
| `mesh.c/.h` | conforming centre-fan triangles, ANUGA geometry, scaled bed, export |
| `hashmap.h` | hash map for nodes, edges and leaves |
| `cl/anuga_common.h`, `cl/anuga_sw.h` | solver kernel bodies shared by C and OpenCL C 1.1 |
| `cl/anuga_kernels.cl`, `kernels_ocl.c/.h` | OpenCL tier |
| `kernels_omp.c` | OpenMP tier |
| `state.c/.h`, `setup.c/.h`, `evolve.c/.h` | solver state, option setup, time stepping |
| `output.c/.h` | raster outputs, space-time datasets, summaries, mass balance |
| `validation/` | comparison scripts run with ANUGA's Python |
| `tests/` | pytest suite |
| `testdata/plumergat/` | scripts building the Plumergat (Morbihan) dual-DEM test case |
| `docs/references/` | papers for the infiltration design (PDFs not committed) |

## Licence

GPL-3.0-or-later (`LICENSE`). Code derived from ANUGA keeps ANUGA's
Apache-2.0 notice (`LICENSE.ANUGA`). 
