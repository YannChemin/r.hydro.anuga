# r.hydro.anuga

GRASS GIS addon: ANUGA's shallow-water flood solver (discontinuous
elevation finite volumes) in OpenCL, on a multi-resolution mesh built
directly from one or more DEMs, with ERA5/STRDS forcing and Green-Ampt
infiltration.

**Status:** phases 0–2 of [PLAN.md](PLAN.md) are done. The build, the
full option set, OpenCL device selection and the `-p` pre-flight report
work. Single-DEM meshes are bitwise identical to
`anuga.rectangular_cross`. The CPU (OpenMP) solver is bitwise identical
to ANUGA's own C kernels (DE0/DE1/DE2, friction, reflective,
transmissive and Dirichlet boundaries) and passes the lake-at-rest and
dam-break validations. Raster outputs, the OpenCL solver, multi-DEM
meshes and forcing come next.

## Build

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

This needs the OpenCL headers and an ICD loader (`-lOpenCL`). Only
devices with double precision (`cl_khr_fp64`) are used; without one,
`device=auto` falls back to OpenMP.

## Try it

```sh
bash testdata/plumergat/run_all.sh    # builds the Plumergat test project
grass ~/grassdata/plumergat/PERMANENT --exec bash -c '
  g.region n=6780000 s=6728992 e=282016 w=230976 res=32
  r.hydro.anuga -p elevation=rgealti_1m,dem_glo30 domain=anuga_domain'
```

## Tests

```sh
grass --tmp-project XY --exec python3 -m pytest tests
```

The tests find the module on `PATH`, in `R_HYDRO_ANUGA_BIN_DIR`, or in
`$HOME/dev/grass/dist.*/bin`. The ANUGA comparison tests need ANUGA in
`.venv-anuga` (or `ANUGA_PYTHON`); they are skipped otherwise:

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
| `PLAN.md` | design, decisions, validation plan, phases |
| `main.c` | options, validation, orchestration |
| `ocl_backend.c/.h` | capability-based OpenCL device selection |
| `preflight.c/.h` | DEM stack at native resolution, levels, estimates |
| `sampler.c/.h` | rasters at native resolution: nearest, bilinear, box mean |
| `quadtree.c/.h` | active leaves in Morton order |
| `mesh.c/.h` | conforming centre-fan triangles, ANUGA geometry, scaled bed, export |
| `output.c/.h` | raster outputs (`mesh_level=`) |
| `cl/anuga_common.h`, `cl/anuga_sw.h` | solver kernel bodies shared by C and OpenCL C 1.1 |
| `kernels_omp.c` | OpenMP tier |
| `state.c/.h`, `setup.c/.h`, `evolve.c/.h` | solver state, option setup, time stepping |
| `validation/` | comparison scripts run with ANUGA's Python |
| `tests/` | pytest suite |
| `testdata/plumergat/` | scripts building the Plumergat (Morbihan) dual-DEM test case |
| `docs/references/` | papers used for the infiltration model (PDFs not committed) |

## Licence

GPL-3.0-or-later (`LICENSE`). Code derived from ANUGA keeps ANUGA's
Apache-2.0 notice (`LICENSE.ANUGA`). Developed with AI assistance
(Claude).
