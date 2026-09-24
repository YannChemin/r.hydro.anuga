# r.hydro.anuga: implementation plan

Status: **phases 0–4 done** (2026-09-24). The OpenMP solver is bitwise
identical to ANUGA's C kernels (phase 2). The OpenCL solver is bitwise
identical to it without friction, on PoCL and on the WX 7100 (phase 3). Earlier:
build, options, device
selection, the `-p` pre-flight report, and single-level mesh construction
(bitwise identical to `anuga.rectangular_cross`) with the scaled integer
bed. The solver itself is not implemented yet.
Revision 9 (2026-09-24), with the user's decisions on the licence, mesh, solver,
infiltration and elevation interpolation, plus the multi-resolution
DEM requirement.

Goal: a compiled GRASS GIS addon, *r.hydro.anuga*, that runs ANUGA's
discontinuous-elevation (DE) shallow-water finite-volume solver in
OpenCL. It reads its inputs straight from GRASS rasters and STRDS: one
**or several DEMs at different resolutions**, friction, soil, initial
water, and forcing from *t.in.era5* or any other STRDS. It writes a
space-time raster dataset of water depth, plus the other hydraulic
outputs ANUGA can produce (velocity, stage, maxima, arrival time,
hazard, infiltration, mass balance, gauges).

Reference source: `$HOME/dev/anuga_core` (git `e2e7bf67`). Sibling
projects whose patterns are reused: `$HOME/dev/r.watershed.opencl`
(OpenCL backend tiering, Module.make build, pytest layout),
`$HOME/dev/RRI.opencl` (OpenCL 1.1 / Clover constraints, validation
discipline), `$HOME/dev/r.hydro.rri` (STRDS forcing input and STRDS
output registration from C), `$HOME/dev/t.in.era5` (forcing producer).

## 0. Decisions recorded

| # | Decision (user, 2026-09-23) |
|---|---|
| D1 | Licence **GPL-3.0-or-later** for the addon. Files ported from ANUGA keep their **Apache-2.0** header and attribution (Apache-2.0 is GPLv3-compatible). The repo's current Unlicense `LICENSE` is replaced in phase 0. |
| D2 | **4 triangles per cell** (ANUGA `rectangular_cross` pattern), generalised to a centre fan on multi-resolution cells (§4). |
| D3 | Default solver **DE1** (RK2, beta = 1.0, `minimum_allowed_height = 1e-5`) with **CFL 0.5** instead of ANUGA's 1.0 (decision R4, after phase 2 finding 3). DE0 and DE2 remain selectable; `cfl=` overrides. |
| D4 | Infiltration: **Green–Ampt with ponded head** (peer-reviewed, physically based, about 15 flops per cell per step), parameters from Rawls et al. (1983). **GAR redistribution** (Ogden & Saghafian 1997) is an option for multi-storm / multi-day runs (§6). |
| D5 | Vertex elevation is **interpolated between cell centres** (bilinear). |
| D6 | **Multiple DEMs at different resolutions**: the mesh refines to the finest DEM available at each location, with a graded fringe between resolution levels (§4). |
| D7 | **Elevation goes to the device only as scaled unsigned integers** (`uint32`, 1 unit = 0.1 mm, i.e. metres × 10000 above an integer datum). Kernels treat elevation as scaled: integer arithmetic for bed-only comparisons and differences, and one exact dequantisation where it meets fp64 stage (§4.7). |

---

## 1. What is being ported, precisely

ANUGA's numerics live in C, not Python. The version-4 code base
already has a Python-free, device-resident time loop written for
OpenMP target offload:

| ANUGA file | Content | Port target |
|---|---|---|
| `anuga/shallow_water/sw_domain.h` | `struct domain`, `struct edge`, flux helpers | `state.h` (SoA buffers) |
| `anuga/shallow_water/sw_domain_math.h` | inline math helpers | `cl/anuga_math.h` (shared C/OpenCL) |
| `anuga/shallow_water/gpu/core_kernels.c` | `core_extrapolate_second_order_edge`, `core_compute_fluxes_central`, `core_update_conserved_quantities`, `core_backup/saxpy`, `core_protect`, `core_fix_negative_cells`, `core_manning_friction_*` | `cl/anuga_kernels.cl` |
| `anuga/shallow_water/gpu/gpu_kernels.c` | `gpu_evolve_one_{euler,rk2,rk3}_step`: the full step orchestrated in C | `evolve.c` (host loop) |
| `anuga/shallow_water/gpu/gpu_boundaries.c` | reflective, transmissive, Dirichlet, time, Flather... | `cl/anuga_boundaries.cl` |
| `anuga/shallow_water/gpu/gpu_rate_operator.c` | rain / extraction on a set of triangles | `cl/anuga_forcing.cl` |
| `anuga/shallow_water/gpu/gpu_inlet_operator.c` | point/line inflow hydrographs | phase 6 |
| `anuga/shallow_water/gpu/gpu_max_quantities_operator.c` | running max of depth, speed, stage | `cl/anuga_stats.cl` |
| `anuga/operators/wind_stress_operator.py`, `barometric_pressure.py` | wind and pressure forcing | phase 6 |

`gpu_evolve_one_rk2_step()` (gpu_kernels.c:381) is the template for the
default DE1 host loop: backup, then two Euler substeps (protect,
extrapolate, boundaries, fluxes returning the local dt, friction,
update), then `saxpy(0.5, 0.5)`. The Python `Domain` layer, `.sww`
output, pmesh/Triangle meshing and MPI are **not** ported.

`algorithm=` sets ANUGA's parameter bundles exactly
(shallow_water_domain.py:1043-1350):

| `algorithm=` | Time stepping | CFL | beta_w/uh/vh | min. allowed height |
|---|---|---|---|---|
| DE0 | Euler | 0.9 | 0.5 | 1e-12 |
| **DE1** (default) | RK2 | **0.5** (ANUGA: 1.0; decision R4) | 1.0 | 1e-5 |
| DE2 | RK3 | as ANUGA | as ANUGA | as ANUGA |

ANUGA's core solver is already unstructured-mesh code (neighbour
tables, per-edge normals and lengths), so the multi-resolution mesh of
§4 needs **no change to the numerics**. Only the mesh generator and the
raster I/O are new.

---

## 2. Target hardware constraints (checked 2026-09-23)

**GPU test host `yann@10.42.0.89`** (`debiantest`), used for all
real-GPU testing of this addon, as for `RRI.opencl` and
`r.watershed.opencl`:

- AMD Radeon Pro WX 7100 (Polaris10, 36 CUs, 16 GiB, max work-group 256).
- **Mesa Clover, OpenCL 1.1**, with `cl_khr_fp64`, int32 and int64
  base/extended atomics.
- Mesa rusticl (`RUSTICL_ENABLE=radeonsi`) reports OpenCL 3.0 for the
  same GPU but has **no fp64**, so it is unusable.
- GRASS 8.6.0dev, `~/dev/grass`, `~/dev/grass-addons`.

**Local dev host**: PoCL CPU device (Skylake-AVX512, fp64), GRASS
8.6.0dev.

Consequences:

1. **Kernels are written for OpenCL C 1.1**: no generic address space,
   no `clCreateCommandQueueWithProperties`, no C11 atomics, no
   `work_group_reduce_*`. Reductions are hand-written in local memory,
   with an explicit `#pragma OPENCL EXTENSION cl_khr_fp64 : enable`.
2. **Double precision is required** (ANUGA is fp64 throughout, and
   wet/dry thresholds are 1e-5 to 1e-12). `ocl_backend.c` chooses the
   device **by capability, not platform order**: the first GPU that
   exposes `cl_khr_fp64`. Otherwise, with rusticl enabled,
   `r.watershed.opencl`'s `find_device()` would pick the fp64-less
   rusticl device. With no fp64 device, `device=gpu` fails loudly and
   `device=auto` falls back to OMP.
3. Polaris fp64 runs at 1/16 of the fp32 rate, but the solver is
   memory-bandwidth bound (~224 GB/s), so fp64 is acceptable.
4. 16 GiB of device memory, and `CL_DEVICE_MAX_MEM_ALLOC_SIZE` (often
   1/4 of global memory on Clover) limits single buffers: §4.6.

---

## 3. Module layout (compiled C, Pattern A)

```
r.hydro.anuga/
├── Makefile                  # Module.make, -lOpenCL, OpenMP, ETCFILES for cl/
├── main.c                    # G_parser, option validation, orchestration
├── ocl_backend.c/.h          # from r.watershed.opencl + fp64-aware device pick
├── dem_stack.c/.h            # multi-DEM reading at native resolution, blending, datum check
├── quadtree.c/.h             # refinement, 2:1 balance, fringe grading
├── mesh.c/.h                 # quadtree leaves -> conforming triangles (centre fan), export
├── hashmap.h                 # 64-bit key hash map (nodes, edges, leaves)
├── preflight.c/.h            # -p report: DEM stack scan, level snapping, estimates
├── sampler.c/.h              # any raster at its native res: nearest, bilinear, box mean
├── state.c/.h                # host SoA arrays, device buffers, upload/download
├── evolve.c/.h               # time loop: euler / rk2 / rk3, dt reduction, yield
├── boundary.c/.h             # boundary tagging + per-side BC setup
├── forcing.c/.h              # rain/evap/wind/pressure: constant, raster, STRDS, CSV
├── infiltration.c/.h         # Green–Ampt / GAR parameters, soil table, init
├── strds_io.c/.h             # t.rast.list resolve, t.create/t.register output
├── output.c/.h               # triangles -> output raster grid(s), per-yield writes
├── gauges.c/.h               # vector points -> time series table/CSV
├── massbal.c/.h              # volume, rain, infiltration, boundary flux accounting
├── kernels_omp.c             # OMP tier: same kernel bodies compiled as C
├── soil_green_ampt.csv       # Rawls et al. (1983) table, installed to ETC
├── cl/
│   ├── anuga_common.h        # types/macros shared by C99 and OpenCL C 1.1
│   ├── anuga_math.h
│   ├── anuga_kernels.cl      # extrapolate, fluxes, update, protect, friction
│   ├── anuga_boundaries.cl
│   ├── anuga_forcing.cl      # rain, evap, wind, pressure, infiltration
│   └── anuga_stats.cl        # max quantities, arrival time, reductions
├── r.hydro.anuga.md          # source of truth
├── r.hydro.anuga.html        # committed, kept in sync
├── tests/                    # pytest, grass.tools.Tools(session=...)
└── validation/               # scripts driving ANUGA Python for reference runs
```

### 3.1 One kernel source, three tiers

Tiers: `device=auto|gpu|cpu|omp`, i.e. GPU OpenCL, then PoCL CPU
OpenCL, then plain C with OpenMP. Each kernel body is written **once**
as a `static inline` function in a header that is valid both as C99 and
as OpenCL C 1.1. `cl/anuga_common.h` defines `GLOBAL`, `KERNEL`,
`REAL` and the index helpers per compiler. Each body has two thin
wrappers: an OpenCL `__kernel` using `get_global_id(0)`, and a C `for`
loop with `#pragma omp parallel for` in `kernels_omp.c`. Kernel
sources are embedded at build time (a generated `ocl_kernels.h`, the
`r.watershed.opencl` pattern) and also installed to
`$(ETC)/r.hydro.anuga/cl/` for development iteration.

### 3.2 Makefile

```makefile
MODULE_TOPDIR = ../..
PGM = r.hydro.anuga
LIBES = $(RASTERLIB) $(VECTORLIB) $(DBMILIB) $(GISLIB) $(MATHLIB)
DEPENDENCIES = $(RASTERDEP) $(VECTORDEP) $(DBMIDEP) $(GISDEP)
EXTRA_INC = $(VECT_INC)
EXTRA_CFLAGS = $(VECT_CFLAGS) $(OPENMP_CFLAGS) -std=c11
EXTRA_LIBS = -lOpenCL $(OPENMP_LIBPATH) $(OPENMP_LIB)
ETCFILES = soil_green_ampt.csv cl/anuga_common.h cl/anuga_math.h \
           cl/anuga_kernels.cl cl/anuga_boundaries.cl \
           cl/anuga_forcing.cl cl/anuga_stats.cl
include $(MODULE_TOPDIR)/include/Make/Module.make
default: cmd
```

---

## 4. Mesh: multi-resolution quadtree to conforming triangles

### 4.1 Concept

The computational domain is a **2:1-balanced quadtree** over the
current region. Each leaf is a square cell of size `res_max / 2^k`,
where `k` is its refinement level. Each leaf is triangulated by a
**fan from its centre** to all vertices on its boundary: its 4 corners
plus the midpoint of any side where the neighbour is one level finer (a
"hanging" node).

- No hanging nodes: 4 triangles. This is exactly the `rectangular_cross`
  pattern of decision D2.
- One to four refined sides: 5 to 8 triangles. All are right triangles
  with angles of 45°/90° or 26.6°/63.4°/90°, so there are no slivers.
- The mesh is **conforming** (every triangle edge is shared with
  exactly one other triangle or is a boundary), which is what ANUGA's
  edge-based flux scheme requires.
- Only **16 templates** exist (4-bit mask of refined sides), so
  per-leaf geometry is implicit: level, position and mask are enough.

A single-DEM run is the special case of a one-level quadtree. It
produces the same mesh as `anuga.rectangular_cross`, so the
triangle-for-triangle cross-check against ANUGA (§8, V4) still holds.
The single-level path is built first; multi-level is the same code with
more levels.

### 4.2 DEM stack (`dem_stack.c`)

`elevation=` accepts **several rasters**, for example
`elevation=lidarhd_mnt_0p5,rge_alti_5m,dem30`. Each raster is read at
**its own native resolution and extent** (from `Rast_get_cellhd`), not
resampled to the region: a per-DEM `Rast_set_input_window()` covers
the DEM's own cells inside the current region's extent. Rasters are
ordered by resolution, finest first. With `-o`, the order given on the
command line is the priority order instead.

At any point, elevation comes from the **finest DEM with a non-NULL
value there**, by bilinear interpolation between that DEM's cell
centres (D5). The coarsest DEM must cover the whole active domain.
Region cells where all DEMs are NULL are outside the domain.

**Vertical datum and bias check, fail loudly.** Real mixtures (IGN
LiDAR HD MNT in NGF-IGN69 heights versus an SRTM/Copernicus 30 m DEM on
an EGM96/EGM2008 geoid) routinely differ by metres, which would create
artificial walls or pits at the seam. For each overlapping pair,
`dem_stack.c` computes the median and MAD of (fine − coarse),
aggregated to the coarse grid, over the overlap:

- If |median| > `dem_bias_tolerance=` (default 0.5 m), the module
  fails with a message that gives the measured offset and suggests
  `dem_offset=` (an explicit per-DEM additive correction) or a
  reprojection or datum fix upstream.
- The offset is **never** applied silently.

**Seam blending.** Even with no bias, fine and coarse surfaces differ
locally. Inside the fine DEM footprint, within `blend_width=`
(default: 2 coarse cells) of the footprint edge, elevation is
`w·z_fine + (1−w)·z_coarse`, with `w` a smooth-step of distance to the
edge. This removes the step at the seam while keeping full fine detail
everywhere else. Blend weights are exported with `mesh_output=` for inspection.

Memory: a 0.5 m DEM over 20 × 20 km is 1.6·10⁹ cells (12.8 GB as
DCELL), so DEMs are **not** loaded whole. The mesh builder sweeps the
region in **bands one base cell high** (`res_max`). For each band it
reads only the needed rows of each DEM (row cache), builds that band's
leaves, and discards the rows. Host memory is then bounded by about
(band height / finest res) × row width per DEM.

### 4.3 Refinement criteria (`quadtree.c`)

1. **Resolution bounds.** `res_min=` defaults to the finest DEM's
   resolution. `res_max=` defaults to the coarsest DEM's resolution,
   **snapped** so that `res_max = res_min · 2^L`. Example: 0.5 m
   LiDAR HD and a 30 m DEM give `res_max = 32 m` and L = 6 levels
   (0.5, 1, 2, 4, 8, 16, 32 m). The snapping is reported, never done
   silently. Base cells tile the current region's extent, which must be
   a multiple of `res_max`. Otherwise the module fails with the aligned
   extent to use (`g.region align=`), and does not quietly shrink the
   domain.
2. **Data-driven target level.** A leaf's target level is the level
   whose size is at or just above the resolution of the finest DEM
   present anywhere in it. A 5 m DEM between the 0.5 m and 30 m ones
   therefore gets its own 4 m level.
3. **Optional user refinement.** `refine=` takes a raster or vector
   area with a target resolution (`refine_res=`), for example to refine
   around a river channel, dike or bridge even where only coarse data
   exists. `coarsen=` does the opposite, for example to keep a large
   flat LiDAR area far from the zone of interest at 2 m instead of
   0.5 m. Both are optional.
4. **Optional terrain criterion** (phase 5, off by default). Within
   the resolution allowed by data, coarsen leaves whose sub-cell
   relief (max − min of the fine DEM inside the leaf) is below
   `relief_tolerance=`, and never coarsen leaves containing a large
   relief. This cuts the triangle count on flat LiDAR areas without
   losing dikes or streets.
5. **Budget.** `max_triangles=` makes the dry run (`-p`) report
   whether the mesh fits. If it does not, the run fails with the
   per-level counts, so the user can choose `coarsen=` or a larger
   `res_min`.

### 4.4 Fringe: graded resolution transition

- **2:1 balance.** Edge-adjacent leaves differ by at most one level.
  This is needed for the centre-fan templates and keeps flux edges
  within a size ratio of 2.
- **Fringe width.** `fringe=N` (default 4) requires at least N leaves
  of each level between two level changes. The transition from 0.5 m
  to 32 m therefore ramps over 6 bands of 4 cells each (at least
  4·(0.5+1+2+4+8+16) = 126 m), not an abrupt 2:1 cascade. This is
  implemented as a morphological dilation of each level's footprint,
  finest to coarsest, before the balance pass.
- **Where the fringe lies.** It is placed **outside** the fine DEM
  footprint, on the coarse-data side, where the bilinear coarse DEM
  (blended per §4.2) supplies elevation. All fine data is then used at
  its native resolution. With `-i`, the fringe goes inside the
  footprint instead, trading some fine data for no fine-on-coarse
  refinement.
- **Why it is smooth numerically.** With the centre fan and 2:1
  balance, adjacent triangle areas differ by at most a factor of 2.
  ANUGA's DE second-order extrapolation and limiter are designed for
  unstructured meshes, so a mesh graded this way is what they expect.
  Lake-at-rest across all level changes (V2m) is the acceptance test.

### 4.5 Triangle construction (`mesh.c`)

1. Leaves are enumerated in **Morton (Z-order)**, which keeps
   neighbouring triangles close in memory for cache and coalescing on
   the GPU. Each leaf's triangles are contiguous.
2. Neighbour tables come from the quadtree's neighbour-finding (same
   level, coarser, or two finer), not from a generic mesh search. An
   edge facing an inactive cell or the region edge becomes a
   **boundary edge**, tagged `north|south|east|west|null`.
3. Geometry (areas, edge lengths, normals, radii, centroids, edge
   midpoints) uses the formulas of ANUGA's `General_mesh`/`Mesh`, so a
   one-level mesh matches `rectangular_cross` bit for bit.
4. Vertex elevation is bilinear between cell centres of the governing
   DEM, per D5 and §4.2. The centre vertex of a leaf whose DEM is finer
   than the leaf uses the **mean of the DEM cells inside the leaf**
   (area-consistent), not a point sample. This matters when the
   `coarsen=` or relief criteria make a leaf coarser than its data.
   ANUGA's DE centroid averaging then gives centroid elevations. These
   steps run on the host in fp64. The resulting centroid elevation is
   then **quantised once** (§4.7), and only the quantised value is used
   from then on, by host and device alike.
5. Coordinates are in map units relative to the region's SW corner
   (ANUGA `geo_reference` style), which keeps doubles well conditioned.
6. Latitude-longitude projects are a **fatal error** in v1. Tell the
   user to reproject.

### 4.6 Memory and CFL cost

With explicit per-triangle arrays, the phase 0 estimator
(`preflight.c:bytes_per_triangle`) uses ANUGA's own count
(`gpu_estimate_required_memory()`: 50 double and 11 int64 arrays, plus
3 backup arrays, i.e. 512 B per triangle). It drops the fp64 bed arrays
(§4.7) and uses int32 indices, giving **440 B per triangle**, plus 8 B
per requested max/time statistic, 8 B for GA, and 52 B for GAR:

| Case | Triangles | Explicit (phase 1–5) | Compact, per-leaf template geometry (phase 8) |
|---|---|---|---|
| 1000 × 1000 single level | 4 M | ~1.8 GB | ~0.8 GB |
| 30 × 30 km at 32 m, plus a 2 × 2 km LiDAR HD patch at 0.5 m | ~0.9 M + ~64 M | ~27 GB, too large | ~12 GB |
| Plumergat as measured by `-p`: 275 km² at 32 m plus 2 × 2 km RGE ALTI at 1 m | 1.05 M + 0.12 M fringe + 16 M | **7.0 GiB**, fits the WX 7100 | ~3 GB |

The fine patch dominates both memory and cost. Two consequences are
documented and enforced:

- The pre-flight check in `-p` and at start-up reports per-level
  triangle counts, host and device bytes, and the largest buffer
  against `CL_DEVICE_MAX_MEM_ALLOC_SIZE` (Clover). It fails loudly
  with an actionable suggestion (`res_min=`, `coarsen=`,
  `relief_tolerance=`, a smaller region).
- **CFL.** A global time step is set by the smallest cells. At 0.5 m
  and 3 m/s the step is about 0.05 s everywhere, including the 32 m
  cells, where it is wasted. This is acceptable when the fine area
  dominates the triangle count (usual). **Local time stepping** (LTS:
  levels advance with 2^k·dt and synchronise at level boundaries) is
  the planned phase 9 optimisation. The quadtree levels map directly
  onto LTS levels.

Index type is `int` (32-bit) on device. Buffers larger than the max
allocation size are split in chunks by Morton range.

### 4.7 Elevation on the device: scaled unsigned integers (D7)

**What is static and what is not.** In ANUGA's DE scheme the only
static elevation read by the time loop is the **centroid bed**
(`bed_centroid_values`, one per triangle). The edge and vertex bed
values are not DEM data: they are fp64 reconstruction values rebuilt
every step. `core_extrapolate_second_order_edge` sets
`bed_edge = stage_edge − height_edge` for **every** triangle at the
end of its iteration (core_kernels.c:309–311). This even overwrites the
`bed_edge = bed_c` assigned earlier to cells with three boundaries
(line 157).
`core_distribute_edges_to_vertices` derives the vertex beds from those
edge values (lines 362–364). The boundary bed is the edge bed copied
(sw_domain_openmp.c:422). Hence:

- **DEM on device = one `uint32` per triangle**, `zq[k]`. It replaces
  the fp64 `bed_centroid_values`: 4 B instead of 8 B per triangle,
  uploaded once and never written.
- **Reconstructed bed values are not stored at all.** `bed_edge`,
  `bed_vertex` and `bed_boundary` become on-the-fly expressions
  (`stage_edge − height_edge`, with the same operands and the same
  single fp64 subtraction, so results are bitwise identical to storing
  them). That drops 3 + 3 + boundary fp64 arrays, about 48 B per
  triangle, and their per-step writes. Sloped Manning friction uses
  ANUGA's **edge-based** variant
  (`core_manning_friction_sloped_semi_implicit_edge_based`), so no
  vertex beds are needed on device.

**Encoding.**

```
Z0      = floor(min_DEM_over_domain · 10000) − 10000     int64, units of 0.1 mm (1 m margin)
zq[k]   = (uint32) llround(z_c[k] · 10000) − Z0          host side, once, fp64 → integer
z(k)    = (double)((int64) zq[k] + Z0) * 1.0e-4          kernel side, exact integer add, then one rounding
```

- **Range:** 2³² × 0.1 mm = 429 km above the datum, so real
  terrain-plus-bathymetry never approaches it. `Z0` absorbs negative
  elevations: −21 m in the Golfe du Morbihan GLO-30, or deep
  bathymetry for surge runs. The host **fails loudly** if any
  quantised value falls outside `[0, 2³²−1]`, or if a DEM value is
  NaN.
- **Resolution:** 0.1 mm, two orders of magnitude finer than LiDAR
  vertical accuracy (~5–10 cm).
- **Dequantisation is one integer add and one correctly rounded fp64
  multiply.** It is never written as `Z0·1e-4 + zq·1e-4`, which an
  OpenCL compiler (FP_CONTRACT defaults on) or gcc could fuse into an
  FMA and round differently from the host. The single-multiply form
  cannot be contracted, so **host (OMP tier) and device produce
  bitwise-identical bed values**. `anuga_common.h` also sets
  `#pragma OPENCL FP_CONTRACT OFF` for the helper, and the C build of
  that header uses `-ffp-contract=off`.
- **Well-balancing is preserved.** Every use of the bed goes through
  the same `z(k)`. A dry cell or a lake at rest is initialised with
  `stage = z(k) + h`, where `z(k)` is exactly the value the kernels
  will subtract, so h is exact (0 when dry). Stage and momenta stay
  fp64, so depths below the 0.1 mm quantum (DE1's 1e-5 m threshold) are
  represented exactly as before. Only the terrain is quantised, not
  the water.

**Kernels "assume scaled":**

| Operation | Form |
|---|---|
| Bed-only comparisons: min/max of centroid beds, dry-neighbour and riverwall tests `zwall > max(zc, zc_n)`, relief criterion, sort/merge on elevation | **integer**, on `zq`, with no dequantisation (riverwall crest heights are quantised with the same `Z0`) |
| Bed differences (slopes, relief, bias statistics): `Δz = (double)((int64)zq_a − zq_b) * 1e-4` | exact integer difference, then one rounding |
| Depth at centroid: `h = fmax(stage − z(k), 0)` | one dequantisation, fp64 subtraction |
| Everything on the reconstructed state (edge/vertex beds, `z_half`, fluxes) | fp64, unchanged from ANUGA |

The host-side DEM stack, blending and bias checks (§4.2) stay in fp64
(GRASS DCELL/FCELL). Quantisation happens exactly once, when the mesh
is finalised. The `mesh_output=` export writes `zq`, `Z0` and the
dequantised z, so external tools (ANUGA Python for V4/V4m) are given
**the same quantised terrain**.

Memory effect (explicit layout, per triangle): the bed goes from 7
fp64 arrays (centroid + 3 edge + 3 vertex, 56 B) to one `uint32`
(4 B). That brings the ~0.8 KB/triangle estimate of §4.6 down to about
0.75 KB. The larger gain is bandwidth: the extrapolation and flux
kernels no longer write or read 6 bed values per triangle per substep.

---

## 5. Numerical core (OpenCL kernels)

One work-item per triangle unless noted:

| Kernel | Ported from | Notes |
|---|---|---|
| `protect` | `core_protect` | returns the mass error, reduced |
| `extrapolate_second_order_edge` | `core_extrapolate_second_order_edge` | the largest kernel (~300 lines); DE limiter, beta and `_dry` variants, velocity extrapolation; reads `zq` (§4.7), no longer writes `bed_edge` |
| `distribute_edges_to_vertices` | `core_distribute_edges_to_vertices` | only on yield steps, if vertex output is needed; vertex bed computed, not stored |
| `compute_fluxes_central` | `core_compute_fluxes_central` | per triangle, 3 edges; each internal edge computed twice (no atomics, as ANUGA); `zl`/`zr` = `stage_edge − height_edge` on the fly; `explicit_update`, `max_speed`, local dt = `radius / max_speed`, `boundary_flux_sum` |
| `reduce_min_dt` | none | two-pass local-memory tree reduction, OpenCL 1.1 |
| `manning_friction_flat` / `_sloped` | `core_manning_friction_flat_semi_implicit` / `..._sloped_semi_implicit_edge_based` | `friction_method=flat` (default, as ANUGA: `use_sloped_mannings=False`) or `sloped`; the sloped variant is the edge-based one, so no static vertex beds are needed |
| `update_conserved_quantities` | `core_update_conserved_quantities` | dt as a kernel argument |
| `backup` / `saxpy` / `saxpy3` | `core_backup_*`, `core_saxpy_*` | RK2/RK3 |
| `fix_negative_cells` | `core_fix_negative_cells` | with a count reduction |
| `compute_water_volume` | `gpu_compute_water_volume` | deterministic pairwise fp64 sum |
| `rate_forcing` | `gpu_rate_operator.c` | rain/evap; negative rates clamped to available water |
| `infiltration_ga` / `_gar` | new (§6) | after the RK update, operator-split |

### 5.1 Time loop (`evolve.c`)

```
while t < end_time:
    next_yield = min(t_next_output, t_next_forcing_change, end_time)
    apply_time_varying_forcing(t)        # upload only when the STRDS interval changes
    dt = step(max_dt = next_yield - t)   # rk2 (DE1 default) | euler | rk3
    rate_forcing(dt); infiltration(dt)   # operator split, mass-conservative clamps
    t += dt
    update_max_quantities(); update_arrival_time(t)
    if output time: write outputs(t)     # asynchronous, double-buffered
```

- The **only per-step blocking read** is the min-dt double. A
  device-side dt buffer comes later (phase 8).
- `max_timestep` and `min_timestep` are honoured. A persistent dt
  below `min_timestep` is fatal, reporting the location and level of
  the limiting triangle.
- Progress via `G_percent` on simulated time. Per-yield dt statistics
  go to `G_verbose_message`.

### 5.2 Boundaries

Per side, `boundary=north:type[:value],south:...,east:...,west:...,null:...`:

| Type | ANUGA class | Values |
|---|---|---|
| `reflective` (default) | `Reflective_boundary` | none |
| `transmissive` | `Transmissive_boundary` | none |
| `dirichlet` | `Dirichlet_boundary` | stage[,xmom,ymom] |
| `stage` | `Transmissive_n_momentum_zero_t_momentum_set_stage_boundary` | constant, or `stage_series=` (CSV `time,stage` or a table) |
| `flather` | `Flather_external_stage_zero_velocity_boundary` | stage or series |

---

## 6. Infiltration model (decision D4)

### 6.1 Choice and references

**Green–Ampt with ponded head**, parameterised by soil texture.

- Green, W. H., & Ampt, G. A. (1911). Studies on soil physics. *The
  Journal of Agricultural Science*, 4(1), 1–24.
  doi:10.1017/S0021859600001441
- Rawls, W. J., Brakensiek, D. L., & Miller, N. (1983). Green-Ampt
  infiltration parameters from soils data. *Journal of Hydraulic
  Engineering*, 109(1), 62–70.
  doi:10.1061/(ASCE)0733-9429(1983)109:1(62). Parameters from about
  5000 soil horizons, averaged by USDA texture class.
- Coupling of Green–Ampt with 2D full shallow-water equations on a
  per-cell, per-time-step basis: Fernández-Pato, J.,
  Caviedes-Voullième, D., & García-Navarro, P. (2016). Rainfall/runoff
  simulation with 2D full shallow water equations: sensitivity analysis
  and calibration of infiltration parameters. *Journal of Hydrology*,
  536, 496–513. doi:10.1016/j.jhydrol.2016.03.021
- Ponded-head term and derivation of Green–Ampt from Richards'
  equation: La Follette, P., Ogden, F. L., & Jan, A. (2023). Layered
  Green and Ampt infiltration with redistribution. *Water Resources
  Research*, 59, e2022WR033742. doi:10.1029/2022WR033742 (eq. 4 and
  §3.2.8).
- Redistribution between storms (option `infiltration=gar`): Ogden,
  F. L., & Saghafian, B. (1997). Green and Ampt infiltration with
  redistribution. *Journal of Irrigation and Drainage Engineering*,
  123(5), 386–393. doi:10.1061/(ASCE)0733-9437(1997)123:5(386).
  Validated against a numerical Richards' equation solution for all 11
  USDA textures with multiple rainfall pulses. The equations are
  transcribed from this paper (§6.3). HEC-RAS 2D implements the same
  method (USACE HEC, *HEC-RAS Technical Reference*, "Green-Ampt"),
  which serves as a secondary cross-check.

Why this model:

- **Physically based.** Green–Ampt is the sharp-front limit of
  Richards' equation, with Darcy flux driven by gravity plus capillary
  and ponding head (La Follette et al. 2023 derive it explicitly).
  Parameters are measurable soil properties, not calibration
  coefficients as in Horton or SCS-CN.
- **Lean.** One state per triangle (cumulative infiltration F) and
  about 15 flops per step. GAR adds one more state (θ₀) and an ODE
  evaluated only during rainfall hiatus.
- **Couples naturally with the shallow-water state.** The ponded depth
  h is the solver's own depth, so infiltration from ponded or flowing
  water (run-on, floodplains) comes for free. That is exactly what a
  rain-on-grid flood model needs, and what the 2016 paper shows in a
  2D SWE finite-volume scheme.

Known limits, documented in the manual: deep homogeneous soil, uniform
initial moisture, no macropores, no groundwater table. Plain GA
**never recovers infiltration capacity**, so for multi-day ERA5
forcing with dry spells use `infiltration=gar`. Layered soils (LGAR)
are out of scope: they need multiple fronts and layer-interface
solves, which are not lean.

### 6.2 Equations (per triangle, operator-split after the RK update)

State: cumulative infiltration F [m], initially 0. Parameters: Ks
[m/s], wetting-front suction head ψ_f [m], Δθ = θ_s − θ_i [−].

```
h      = max(stage − bed, 0)                      # ponded depth after rain this step
if F > F_eps:
    f_p = Ks · (1 + (ψ_f + h) · Δθ / F)           # GA capacity with ponded head
    dF  = f_p · dt
else:                                             # F → 0: f_p unbounded; use early-time GA
    dF  = sqrt(2 · Ks · (ψ_f + h) · Δθ · dt)      # (the small-t limit of the GA solution)
dF     = min(dF, h)                               # supply limit: never infiltrate more than is there
if soil_depth given: dF = min(dF, Δθ · soil_depth − F)   # saturated profile: no further infiltration
F     += dF
stage −= dF                                       # momenta scaled by (h − dF)/h to keep velocity
```

- **Mass conservation is exact.** Every metre removed from `stage` is
  added to F and to the mass-balance infiltration account.
- The explicit update is stable because dt is the hydraulic CFL step
  (sub-second to seconds), much smaller than the Green–Ampt time scale
  (ψ_f·Δθ/Ks), and the `min(dF, h)` clamp bounds it.
- Cost: one division, one fused multiply-add chain, and a rare `sqrt`.

### 6.3 GAR option (`infiltration=gar`)

Transcribed from the primary source, Ogden & Saghafian (1997),
`docs/references/GARedistributonOgden1997.pdf` (J. Irrig. Drain. Eng.
123(5):386–393), with equation numbers as in the paper. θi is the
initial water content, θ0 the surface/profile water content, θe the
water content at natural saturation, θr the residual content, Z the
wetting-front depth, and Θ = (θ − θr)/(θe − θr).

```
(1)  f_p = Ks · (1 + Hc·Δθ / F)                    GA capacity (+ ponded head h, §6.2)
(9)  K(θ) = Ks · Θ^(3 + 2/λ)                        Brooks–Corey, a = 2, b = 3
(15) G(θi, θ) = Hc · (Θ^(3+1/λ) − Θi^(3+1/λ)) / (1 − Θi^(3+1/λ))
(17) Hc = −Ψb · (2 + 3λ) / (1 + 3λ)                 GA suction from Brooks–Corey
(2)  (θ0 − θi)·Z = F_h + (r − K_i)·(t − t_h)        mass during hiatus
(5)  dθ0/dt = (1/Z) · [ r_h − K_i − ( K(θ0) + Ks·G(θi, θ0)/Z ) ]
     dZ/dt  from (4): (θ0 − θi)·dZ/dt = Ks·G(θi, θ0)/Z + K(θ0)
```

- **Hiatus**: "after ponding, the rainfall rate r is less than Ks, all
  ponded surface water is infiltrated". In this model that means the
  triangle's depth h reaches 0. The profile stays rectangular while θ0
  drops and Z grows, conserving mass (eq. 2).
- **Two-front scheme** (at most 2 profiles, so per-cell state is
  fixed-size, which suits the GPU):
  - When r > Ks again after a hiatus, a second saturated profile
    forms. Its capacity is eq. (1) with `Δθ' = θe − θ1` (18) and
    `F' = F − F1` (19), where θ1 and F1 belong to the first profile.
    The first profile keeps redistributing with (5).
  - When Z2 = Z1 the profiles merge, and eq. (1) applies again with the
    original Δθ.
  - If r < Ks before Z2 reaches Z1, the merge is **forced**, at the
    first-moment depth
    `Zm = [(θ1 − θi)·Z1 + (θe − θ1)·Z2] / (θe − θi)` (20).
  - The authors note this approximation holds for relatively short
    redistribution periods ((θ1 − θi) > (θe − θ1)). The manual documents
    that limit.
- **Before the first ponding**: eq. (5) also raises the surface
  saturation from θ0 = θi, with (θ0 − θi)/Z = 1 at the very first
  rainy step, because Z = 0 there. The paper integrates (5) explicitly
  over O(30 s) steps (RK4 before the first ponding).
- **Implementation:**
  - Per triangle: θ0, Z1, θ1, F1, F2, a phase flag (none, pre-ponding,
    ponded, hiatus, two-front), plus the GA F. That is 6 doubles and
    1 byte.
  - The soil update runs every `infil_substep=` (default 30 s of
    simulated time, as in the paper) using an accumulated supply r_h
    over the substep. On the hydraulic steps in between, only eq. (1)
    and the mass clamp run, so it stays lean.
  - `K(θ)` and `G(θi,·)` are pre-tabulated per soil class at start-up
    (64-entry tables, linear interpolation), so there are no `pow`
    calls in the kernel.
  - RK4 in the pre-ponding stage and explicit Euler/Heun elsewhere,
    with Z and θ0 kept within [θi, θe].
- **Evapotranspiration**: the paper has no ET term (r_h only). An
  optional sink `E_v` from `evap_strds=` subtracts from r_h during a
  hiatus, as HEC-RAS does (its version of eq. 5 includes `− E_v`).
  Marked as an extension beyond the paper and off by default.
- **V9i validation** uses the paper's own experiment: the 11 textures
  of Table 1, initial θi = wilting point, and the two rain pulses of
  Table 2 (the second starting at t = 3 h), compared with its Fig. 3
  (infiltration rate) and Fig. 4 (surface Θ0), digitised. Mass must
  close exactly.

### 6.4 Parameters and inputs

- `soil_texture=` (raster, USDA class codes 1–11) looks up
  `soil_green_ampt.csv`, installed to ETC and pinned by a unit test. A
  user-supplied table (`soil_table=`) overrides it. There are **two
  parameter sets, one per method**, because each must be internally
  consistent with its equations:

  **GA set: Rawls, Brakensiek & Miller (1983), Table 2, texture-class
  rows** (transcribed from `docs/references/rawls_et_al_1983.pdf`,
  p. 67; ψ_f is the antilog of the log mean):

  | Code | Texture | n | Total porosity φ | Effective porosity θe | ψ_f (cm) | Ks (cm/h) |
  |---|---|---|---|---|---|---|
  | 1 | Sand | 762 | 0.437 | 0.417 | 4.95 | 11.78 |
  | 2 | Loamy sand | 338 | 0.437 | 0.401 | 6.13 | 2.99 |
  | 3 | Sandy loam | 666 | 0.453 | 0.412 | 11.01 | 1.09 |
  | 4 | Loam | 383 | 0.463 | 0.434 | 8.89 | 0.34 |
  | 5 | Silt loam | 1206 | 0.501 | 0.486 | 16.68 | 0.65 |
  | 6 | Sandy clay loam | 498 | 0.398 | 0.330 | 21.85 | 0.15 |
  | 7 | Clay loam | 366 | 0.464 | **0.390** (printed 0.309) | 20.88 | 0.10 |
  | 8 | Silty clay loam | 689 | 0.471 | 0.432 | 27.30 | 0.10 |
  | 9 | Sandy clay | 45 | 0.430 | 0.321 | 23.90 | 0.06 |
  | 10 | Silty clay | 127 | 0.479 | 0.423 | 29.22 | 0.05 |
  | 11 | Clay | 291 | 0.475 | 0.385 | 31.63 | 0.03 |

  The original confirms loam Ks = 0.34 cm/h and silt loam 0.65 cm/h.
  The TUFLOW wiki transcription (loam 7.6 mm/h, silt loam 3.4 mm/h) is
  wrong and must not be used. **Probable typo in the original:** the
  clay loam effective porosity is printed as 0.309, but its ±1σ range
  (0.279–0.501) is centred on 0.390, as every other row's mean is on
  its range. Ogden & Saghafian (1997) Table 1 uses 0.390, so 0.390 is
  adopted, with a comment in the CSV. Table 2 also gives A/B/C-horizon rows and
  ±1σ ranges. The horizon rows are selectable with `soil_horizon=A|B|C`
  (with a fallback to the class row where the paper lacks a value), and
  the ranges are kept for later sensitivity runs.

  **GAR set: Ogden & Saghafian (1997) Table 1** ("after Rawls et al.
  1982, 1983", the parameters the paper's own validation used;
  transcribed from `docs/references/GARedistributonOgden1997.pdf`,
  p. 389). Hc is computed from Ψb and λ with eq. (17). The table was
  spot-checked: loam 11.15·(2 + 0.756)/(1 + 0.756) = 17.50 ✓, sand
  7.26·(2 + 2.082)/(1 + 2.082) = 9.62 ✓. The unit test recomputes all
  11 rows.

  | Code | Texture | φ | θe | θr | θ wilting | Ψb (cm) | λ | Ks (cm/h) | Hc (cm) |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | Sand | 0.437 | 0.417 | 0.020 | 0.033 | 7.26 | 0.694 | 23.56 | 9.62 |
  | 2 | Loamy sand | 0.437 | 0.401 | 0.035 | 0.055 | 8.69 | 0.553 | 5.98 | 11.96 |
  | 3 | Sandy loam | 0.453 | 0.412 | 0.041 | 0.095 | 14.66 | 0.378 | 2.18 | 21.53 |
  | 4 | Loam | 0.463 | 0.434 | 0.027 | 0.117 | 11.15 | 0.252 | 1.32 | 17.50 |
  | 5 | Silt loam | 0.501 | 0.486 | 0.015 | 0.133 | 20.79 | 0.234 | 0.68 | 32.96 |
  | 6 | Sandy clay loam | 0.398 | 0.330 | 0.068 | 0.148 | 28.08 | 0.319 | 0.30 | 42.43 |
  | 7 | Clay loam | 0.464 | 0.390 | 0.075 | 0.197 | 25.89 | 0.242 | 0.20 | 40.89 |
  | 8 | Silty clay loam | 0.471 | 0.432 | 0.040 | 0.208 | 32.56 | 0.177 | 0.20 | 53.83 |
  | 9 | Sandy clay | 0.430 | 0.321 | 0.109 | 0.239 | 29.17 | 0.223 | 0.12 | 46.65 |
  | 10 | Silty clay | 0.479 | 0.423 | 0.056 | 0.250 | 34.19 | 0.150 | 0.10 | 57.77 |
  | 11 | Clay | 0.475 | 0.385 | 0.090 | 0.272 | 37.30 | 0.165 | 0.06 | 62.25 |

  The GA and GAR sets differ (for loam, Ks 0.34 vs 1.32 cm/h and Hc
  8.89 vs 17.50 cm) because the 1983 GA values are direct texture
  means, while the GAR values come from Brooks–Corey fits (Rawls et
  al. 1982) with Hc from eq. (17). **They are never mixed:**
  `infiltration=ga` uses the 1983 set, `infiltration=gar` uses
  Ogden & Saghafian's Table 1 (for both the GA ponding phase and the
  redistribution), and the manual explains why. The HEC-RAS slide
  table (loam Ψ = 31.5 cm, sand Ks = 21.0 cm/h) disagrees with both
  papers and is **not** used. The wilting-point column gives the
  default initial θi for GAR validation runs (the paper's setup).
- Direct rasters instead: `ks=`, `suction=`, `porosity=`, with `-k` to
  give Ks in mm/h.
- Initial moisture: `initial_saturation=` (0–1, constant or raster).
  θ_i = θ_r + S_e·(θ_s − θ_r), so Δθ = θ_e·(1 − S_e). ERA5-Land
  `volumetric_soil_water_layer_1` is a natural source. Adding it to
  *t.in.era5* is a small, separate change.
- `soil_depth=` (optional raster, m): depth to an impervious layer.
- `impervious=` (optional raster, fraction 0–1): scales Ks by
  (1 − fraction), for urban areas (for example from OSO or Urban Atlas
  land cover in France).
- Outputs: `infiltration=` STRDS (cumulative F, mm), `max_infiltration`
  and the mass-balance column.
- For France, `soil_texture` can be derived from the ISRIC SoilGrids
  sand/silt/clay fractions with the USDA triangle in *r.mapcalc*. The
  manual will give the expression.

---

## 7. Inputs (`main.c` options)

### 7.1 Static

| Option | Type | Meaning |
|---|---|---|
| `elevation=` | `G_OPT_R_ELEVS` (multiple) | **required**; one or more DEMs, finest first (§4.2) |
| `domain=` | raster | active domain (non-NULL cells), e.g. the Plumergat `anuga_domain` watershed mask; default: the coarsest DEM's valid cells |
| `dem_offset=` | double list | explicit per-DEM vertical correction |
| `dem_bias_tolerance=`, `blend_width=` | double | §4.2 |
| `res_min=`, `res_max=`, `fringe=`, `refine=`, `refine_res=`, `coarsen=`, `relief_tolerance=`, `max_triangles=` | | §4.3–4.4 |
| `manning=` / `manning_value=` / `landcover=` + `manning_rules=` | raster / double / raster + file | Manning's n (ANUGA default 0.03) |
| `initial_depth=` / `initial_stage=`, `initial_xmom=`, `initial_ymom=` | raster | default dry; hot start |
| `buildings=` | raster/vector | raise elevation by `building_height=`, the usual ANUGA practice; best combined with `refine=` at LiDAR resolution |
| infiltration options | | §6.4 |

All non-DEM rasters are sampled at triangle centroids **at their own
native resolution** by `sampler.c`. They are not resampled to the
region, so a 10 m land-cover map is used as 10 m data under both the
0.5 m and 32 m parts of the mesh. Where a triangle is larger than the
raster cells, the sampler uses the area mean. Where it is smaller, it
uses nearest (categorical) or bilinear (continuous) sampling, selected
per option.

### 7.2 Time-varying forcing

Each forcing accepts **one of** a constant, a raster, a STRDS, or a CSV
time series (spatially uniform). A STRDS is resolved once via
`t.rast.list columns=name,start_time,end_time` (the
`r.hydro.rri/main.c` `resolve_strds_steps` pattern). The value is held
over each map's `[start,end)` interval (ANUGA
`Raster_time_slice_data` semantics) and uploaded **only when the
interval changes**.

| Forcing | Options | ERA5 source (*t.in.era5*) | Solver effect |
|---|---|---|---|
| Rainfall | `rain=`, `rain_strds=`, `rain_value=`, `rain_hyetograph=`, `rain_units=` | `precipitation` (mm/d; mm/h with **-h**) | rate operator |
| Evaporation | `evap=`, `evap_strds=`, `evap_units=` | `potential_evaporation` (mm/d) | negative rate, clamped to available water |
| Infiltration | §6 | (`volumetric_soil_water_layer_1` for S_e, future) | §6.2 |
| Wind stress | `wind_u_strds=`, `wind_v_strds=` | `wind_u`, `wind_v` (m/s) | port of `wind_stress_operator` |
| Pressure | `pressure_strds=`, `pressure_units=` | `surface_pressure` (kPa) | port of `barometric_pressure` |
| Inflow | `inflow=` (points) + `inflow_series=` | discharge from *r.hydro.rri* / *r.hydro.hbv* | port of `gpu_inlet_operator.c` |

- Units are explicit and never guessed. A STRDS with no units option
  is fatal and lists the choices. A granularity/units mismatch (1-day
  STRDS with mm/h) raises a warning.
- Snowfall is ignored (no snow model) and documented as such.
- Any STRDS with absolute time works: GPM IMERG, radar via
  *t.rast.import*, or design storms from *t.rast.mapcalc*.
- ERA5 cells (9–31 km) are far coarser than the mesh. The sampler
  reads them at native resolution, so forcing is blocky. Smooth it
  upstream (*r.resamp.interp*) if wanted; there is no silent
  interpolation.

### 7.3 Time control and backend

`start=`, `end=` / `duration=`, `output_step=`, `max_timestep=`,
`min_timestep=`, `cfl=`, `algorithm=DE1|DE0|DE2`,
`friction_method=flat|sloped`, `device=auto|gpu|cpu|omp`, `nprocs=`.
`-p` is a dry run: device, mesh level counts, memory estimate, DEM
bias report, then exit.

---

## 8. Outputs

### 8.1 Output grids for a multi-resolution mesh

GRASS rasters have a single resolution, and writing the whole region
at 0.5 m is usually infeasible (30 km × 30 km would be 3.6·10⁹ cells
per map). Output therefore goes to **one or more output grids**:

- **Main grid**: the current region, at the region resolution. By
  default the user sets the region to the coarse DEM, so the 30 m or
  32 m main maps cover the whole domain.
- **Detail grids**, with `-f` or `detail_output=`: for each finer DEM
  footprint (or `refine=` area), a second STRDS
  (`<name>_detail<k>`) at that level's resolution, clipped to the
  footprint's bounding box.
- **Transfer** (`output.c`): an output cell coarser than the
  triangles gets the **area-weighted mean** of the triangles
  overlapping it. For depth this is volume-preserving: Σ depth·area is
  identical. An output cell finer than a triangle gets the value of the
  triangle containing its centre. Lookup is by quadtree descent,
  O(levels). Velocities in coarse cells are momentum-weighted
  (Σ uh·A / Σ h·A), not a mean of velocities.

### 8.2 Time series (STRDS per output grid, one map per `output_step`)

Selected with `outputs=depth,speed,...`; `depth` is always produced.

| Quantity | Definition |
|---|---|
| `depth` | h (m); below `min_depth=` written as 0, or NULL with **-n** |
| `stage` | w (m) |
| `xvelocity`, `yvelocity`, `speed`, `direction` | from momentum with ANUGA's `velocity_zero_height` protection; direction in degrees CCW from east |
| `discharge` | unit discharge h·speed (m²/s) |
| `xmomentum`, `ymomentum` | hot-start inputs |
| `froude` | speed / sqrt(g·h) |
| `hazard` | h·(speed + 0.5), a documented hazard rating |
| `infiltration` | cumulative F (mm) |

Each map gets units, a title, a colour table (`water` for depth, `bcyr`
for speed) and history. The output type is FCELL by default, DCELL
with `-d`. STRDS are created and registered at the end with `t.create`
and a single `t.register file=` (the `r.hydro.rri` pattern), in
absolute time `start + t`. Raster writing runs on a host thread while
the device keeps stepping (double-buffered read-back).

### 8.3 Summary rasters, gauges and mass balance

- On each output grid: `max_depth`, `max_speed`, `max_stage`,
  `max_hazard`, `arrival_time` (first h > `arrival_depth=`),
  `inundation_duration`, `final_*` for hot start, and `mesh_level`
  (the quadtree level per cell, useful to show where the fine DEM was
  used). Maxima are updated on device **every step**.
- `gauges=` (vector points) with `gauge_output=`: time series of
  stage, depth, u, v and speed from the containing triangle.
- `massbalance=` CSV per output step: volume, cumulative rain,
  evaporation, infiltration, boundary flux, inflows, and relative
  error. **-m** prints the error at the end.
- Outputs are never overwritten without `--overwrite`.

---

## 9. Validation plan

The ANUGA Python package, installed from `$HOME/dev/anuga_core`, is
the reference. Each test runs first on local PoCL, then on the GPU host
`yann@10.42.0.89` (build in `~/dev/r.hydro.anuga` against its
`~/dev/grass`, synced with git). Tests use pytest, `tests/*_test.py`,
with `grass.tools.Tools(session=...)`.

| # | Test | Pass criterion |
|---|---|---|
| V1 | Superseded in phase 2 by V4, which is stricter: whole runs, including every kernel, are bitwise identical to ANUGA's compiled `core_kernels.c`. OpenCL vs OMP is V13 | — |
| V2 | Lake at rest, immersed bump / steep island (ANUGA `validation_tests/analytical_exact`), on the quantised bed | velocities < 1e-10 |
| **V1q** | Elevation quantisation: `zq` round-trip on real DEMs (Plumergat GLO-30 incl. −21 m cells, RGE ALTI 1 m); dequantised z bitwise equal on host (OMP), PoCL and Clover; out-of-range and NaN DEM values are fatal | max \|z − z_dem\| ≤ 0.05 mm; bitwise equality across tiers |
| **V2m** | **Lake at rest on a 3-level quadtree** (fine patch in the middle, fringe 4) with bumpy terrain crossing the level transitions | velocities < 1e-10: the well-balanced property survives hanging-node templates |
| V3 | Dam break, wet and dry | L1 vs analytical within ANUGA's reported error |
| V4 | Same `rectangular_cross` mesh in ANUGA Python and in the module, DE1 and DE0, 200 steps; ANUGA is given the dequantised elevation from `mesh_output=` | per-triangle stage diff < 1e-9 relative; identical dt sequence |
| **V4m** | ANUGA Python **on the module's exported multi-level mesh** (vertices and triangles are written with `mesh_output=` and loaded through `anuga.Domain(points, elements)`) | per-triangle diff < 1e-9: proves the unstructured path is ANUGA-exact |
| **V5m** | Radial dam break or plane wave crossing 32 → 0.5 m levels, compared with a uniform 0.5 m run | reflected-wave amplitude at the fringe < 1% of the incident; arrival time within one coarse cell |
| V6 | Rain on a tilted plane, transmissive outlet; then the same on a 2-level mesh | mass error < 1e-10 relative |
| **V7i** | Green–Ampt: ponded column (no flow), compared with the implicit analytic GA solution t(F) for 3 textures | F(t) error < 1%; exact mass accounting |
| **V8i** | Rain on a slope with GA: runoff starts at the analytic ponding time t_p = Ks·ψ_f·Δθ / (r·(r − Ks)) | t_p within one output step |
| V9i | GAR: Ogden & Saghafian (1997) experiment: 11 textures (Table 1), θi = wilting point, two pulses (Table 2), compared with Figs. 3–4 (Richards-equation reference) digitised; plus the HEC-RAS GA case (Daliakopoulos 2015 parameters) | within the digitising error of the GAR curves; mass exact |
| V10 | STRDS forcing: synthetic 3-map STRDS | volume = Σ rate·area·interval exactly; one upload per interval |
| V11 | Multi-DEM bias check: synthetic 5 m offset | fails with the measured offset; passes with `dem_offset=` |
| V12 | *t.in.era5* end to end, plus a real LiDAR HD 0.5 m tile embedded in a 30 m DEM (France) | runs; mass balance closes; `mesh_level` map correct; detail STRDS registered |
| V13 | Clover vs PoCL vs OMP on V4/V4m/V6 | < 1e-9 |

Performance (phase 8): triangles/s and steps/s on single- and
multi-level meshes on the WX 7100, compared with ANUGA's own CPU
OpenMP build on the same host.

---

## 10. Phased implementation

Each phase ends validated, committed, and with the manual updated.

- **Phase 0 — scaffolding. DONE 2026-09-23.** Makefile, GPL-3.0-or-later
  `LICENSE` plus `LICENSE.ANUGA` (D1), and `main.c` with the full
  `G_parser` option set and data-free validation (units required,
  exclusive/required option groups). A simulation run is fatal with
  "not implemented yet"; nothing is silently ignored.
  - `ocl_backend.c` selects by capability: it scans every platform and
    device, requires `cl_khr_fp64`, and requires a successful build of
    an fp64 OpenCL C 1.1 probe kernel that uses the §4.7
    dequantisation. On `10.42.0.89` with `RUSTICL_ENABLE=radeonsi` it
    skips rusticl and picks Clover (WX 7100, 36 CUs, 16 GiB, 4 GiB max
    buffer).
  - `preflight.c` scans each DEM at native resolution through a split
    read window, snaps `res_max`, assigns DEM levels, fails on a
    non-aligned region with the aligned `g.region` command, and
    estimates triangles per level (fringe included) and device memory.
    Plumergat: 17.2 M triangles, 7.0 GiB, in 1.4 s.
  - 10 pytest tests (`tests/preflight_test.py`) pass locally (PoCL
    host) and on the GPU host. Formatting uses clang-format 22.1.5 and
    ruff 0.15.17, the versions GRASS's pre-commit config pins.
  - Manual in `.md` + `.html`.
- **Phase 1 — single-level mesh and I/O. DONE 2026-09-23.**
  - `sampler.c` reads any raster at native resolution (split read
    window), with nearest, bilinear (NULL-aware, renormalised) and
    box-mean queries.
  - `quadtree.c` builds a uniform level-0 quadtree of active leaves (DEM
    has valid cells, centre inside `domain=`), in Morton order.
  - `mesh.c` fans each leaf from its centre by walking the perimeter
    clockwise. That is ANUGA's left/bottom/right/top order with edge 1
    external, and the same code handles hanging nodes for phase 5.
    Nodes are hashed on a doubled finest-level integer grid. Neighbours
    come from edge matching, boundary edges are enumerated in sorted
    `(triangle, edge)` order with tags N/S/E/W/null, and the geometry
    uses ANUGA's exact expressions. The centre point follows
    `rectangular_cross_construct` (`(i + 0.5) * delta`), not the Python
    mean of the corners, which can differ in the last bit. Node
    elevations are bilinear at corners and the leaf's box mean at
    centres; the centroid is `(q0 + q1 + q2) / 3.0`, as ANUGA's
    `_interpolate`, then quantised (§4.7).
  - `mesh_output=` (a directory: `manifest.json` plus raw arrays)
    replaces the planned `-x` flag, because an export needs a path.
    `mesh_level=` writes the leaf level per region cell. With either of
    them and no `output=`, the module runs in **mesh-only mode** and
    exits successfully. This is a permanent inspection feature.
  - After quantisation, the selected OpenCL device dequantises the real
    `zq` array, and any value not bitwise equal to the host's is fatal
    (V1q). Host code is built with `-ffp-contract=off`.
  - **Results:**
    - `validation/compare_mesh_anuga.py`, run with ANUGA 4.0.1.dev8
      built from `~/dev/anuga_core` into `.venv-anuga`, finds every
      geometry, connectivity, boundary-tag and centroid-bed array
      **bitwise identical** to `anuga.rectangular_cross` + `anuga.Domain`.
      That holds at 30 m, at Lambert-93 coordinates and at 0.3 m.
    - On the WX 7100 (Clover), 16 M triangles dequantise bitwise equal
      to the host; mesh build plus export takes 9 s.
    - Plumergat: the GLO-30 watershed domain gives 1.22 M triangles in
      0.9 s (305,559 cells = 275 km²; the −21 m pit is kept; every
      boundary is tagged null), and RGE ALTI 1 m gives 16 M triangles in
      8.2 s.
    - 21 pytest tests pass locally; 18 pass on the GPU host, where the 3
      ANUGA comparisons skip because ANUGA is not installed there.
- **Phase 2 — numerical core (OMP tier). DONE 2026-09-23.**
  - **Structure.**
    - `cl/anuga_common.h` + `cl/anuga_sw.h`: every per-triangle body
      written once, valid as C99 and OpenCL C 1.1, with the bed read
      only through `anuga_bed()`.
    - `kernels_omp.c`: the OpenMP loops.
    - `evolve.c`: Euler/RK2/RK3 steps and the evolve loop behind a
      `solver_ops` table, so phase 3 only adds an OpenCL table.
    - `state.c`: structure-of-arrays state.
    - `setup.c`: algorithm bundles, `boundary=`
      (reflective/transmissive/`dirichlet:stage[:xmom:ymom]`),
      `initial_depth=`/`initial_stage=`, `manning=`/`manning_value=`.
    - `state_output=`: raw initial and final state, step log and mass
      budget. This is the validation interface until raster outputs
      arrive in phase 4.
  - **V4 result: bitwise identical to ANUGA.** Final stage, xmom and
    ymom, and the number of steps, match ANUGA 4.0.1.dev8's
    unified-mode C step functions exactly on every case tested:
    - DE0, DE1 and DE2;
    - flat and sloped friction;
    - reflective, transmissive and Dirichlet boundaries (with and
      without momentum);
    - moving wet/dry fronts;
    - up to 505 steps;
    - a 3 × 3 km window of the real Plumergat GLO-30 terrain.

    `validation/compare_state_anuga.py` drives ANUGA's
    `evolve_one_*_step_gpu` with `evolve()`'s time capping, for the
    reason in finding 2 below. 6 cases are in pytest.
  - **Physics:**
    - *V2, lake at rest* (immersed bump, and emergent island with
      wet/dry edges): velocities < 1e-10 after 200 s.
    - *V3, dam break vs analytical, relative L1 at t = 5/25/50 s:*

      | Case | Stage | xmom | ANUGA's tolerance |
      |---|---|---|---|
      | Wet (Stoker) | 2.7e-4 / 2.9e-4 / 2.7e-4 | 3.9e-3 / 9.1e-4 / 4.6e-4 | 1e-2 (2e-2) |
      | Dry (Ritter) | 2.9e-4 / 3.0e-4 / 2.1e-4 | 5.7e-3 / 1.1e-3 / 3.2e-4 | 0.1–0.15 / 0.25 |
  - **Mass budget.**
    - Tracked quantities: boundary flux, the water added by `protect()`
      (clamping negative depths), and positive-only and signed volumes.
    - Clamping is weighted like the boundary fluxes (½, ½ for RK2; ⅙,
      ⅙, ⅔ for RK3), because the substep states enter the result with
      those coefficients.
    - With that, Δ signed volume = boundary + clamping, to 1e-12
      relative (tested for all three algorithms).
  - **Performance (OpenMP, 8 threads, i7-1165G7):** the Plumergat
    watershed at 30 m (1.22 M triangles) runs at about 0.36 s per RK2
    step.
  - **Findings about ANUGA itself (worth reporting upstream):**
    1. *Extrapolation race.* In
       `core_extrapolate_second_order_edge`, a triangle zeroes its own
       centroid velocity while neighbours read it in the same parallel
       loop. This is harmless when triangles have ≤ 1 boundary edge and
       the dry betas are 0; the port keeps the zero local (header of
       `cl/anuga_sw.h`).
    2. *CPU unified trajectory depends on `yieldstep`.* In a CPU build,
       `distribute_to_vertices_and_edges()` at t = 0 and every yield
       runs the host `protect_new` + openmp extrapolation on arrays
       shared with the "device". Contrary to its own comment, the CPU
       trajectory therefore depends on `yieldstep`; on a GPU it does
       not. r.hydro.anuga follows the device trajectory.
    3. *DE1 (and DE2) at CFL 1 are not positivity-preserving for thin
       films on coarse steep terrain.* In the Plumergat window (5 cm of
       water on 30 m cells, 60 s), clamping adds 33% of the initial
       volume with DE1 at CFL 1.0. It adds 3.8% at CFL 0.5, about 0 at
       CFL 0.25, 22% with DE2, and 0 with DE0 (Euler, beta 0.5, CFL
       0.9). ANUGA does exactly the same (bitwise). This bears on the
       default algorithm (D3) for rain-on-grid; see R4.
  - `friction_method=` defaults to **flat**, as ANUGA does; the plan
    previously said sloped, which was wrong.
- **Phase 3 — OpenCL tier. DONE 2026-09-23.**
  - **Kernels.** `cl/anuga_kernels.cl` holds 12 thin `__kernel` wrappers
    around the bodies of `cl/anuga_sw.h`, so there is still one copy of
    the physics.
    - The time-step minimum and the boundary-flux, clamping-mass and
      volume sums reduce per work-group in local memory (OpenCL 1.1).
    - The host combines the partials in group order, so results are
      deterministic.
    - The program is `anuga_common.h + anuga_sw.h + anuga_kernels.cl`,
      embedded by the Makefile as an array of lines
      (`ocl_kernels_src.h`) and built with `-cl-std=CL1.1 -DWG=<size>`.
      There are no static or inline functions in the OpenCL build.
  - **Host.** `kernels_ocl.c` provides a second `solver_ops` table over
    device buffers.
    - Every array is uploaded once after setup, and synchronised back
      only at outputs.
    - The work-group size is the largest power of two, at most 256, that
      every kernel accepts.
    - `evolve.c` is unchanged. `device=` now selects the solver tier too.
  - **V13 results (all tiers):**
    - Without friction, OpenCL is **bitwise identical** to OpenMP, and
      hence to ANUGA, on both PoCL and Clover (WX 7100), for DE0, DE1
      and DE2 and all boundary types. OpenCL requires correctly rounded
      double `+ − × ÷` and `sqrt`.
    - With Manning friction, `pow()` is only accurate to a few ulp in
      OpenCL. The largest relative difference after 903 steps is 3e-15
      (stage), with identical step counts.
    - Lake at rest and the mass-balance identity also hold on OpenCL.
  - **Performance.** 4 M triangles (10 km × 10 km at 10 m), 394 DE1
    steps, 60 s simulated:

    | Tier (server) | Wall time |
    |---|---|
    | OpenMP, Ryzen 9 3950X, 32 threads | 205 s |
    | OpenCL, WX 7100 | 39 s, end to end, **5.3×** |

    Each substep still does three small blocking reads (clamping mass,
    time-step and boundary-flux partials); removing them is phase 8.
  - 43 pytest tests pass on both hosts. `tests/opencl_test.py` skips
    when no OpenCL device exists.
- **Phase 4 — outputs, single grid. DONE 2026-09-24.**
  - **Transfer.** `output.c` builds a triangle → region-cell map once, as
    compressed rows of (triangle, weight).
    - A cell at least as large as a leaf takes the area-weighted leaf
      triangles.
    - A finer cell takes the triangle containing its centre (a generic
      point-in-triangle test, ready for phase 5's hanging-node fans).
    - Depth and stage are weighted means; velocities are
      momentum-weighted.
  - **Time series.** `output=` + `outputs=` write the 11 quantities of
    §8.2 at t = 0, every `output_step` and at the end, as
    `<output>_<q>_<index>`.
    - Each quantity is registered with `t.create` + one `t.register
      file=` (absolute time with `start=`, else relative in seconds) and
      coloured with `t.rast.colors`.
    - Maps get units, titles and history. FCELL by default, DCELL with
      `-d`; dry cells are 0, or NULL with `-n`.
    - `output_step` must be whole seconds.
  - **Summaries.** `max_depth`, `max_speed`, `max_stage`, `max_hazard`,
    `arrival_time`, `inundation_duration` and `final_prefix` use running
    statistics updated **on the device every step** (`sw_stats`, both
    tiers).
    - Speed follows ANUGA's max-quantities operator, with `min_depth` as
      the velocity-zero height.
    - Cells take the maximum (the earliest for arrival) over their
      triangles.
  - **Mass balance.** `massbalance=` writes a CSV per output time
    (volumes, boundary inflow, clamping, errors); `-m` prints the final
    error.
  - **Safety.** Output names (every indexed map) are checked for legality
    and existence **before** the mesh is built.
  - **Results:**
    - Σ depth × cell area equals the solver volume to 1e-12 at output
      resolutions of 10, 20 and 40 m on a 10 m mesh.
    - OpenCL and OpenMP rasters are identical (no friction).
    - 4 M triangles on the WX 7100 with 14 time-series maps and 2
      summaries: 38 s, versus 39 s without outputs.
    - 55 pytest tests pass on both hosts.
  - **Deferred.**
    - Asynchronous double-buffered writing (phase 7). Output cost is
      currently negligible, as measured.
    - Fine detail grids (`-f`) need multi-level meshes (phase 5).
- **Phase 5 — multi-resolution.** Multi-DEM stack (native-resolution
  banded reading, bias check, blending), quadtree refinement, 2:1
  balance, fringe dilation, all 16 templates, Morton ordering,
  multi-grid output transfer, `mesh_level` map, pre-flight per level.
  V2m, V4m, V5m, V11, and the multi-level part of V6.
- **Phase 6 — forcing and infiltration.** (a) Rain, evaporation,
  hyetograph, STRDS with explicit units, then Green–Ampt with soil
  table, direct rasters, `soil_depth`, `impervious`: V7i, V8i, V10.
  (b) GAR (Ogden & Saghafian 1997, §6.3): V9i. (c) Wind,
  pressure, inflow hydrographs, stage/Flather series: V12.
- **Phase 7 — robustness and user experience.** Asynchronous output,
  hot start, gauges, stall detection, `G_percent`, and documented
  examples: ERA5 rain on a mixed LiDAR HD / 30 m DEM, dam break, storm
  surge.
- **Phase 8 — performance.** Compact per-leaf template geometry,
  device-side dt, work-group tuning for Polaris, kernel fusion,
  dry-cell skipping, and an fp32-state experiment kept only if V4/V13
  still hold. Benchmarks.
- **Phase 9 (post-v1).** Local time stepping by quadtree level,
  riverwalls from vector lines, buildings as holes, culverts,
  multi-GPU or domain tiling beyond device memory, spherical
  coordinates.

---

## 11. Conventions

- C: `clang-format` (LLVM base, 4-space indent, Stroustrup braces),
  snake_case. Includes: system, then OpenCL, then GRASS, then local.
- Headers: `MODULE:` / `AUTHOR(S):` / `PURPOSE:` / `COPYRIGHT: (C) 2026 by
  Yann Chemin and the GRASS Development Team` /
  `SPDX-License-Identifier: GPL-3.0-or-later`. Files ported from ANUGA
  add "Derived from ANUGA, Copyright 2004-2015 ANU and Geoscience
  Australia, Apache-2.0" and ship ANUGA's licence text in
  `LICENSE.ANUGA`.
- Messages: `G_fatal_error` / `G_warning` / `G_message`, map names in
  `<angle brackets>`, ellipsis for in-progress actions. Fail loudly
  (DEM bias, extent alignment, units, memory) rather than guess.
- The module reads the current region and never changes it.
- Commit messages `r.hydro.anuga: <imperative>`, ASCII;
  `pre-commit run --all-files` is the gate.
- Docs: `.md` is the source of truth, `.html` in sync; tool names in
  italics, parameters in bold. SEE ALSO, alphabetised: *r.hydro.hbv*,
  *r.hydro.rri*, *r.sim.water*, *r.watershed.opencl*, *t.in.era5*,
  *t.rast.list*. REFERENCES lists the papers of §6.1.
- AI assistance is disclosed in the README and the submission PR.

## 12. Remaining open items

- **R1 (resolved):** `docs/references/` now holds Rawls et al.
  (1983), La Follette et al. (2023, NOAA IR) and the HEC-RAS 2D
  precipitation/infiltration/wind deck. The GA table (§6.4) is
  transcribed from the original, and the GAR equations (§6.3) come from
  the primary source, Ogden & Saghafian (1997), which is now in
  `docs/references/`, as is its Table 1 (the GAR parameter set, §6.4).
  La Follette et al. (2023) (NOAA IR 51167, the 18-page article; 53741,
  29 pages with supporting information) is kept as the modern
  derivation and a possible LGAR extension. Rawls, Brakensiek & Saxton
  (1982) is only needed if more Brooks–Corey horizon data is wanted. The HEC-RAS ras1dtechref
  "6.7_beta2" Green-Ampt link returns 404; the 6.5 page was used.
- **R2 (resolved, 2026-09-23):** V12 uses the **Plumergat** dataset
  (`testdata/plumergat/`, scripts plus README with results). Copernicus
  GLO-30 via *r.in.dem* covers 51 km at 30 m. BD TOPAGE watersheds come
  from the Sandre WFS, and *r.watershed.opencl* delineation agrees with
  them at IoU 0.88 (Bono) and 0.89 (upper Loc'h). The domain is
  `anuga_domain`, 275 km². **LiDAR HD MNT is not yet published around
  Plumergat**, so the fine DEM is IGN RGE ALTI 1 m (2 × 2 km),
  selectable back to LiDAR HD in `config.sh`. The DEM pair's median
  vertical difference is −0.29 m, within the 0.5 m default tolerance.
  GLO-30 is a surface model, so a bare-earth coarse DEM (RGE ALTI
  5/25 m) is worth adding.
- **R3:** default `fringe=4` and `blend_width=2` coarse cells are
  initial guesses, to be tuned by V5m.
- **R4 (decided 2026-09-23): DE1 with CFL 0.5** by default. The module
  also warns when clamping negative depths adds more than 1% of the
  initial water volume. The ANUGA comparisons pass the CFL explicitly,
  so they are unaffected.
