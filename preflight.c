/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Pre-flight checks: DEM stack description at native
 *               resolution, quadtree level snapping, triangle count and
 *               memory estimates (PLAN.md sections 4.2, 4.3, 4.6).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "preflight.h"

/* Relative tolerance for "square cell" and "whole number of cells". */
#define REL_TOL 1.0e-6

static int is_multiple(double length, double step)
{
    double n = length / step;

    return fabs(n - floor(n + 0.5)) < REL_TOL * (n > 1.0 ? n : 1.0);
}

/* Count valid cells and the elevation range of one DEM over its native
 * grid clipped to the current region, using a split read window so the
 * current region's resolution does not resample the DEM. */
static void scan_dem(struct dem_desc *dem, const struct Cell_head *region)
{
    struct Cell_head win = *region;
    DCELL *row_buf;
    int fd, row, col;

    win.north = fmin(region->north, dem->cellhd.north);
    win.south = fmax(region->south, dem->cellhd.south);
    win.east = fmin(region->east, dem->cellhd.east);
    win.west = fmax(region->west, dem->cellhd.west);
    if (win.north <= win.south || win.east <= win.west)
        G_fatal_error(_("Raster map <%s> does not overlap the current "
                        "region"),
                      dem->name);
    Rast_align_window(&win, &dem->cellhd);
    dem->window = win;

    Rast_set_input_window(&win);
    fd = Rast_open_old(dem->name, dem->mapset);
    row_buf = Rast_allocate_d_input_buf();

    dem->n_valid = 0;
    dem->z_min = INFINITY;
    dem->z_max = -INFINITY;
    for (row = 0; row < win.rows; row++) {
        G_percent(row, win.rows, 5);
        Rast_get_d_row(fd, row_buf, row);
        for (col = 0; col < win.cols; col++) {
            DCELL z = row_buf[col];

            if (Rast_is_d_null_value(&z))
                continue;
            if (!isfinite(z))
                G_fatal_error(_("Raster map <%s> contains a non-finite "
                                "value at row %d, col %d"),
                              dem->name, row, col);
            dem->n_valid++;
            if (z < dem->z_min)
                dem->z_min = z;
            if (z > dem->z_max)
                dem->z_max = z;
        }
    }
    G_percent(1, 1, 1);
    G_free(row_buf);
    Rast_close(fd);

    if (dem->n_valid == 0)
        G_fatal_error(_("Raster map <%s> has no valid cells inside the "
                        "current region"),
                      dem->name);
    dem->area = (double)dem->n_valid * dem->res * dem->res;
}

static int cmp_dem_res(const void *a, const void *b)
{
    const struct dem_desc *da = a, *db = b;

    return (da->res > db->res) - (da->res < db->res);
}

void preflight_describe_dems(struct preflight *pf, char **names, int keep_order)
{
    struct Cell_head region;
    int i;

    G_get_window(&region);
    if (region.proj == PROJECTION_LL)
        G_fatal_error(_("Latitude-longitude projects are not supported; "
                        "reproject to a metric CRS first"));

    pf->n_dems = 0;
    for (i = 0; names[i]; i++) {
        struct dem_desc *dem;
        const char *mapset;

        if (pf->n_dems == MAX_DEMS)
            G_fatal_error(_("Too many elevation maps (maximum %d)"), MAX_DEMS);
        dem = &pf->dems[pf->n_dems];
        memset(dem, 0, sizeof(*dem));
        mapset = G_find_raster2(names[i], "");
        if (!mapset)
            G_fatal_error(_("Raster map <%s> not found"), names[i]);
        G_strlcpy(dem->name, names[i], sizeof(dem->name));
        G_strlcpy(dem->mapset, mapset, sizeof(dem->mapset));
        Rast_get_cellhd(dem->name, dem->mapset, &dem->cellhd);

        if (fabs(dem->cellhd.ns_res - dem->cellhd.ew_res) >
            REL_TOL * dem->cellhd.ew_res)
            G_fatal_error(_("Raster map <%s> has non-square cells "
                            "(nsres=%g, ewres=%g); resample it to square "
                            "cells first"),
                          dem->name, dem->cellhd.ns_res, dem->cellhd.ew_res);
        dem->res = dem->cellhd.ew_res;
        pf->n_dems++;
    }

    if (!keep_order)
        qsort(pf->dems, pf->n_dems, sizeof(struct dem_desc), cmp_dem_res);

    pf->z_min = INFINITY;
    pf->z_max = -INFINITY;
    for (i = 0; i < pf->n_dems; i++) {
        G_verbose_message(_("Scanning elevation map <%s> at its native "
                            "resolution %g..."),
                          pf->dems[i].name, pf->dems[i].res);
        scan_dem(&pf->dems[i], &region);
        pf->z_min = fmin(pf->z_min, pf->dems[i].z_min);
        pf->z_max = fmax(pf->z_max, pf->dems[i].z_max);
    }

    /* Read later maps on the current region again. The output window was
     * never changed, so restoring the input window is enough. */
    Rast_set_input_window(&region);
}

void preflight_levels(struct preflight *pf, double res_min, double res_max)
{
    struct Cell_head region;
    double finest = INFINITY, coarsest = 0.0, ratio;
    int i, n_steps;

    for (i = 0; i < pf->n_dems; i++) {
        finest = fmin(finest, pf->dems[i].res);
        coarsest = fmax(coarsest, pf->dems[i].res);
    }
    pf->res_min = res_min > 0.0 ? res_min : finest;
    pf->res_max_requested = res_max > 0.0 ? res_max : coarsest;
    if (pf->res_max_requested < pf->res_min)
        G_fatal_error(_("res_max (%g) is smaller than res_min (%g)"),
                      pf->res_max_requested, pf->res_min);

    /* res_max = res_min * 2^L with L the smallest integer reaching the
     * requested value: base cells are never finer than asked for. */
    ratio = pf->res_max_requested / pf->res_min;
    n_steps = (int)ceil(log2(ratio) - REL_TOL);
    if (n_steps < 0)
        n_steps = 0;
    if (n_steps + 1 > MAX_LEVELS)
        G_fatal_error(_("Too many resolution levels (%d, maximum %d)"),
                      n_steps + 1, MAX_LEVELS);
    pf->res_max = pf->res_min * ldexp(1.0, n_steps);
    pf->res_max_snapped = fabs(pf->res_max - pf->res_max_requested) >
                          REL_TOL * pf->res_max_requested;
    pf->n_levels = n_steps + 1;
    for (i = 0; i < pf->n_levels; i++) {
        pf->levels[i].size = pf->res_max / ldexp(1.0, i);
        pf->levels[i].area = 0.0;
        pf->levels[i].triangles = 0;
    }

    /* Target level of each DEM: the finest level whose cell size is not
     * finer than the DEM (and not finer than res_min). */
    for (i = 0; i < pf->n_dems; i++) {
        double r = fmax(pf->dems[i].res, pf->res_min);
        int level = (int)floor(log2(pf->res_max / r) + REL_TOL);

        if (level < 0)
            level = 0;
        if (level > pf->n_levels - 1)
            level = pf->n_levels - 1;
        pf->dems[i].level = level;
    }

    G_get_window(&region);
    if (!is_multiple(region.east - region.west, pf->res_max) ||
        !is_multiple(region.north - region.south, pf->res_max)) {
        double w = floor(region.west / pf->res_max) * pf->res_max;
        double s = floor(region.south / pf->res_max) * pf->res_max;
        double e = ceil(region.east / pf->res_max) * pf->res_max;
        double n = ceil(region.north / pf->res_max) * pf->res_max;

        G_fatal_error(_("The current region extent is not a whole number "
                        "of %g base cells (res_max). Use e.g. "
                        "'g.region n=%.12g s=%.12g e=%.12g w=%.12g'"),
                      pf->res_max, n, s, e, w);
    }
}

/* Count non-NULL cells of the domain raster in the current region. */
static double domain_area(const char *domain)
{
    struct Cell_head region;
    const char *mapset;
    DCELL *buf;
    long long n = 0;
    int fd, row, col;

    G_get_window(&region);
    mapset = G_find_raster2(domain, "");
    if (!mapset)
        G_fatal_error(_("Raster map <%s> not found"), domain);
    fd = Rast_open_old(domain, mapset);
    /* The read window is split from the write window (scan_dem), so use
     * the input-window API throughout. */
    buf = Rast_allocate_d_input_buf();
    for (row = 0; row < region.rows; row++) {
        Rast_get_d_row(fd, buf, row);
        for (col = 0; col < region.cols; col++)
            if (!Rast_is_d_null_value(&buf[col]))
                n++;
    }
    G_free(buf);
    Rast_close(fd);
    if (n == 0)
        G_fatal_error(_("Domain raster <%s> has no valid cells in the "
                        "current region"),
                      domain);

    return (double)n * region.ew_res * region.ns_res;
}

/* Bytes per triangle on the device, following ANUGA's own
 * gpu_estimate_required_memory() (gpu_domain_core.c:41: 50 double and 11
 * int64 arrays per triangle, plus 3 backup arrays) with the changes of
 * PLAN.md section 4.7: no fp64 bed centroid or bed edge arrays (4
 * doubles), one uint32 scaled elevation, and int32 indices. */
static double bytes_per_triangle(const struct memory_options *mopt)
{
    double doubles = 50.0 - 4.0 + 3.0;
    double int32s = 11.0;
    double uint32s = 1.0;

    doubles += mopt->n_max_outputs;
    if (mopt->infiltration == 1)
        doubles += 1.0; /* F */
    else if (mopt->infiltration == 2)
        doubles += 1.0 + 5.0; /* F, theta0, Z1, theta1, F1, F2 */
    if (mopt->infiltration)
        uint32s += 1.0; /* Soil class / phase flag. */

    return doubles * 8.0 + int32s * 4.0 + uint32s * 4.0;
}

void preflight_estimate(struct preflight *pf, const char *domain, int fringe,
                        const struct memory_options *mopt)
{
    double claimed = 0.0;
    int i, l;

    pf->fringe = fringe;
    pf->domain_area =
        domain ? domain_area(domain) : pf->dems[pf->n_dems - 1].area;

    /* Assume nested footprints: each DEM governs its own area minus what
     * finer DEMs already claimed; the coarsest DEM fills the domain. */
    for (i = 0; i < pf->n_dems; i++) {
        double a = (i == pf->n_dems - 1) ? pf->domain_area : pf->dems[i].area;

        a = fmax(a - claimed, 0.0);
        pf->dems[i].claimed_area = a;
        claimed += a;
        pf->levels[pf->dems[i].level].area += a;
    }

    /* Fringe bands: around each finer footprint, every intermediate
     * level gets a band `fringe` leaves wide. The footprint's
     * bounding-box perimeter grows as bands are added; band area is
     * taken from the coarsest level it replaces. */
    for (i = 0; i < pf->n_dems; i++) {
        const struct Cell_head *w = &pf->dems[i].window;
        double half_w = 0.5 * (w->east - w->west);
        double half_h = 0.5 * (w->north - w->south);
        int coarse_level = pf->dems[pf->n_dems - 1].level;

        if (pf->dems[i].level <= coarse_level || i == pf->n_dems - 1)
            continue;
        for (l = pf->dems[i].level - 1; l > coarse_level; l--) {
            double width = fringe * pf->levels[l].size;
            double band = 4.0 * (half_w + half_h) * width + 4.0 * width * width;

            pf->levels[l].area += band;
            pf->levels[coarse_level].area =
                fmax(pf->levels[coarse_level].area - band, 0.0);
            half_w += width;
            half_h += width;
        }
    }

    pf->triangles = 0;
    for (l = 0; l < pf->n_levels; l++) {
        double s = pf->levels[l].size;

        pf->levels[l].triangles =
            (long long)ceil(4.0 * pf->levels[l].area / (s * s));
        pf->triangles += pf->levels[l].triangles;
    }
    pf->bytes_per_triangle = bytes_per_triangle(mopt);
    pf->device_bytes = pf->bytes_per_triangle * (double)pf->triangles;
    /* Largest single buffer: an edge array (3 doubles per triangle). */
    pf->largest_buffer_bytes = 24.0 * (double)pf->triangles;
}

void preflight_print(const struct preflight *pf, const char *format)
{
    int i, shell = format && strcmp(format, "shell") == 0;

    if (shell) {
        fprintf(stdout, "n_dems=%d\n", pf->n_dems);
        for (i = 0; i < pf->n_dems; i++) {
            const struct dem_desc *d = &pf->dems[i];

            fprintf(stdout,
                    "dem%d=%s@%s\ndem%d_res=%.12g\ndem%d_level=%d\n"
                    "dem%d_valid_cells=%lld\ndem%d_area_km2=%.6f\n"
                    "dem%d_governed_km2=%.6f\ndem%d_zmin=%.6f\n"
                    "dem%d_zmax=%.6f\n",
                    i, d->name, d->mapset, i, d->res, i, d->level, i,
                    d->n_valid, i, d->area / 1e6, i, d->claimed_area / 1e6, i,
                    d->z_min, i, d->z_max);
        }
        fprintf(stdout,
                "res_min=%.12g\nres_max=%.12g\nres_max_requested=%.12g\n"
                "res_max_snapped=%d\nn_levels=%d\nfringe=%d\n"
                "domain_km2=%.6f\n",
                pf->res_min, pf->res_max, pf->res_max_requested,
                pf->res_max_snapped, pf->n_levels, pf->fringe,
                pf->domain_area / 1e6);
        for (i = 0; i < pf->n_levels; i++)
            fprintf(stdout,
                    "level%d_size=%.12g\nlevel%d_km2=%.6f\n"
                    "level%d_triangles=%lld\n",
                    i, pf->levels[i].size, i, pf->levels[i].area / 1e6, i,
                    pf->levels[i].triangles);
        fprintf(stdout,
                "triangles=%lld\nbytes_per_triangle=%.1f\n"
                "device_bytes=%.0f\nlargest_buffer_bytes=%.0f\n"
                "zmin=%.6f\nzmax=%.6f\n",
                pf->triangles, pf->bytes_per_triangle, pf->device_bytes,
                pf->largest_buffer_bytes, pf->z_min, pf->z_max);
        return;
    }

    fprintf(stdout, "Elevation stack (%d map%s):\n", pf->n_dems,
            pf->n_dems > 1 ? "s" : "");
    for (i = 0; i < pf->n_dems; i++) {
        const struct dem_desc *d = &pf->dems[i];

        fprintf(stdout,
                "  <%s@%s>  res %g m  level %d  valid %.3f km2  "
                "governs %.3f km2  z %.2f .. %.2f m\n",
                d->name, d->mapset, d->res, d->level, d->area / 1e6,
                d->claimed_area / 1e6, d->z_min, d->z_max);
    }
    fprintf(stdout, "Resolution levels: res_min %g m, res_max %g m",
            pf->res_min, pf->res_max);
    if (pf->res_max_snapped)
        fprintf(stdout, " (snapped from %g m to res_min * 2^%d)",
                pf->res_max_requested, pf->n_levels - 1);
    fprintf(stdout, ", fringe %d leaves\n", pf->fringe);
    for (i = 0; i < pf->n_levels; i++)
        fprintf(stdout, "  level %2d  %10g m  %12.3f km2  %14lld triangles\n",
                i, pf->levels[i].size, pf->levels[i].area / 1e6,
                pf->levels[i].triangles);
    fprintf(stdout, "Domain: %.3f km2\n", pf->domain_area / 1e6);
    fprintf(stdout,
            "Estimated triangles: %lld (phase 0 estimate; exact counts come "
            "from the quadtree)\n",
            pf->triangles);
    fprintf(stdout,
            "Estimated device memory: %.2f GiB (%.0f B/triangle), largest "
            "buffer %.2f GiB\n",
            pf->device_bytes / 1073741824.0, pf->bytes_per_triangle,
            pf->largest_buffer_bytes / 1073741824.0);
}
