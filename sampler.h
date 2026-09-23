/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Raster maps loaded at their native resolution (clipped to
 *               the current region) and sampled at arbitrary points or
 *               boxes (PLAN.md sections 4.2 and 7.1).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_SAMPLER_H
#define R_HYDRO_ANUGA_SAMPLER_H

#include <grass/gis.h>

/* A raster held in memory on its own grid. NULL cells are stored as NaN.
 * Coordinates passed to the query functions are absolute map
 * coordinates. */
struct raster_grid {
    char name[GNAME_MAX];
    struct Cell_head win; /* Native grid clipped to the region. */
    double *data;         /* rows * cols, row 0 is the north row. */
};

/* Load raster `name` at its native resolution over its intersection with
 * the current region. Fails loudly if the map is missing, contains
 * non-finite values, or does not overlap the region. */
void raster_grid_load(struct raster_grid *grid, const char *name);

void raster_grid_free(struct raster_grid *grid);

/* Value of the cell containing (x, y), NaN if NULL or outside. */
double raster_grid_nearest(const struct raster_grid *grid, double x, double y);

/* Bilinear interpolation between cell centres at (x, y). NULL or missing
 * cells are left out and the remaining weights renormalised; NaN if no
 * valid cell contributes. At a point shared by four cell corners this is
 * the mean of the valid cells among the four. */
double raster_grid_bilinear(const struct raster_grid *grid, double x, double y);

/* Mean of the valid cells whose centres lie in [x0, x1) x [y0, y1); NaN
 * if none. Also returns the number of valid cells in *count if not
 * NULL. */
double raster_grid_box_mean(const struct raster_grid *grid, double x0,
                            double y0, double x1, double y1, long *count);

#endif /* R_HYDRO_ANUGA_SAMPLER_H */
