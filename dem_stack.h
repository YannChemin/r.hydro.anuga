/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Stack of DEMs at their native resolutions: elevation from
 *               the finest DEM available, vertical bias check between
 *               overlapping DEMs, and smooth blending across the seams
 *               (PLAN.md section 4.2).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_DEM_STACK_H
#define R_HYDRO_ANUGA_DEM_STACK_H

#include "preflight.h"
#include "sampler.h"

/* One DEM of the stack, with summed-area tables of valid cells and
 * values (for O(1) coverage counts and box means) and, for all but the
 * coarsest DEM, the distance of each valid cell to the edge of the DEM's
 * footprint (for seam blending). */
struct dem_layer {
    struct raster_grid grid;
    double res, offset, blend_width;
    int level;
    long long *count_sat; /* (rows + 1) * (cols + 1) */
    double *sum_sat;      /* (rows + 1) * (cols + 1) */
    float *edge_distance; /* rows * cols, map units; NULL for the coarsest */
};

struct dem_stack {
    int n; /* Finest first. */
    struct dem_layer layers[MAX_DEMS];
};

/* Load the DEMs described by the pre-flight (finest first), apply the
 * per-DEM offsets (offset_answers in the order the maps were given in
 * names, or NULL), run the vertical bias check between each DEM and the
 * next coarser one (fatal above tolerance unless that DEM has an explicit
 * offset), and prepare seam blending over blend_width coarse cells. */
void dem_stack_load(struct dem_stack *st, const struct preflight *pf,
                    char **names, char **offset_answers, double tolerance,
                    double blend_width);

void dem_stack_free(struct dem_stack *st);

/* Elevation at (x, y), absolute coordinates: bilinear between cell
 * centres of the finest DEM covering the point, blended near its
 * footprint edge with the next coarser DEM. NaN if no DEM covers it. */
double dem_stack_point(const struct dem_stack *st, double x, double y);

/* Mean elevation of the box [x0, x1) x [y0, y1): the mean of the cells of
 * the finest DEM with valid cells in the box, blended like
 * dem_stack_point() at the box centre. NaN if no DEM covers it. */
double dem_stack_box_mean(const struct dem_stack *st, double x0, double y0,
                          double x1, double y1);

/* Number of valid cells of layer i whose centres lie in the box. */
long long dem_layer_count(const struct dem_layer *l, double x0, double y0,
                          double x1, double y1);

#endif /* R_HYDRO_ANUGA_DEM_STACK_H */
