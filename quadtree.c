/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Quadtree leaves over the current region (PLAN.md sections
 *               4.1, 4.3 and 4.5). Phase 1 builds a single level;
 *               refinement, 2:1 balance and fringe grading come in phase 5.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "quadtree.h"

/* Spread the low 32 bits of v over the even bits of a 64-bit word. */
static uint64_t spread_bits(uint32_t v)
{
    uint64_t x = v;

    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2)) & 0x3333333333333333ULL;
    x = (x | (x << 1)) & 0x5555555555555555ULL;

    return x;
}

uint64_t quadtree_morton(const struct quadtree *qt, const struct leaf *lf)
{
    int shift = qt->n_levels - 1 - lf->level;
    uint32_t fx = (uint32_t)lf->ix << shift;
    uint32_t fy = (uint32_t)lf->iy << shift;

    return spread_bits(fx) | (spread_bits(fy) << 1);
}

double quadtree_leaf_size(const struct quadtree *qt, const struct leaf *lf)
{
    return ldexp(qt->res_max, -lf->level);
}

void quadtree_leaf_origin(const struct quadtree *qt, const struct leaf *lf,
                          double *x, double *y)
{
    double size = quadtree_leaf_size(qt, lf);

    *x = lf->ix * size;
    *y = lf->iy * size;
}

static const struct quadtree *sort_qt;

static int cmp_morton(const void *a, const void *b)
{
    uint64_t ka = quadtree_morton(sort_qt, a);
    uint64_t kb = quadtree_morton(sort_qt, b);

    return (ka > kb) - (ka < kb);
}

void quadtree_build_uniform(struct quadtree *qt, double res_max,
                            leaf_active_fn active, void *data)
{
    struct Cell_head region;
    double nx, ny;
    long capacity;
    int32_t ix, iy;

    G_get_window(&region);
    memset(qt, 0, sizeof(*qt));
    qt->west = region.west;
    qt->south = region.south;
    qt->res_max = res_max;
    qt->n_levels = 1;
    nx = (region.east - region.west) / res_max;
    ny = (region.north - region.south) / res_max;
    qt->nx0 = (int32_t)floor(nx + 0.5);
    qt->ny0 = (int32_t)floor(ny + 0.5);
    if (fabs(nx - qt->nx0) > 1e-6 * nx || fabs(ny - qt->ny0) > 1e-6 * ny)
        G_fatal_error(_("The region extent is not a whole number of %g cells"),
                      res_max);

    capacity = (long)qt->nx0 * qt->ny0;
    qt->leaves = G_malloc(capacity * sizeof(struct leaf));
    for (iy = 0; iy < qt->ny0; iy++) {
        G_percent(iy, qt->ny0, 5);
        for (ix = 0; ix < qt->nx0; ix++) {
            struct leaf lf = {0, ix, iy};

            if (active(qt, &lf, data))
                qt->leaves[qt->n_leaves++] = lf;
        }
    }
    G_percent(1, 1, 1);
    if (qt->n_leaves == 0)
        G_fatal_error(_("The domain has no active cells"));
    qt->leaves = G_realloc(qt->leaves, qt->n_leaves * sizeof(struct leaf));

    sort_qt = qt;
    qsort(qt->leaves, qt->n_leaves, sizeof(struct leaf), cmp_morton);
    sort_qt = NULL;
}

void quadtree_free(struct quadtree *qt)
{
    G_free(qt->leaves);
    qt->leaves = NULL;
    qt->n_leaves = 0;
}
