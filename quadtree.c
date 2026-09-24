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

#include "hashmap.h"
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
            struct leaf lf = {0, ix, iy, 0};

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

/* ---- Graded multi-level build ------------------------------------------ */

/* A bitmap of level-m cells over the index box [x0, x0 + nx) x
 * [y0, y0 + ny), indices counted from the region's south-west corner. */
struct bitmap {
    int64_t x0, y0, nx, ny;
    uint8_t *bits;
};

static int bm_get(const struct bitmap *b, int64_t ix, int64_t iy)
{
    if (!b->bits || ix < b->x0 || iy < b->y0 || ix >= b->x0 + b->nx ||
        iy >= b->y0 + b->ny)
        return 0;

    return b->bits[(iy - b->y0) * b->nx + (ix - b->x0)];
}

static void bm_alloc(struct bitmap *b, int64_t x0, int64_t y0, int64_t x1,
                     int64_t y1)
{
    b->x0 = x0;
    b->y0 = y0;
    b->nx = x1 > x0 ? x1 - x0 : 0;
    b->ny = y1 > y0 ? y1 - y0 : 0;
    b->bits = b->nx && b->ny ? G_calloc(b->nx * b->ny, 1) : NULL;
}

static void bm_free(struct bitmap *b)
{
    G_free(b->bits);
    memset(b, 0, sizeof(*b));
}

/* Separable Chebyshev dilation by r cells, in place. */
static void bm_dilate(struct bitmap *b, int r)
{
    uint8_t *tmp;
    int64_t i, j;

    if (!b->bits || r <= 0)
        return;
    tmp = G_calloc(b->nx * b->ny, 1);
    for (j = 0; j < b->ny; j++) {
        int64_t last = -(int64_t)r - 1; /* Last set column seen. */

        for (i = 0; i < b->nx; i++)
            if (b->bits[j * b->nx + i]) {
                int64_t a = i - r > last + 1 ? i - r : last + 1, k;

                for (k = a < 0 ? 0 : a; k <= i + r && k < b->nx; k++)
                    tmp[j * b->nx + k] = 1;
                last = i + r;
            }
    }
    memset(b->bits, 0, b->nx * b->ny);
    for (i = 0; i < b->nx; i++) {
        int64_t last = -(int64_t)r - 1;

        for (j = 0; j < b->ny; j++)
            if (tmp[j * b->nx + i]) {
                int64_t a = j - r > last + 1 ? j - r : last + 1, k;

                for (k = a < 0 ? 0 : a; k <= j + r && k < b->ny; k++)
                    b->bits[k * b->nx + i] = 1;
                last = j + r;
            }
    }
    G_free(tmp);
}

struct graded {
    struct quadtree *qt;
    int max_level;
    struct bitmap *split; /* split[l]: level-l cells to subdivide */
    leaf_active_fn active;
    void *data;
    long capacity;
};

static void visit(struct graded *gr, int level, int64_t ix, int64_t iy)
{
    struct quadtree *qt = gr->qt;

    if (level < gr->max_level && bm_get(&gr->split[level], ix, iy)) {
        visit(gr, level + 1, 2 * ix, 2 * iy);
        visit(gr, level + 1, 2 * ix + 1, 2 * iy);
        visit(gr, level + 1, 2 * ix, 2 * iy + 1);
        visit(gr, level + 1, 2 * ix + 1, 2 * iy + 1);
        return;
    }
    {
        struct leaf lf = {level, (int32_t)ix, (int32_t)iy, 0};

        if (!gr->active(qt, &lf, gr->data))
            return;
        if (qt->n_leaves == gr->capacity) {
            gr->capacity = gr->capacity ? 2 * gr->capacity : 4096;
            qt->leaves = G_realloc(qt->leaves, gr->capacity * sizeof(struct leaf));
        }
        qt->leaves[qt->n_leaves++] = lf;
    }
}

static uint64_t level_key(int level, int64_t ix, int64_t iy)
{
    return ((uint64_t)level << 58) | ((uint64_t)ix << 29) | (uint64_t)iy;
}

/* Level of the leaf containing the point (x, y) relative to the region
 * origin, or -1. */
static int leaf_level_at(const struct quadtree *qt, const struct hashmap *h,
                         double x, double y)
{
    int level;

    if (x < 0 || y < 0)
        return -1;
    for (level = qt->n_levels - 1; level >= 0; level--) {
        double size = ldexp(qt->res_max, -level);

        if (hashmap_get(h, level_key(level, (int64_t)floor(x / size),
                                     (int64_t)floor(y / size))) >= 0)
            return level;
    }

    return -1;
}

/* Hanging-node masks from the neighbours just outside each side at its
 * quarter points; fatal if 2:1 balance does not hold. */
static void compute_hanging(struct quadtree *qt)
{
    struct hashmap h;
    long k;

    hashmap_init(&h, qt->n_leaves);
    for (k = 0; k < qt->n_leaves; k++)
        hashmap_put_new(&h,
                        level_key(qt->leaves[k].level, qt->leaves[k].ix,
                                  qt->leaves[k].iy),
                        k);
    for (k = 0; k < qt->n_leaves; k++) {
        struct leaf *lf = &qt->leaves[k];
        double size = quadtree_leaf_size(qt, lf), x0, y0, e;
        /* Outside points per side (W, S, E, N), at 1/4 and 3/4. */
        double px[4][2], py[4][2];
        int side, i;

        quadtree_leaf_origin(qt, lf, &x0, &y0);
        e = 1e-3 * ldexp(qt->res_max, -(qt->n_levels - 1));
        for (i = 0; i < 2; i++) {
            double f = (i == 0 ? 0.25 : 0.75) * size;

            px[0][i] = x0 - e, py[0][i] = y0 + f;
            px[1][i] = x0 + f, py[1][i] = y0 - e;
            px[2][i] = x0 + size + e, py[2][i] = y0 + f;
            px[3][i] = x0 + f, py[3][i] = y0 + size + e;
        }
        lf->hanging = 0;
        for (side = 0; side < 4; side++)
            for (i = 0; i < 2; i++) {
                int nl = leaf_level_at(qt, &h, px[side][i], py[side][i]);

                if (nl > lf->level + 1)
                    G_fatal_error("Internal error: 2:1 balance violated at "
                                  "leaf level %d (%d, %d)",
                                  lf->level, lf->ix, lf->iy);
                if (nl == lf->level + 1)
                    lf->hanging |= (uint8_t)(1 << side);
            }
    }
    hashmap_free(&h);
}

void quadtree_build_graded(struct quadtree *qt, double res_max, int n_levels,
                           int fringe, const struct footprint *footprints,
                           int n_footprints, leaf_active_fn active,
                           void *data)
{
    struct Cell_head region;
    struct graded gr;
    struct bitmap *need, *split;
    double nx, ny;
    int m, f;
    int64_t ix, iy;

    G_get_window(&region);
    memset(qt, 0, sizeof(*qt));
    qt->west = region.west;
    qt->south = region.south;
    qt->res_max = res_max;
    qt->n_levels = n_levels;
    nx = (region.east - region.west) / res_max;
    ny = (region.north - region.south) / res_max;
    qt->nx0 = (int32_t)floor(nx + 0.5);
    qt->ny0 = (int32_t)floor(ny + 0.5);
    if (fabs(nx - qt->nx0) > 1e-6 * nx || fabs(ny - qt->ny0) > 1e-6 * ny)
        G_fatal_error(_("The region extent is not a whole number of %g cells"),
                      res_max);

    need = G_calloc(n_levels + 1, sizeof(struct bitmap));
    split = G_calloc(n_levels + 1, sizeof(struct bitmap));

    /* From the finest level down: need[m] = cells requiring level >= m;
     * split[m - 1] = their parents; need[m - 1] adds the parents dilated
     * by the fringe. */
    for (m = n_levels - 1; m >= 1; m--) {
        double size = ldexp(res_max, -m);
        int64_t lx0 = INT64_MAX, ly0 = INT64_MAX, lx1 = 0, ly1 = 0;
        int64_t lim_x = (int64_t)qt->nx0 << m, lim_y = (int64_t)qt->ny0 << m;
        const struct bitmap *finer = &need[m + 1];

        /* Bounding box: footprints of this level, and the dilated parents
         * of the finer level. */
        for (f = 0; f < n_footprints; f++) {
            const struct footprint *fp = &footprints[f];

            if (fp->level != m)
                continue;
            lx0 = fmin(lx0, floor((fp->west - region.west) / size));
            ly0 = fmin(ly0, floor((fp->south - region.south) / size));
            lx1 = fmax(lx1, ceil((fp->east - region.west) / size));
            ly1 = fmax(ly1, ceil((fp->north - region.south) / size));
        }
        if (m + 1 < n_levels && finer->bits) {
            lx0 = fmin(lx0, (finer->x0 >> 1) - fringe);
            ly0 = fmin(ly0, (finer->y0 >> 1) - fringe);
            lx1 = fmax(lx1, ((finer->x0 + finer->nx + 1) >> 1) + fringe);
            ly1 = fmax(ly1, ((finer->y0 + finer->ny + 1) >> 1) + fringe);
        }
        if (lx0 == INT64_MAX)
            continue;
        lx0 = lx0 < 0 ? 0 : lx0;
        ly0 = ly0 < 0 ? 0 : ly0;
        lx1 = lx1 > lim_x ? lim_x : lx1;
        ly1 = ly1 > lim_y ? lim_y : ly1;
        bm_alloc(&need[m], lx0, ly0, lx1, ly1);
        if (!need[m].bits)
            continue;

        /* Parents of the finer level, dilated by the fringe. */
        if (m + 1 < n_levels && finer->bits) {
            for (iy = 0; iy < finer->ny; iy++)
                for (ix = 0; ix < finer->nx; ix++)
                    if (finer->bits[iy * finer->nx + ix]) {
                        int64_t px = ((finer->x0 + ix) >> 1) - need[m].x0;
                        int64_t py = ((finer->y0 + iy) >> 1) - need[m].y0;

                        need[m].bits[py * need[m].nx + px] = 1;
                    }
            bm_dilate(&need[m], fringe);
        }

        /* Footprints of this level. */
        for (f = 0; f < n_footprints; f++) {
            const struct footprint *fp = &footprints[f];

            if (fp->level != m)
                continue;
            for (iy = 0; iy < need[m].ny; iy++) {
                double y0 = region.south + (need[m].y0 + iy) * size;

                if (y0 >= fp->north || y0 + size <= fp->south)
                    continue;
                for (ix = 0; ix < need[m].nx; ix++) {
                    double x0 = region.west + (need[m].x0 + ix) * size;

                    if (x0 >= fp->east || x0 + size <= fp->west)
                        continue;
                    if (!need[m].bits[iy * need[m].nx + ix] &&
                        fp->count(fp->data, x0, y0, x0 + size, y0 + size) > 0)
                        need[m].bits[iy * need[m].nx + ix] = 1;
                }
            }
        }
    }

    /* split[m] = parents of need[m + 1]. */
    for (m = 0; m + 1 < n_levels; m++) {
        const struct bitmap *finer = &need[m + 1];

        if (!finer->bits)
            continue;
        bm_alloc(&split[m], finer->x0 >> 1, finer->y0 >> 1,
                 ((finer->x0 + finer->nx + 1) >> 1),
                 ((finer->y0 + finer->ny + 1) >> 1));
        for (iy = 0; iy < finer->ny; iy++)
            for (ix = 0; ix < finer->nx; ix++)
                if (finer->bits[iy * finer->nx + ix])
                    split[m].bits[(((finer->y0 + iy) >> 1) - split[m].y0) *
                                      split[m].nx +
                                  (((finer->x0 + ix) >> 1) - split[m].x0)] = 1;
    }

    gr.qt = qt;
    gr.max_level = n_levels - 1;
    gr.split = split;
    gr.active = active;
    gr.data = data;
    gr.capacity = 0;
    for (iy = 0; iy < qt->ny0; iy++) {
        G_percent(iy, qt->ny0, 5);
        for (ix = 0; ix < qt->nx0; ix++)
            visit(&gr, 0, ix, iy);
    }
    G_percent(1, 1, 1);
    for (m = 0; m <= n_levels; m++) {
        bm_free(&need[m]);
        bm_free(&split[m]);
    }
    G_free(need);
    G_free(split);
    if (qt->n_leaves == 0)
        G_fatal_error(_("The domain has no active cells"));

    sort_qt = qt;
    qsort(qt->leaves, qt->n_leaves, sizeof(struct leaf), cmp_morton);
    sort_qt = NULL;
    compute_hanging(qt);
}
