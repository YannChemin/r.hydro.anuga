/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Quadtree leaves over the current region (PLAN.md sections
 *               4.1, 4.3 and 4.5).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_QUADTREE_H
#define R_HYDRO_ANUGA_QUADTREE_H

#include <stdint.h>

/* A square leaf cell. (ix, iy) index the cell at its own level, from the
 * region's south-west corner, x eastward and y northward. Level 0 cells
 * have size res_max; each level halves the size. */
struct leaf {
    int level;
    int32_t ix, iy;
    uint8_t hanging; /* Sides (1 << W, S, E, N) split by a finer leaf. */
};

/* An area requiring at least a given level (a DEM footprint or a refine=
 * map). count() returns the number of valid cells whose centres lie in the
 * box [x0, x1) x [y0, y1) (absolute coordinates); the bounding box limits
 * where it is queried. */
struct footprint {
    int level;
    double west, south, east, north;
    long long (*count)(const void *data, double x0, double y0, double x1,
                       double y1);
    const void *data;
};

struct quadtree {
    double west, south; /* Region south-west corner (absolute). */
    double res_max;     /* Level 0 cell size. */
    int n_levels;       /* Levels 0 .. n_levels - 1. */
    int32_t nx0, ny0;   /* Level 0 cells across the region. */
    long n_leaves;
    struct leaf *leaves; /* Morton order at the finest level. */
};

/* Callback deciding whether a leaf is part of the domain. */
typedef int (*leaf_active_fn)(const struct quadtree *qt, const struct leaf *lf,
                              void *data);

/* Build a single-level quadtree (every leaf at level 0) over the current
 * region, keeping the leaves for which active() returns non-zero. The
 * region extent must be a whole number of res_max cells. */
void quadtree_build_uniform(struct quadtree *qt, double res_max,
                            leaf_active_fn active, void *data);

/* Build a graded quadtree with levels 0 .. n_levels - 1: every footprint
 * is covered by leaves of at least its level, and around every region of
 * level m there are at least `fringe` leaves of level m - 1 (PLAN.md
 * section 4.4), which also guarantees 2:1 balance. Leaves are kept if
 * active() returns non-zero, then sorted in Morton order and their
 * hanging-node masks computed. */
void quadtree_build_graded(struct quadtree *qt, double res_max, int n_levels,
                           int fringe, const struct footprint *footprints,
                           int n_footprints, leaf_active_fn active,
                           void *data);

void quadtree_free(struct quadtree *qt);

/* Size of a leaf and its south-west corner relative to the region's
 * south-west corner. */
double quadtree_leaf_size(const struct quadtree *qt, const struct leaf *lf);
void quadtree_leaf_origin(const struct quadtree *qt, const struct leaf *lf,
                          double *x, double *y);

/* Morton key of a leaf's south-west corner at the finest level. */
uint64_t quadtree_morton(const struct quadtree *qt, const struct leaf *lf);

#endif /* R_HYDRO_ANUGA_QUADTREE_H */
