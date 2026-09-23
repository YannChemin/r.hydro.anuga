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

void quadtree_free(struct quadtree *qt);

/* Size of a leaf and its south-west corner relative to the region's
 * south-west corner. */
double quadtree_leaf_size(const struct quadtree *qt, const struct leaf *lf);
void quadtree_leaf_origin(const struct quadtree *qt, const struct leaf *lf,
                          double *x, double *y);

/* Morton key of a leaf's south-west corner at the finest level. */
uint64_t quadtree_morton(const struct quadtree *qt, const struct leaf *lf);

#endif /* R_HYDRO_ANUGA_QUADTREE_H */
