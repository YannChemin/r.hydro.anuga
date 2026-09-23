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

#ifndef R_HYDRO_ANUGA_PREFLIGHT_H
#define R_HYDRO_ANUGA_PREFLIGHT_H

#include <grass/gis.h>

#define MAX_DEMS   16
#define MAX_LEVELS 24

/* One input DEM, described at its own native resolution over its
 * intersection with the current region. */
struct dem_desc {
    char name[GNAME_MAX];
    char mapset[GMAPSET_MAX];
    struct Cell_head cellhd; /* The map's own header. */
    struct Cell_head window; /* Native grid clipped to the region. */
    double res;              /* Square cell size (map units). */
    long long n_valid;       /* Non-NULL cells inside window. */
    double area;             /* n_valid * res * res. */
    double z_min, z_max;     /* Over valid cells. */
    int level;               /* Target quadtree level. */
    double claimed_area;     /* Area this DEM governs (finer DEMs removed). */
};

struct level_desc {
    double size;         /* Leaf cell size at this level. */
    double area;         /* Area covered by leaves of this level. */
    long long triangles; /* Estimated triangles (4 per leaf). */
};

struct preflight {
    int n_dems;
    struct dem_desc dems[MAX_DEMS]; /* Sorted finest first (or as given). */
    double res_min, res_max;
    int res_max_snapped; /* 1 if res_max was rounded up. */
    double res_max_requested;
    int n_levels; /* Levels 0 (coarsest) .. n_levels-1. */
    struct level_desc levels[MAX_LEVELS];
    double domain_area;
    int fringe;
    long long triangles;
    double bytes_per_triangle;
    double device_bytes;
    double largest_buffer_bytes;
    double z_min, z_max; /* Over all DEMs. */
};

/* Options affecting the per-triangle memory estimate. */
struct memory_options {
    int infiltration;  /* 0 none, 1 GA, 2 GAR. */
    int n_max_outputs; /* Number of running max/time statistics. */
};

/* Describe each DEM at native resolution. Fails loudly on missing maps,
 * non-square cells, or a DEM that does not overlap the region. If
 * keep_order is 0, DEMs are sorted finest first. */
void preflight_describe_dems(struct preflight *pf, char **names,
                             int keep_order);

/* Resolve res_min/res_max (0 means default), snap res_max to
 * res_min * 2^L, assign DEM levels, and check the region extent is a
 * whole number of base cells. */
void preflight_levels(struct preflight *pf, double res_min, double res_max);

/* Estimate triangles per level (including fringe bands) and memory. The
 * domain area is taken from the domain raster if given, otherwise from
 * the coarsest DEM. */
void preflight_estimate(struct preflight *pf, const char *domain, int fringe,
                        const struct memory_options *mopt);

/* Print the report in "plain" or "shell" format to stdout. */
void preflight_print(const struct preflight *pf, const char *format);

#endif /* R_HYDRO_ANUGA_PREFLIGHT_H */
