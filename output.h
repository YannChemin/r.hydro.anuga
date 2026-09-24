/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Raster outputs on the current region: mesh level, time
 *               series registered as space-time raster datasets, summary
 *               rasters and the mass balance table (PLAN.md section 8).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_OUTPUT_H
#define R_HYDRO_ANUGA_OUTPUT_H

#include <stdio.h>

#include <grass/gis.h>

#include "evolve.h"
#include "mesh.h"
#include "quadtree.h"
#include "state.h"

/* Write the quadtree level of the leaf containing each region cell centre
 * (NULL outside the domain). */
void output_mesh_level(const struct quadtree *qt, const char *name);

/* Transfer from triangles to the cells of the current region (PLAN.md
 * section 8.1): a cell at least as large as a leaf takes the area-weighted
 * mean of the triangles of the leaves whose centres it contains; a finer
 * cell takes the triangle containing its centre. */
struct out_grid {
    struct Cell_head win;
    char suffix[32]; /* Appended to output names ("" for the main grid). */
    long n_cells;
    long *start;    /* n_cells + 1 offsets into tri/weight. */
    int32_t *tri;   /* Contributing triangles. */
    double *weight; /* Normalised weights (sum 1 per covered cell). */
};

/* Build the transfer onto win (NULL: the current region). suffix is
 * appended to the names of the maps written on this grid. */
void out_grid_build(struct out_grid *g, const struct quadtree *qt,
                    const struct mesh *m, const struct Cell_head *win,
                    const char *suffix);
void out_grid_free(struct out_grid *g);

/* Time series quantities. */
enum ts_quantity {
    Q_DEPTH,
    Q_STAGE,
    Q_XVELOCITY,
    Q_YVELOCITY,
    Q_SPEED,
    Q_DIRECTION,
    Q_DISCHARGE,
    Q_XMOMENTUM,
    Q_YMOMENTUM,
    Q_FROUDE,
    Q_HAZARD,
    N_QUANTITIES
};

struct output_options {
    const char *basename;       /* output=, NULL for no time series */
    int quantity[N_QUANTITIES]; /* Selected by outputs= */
    double min_depth;           /* Dry threshold for writing */
    int null_dry;               /* -n */
    int dcell;                  /* -d */
    int detail;                 /* -f: also write on the fine DEM footprints */
    int absolute;               /* start= given */
    double start_epoch;         /* start= as seconds since the epoch (UTC) */
    double output_step, duration;
    const char *massbalance; /* CSV file or NULL */
    int print_mass_error;    /* -m */
    /* Summary rasters (NULL when not requested). */
    const char *max_depth, *max_speed, *max_stage, *max_hazard;
    const char *arrival_time, *inundation_duration, *final_prefix;
};

#define MAX_OUT_GRIDS (1 + MAX_DEMS)

struct output_context {
    const struct output_options *opt;
    const struct out_grid *grids; /* Main grid first, then detail grids. */
    int n_grids;
    struct Cell_head region;
    const struct evolve_log *log;
    double initial_signed_volume;
    long n_outputs, next_index;
    int index_width;
    double *h; /* Per-triangle depth at the current output. */
    FILE *massbal;
    FILE *register_file[MAX_OUT_GRIDS][N_QUANTITIES];
    char *register_path[MAX_OUT_GRIDS][N_QUANTITIES];
};

/* Parse outputs= answers into opt->quantity; fatal for unknown names. */
void output_parse_quantities(struct output_options *opt, char **answers);

/* Parse start= (YYYY-MM-DD[ HH:MM[:SS]], T separator allowed) into
 * seconds since the epoch, UTC. */
double output_parse_start(const char *text);

/* Check that no output map or dataset exists unless --overwrite, before
 * any computation. n_detail is the number of detail grids (-f), named with
 * the suffixes _detail1, _detail2, ... */
void output_check_names(const struct output_options *opt, int n_detail);

void output_begin(struct output_context *ctx, const struct output_options *opt,
                  const struct out_grid *grids, int n_grids,
                  const struct evolve_log *log, const struct sw_state *s);

/* output_fn for evolve_run(): writes the rasters of one output time. */
void output_step_fn(struct sw_state *s, double t, void *data);

/* Summary rasters, mass balance closing line, and registration of the
 * time series as space-time raster datasets. */
void output_end(struct output_context *ctx, const struct sw_state *s);

#endif /* R_HYDRO_ANUGA_OUTPUT_H */
