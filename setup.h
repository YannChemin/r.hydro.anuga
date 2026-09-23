/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Solver setup from module options: configuration, boundary
 *               conditions, initial conditions, friction, and the state
 *               export used for validation (PLAN.md sections 5 and 7).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_SETUP_H
#define R_HYDRO_ANUGA_SETUP_H

#include "evolve.h"
#include "mesh.h"
#include "state.h"

/* algorithm: DE0|DE1|DE2; cfl: NULL for the algorithm's default;
 * friction_method: sloped|flat. */
void setup_config(struct solver_config *cfg, const char *algorithm,
                  const char *cfl, const char *friction_method);

/* Parse boundary= answers ("side:type[:stage[:xmom:ymom]]", side one of
 * north, south, east, west, null) and set the type of every boundary
 * edge. Unlisted sides stay reflective. */
void setup_boundaries(struct sw_state *s, const struct mesh *m, char **answers);

/* Initial water from a depth or a stage raster (at most one), sampled at
 * triangle centroids; NULL cells are dry. */
void setup_initial(struct sw_state *s, const struct mesh *m,
                   const char *initial_depth, const char *initial_stage);

/* Manning's n from a raster sampled at centroids, else the constant. */
void setup_friction(struct sw_state *s, const struct mesh *m,
                    const char *manning, double manning_value);

/* Snapshot of the conserved quantities, for the export. */
struct state_snapshot {
    double *stage, *xmom, *ymom;
    double volume;        /* Positive depths only, as ANUGA reports it. */
    double signed_volume; /* Sum of (stage - bed) * area. */
};

void snapshot_take(struct state_snapshot *snap, struct sw_state *s,
                   const struct solver_ops *ops);
void snapshot_free(struct state_snapshot *snap);

/* Write the initial and final states, bed, friction, boundary setup and
 * the time step log as raw arrays plus manifest.json into dir. */
void state_export(const char *dir, const struct sw_state *s,
                  const struct state_snapshot *initial,
                  const struct state_snapshot *final,
                  const struct evolve_log *log, double duration,
                  double yieldstep);

#endif /* R_HYDRO_ANUGA_SETUP_H */
