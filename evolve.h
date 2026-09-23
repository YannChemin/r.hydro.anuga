/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Time stepping: ANUGA's Euler, RK2 and RK3 steps and the
 *               evolve loop with output times (PLAN.md section 5.1).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_EVOLVE_H
#define R_HYDRO_ANUGA_EVOLVE_H

#include "state.h"

/* Kernel operations of one compute tier. The OpenMP tier works on the
 * host arrays of struct sw_state; the OpenCL tier (phase 3) on device
 * buffers, synchronising with sync_to_host() before outputs. */
struct solver_ops {
    const char *name;
    double (*protect)(struct sw_state *s);
    void (*extrapolate)(struct sw_state *s);
    void (*boundaries)(struct sw_state *s);
    /* Returns the minimum cell time step (1e100 if none) and the summed
     * boundary mass flux in *bflux. */
    double (*fluxes)(struct sw_state *s, int first_substep, double *bflux);
    void (*friction)(struct sw_state *s);
    void (*update)(struct sw_state *s, double dt);
    void (*backup)(struct sw_state *s);
    void (*saxpy)(struct sw_state *s, double a, double b, double c);
    double (*volume)(struct sw_state *s);
    void (*sync_to_host)(struct sw_state *s);
};

extern const struct solver_ops omp_ops;

/* Called at every output time (and at the final time) with the state
 * synchronised to the host. */
typedef void (*output_fn)(struct sw_state *s, double t, void *data);

struct evolve_log {
    long n_steps, capacity;
    double *t, *dt;       /* Time after each step and the step taken. */
    double boundary_mass; /* Integrated boundary mass flux (m3, + = in). */
    double protect_mass;  /* Water added by clamping negative depths (m3). */
};

/* Advance from t = 0 to duration, calling output at every yieldstep and at
 * the end, exactly as ANUGA's evolve() loop (generic_domain.py) caps and
 * accumulates time steps. */
void evolve_run(struct sw_state *s, const struct solver_ops *ops,
                double duration, double yieldstep, output_fn output,
                void *output_data, struct evolve_log *log);

void evolve_log_free(struct evolve_log *log);

#endif /* R_HYDRO_ANUGA_EVOLVE_H */
