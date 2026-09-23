/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Time stepping: ANUGA's Euler, RK2 and RK3 steps and the
 *               evolve loop with output times (PLAN.md section 5.1).
 *               The step sequences follow gpu_evolve_one_{euler,rk2,rk3}_step
 *               (anuga/shallow_water/gpu/gpu_kernels.c) and the loop follows
 *               Generic_Domain.evolve() (generic_domain.py), so that time
 *               steps and states match ANUGA's "unified" compute mode.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *               Step orchestration derived from ANUGA, (C) 2004-2015
 *               Australian National University and Geoscience Australia,
 *               Apache License 2.0 (see LICENSE.ANUGA).
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "evolve.h"

/* anuga.config.epsilon, used by evolve() to detect the final time. */
#define EVOLVE_EPSILON 1.0e-12

/* One flux evaluation: protect, extrapolate, boundaries, fluxes. The
 * water added by protect() is returned in *pmass. */
static double evaluate(struct sw_state *s, const struct solver_ops *ops,
                       int first_substep, double *bflux, double *pmass)
{
    *pmass = ops->protect(s);
    ops->extrapolate(s);
    ops->boundaries(s);

    return ops->fluxes(s, first_substep, bflux);
}

static double cfl_timestep(const struct sw_state *s, double local_timestep,
                           double max_timestep)
{
    double timestep = s->cfg.cfl * local_timestep;

    if (timestep > max_timestep)
        timestep = max_timestep;

    return timestep;
}

/* Each step returns its boundary mass flux in *bmass and the water added
 * by protect() in *pmass, both weighted by the Runge-Kutta coefficients
 * with which the substep states enter the result, so that the change of
 * the signed volume equals *bmass + *pmass. */
static double step_euler(struct sw_state *s, const struct solver_ops *ops,
                         double max_timestep, double *bmass, double *pmass)
{
    double b0, p0, dt;

    dt = cfl_timestep(s, evaluate(s, ops, 1, &b0, &p0), max_timestep);
    ops->friction(s);
    ops->update(s, dt);
    *bmass = b0 * dt;
    *pmass = p0;

    return dt;
}

static double step_rk2(struct sw_state *s, const struct solver_ops *ops,
                       double max_timestep, double *bmass, double *pmass)
{
    double b0, b1, p0, p1, dt;

    ops->backup(s);

    dt = cfl_timestep(s, evaluate(s, ops, 1, &b0, &p0), max_timestep);
    ops->friction(s);
    ops->update(s, dt);

    evaluate(s, ops, 0, &b1, &p1);
    ops->friction(s);
    ops->update(s, dt);

    ops->saxpy(s, 0.5, 0.5, 0.0);
    *bmass = 0.5 * (b0 + b1) * dt;
    *pmass = 0.5 * (p0 + p1);

    return dt;
}

static double step_rk3(struct sw_state *s, const struct solver_ops *ops,
                       double max_timestep, double *bmass, double *pmass)
{
    double b0, b1, b2, p0, p1, p2, dt;

    ops->backup(s);

    dt = cfl_timestep(s, evaluate(s, ops, 1, &b0, &p0), max_timestep);
    ops->friction(s);
    ops->update(s, dt);

    evaluate(s, ops, 0, &b1, &p1);
    ops->friction(s);
    ops->update(s, dt);
    ops->saxpy(s, 0.25, 0.75, 0.0);

    evaluate(s, ops, 0, &b2, &p2);
    ops->friction(s);
    ops->update(s, dt);
    ops->saxpy(s, 2.0, 1.0, 3.0);
    /* Shu-Osher weights of the three flux evaluations. */
    *bmass = (b0 / 6.0 + b1 / 6.0 + 2.0 * b2 / 3.0) * dt;
    *pmass = p0 / 6.0 + p1 / 6.0 + 2.0 * p2 / 3.0;

    return dt;
}

static void log_step(struct evolve_log *log, double t, double dt)
{
    if (!log)
        return;
    if (log->n_steps == log->capacity) {
        log->capacity = log->capacity ? 2 * log->capacity : 1024;
        log->t = G_realloc(log->t, log->capacity * sizeof(double));
        log->dt = G_realloc(log->dt, log->capacity * sizeof(double));
    }
    log->t[log->n_steps] = t;
    log->dt[log->n_steps] = dt;
    log->n_steps++;
}

void evolve_run(struct sw_state *s, const struct solver_ops *ops,
                double duration, double yieldstep, output_fn output,
                void *output_data, struct evolve_log *log)
{
    double t = 0.0, yield_t = yieldstep, bmass, pmass;
    long steps = 0;

    if (!(duration > 0.0) || !(yieldstep > 0.0))
        G_fatal_error(_("Duration and output step must be positive"));
    if (log) {
        log->n_steps = 0;
        log->boundary_mass = 0.0;
        log->protect_mass = 0.0;
    }

    while (1) {
        double max_dt = s->cfg.evolve_max_timestep, remaining, dt;

        /* _clip_timestep_to_output_times */
        remaining = fmax(duration - t, 0.0);
        if (max_dt > remaining)
            max_dt = remaining;
        remaining = fmax(yield_t - t, 0.0);
        if (max_dt > remaining)
            max_dt = remaining;

        switch (s->cfg.algorithm) {
        case ALG_DE0:
            dt = step_euler(s, ops, max_dt, &bmass, &pmass);
            break;
        case ALG_DE2:
            dt = step_rk3(s, ops, max_dt, &bmass, &pmass);
            break;
        default:
            dt = step_rk2(s, ops, max_dt, &bmass, &pmass);
            break;
        }

        t = t + dt;
        steps++;
        if (log) {
            log_step(log, t, dt);
            log->boundary_mass += bmass;
            log->protect_mass += pmass;
        }
        G_percent((long)(1000.0 * fmin(t / duration, 1.0)), 1000, 2);

        if (t >= duration - EVOLVE_EPSILON) {
            if (t > duration)
                G_fatal_error(_("Internal error: time overshot the end time"));
            ops->sync_to_host(s);
            if (output)
                output(s, duration, output_data);
            break;
        }
        if (t >= yield_t) {
            ops->sync_to_host(s);
            if (output)
                output(s, t, output_data);
            yield_t += yieldstep;
        }
    }
    G_verbose_message(_("%ld time steps"), steps);
}

void evolve_log_free(struct evolve_log *log)
{
    G_free(log->t);
    G_free(log->dt);
    log->t = log->dt = NULL;
    log->n_steps = log->capacity = 0;
}
