/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenMP tier: loops over the shared kernel bodies of
 *               cl/anuga_sw.h (PLAN.md section 3.1).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>

#include "cl/anuga_sw.h"
#include "evolve.h"

static double omp_protect(struct sw_state *s)
{
    double mass_error = 0.0;
    long k;

#pragma omp parallel for reduction(+ : mass_error) schedule(static)
    for (k = 0; k < s->n; k++)
        mass_error +=
            sw_protect((anuga_idx)k, s->cfg.P, s->zq, s->z0, s->stage_c,
                       s->xmom_c, s->ymom_c, s->height_c, s->areas);

    return mass_error;
}

static void omp_extrapolate(struct sw_state *s)
{
    long k;

#pragma omp parallel for schedule(static)
    for (k = 0; k < s->n; k++)
        sw_extrapolate_pass1((anuga_idx)k, s->cfg.P, s->zq, s->z0, s->stage_c,
                             s->xmom_c, s->ymom_c, s->height_c, s->xwork,
                             s->ywork);

#pragma omp parallel for schedule(static)
    for (k = 0; k < s->n; k++)
        sw_extrapolate_pass2((anuga_idx)k, s->cfg.P, s->stage_c, s->xmom_c,
                             s->ymom_c, s->height_c, s->xwork, s->ywork,
                             s->stage_e, s->xmom_e, s->ymom_e, s->height_e,
                             s->centroid_coords, s->edge_coords, s->surrogate,
                             s->number_of_boundaries);

    if (s->cfg.P.extrapolate_velocity_second_order == 1.0) {
#pragma omp parallel for schedule(static)
        for (k = 0; k < s->n; k++)
            sw_extrapolate_pass3((anuga_idx)k, s->xmom_c, s->ymom_c, s->xwork,
                                 s->ywork);
    }
}

static void omp_boundaries(struct sw_state *s)
{
    long j;

#pragma omp parallel for schedule(static)
    for (j = 0; j < s->nb; j++)
        sw_boundary((anuga_idx)j, s->bnd_tri, s->bnd_edge, s->bnd_type,
                    s->bnd_value, s->cfg.transmissive_use_centroid, s->stage_c,
                    s->xmom_c, s->ymom_c, s->stage_e, s->xmom_e, s->ymom_e,
                    s->normals, s->stage_bv, s->xmom_bv, s->ymom_bv);
}

static double omp_fluxes(struct sw_state *s, int first_substep, double *bflux)
{
    double dt_min = 1.0e+100, bsum = 0.0;
    long k;

#pragma omp parallel for reduction(min : dt_min) reduction(+ : bsum) \
    schedule(static)
    for (k = 0; k < s->n; k++) {
        double cell_dt, cell_bflux;

        sw_fluxes((anuga_idx)k, s->cfg.P, first_substep, s->zq, s->z0,
                  s->height_c, s->stage_e, s->xmom_e, s->ymom_e, s->height_e,
                  s->stage_bv, s->xmom_bv, s->ymom_bv, s->stage_eu, s->xmom_eu,
                  s->ymom_eu, s->neighbours, s->neighbour_edges, s->normals,
                  s->edgelengths, s->radii, s->areas, s->max_speed, &cell_dt,
                  &cell_bflux);
        dt_min = fmin(dt_min, cell_dt);
        bsum += cell_bflux;
    }
    *bflux = bsum;

    return dt_min;
}

static void omp_friction(struct sw_state *s)
{
    long k;

    if (s->cfg.sloped_friction) {
#pragma omp parallel for schedule(static)
        for (k = 0; k < s->n; k++)
            sw_manning_sloped((anuga_idx)k, s->cfg.P, s->stage_c, s->stage_e,
                              s->height_e, s->xmom_c, s->ymom_c, s->friction,
                              s->edge_coords, s->xmom_siu, s->ymom_siu);
    }
    else {
#pragma omp parallel for schedule(static)
        for (k = 0; k < s->n; k++)
            sw_manning_flat((anuga_idx)k, s->cfg.P, s->zq, s->z0, s->stage_c,
                            s->xmom_c, s->ymom_c, s->friction, s->xmom_siu,
                            s->ymom_siu);
    }
}

static void omp_update(struct sw_state *s, double dt)
{
    long k;

#pragma omp parallel for schedule(static)
    for (k = 0; k < s->n; k++)
        sw_update((anuga_idx)k, dt, s->stage_c, s->xmom_c, s->ymom_c,
                  s->stage_eu, s->xmom_eu, s->ymom_eu, s->stage_siu,
                  s->xmom_siu, s->ymom_siu);
}

static void omp_backup(struct sw_state *s)
{
    long k;

#pragma omp parallel for schedule(static)
    for (k = 0; k < s->n; k++)
        sw_backup((anuga_idx)k, s->stage_c, s->xmom_c, s->ymom_c, s->stage_bk,
                  s->xmom_bk, s->ymom_bk);
}

static void omp_saxpy(struct sw_state *s, double a, double b, double c)
{
    int scale = (c != 1.0 && c != 0.0);
    double c_inv = scale ? 1.0 / c : 1.0;
    long k;

#pragma omp parallel for schedule(static)
    for (k = 0; k < s->n; k++)
        sw_saxpy((anuga_idx)k, a, b, scale, c_inv, s->zq, s->z0, s->stage_c,
                 s->xmom_c, s->ymom_c, s->height_c, s->stage_bk, s->xmom_bk,
                 s->ymom_bk);
}

static double omp_volume(struct sw_state *s)
{
    double volume = 0.0;
    long k;

#pragma omp parallel for reduction(+ : volume) schedule(static)
    for (k = 0; k < s->n; k++)
        volume += sw_volume((anuga_idx)k, s->zq, s->z0, s->stage_c, s->areas);

    return volume;
}

static void omp_sync_to_host(struct sw_state *s)
{
    (void)s;
}

static void omp_stats(struct sw_state *s, double t, double dt)
{
    long k;

#pragma omp parallel for schedule(static)
    for (k = 0; k < s->n; k++)
        sw_stats((anuga_idx)k, t, dt, s->velocity_zero_height, s->arrival_depth,
                 s->zq, s->z0, s->stage_c, s->xmom_c, s->ymom_c,
                 s->stat_max_stage, s->stat_max_depth, s->stat_max_speed,
                 s->stat_max_hazard, s->stat_arrival, s->stat_duration);
}

const struct solver_ops omp_ops = {
    "OpenMP",         omp_protect, omp_extrapolate, omp_boundaries, omp_fluxes,
    omp_friction,     omp_update,  omp_backup,      omp_saxpy,      omp_volume,
    omp_sync_to_host, omp_stats,   omp_sync_to_host};
