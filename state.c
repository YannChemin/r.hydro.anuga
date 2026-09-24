/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Solver state: structure-of-arrays of conserved quantities,
 *               work arrays and the mesh arrays the kernels read (PLAN.md
 *               sections 4.7 and 5).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <string.h>

#include <grass/gis.h>

#include "state.h"

/* Parameter bundles of ANUGA's set_flow_algorithm (shallow_water_domain.py
 * _set_DE0/DE1/DE2_defaults), as reported by a live anuga.Domain. */
void solver_config_init(struct solver_config *cfg, enum flow_algorithm alg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->algorithm = alg;
    cfg->P.g = 9.8;
    cfg->P.epsilon = 1.0e-12;
    cfg->P.low_froude = 0.0;
    cfg->P.extrapolate_velocity_second_order = 1.0;
    cfg->P.beta_w_dry = 0.0;
    cfg->P.beta_uh_dry = 0.0;
    cfg->P.beta_vh_dry = 0.0;
    cfg->evolve_max_timestep = 1000.0;
    cfg->sloped_friction = 0;
    cfg->transmissive_use_centroid = 0;

    if (alg == ALG_DE0) {
        cfg->cfl = 0.9;
        cfg->P.minimum_allowed_height = 1.0e-12;
        cfg->P.beta_w = cfg->P.beta_uh = cfg->P.beta_vh = 0.5;
    }
    else {
        cfg->cfl = 1.0;
        cfg->P.minimum_allowed_height = 1.0e-5;
        /* Decision R4 (PLAN.md): DE1 defaults to CFL 0.5 instead of
         * ANUGA's 1.0. At CFL 1, thin films on coarse steep terrain are
         * driven to negative depths often enough that clamping them adds
         * a large volume of water (33% in 60 s on the Plumergat test);
         * at 0.5 it is about 10x less. */
        if (alg == ALG_DE1)
            cfg->cfl = 0.5;
        cfg->P.beta_w = cfg->P.beta_uh = cfg->P.beta_vh = 1.0;
    }
}

static double *zeros(long n)
{
    return G_calloc(n > 0 ? n : 1, sizeof(double));
}

void state_init(struct sw_state *s, const struct mesh *m,
                const struct solver_config *cfg)
{
    long n = m->n_tri, nb = m->n_boundary, k;

    memset(s, 0, sizeof(*s));
    s->n = n;
    s->nb = nb;
    s->cfg = *cfg;

    s->zq = m->zq;
    s->z0 = m->z0;
    s->centroid_coords = m->centroid_coordinates;
    s->edge_coords = m->edge_coordinates;
    s->normals = m->normals;
    s->edgelengths = m->edgelengths;
    s->radii = m->radii;
    s->areas = m->areas;
    s->neighbours = m->neighbours;
    s->neighbour_edges = m->neighbour_edges;
    s->surrogate = m->surrogate_neighbours;
    s->number_of_boundaries = m->number_of_boundaries;
    s->bnd_tri = m->boundary_tri;
    s->bnd_edge = m->boundary_edge;

    s->bnd_type = G_calloc(nb > 0 ? nb : 1, sizeof(anuga_u8));
    s->bnd_value = zeros(3 * nb);

    s->stage_c = zeros(n);
    s->xmom_c = zeros(n);
    s->ymom_c = zeros(n);
    s->height_c = zeros(n);
    s->friction = zeros(n);
    s->stage_e = zeros(3 * n);
    s->xmom_e = zeros(3 * n);
    s->ymom_e = zeros(3 * n);
    s->height_e = zeros(3 * n);
    s->stage_bv = zeros(nb);
    s->xmom_bv = zeros(nb);
    s->ymom_bv = zeros(nb);
    s->stage_eu = zeros(n);
    s->xmom_eu = zeros(n);
    s->ymom_eu = zeros(n);
    s->stage_siu = zeros(n);
    s->xmom_siu = zeros(n);
    s->ymom_siu = zeros(n);
    s->stage_bk = zeros(n);
    s->xmom_bk = zeros(n);
    s->ymom_bk = zeros(n);
    s->xwork = zeros(n);
    s->ywork = zeros(n);
    s->max_speed = zeros(n);

    for (k = 0; k < n; k++)
        s->stage_c[k] = anuga_bed(s->zq, s->z0, (anuga_idx)k);
}

void state_enable_stats(struct sw_state *s, double velocity_zero_height,
                        double arrival_depth)
{
    long k;

    s->stats = 1;
    s->velocity_zero_height = velocity_zero_height;
    s->arrival_depth = arrival_depth;
    s->stat_max_stage = zeros(s->n);
    s->stat_max_depth = zeros(s->n);
    s->stat_max_speed = zeros(s->n);
    s->stat_max_hazard = zeros(s->n);
    s->stat_arrival = zeros(s->n);
    s->stat_duration = zeros(s->n);
    for (k = 0; k < s->n; k++) {
        s->stat_max_stage[k] = -1.0e100;
        s->stat_arrival[k] = -1.0;
    }
}

void state_free(struct sw_state *s)
{
    double *arrays[] = {
        s->bnd_value, s->stage_c,   s->xmom_c,   s->ymom_c,   s->height_c,
        s->friction,  s->stage_e,   s->xmom_e,   s->ymom_e,   s->height_e,
        s->stage_bv,  s->xmom_bv,   s->ymom_bv,  s->stage_eu, s->xmom_eu,
        s->ymom_eu,   s->stage_siu, s->xmom_siu, s->ymom_siu, s->stage_bk,
        s->xmom_bk,   s->ymom_bk,   s->xwork,    s->ywork,    s->max_speed};
    size_t i;

    for (i = 0; i < sizeof(arrays) / sizeof(arrays[0]); i++)
        G_free(arrays[i]);
    G_free(s->bnd_type);
    if (s->stats) {
        G_free(s->stat_max_stage);
        G_free(s->stat_max_depth);
        G_free(s->stat_max_speed);
        G_free(s->stat_max_hazard);
        G_free(s->stat_arrival);
        G_free(s->stat_duration);
    }
    memset(s, 0, sizeof(*s));
}
