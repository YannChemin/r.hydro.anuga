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

#ifndef R_HYDRO_ANUGA_STATE_H
#define R_HYDRO_ANUGA_STATE_H

#include "cl/anuga_common.h"
#include "mesh.h"

enum flow_algorithm { ALG_DE0, ALG_DE1, ALG_DE2 };

/* Solver configuration derived from algorithm= (ANUGA's parameter
 * bundles, checked against anuga.Domain.set_flow_algorithm) and options. */
struct solver_config {
    enum flow_algorithm algorithm;
    struct sw_params P;
    double cfl;
    double evolve_max_timestep;
    int sloped_friction;
    int transmissive_use_centroid;
};

void solver_config_init(struct solver_config *cfg, enum flow_algorithm alg);

struct sw_state {
    long n, nb;
    struct solver_config cfg;

    /* Mesh arrays (borrowed from struct mesh, not owned). */
    const anuga_zq *zq;
    anuga_z0 z0;
    const double *centroid_coords, *edge_coords, *normals, *edgelengths;
    const double *radii, *areas;
    const anuga_idx *neighbours, *neighbour_edges, *surrogate;
    const anuga_idx *number_of_boundaries;
    const anuga_idx *bnd_tri, *bnd_edge;

    /* Boundary conditions (owned). */
    anuga_u8 *bnd_type;
    double *bnd_value; /* nb * 3: Dirichlet stage, xmom, ymom */

    /* Centroid values. */
    double *stage_c, *xmom_c, *ymom_c, *height_c, *friction;
    /* Edge values (3 per triangle). */
    double *stage_e, *xmom_e, *ymom_e, *height_e;
    /* Boundary values. */
    double *stage_bv, *xmom_bv, *ymom_bv;
    /* Updates, backups and work arrays. */
    double *stage_eu, *xmom_eu, *ymom_eu;
    double *stage_siu, *xmom_siu, *ymom_siu;
    double *stage_bk, *xmom_bk, *ymom_bk;
    double *xwork, *ywork, *max_speed;
};

/* Allocate the state for mesh m. Conserved quantities start dry (stage =
 * bed, zero momentum), friction zero, all boundaries reflective. */
void state_init(struct sw_state *s, const struct mesh *m,
                const struct solver_config *cfg);

void state_free(struct sw_state *s);

#endif /* R_HYDRO_ANUGA_STATE_H */
