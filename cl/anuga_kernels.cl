/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL C 1.1 kernels: thin wrappers around the shared
 *               per-triangle bodies of anuga_sw.h, plus work-group
 *               reductions (PLAN.md sections 3.1 and 5). The program
 *               source is anuga_common.h + anuga_sw.h + this file.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

/* Work-group reductions over local memory. The work-group size is a power
 * of two, set by the host with -DWG=... . One partial result per group is
 * written; the host combines the partials in group order, so results do
 * not depend on scheduling. */

void wg_reduce_sum(__local double *buf, __global double *partial)
{
    int lid = get_local_id(0);

    for (int s = WG / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s)
            buf[lid] += buf[lid + s];
    }
    if (lid == 0)
        partial[get_group_id(0)] = buf[0];
}

void wg_reduce_min(__local double *buf, __global double *partial)
{
    int lid = get_local_id(0);

    for (int s = WG / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s)
            buf[lid] = fmin(buf[lid], buf[lid + s]);
    }
    if (lid == 0)
        partial[get_group_id(0)] = buf[0];
}

__kernel void k_protect(int n, struct sw_params P, __global const uint *zq,
                        long z0, __global double *stage_c,
                        __global double *xmom_c, __global double *ymom_c,
                        __global double *height_c,
                        __global const double *areas,
                        __global double *partial)
{
    __local double buf[WG];
    int k = get_global_id(0);

    buf[get_local_id(0)] =
        k < n ? sw_protect(k, P, zq, z0, stage_c, xmom_c, ymom_c, height_c,
                           areas)
              : 0.0;
    wg_reduce_sum(buf, partial);
}

__kernel void k_extrapolate_pass1(int n, struct sw_params P,
                                  __global const uint *zq, long z0,
                                  __global const double *stage_c,
                                  __global double *xmom_c,
                                  __global double *ymom_c,
                                  __global double *height_c,
                                  __global double *xwork,
                                  __global double *ywork)
{
    int k = get_global_id(0);

    if (k < n)
        sw_extrapolate_pass1(k, P, zq, z0, stage_c, xmom_c, ymom_c, height_c,
                             xwork, ywork);
}

__kernel void k_extrapolate_pass2(
    int n, struct sw_params P, __global const double *stage_c,
    __global const double *xmom_c, __global const double *ymom_c,
    __global const double *height_c, __global double *xwork,
    __global double *ywork, __global double *stage_e, __global double *xmom_e,
    __global double *ymom_e, __global double *height_e,
    __global const double *centroid_coords, __global const double *edge_coords,
    __global const int *surrogate, __global const int *number_of_boundaries)
{
    int k = get_global_id(0);

    if (k < n)
        sw_extrapolate_pass2(k, P, stage_c, xmom_c, ymom_c, height_c, xwork,
                             ywork, stage_e, xmom_e, ymom_e, height_e,
                             centroid_coords, edge_coords, surrogate,
                             number_of_boundaries);
}

__kernel void k_extrapolate_pass3(int n, __global double *xmom_c,
                                  __global double *ymom_c,
                                  __global const double *xwork,
                                  __global const double *ywork)
{
    int k = get_global_id(0);

    if (k < n)
        sw_extrapolate_pass3(k, xmom_c, ymom_c, xwork, ywork);
}

__kernel void k_boundaries(
    int nb, __global const int *bnd_tri, __global const int *bnd_edge,
    __global const uchar *bnd_type, __global const double *bnd_value,
    int use_centroid, __global const double *stage_c,
    __global const double *xmom_c, __global const double *ymom_c,
    __global const double *stage_e, __global const double *xmom_e,
    __global const double *ymom_e, __global const double *normals,
    __global double *stage_bv, __global double *xmom_bv,
    __global double *ymom_bv)
{
    int j = get_global_id(0);

    if (j < nb)
        sw_boundary(j, bnd_tri, bnd_edge, bnd_type, bnd_value, use_centroid,
                    stage_c, xmom_c, ymom_c, stage_e, xmom_e, ymom_e, normals,
                    stage_bv, xmom_bv, ymom_bv);
}

__kernel void k_fluxes(
    int n, struct sw_params P, int first_substep, __global const uint *zq,
    long z0, __global const double *height_c, __global const double *stage_e,
    __global const double *xmom_e, __global const double *ymom_e,
    __global const double *height_e, __global const double *stage_bv,
    __global const double *xmom_bv, __global const double *ymom_bv,
    __global double *stage_eu, __global double *xmom_eu,
    __global double *ymom_eu, __global const int *neighbours,
    __global const int *neighbour_edges, __global const double *normals,
    __global const double *edgelengths, __global const double *radii,
    __global const double *areas, __global double *max_speed,
    __global double *dt_partial, __global double *bflux_partial)
{
    __local double buf[WG];
    int k = get_global_id(0), lid = get_local_id(0);
    double cell_dt = 1.0e+100, cell_bflux = 0.0;

    if (k < n)
        sw_fluxes(k, P, first_substep, zq, z0, height_c, stage_e, xmom_e,
                  ymom_e, height_e, stage_bv, xmom_bv, ymom_bv, stage_eu,
                  xmom_eu, ymom_eu, neighbours, neighbour_edges, normals,
                  edgelengths, radii, areas, max_speed, &cell_dt,
                  &cell_bflux);

    buf[lid] = cell_dt;
    wg_reduce_min(buf, dt_partial);
    barrier(CLK_LOCAL_MEM_FENCE);
    buf[lid] = cell_bflux;
    wg_reduce_sum(buf, bflux_partial);
}

__kernel void k_friction_flat(int n, struct sw_params P,
                              __global const uint *zq, long z0,
                              __global const double *stage_c,
                              __global const double *xmom_c,
                              __global const double *ymom_c,
                              __global const double *friction,
                              __global double *xmom_siu,
                              __global double *ymom_siu)
{
    int k = get_global_id(0);

    if (k < n)
        sw_manning_flat(k, P, zq, z0, stage_c, xmom_c, ymom_c, friction,
                        xmom_siu, ymom_siu);
}

__kernel void k_friction_sloped(
    int n, struct sw_params P, __global const double *stage_c,
    __global const double *stage_e, __global const double *height_e,
    __global const double *xmom_c, __global const double *ymom_c,
    __global const double *friction, __global const double *edge_coords,
    __global double *xmom_siu, __global double *ymom_siu)
{
    int k = get_global_id(0);

    if (k < n)
        sw_manning_sloped(k, P, stage_c, stage_e, height_e, xmom_c, ymom_c,
                          friction, edge_coords, xmom_siu, ymom_siu);
}

__kernel void k_update(int n, double dt, __global double *stage_c,
                       __global double *xmom_c, __global double *ymom_c,
                       __global const double *stage_eu,
                       __global const double *xmom_eu,
                       __global const double *ymom_eu,
                       __global double *stage_siu, __global double *xmom_siu,
                       __global double *ymom_siu)
{
    int k = get_global_id(0);

    if (k < n)
        sw_update(k, dt, stage_c, xmom_c, ymom_c, stage_eu, xmom_eu, ymom_eu,
                  stage_siu, xmom_siu, ymom_siu);
}

__kernel void k_backup(int n, __global const double *stage_c,
                       __global const double *xmom_c,
                       __global const double *ymom_c,
                       __global double *stage_bk, __global double *xmom_bk,
                       __global double *ymom_bk)
{
    int k = get_global_id(0);

    if (k < n)
        sw_backup(k, stage_c, xmom_c, ymom_c, stage_bk, xmom_bk, ymom_bk);
}

__kernel void k_saxpy(int n, double a, double b, int scale, double c_inv,
                      __global const uint *zq, long z0,
                      __global double *stage_c, __global double *xmom_c,
                      __global double *ymom_c, __global double *height_c,
                      __global const double *stage_bk,
                      __global const double *xmom_bk,
                      __global const double *ymom_bk)
{
    int k = get_global_id(0);

    if (k < n)
        sw_saxpy(k, a, b, scale, c_inv, zq, z0, stage_c, xmom_c, ymom_c,
                 height_c, stage_bk, xmom_bk, ymom_bk);
}

__kernel void k_volume(int n, __global const uint *zq, long z0,
                       __global const double *stage_c,
                       __global const double *areas, __global double *partial)
{
    __local double buf[WG];
    int k = get_global_id(0);

    buf[get_local_id(0)] = k < n ? sw_volume(k, zq, z0, stage_c, areas) : 0.0;
    wg_reduce_sum(buf, partial);
}
