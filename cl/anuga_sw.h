/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Per-triangle bodies of ANUGA's discontinuous-elevation
 *               shallow-water kernels, written once for both the C
 *               (OpenMP) and the OpenCL C 1.1 compilations (PLAN.md
 *               sections 3.1, 4.7 and 5).
 *
 *               Ported from ANUGA 4.0 anuga/shallow_water/gpu/
 *               core_kernels.c, gpu_device_helpers.h, gpu_boundaries.c
 *               and gpu_kernels.c, keeping every floating-point operation
 *               and its order, so results are bitwise identical to ANUGA's
 *               "unified" compute mode on the same mesh. Differences, all
 *               value-preserving:
 *               - The bed is read through anuga_bed() from the scaled
 *                 integer zq; bed edge values are not stored but computed
 *                 as stage_edge - height_edge, which is exactly how ANUGA
 *                 sets them at the end of the extrapolation.
 *               - In the extrapolation, the zeroing of a triangle's own
 *                 centroid velocity when all its neighbours are dry is
 *                 kept local (neighbours read the pass-1 value), removing
 *                 a read/write race of ANUGA's parallel loop. The race
 *                 cannot change results when every triangle has at most
 *                 one boundary edge (always true for this module's
 *                 meshes) and the dry betas are zero (DE0/DE1/DE2).
 *               - protect() and saxpy() also update the centroid height,
 *                 as ANUGA's gpu_protect()/gpu_saxpy_*() wrappers do in a
 *                 second loop.
 *               Riverwalls, ADER and the other boundary types are not
 *               ported (PLAN.md section 9).
 *
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *               Derived from ANUGA, (C) 2004-2015 Australian National
 *               University and Geoscience Australia, Apache License 2.0
 *               (see LICENSE.ANUGA).
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_CL_SW_H
#define R_HYDRO_ANUGA_CL_SW_H

#include "anuga_common.h"

#define SW_TINY 1.0e-100

/* ---- Extrapolation helpers (gpu_device_helpers.h) ---------------------- */

KINLINE void sw_find_qmin_and_qmax_dq1_dq2(double dq0, double dq1, double dq2,
                                           double *qmin, double *qmax)
{
    *qmax = fmax(fmax(dq0, fmax(dq0 + dq1, dq0 + dq2)), 0.0);
    *qmin = fmin(fmin(dq0, fmin(dq0 + dq1, dq0 + dq2)), 0.0);
}

KINLINE void sw_compute_qmin_qmax_from_dq1(double dq1, double *qmin,
                                           double *qmax)
{
    if (dq1 >= 0.0) {
        *qmin = 0.0;
        *qmax = dq1;
    }
    else {
        *qmin = dq1;
        *qmax = 0.0;
    }
}

KINLINE void sw_limit_gradient(double *dqv, double qmin, double qmax,
                               double beta_w)
{
    double r = 1000.0;
    double dq_x = dqv[0], dq_y = dqv[1], dq_z = dqv[2];
    double phi;

    if (dq_x < -SW_TINY)
        r = fmin(r, qmin / dq_x);
    else if (dq_x > SW_TINY)
        r = fmin(r, qmax / dq_x);
    if (dq_y < -SW_TINY)
        r = fmin(r, qmin / dq_y);
    else if (dq_y > SW_TINY)
        r = fmin(r, qmax / dq_y);
    if (dq_z < -SW_TINY)
        r = fmin(r, qmin / dq_z);
    else if (dq_z > SW_TINY)
        r = fmin(r, qmax / dq_z);

    phi = fmin(r * beta_w, 1.0);
    dqv[0] *= phi;
    dqv[1] *= phi;
    dqv[2] *= phi;
}

KINLINE void sw_calc_edge_values_with_gradient(
    double cv_k, double cv_k0, double cv_k1, double cv_k2, double dxv0,
    double dxv1, double dxv2, double dyv0, double dyv1, double dyv2, double dx1,
    double dx2, double dy1, double dy2, double inv_area2, double beta_tmp,
    double *edge_values)
{
    double dqv[3], qmin, qmax;
    double dq0 = cv_k0 - cv_k;
    double dq1 = cv_k1 - cv_k0;
    double dq2 = cv_k2 - cv_k0;
    double a = (dy2 * dq1 - dy1 * dq2) * inv_area2;
    double b = (dx1 * dq2 - dx2 * dq1) * inv_area2;

    dqv[0] = a * dxv0 + b * dyv0;
    dqv[1] = a * dxv1 + b * dyv1;
    dqv[2] = a * dxv2 + b * dyv2;
    sw_find_qmin_and_qmax_dq1_dq2(dq0, dq1, dq2, &qmin, &qmax);
    sw_limit_gradient(dqv, qmin, qmax, beta_tmp);
    edge_values[0] = cv_k + dqv[0];
    edge_values[1] = cv_k + dqv[1];
    edge_values[2] = cv_k + dqv[2];
}

KINLINE void sw_compute_dqv_from_gradient(double dq1, double dx2, double dy2,
                                          double dxv0, double dxv1, double dxv2,
                                          double dyv0, double dyv1, double dyv2,
                                          double *dqv)
{
    double a = dq1 * dx2;
    double b = dq1 * dy2;

    dqv[0] = a * dxv0 + b * dyv0;
    dqv[1] = a * dxv1 + b * dyv1;
    dqv[2] = a * dxv2 + b * dyv2;
}

/* ---- Flux helpers (gpu_device_helpers.h) -------------------------------- */

KINLINE void sw_rotate(double *q, double n1, double n2)
{
    double q1 = q[1];
    double q2 = q[2];

    q[1] = n1 * q1 + n2 * q2;
    q[2] = -n2 * q1 + n1 * q2;
}

KINLINE void sw_compute_velocity_terms(double h, double h_edge, double uh_raw,
                                       double vh_raw, double *u, double *uh,
                                       double *v, double *vh)
{
    if (h_edge > 0.0) {
        double inv_h_edge = 1.0 / h_edge;

        *u = uh_raw * inv_h_edge;
        *uh = h * (*u);
        *v = vh_raw * inv_h_edge;
        *vh = h * inv_h_edge * vh_raw;
    }
    else {
        *u = 0.0;
        *uh = 0.0;
        *v = 0.0;
        *vh = 0.0;
    }
}

KINLINE double sw_compute_local_froude(double low_froude, double u_left,
                                       double u_right, double v_left,
                                       double v_right, double soundspeed_left,
                                       double soundspeed_right)
{
    double numerator = u_right * u_right + u_left * u_left + v_right * v_right +
                       v_left * v_left;
    double denominator = soundspeed_left * soundspeed_left +
                         soundspeed_right * soundspeed_right + 1.0e-10;

    if (low_froude == 1.0)
        return sqrt(fmax(0.001, fmin(1.0, numerator / denominator)));
    if (low_froude == 2.0) {
        double fr = sqrt(numerator / denominator);

        return sqrt(fmin(1.0, 0.01 + fmax(fr - 0.01, 0.0)));
    }

    return 1.0;
}

KINLINE double sw_compute_s_max(double u_left, double u_right, double c_left,
                                double c_right)
{
    double s = fmax(u_left + c_left, u_right + c_right);

    return (s < 0.0) ? 0.0 : s;
}

KINLINE double sw_compute_s_min(double u_left, double u_right, double c_left,
                                double c_right)
{
    double s = fmin(u_left - c_left, u_right - c_right);

    return (s > 0.0) ? 0.0 : s;
}

/* Central-upwind (Kurganov-Noelle-Petrova) flux across one edge. */
KINLINE void sw_flux_function_central(double *q_left, double *q_right,
                                      double h_left, double h_right, double hle,
                                      double hre, double n1, double n2,
                                      double epsilon, double ze, double g,
                                      double *edgeflux, double *max_speed,
                                      double *pressure_flux, double low_froude)
{
    double uh_left, vh_left, u_left, v_left;
    double uh_right, vh_right, u_right, v_right;
    double soundspeed_left, soundspeed_right;
    double q_left_rotated[3], q_right_rotated[3];
    double flux_left[3], flux_right[3];
    double local_fr, s_max, s_min, denom, inverse_denominator, s_max_s_min;
    int i;

    for (i = 0; i < 3; i++) {
        q_left_rotated[i] = q_left[i];
        q_right_rotated[i] = q_right[i];
    }
    sw_rotate(q_left_rotated, n1, n2);
    sw_rotate(q_right_rotated, n1, n2);

    uh_left = q_left_rotated[1];
    vh_left = q_left_rotated[2];
    sw_compute_velocity_terms(h_left, hle, q_left_rotated[1], q_left_rotated[2],
                              &u_left, &uh_left, &v_left, &vh_left);
    uh_right = q_right_rotated[1];
    vh_right = q_right_rotated[2];
    sw_compute_velocity_terms(h_right, hre, q_right_rotated[1],
                              q_right_rotated[2], &u_right, &uh_right, &v_right,
                              &vh_right);

    soundspeed_left = sqrt(g * h_left);
    soundspeed_right = sqrt(g * h_right);

    local_fr =
        sw_compute_local_froude(low_froude, u_left, u_right, v_left, v_right,
                                soundspeed_left, soundspeed_right);
    s_max =
        sw_compute_s_max(u_left, u_right, soundspeed_left, soundspeed_right);
    s_min =
        sw_compute_s_min(u_left, u_right, soundspeed_left, soundspeed_right);

    flux_left[0] = u_left * h_left;
    flux_left[1] = u_left * uh_left;
    flux_left[2] = u_left * vh_left;
    flux_right[0] = u_right * h_right;
    flux_right[1] = u_right * uh_right;
    flux_right[2] = u_right * vh_right;

    denom = s_max - s_min;
    inverse_denominator = 1.0 / fmax(denom, 1.0e-100);
    s_max_s_min = s_max * s_min;

    if (denom < epsilon) {
        edgeflux[0] = 0.0;
        edgeflux[1] = 0.0;
        edgeflux[2] = 0.0;
        *max_speed = 0.0;
        *pressure_flux = 0.5 * g * 0.5 * (h_left * h_left + h_right * h_right);
    }
    else {
        double flux_0, flux_1, flux_2;

        *max_speed = fmax(s_max, -s_min);

        flux_0 = s_max * flux_left[0] - s_min * flux_right[0];
        flux_0 += s_max_s_min *
                  (fmax(q_right_rotated[0], ze) - fmax(q_left_rotated[0], ze));
        edgeflux[0] = flux_0 * inverse_denominator;

        flux_1 = s_max * flux_left[1] - s_min * flux_right[1];
        flux_1 += local_fr * s_max_s_min * (uh_right - uh_left);
        edgeflux[1] = flux_1 * inverse_denominator;

        flux_2 = s_max * flux_left[2] - s_min * flux_right[2];
        flux_2 += local_fr * s_max_s_min * (vh_right - vh_left);
        edgeflux[2] = flux_2 * inverse_denominator;

        *pressure_flux = 0.5 * g *
                         (s_max * h_left * h_left - s_min * h_right * h_right) *
                         inverse_denominator;

        sw_rotate(edgeflux, n1, -n2);
    }
}

/* ---- Extrapolation (core_extrapolate_second_order_edge) ---------------- */

/* Pass 1: centroid height; momentum to velocity where extrapolated. */
KINLINE void sw_extrapolate_pass1(anuga_idx k, struct sw_params P,
                                  GLOBAL const anuga_zq *zq, anuga_z0 z0,
                                  GLOBAL const double *stage_c,
                                  GLOBAL double *xmom_c, GLOBAL double *ymom_c,
                                  GLOBAL double *height_c, GLOBAL double *xwork,
                                  GLOBAL double *ywork)
{
    double stage = stage_c[k];
    double bed = anuga_bed(zq, z0, k);
    double xmom = xmom_c[k];
    double ymom = ymom_c[k];
    double dk = fmax(stage - bed, 0.0);
    int is_dry = (dk <= P.minimum_allowed_height);
    int extrapolate = (P.extrapolate_velocity_second_order == 1.0) &&
                      (dk > P.minimum_allowed_height);
    double xmom_out = is_dry ? 0.0 : xmom;
    double ymom_out = is_dry ? 0.0 : ymom;
    double inv_dk = extrapolate ? (1.0 / dk) : 1.0;

    height_c[k] = dk;
    xwork[k] = extrapolate ? xmom_out : 0.0;
    ywork[k] = extrapolate ? ymom_out : 0.0;
    xmom_c[k] = xmom_out * inv_dk;
    ymom_c[k] = ymom_out * inv_dk;
}

KINLINE void sw_store3(GLOBAL double *dst, anuga_idx k3, const double *v)
{
    dst[k3] = v[0];
    dst[k3 + 1] = v[1];
    dst[k3 + 2] = v[2];
}

/* Pass 2: limited second-order edge values. Reads pass-1 centroid values
 * of the triangle and its surrogate neighbours; writes edge values of the
 * triangle only, plus its own work arrays. */
KINLINE void sw_extrapolate_pass2(
    anuga_idx k, struct sw_params P, GLOBAL const double *stage_c,
    GLOBAL const double *xmom_c, GLOBAL const double *ymom_c,
    GLOBAL const double *height_c, GLOBAL double *xwork, GLOBAL double *ywork,
    GLOBAL double *stage_e, GLOBAL double *xmom_e, GLOBAL double *ymom_e,
    GLOBAL double *height_e, GLOBAL const double *centroid_coords,
    GLOBAL const double *edge_coords, GLOBAL const anuga_idx *surrogate,
    GLOBAL const anuga_idx *number_of_boundaries)
{
    const double a_tmp = 0.3, b_tmp = 0.1;
    const double c_tmp = 1.0 / (a_tmp - b_tmp);
    const double d_tmp = 1.0 - (c_tmp * a_tmp);
    const double mah = P.minimum_allowed_height;
    anuga_idx k2 = 2 * k, k3 = 3 * k, k6 = 6 * k;
    double xv0 = edge_coords[k6], yv0 = edge_coords[k6 + 1];
    double xv1 = edge_coords[k6 + 2], yv1 = edge_coords[k6 + 3];
    double xv2 = edge_coords[k6 + 4], yv2 = edge_coords[k6 + 5];
    double x = centroid_coords[k2], y = centroid_coords[k2 + 1];
    double dxv0 = xv0 - x, dxv1 = xv1 - x, dxv2 = xv2 - x;
    double dyv0 = yv0 - y, dyv1 = yv1 - y, dyv2 = yv2 - y;
    anuga_idx k0 = surrogate[k3], k1 = surrogate[k3 + 1];
    anuga_idx sn2 = surrogate[k3 + 2];
    double x0 = centroid_coords[2 * k0], y0 = centroid_coords[2 * k0 + 1];
    double x1 = centroid_coords[2 * k1], y1 = centroid_coords[2 * k1 + 1];
    double x2 = centroid_coords[2 * sn2], y2 = centroid_coords[2 * sn2 + 1];
    double dx1 = x1 - x0, dx2 = x2 - x0, dy1 = y1 - y0, dy2 = y2 - y0;
    double area2 = dy2 * dx1 - dy1 * dx2;
    int dry = ((height_c[k0] < mah) || (k0 == k)) &&
              ((height_c[k1] < mah) || (k1 == k)) &&
              ((height_c[sn2] < mah) || (sn2 == k));
    /* The triangle's own centroid velocity, zeroed when dry (ANUGA writes
     * this zero to xmom_cv[k]; see the file header). */
    double xk = dry ? 0.0 : xmom_c[k];
    double yk = dry ? 0.0 : ymom_c[k];
    anuga_idx nb = number_of_boundaries[k];
    double ev[3];
    int i;

    if (dry) {
        xwork[k] = 0.0;
        ywork[k] = 0.0;
    }

    if (nb == 3) {
        ev[0] = ev[1] = ev[2] = stage_c[k];
        sw_store3(stage_e, k3, ev);
        ev[0] = ev[1] = ev[2] = xk;
        sw_store3(xmom_e, k3, ev);
        ev[0] = ev[1] = ev[2] = yk;
        sw_store3(ymom_e, k3, ev);
        ev[0] = ev[1] = ev[2] = height_c[k];
        sw_store3(height_e, k3, ev);
    }
    else if (nb <= 1) {
        double hc = height_c[k], h0 = height_c[k0], h1 = height_c[k1];
        double h2 = height_c[sn2];
        double hmin = fmin(fmin(h0, fmin(h1, h2)), hc);
        double hmax = fmax(fmax(h0, fmax(h1, h2)), hc);
        double tmp1 = c_tmp * fmax(hmin, 0.0) / fmax(hc, 1.0e-06) + d_tmp;
        double tmp2 = c_tmp * fmax(hc, 0.0) / fmax(hmax, 1.0e-06) + d_tmp;
        double hfactor = fmax(0.0, fmin(tmp1, fmin(tmp2, 1.0)));
        double inv_area2, beta_stage, beta_xmom, beta_ymom;
        double xk0 = (k0 == k) ? xk : xmom_c[k0];
        double xk1 = (k1 == k) ? xk : xmom_c[k1];
        double xk2 = (sn2 == k) ? xk : xmom_c[sn2];
        double yk0 = (k0 == k) ? yk : ymom_c[k0];
        double yk1 = (k1 == k) ? yk : ymom_c[k1];
        double yk2 = (sn2 == k) ? yk : ymom_c[sn2];

        hfactor = fmin(1.2 * fmax(hmin - mah, 0.0) / (fmax(hmin, 0.0) + mah),
                       hfactor);
        inv_area2 = 1.0 / area2;

        beta_stage = P.beta_w_dry + (P.beta_w - P.beta_w_dry) * hfactor;
        if (beta_stage > 0.0)
            sw_calc_edge_values_with_gradient(
                stage_c[k], stage_c[k0], stage_c[k1], stage_c[sn2], dxv0, dxv1,
                dxv2, dyv0, dyv1, dyv2, dx1, dx2, dy1, dy2, inv_area2,
                beta_stage, ev);
        else
            ev[0] = ev[1] = ev[2] = stage_c[k];
        sw_store3(stage_e, k3, ev);

        if (beta_stage > 0.0)
            sw_calc_edge_values_with_gradient(
                height_c[k], height_c[k0], height_c[k1], height_c[sn2], dxv0,
                dxv1, dxv2, dyv0, dyv1, dyv2, dx1, dx2, dy1, dy2, inv_area2,
                beta_stage, ev);
        else
            ev[0] = ev[1] = ev[2] = height_c[k];
        sw_store3(height_e, k3, ev);

        beta_xmom = P.beta_uh_dry + (P.beta_uh - P.beta_uh_dry) * hfactor;
        if (beta_xmom > 0.0)
            sw_calc_edge_values_with_gradient(
                xk, xk0, xk1, xk2, dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dx1, dx2,
                dy1, dy2, inv_area2, beta_xmom, ev);
        else
            ev[0] = ev[1] = ev[2] = xk;
        sw_store3(xmom_e, k3, ev);

        beta_ymom = P.beta_vh_dry + (P.beta_vh - P.beta_vh_dry) * hfactor;
        if (beta_ymom > 0.0)
            sw_calc_edge_values_with_gradient(
                yk, yk0, yk1, yk2, dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dx1, dx2,
                dy1, dy2, inv_area2, beta_ymom, ev);
        else
            ev[0] = ev[1] = ev[2] = yk;
        sw_store3(ymom_e, k3, ev);
    }
    else {
        /* Two boundary edges: gradient towards the single neighbour. */
        anuga_idx kn = k;
        double xn, yn, dx, dy, dist2, grad_dx2, grad_dy2;
        double dqv[3], qmin, qmax, dq1, xkn, ykn;

        for (i = 0; i < 3; i++) {
            anuga_idx sn = surrogate[k3 + i];

            if (sn != k) {
                kn = sn;
                break;
            }
        }
        xn = centroid_coords[2 * kn];
        yn = centroid_coords[2 * kn + 1];
        dx = xn - x;
        dy = yn - y;
        dist2 = dx * dx + dy * dy;
        grad_dx2 = (dist2 > 0.0) ? dx / dist2 : 0.0;
        grad_dy2 = (dist2 > 0.0) ? dy / dist2 : 0.0;
        xkn = (kn == k) ? xk : xmom_c[kn];
        ykn = (kn == k) ? yk : ymom_c[kn];

        dq1 = stage_c[kn] - stage_c[k];
        sw_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2, dxv0, dxv1, dxv2,
                                     dyv0, dyv1, dyv2, dqv);
        sw_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
        sw_limit_gradient(dqv, qmin, qmax, P.beta_w);
        for (i = 0; i < 3; i++)
            stage_e[k3 + i] = stage_c[k] + dqv[i];

        dq1 = height_c[kn] - height_c[k];
        sw_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2, dxv0, dxv1, dxv2,
                                     dyv0, dyv1, dyv2, dqv);
        sw_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
        sw_limit_gradient(dqv, qmin, qmax, P.beta_w);
        for (i = 0; i < 3; i++)
            height_e[k3 + i] = height_c[k] + dqv[i];

        dq1 = xkn - xk;
        sw_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2, dxv0, dxv1, dxv2,
                                     dyv0, dyv1, dyv2, dqv);
        sw_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
        sw_limit_gradient(dqv, qmin, qmax, P.beta_w);
        for (i = 0; i < 3; i++)
            xmom_e[k3 + i] = xk + dqv[i];

        dq1 = ykn - yk;
        sw_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2, dxv0, dxv1, dxv2,
                                     dyv0, dyv1, dyv2, dqv);
        sw_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
        sw_limit_gradient(dqv, qmin, qmax, P.beta_w);
        for (i = 0; i < 3; i++)
            ymom_e[k3 + i] = yk + dqv[i];
    }

    /* Velocity edge values back to momentum. The bed edge value that
     * ANUGA stores next (stage_e - height_e) is computed where needed. */
    if (P.extrapolate_velocity_second_order == 1.0) {
        for (i = 0; i < 3; i++) {
            double dk = height_e[k3 + i];

            xmom_e[k3 + i] *= dk;
            ymom_e[k3 + i] *= dk;
        }
    }
}

/* Pass 3: restore centroid momentum from the work arrays. */
KINLINE void sw_extrapolate_pass3(anuga_idx k, GLOBAL double *xmom_c,
                                  GLOBAL double *ymom_c,
                                  GLOBAL const double *xwork,
                                  GLOBAL const double *ywork)
{
    xmom_c[k] = xwork[k];
    ymom_c[k] = ywork[k];
}

/* ---- Boundaries (gpu_boundaries.c) ------------------------------------- */

/* Boundary edge j of triangle vid, edge eid. use_centroid selects ANUGA's
 * centroid_transmissive_bc variant for transmissive edges. */
KINLINE void
sw_boundary(anuga_idx j, GLOBAL const anuga_idx *bnd_tri,
            GLOBAL const anuga_idx *bnd_edge, GLOBAL const anuga_u8 *bnd_type,
            GLOBAL const double *bnd_value, int use_centroid,
            GLOBAL const double *stage_c, GLOBAL const double *xmom_c,
            GLOBAL const double *ymom_c, GLOBAL const double *stage_e,
            GLOBAL const double *xmom_e, GLOBAL const double *ymom_e,
            GLOBAL const double *normals, GLOBAL double *stage_bv,
            GLOBAL double *xmom_bv, GLOBAL double *ymom_bv)
{
    anuga_idx vid = bnd_tri[j], eid = bnd_edge[j];
    anuga_idx ve = 3 * vid + eid;

    if (bnd_type[j] == BC_REFLECTIVE) {
        double n1 = normals[vid * 6 + 2 * eid];
        double n2 = normals[vid * 6 + 2 * eid + 1];
        double q1 = xmom_e[ve];
        double q2 = ymom_e[ve];
        double r1 = -q1 * n1 - q2 * n2;
        double r2 = -q1 * n2 + q2 * n1;

        stage_bv[j] = stage_e[ve];
        xmom_bv[j] = n1 * r1 - n2 * r2;
        ymom_bv[j] = n2 * r1 + n1 * r2;
    }
    else if (bnd_type[j] == BC_TRANSMISSIVE) {
        if (use_centroid) {
            stage_bv[j] = stage_c[vid];
            xmom_bv[j] = xmom_c[vid];
            ymom_bv[j] = ymom_c[vid];
        }
        else {
            stage_bv[j] = stage_e[ve];
            xmom_bv[j] = xmom_e[ve];
            ymom_bv[j] = ymom_e[ve];
        }
    }
    else { /* BC_DIRICHLET */
        stage_bv[j] = bnd_value[3 * j];
        xmom_bv[j] = bnd_value[3 * j + 1];
        ymom_bv[j] = bnd_value[3 * j + 2];
    }
}

/* ---- Fluxes (core_compute_fluxes_central) ------------------------------ */

/* Fluxes of triangle k. Writes its explicit updates and, on the first
 * substep, its max speed. Returns the cell's time step candidate in
 * *cell_dt (1e100 if none) and its boundary mass flux in *bflux. */
KINLINE void
sw_fluxes(anuga_idx k, struct sw_params P, int first_substep,
          GLOBAL const anuga_zq *zq, anuga_z0 z0, GLOBAL const double *height_c,
          GLOBAL const double *stage_e, GLOBAL const double *xmom_e,
          GLOBAL const double *ymom_e, GLOBAL const double *height_e,
          GLOBAL const double *stage_bv, GLOBAL const double *xmom_bv,
          GLOBAL const double *ymom_bv, GLOBAL double *stage_eu,
          GLOBAL double *xmom_eu, GLOBAL double *ymom_eu,
          GLOBAL const anuga_idx *neighbours,
          GLOBAL const anuga_idx *neighbour_edges, GLOBAL const double *normals,
          GLOBAL const double *edgelengths, GLOBAL const double *radii,
          GLOBAL const double *areas, GLOBAL double *max_speed, double *cell_dt,
          double *bflux)
{
    const double g = P.g;
    double edgeflux[3], ql[3], qr[3];
    double speed_max_last = 0.0;
    double seu = 0.0, xeu = 0.0, yeu = 0.0, bsum = 0.0;
    double hc = height_c[k];
    double zc = anuga_bed(zq, z0, k);
    double inv_area;
    int i;

    for (i = 0; i < 3; i++) {
        anuga_idx ki = 3 * k + i, ki2 = 2 * ki;
        double zl, hle, length, n1, n2, zr, hre, z_half, h_left, h_right;
        double max_speed_local = 0.0, pressure_flux = 0.0, pressuregrad_work;
        anuga_idx neighbour = neighbours[ki];
        int is_boundary = (neighbour < 0);

        ql[0] = stage_e[ki];
        ql[1] = xmom_e[ki];
        ql[2] = ymom_e[ki];
        hle = height_e[ki];
        zl = stage_e[ki] - height_e[ki];
        length = edgelengths[ki];
        n1 = normals[ki2];
        n2 = normals[ki2 + 1];

        if (is_boundary) {
            anuga_idx m = -neighbour - 1;

            qr[0] = stage_bv[m];
            qr[1] = xmom_bv[m];
            qr[2] = ymom_bv[m];
            zr = zl;
            hre = fmax(qr[0] - zr, 0.0);
        }
        else {
            anuga_idx nm = neighbour * 3 + neighbour_edges[ki];

            qr[0] = stage_e[nm];
            qr[1] = xmom_e[nm];
            qr[2] = ymom_e[nm];
            zr = stage_e[nm] - height_e[nm];
            hre = height_e[nm];
        }

        z_half = fmax(zl, zr);
        h_left = fmax(hle + zl - z_half, 0.0);
        h_right = fmax(hre + zr - z_half, 0.0);

        if (h_left == 0.0 && h_right == 0.0) {
            edgeflux[0] = 0.0;
            edgeflux[1] = 0.0;
            edgeflux[2] = 0.0;
        }
        else {
            sw_flux_function_central(
                ql, qr, h_left, h_right, hle, hre, n1, n2, P.epsilon, z_half, g,
                edgeflux, &max_speed_local, &pressure_flux, P.low_froude);
        }

        edgeflux[0] *= -length;
        edgeflux[1] *= -length;
        edgeflux[2] *= -length;

        speed_max_last = fmax(speed_max_last, max_speed_local);

        seu += edgeflux[0];
        xeu += edgeflux[1];
        yeu += edgeflux[2];

        /* All triangles are full (no MPI ghosts): only boundary edges
         * contribute to the boundary flux. */
        if (is_boundary)
            bsum += edgeflux[0];

        pressuregrad_work =
            length *
            (-g * 0.5 * (h_left * h_left - hle * hle - (hle + hc) * (zl - zc)) +
             pressure_flux);
        xeu -= normals[ki2] * pressuregrad_work;
        yeu -= normals[ki2 + 1] * pressuregrad_work;
    }

    *cell_dt = 1.0e+100;
    if (first_substep) {
        if (speed_max_last > P.epsilon)
            *cell_dt = radii[k] / speed_max_last;
        max_speed[k] = speed_max_last;
    }

    inv_area = 1.0 / areas[k];
    stage_eu[k] = seu * inv_area;
    xmom_eu[k] = xeu * inv_area;
    ymom_eu[k] = yeu * inv_area;
    *bflux = bsum;
}

/* ---- Friction ----------------------------------------------------------- */

/* core_manning_friction_flat_semi_implicit */
KINLINE void sw_manning_flat(anuga_idx k, struct sw_params P,
                             GLOBAL const anuga_zq *zq, anuga_z0 z0,
                             GLOBAL const double *stage_c,
                             GLOBAL const double *xmom_c,
                             GLOBAL const double *ymom_c,
                             GLOBAL const double *friction,
                             GLOBAL double *xmom_siu, GLOBAL double *ymom_siu)
{
    const double seven_thirds = 7.0 / 3.0;
    double S = 0.0;
    double uh = xmom_c[k];
    double vh = ymom_c[k];
    double eta = friction[k];
    double abs_mom = sqrt(uh * uh + vh * vh);

    if (eta > 1.0e-15) {
        double h = stage_c[k] - anuga_bed(zq, z0, k);

        if (h >= P.minimum_allowed_height) {
            S = -P.g * eta * eta * abs_mom;
            S /= pow(h, seven_thirds);
        }
    }
    xmom_siu[k] += S * uh;
    ymom_siu[k] += S * vh;
}

/* core_manning_friction_sloped_semi_implicit_edge_based, with the bed
 * edge values computed as stage_edge - height_edge. */
KINLINE void
sw_manning_sloped(anuga_idx k, struct sw_params P, GLOBAL const double *stage_c,
                  GLOBAL const double *stage_e, GLOBAL const double *height_e,
                  GLOBAL const double *xmom_c, GLOBAL const double *ymom_c,
                  GLOBAL const double *friction,
                  GLOBAL const double *edge_coords, GLOBAL double *xmom_siu,
                  GLOBAL double *ymom_siu)
{
    const double one_third = 1.0 / 3.0;
    const double seven_thirds = 7.0 / 3.0;
    double S = 0.0;
    double eta = friction[k];

    if (eta > 1.0e-16) {
        anuga_idx k3 = 3 * k, k6 = 6 * k;
        double z0 = stage_e[k3] - height_e[k3];
        double z1 = stage_e[k3 + 1] - height_e[k3 + 1];
        double z2 = stage_e[k3 + 2] - height_e[k3 + 2];
        double x0 = edge_coords[k6], y0 = edge_coords[k6 + 1];
        double x1 = edge_coords[k6 + 2], y1 = edge_coords[k6 + 3];
        double x2 = edge_coords[k6 + 4], y2 = edge_coords[k6 + 5];
        double det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
        double zx = ((y2 - y0) * (z1 - z0) - (y1 - y0) * (z2 - z0)) / det;
        double zy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det;
        double zs = sqrt(1.0 + zx * zx + zy * zy);
        double z = (z0 + z1 + z2) * one_third;
        double h = stage_c[k] - z;

        if (h >= P.minimum_allowed_height) {
            double uh = xmom_c[k];
            double vh = ymom_c[k];

            S = -P.g * eta * eta * zs * sqrt(uh * uh + vh * vh);
            S /= pow(h, seven_thirds);
        }
    }
    xmom_siu[k] += S * xmom_c[k];
    ymom_siu[k] += S * ymom_c[k];
}

/* ---- Updates ------------------------------------------------------------ */

/* core_protect followed by gpu_protect's height update. Returns the mass
 * added by clamping negative depths. */
KINLINE double sw_protect(anuga_idx k, struct sw_params P,
                          GLOBAL const anuga_zq *zq, anuga_z0 z0,
                          GLOBAL double *stage_c, GLOBAL double *xmom_c,
                          GLOBAL double *ymom_c, GLOBAL double *height_c,
                          GLOBAL const double *areas)
{
    double bed = anuga_bed(zq, z0, k);
    double h = stage_c[k] - bed;
    double mass_error = 0.0;

    if (h < P.minimum_allowed_height) {
        xmom_c[k] = 0.0;
        ymom_c[k] = 0.0;
    }
    if (h < 0.0) {
        mass_error = (-h) * areas[k];
        stage_c[k] = bed;
    }
    height_c[k] = fmax(stage_c[k] - bed, 0.0);

    return mass_error;
}

/* core_update_conserved_quantities */
KINLINE void sw_update(anuga_idx k, double timestep, GLOBAL double *stage_c,
                       GLOBAL double *xmom_c, GLOBAL double *ymom_c,
                       GLOBAL const double *stage_eu,
                       GLOBAL const double *xmom_eu,
                       GLOBAL const double *ymom_eu, GLOBAL double *stage_siu,
                       GLOBAL double *xmom_siu, GLOBAL double *ymom_siu)
{
    double stage_c0 = stage_c[k];
    double xmom_c0 = xmom_c[k];
    double ymom_c0 = ymom_c[k];
    double stage_new = stage_c0 + timestep * stage_eu[k];
    double xmom_new = xmom_c0 + timestep * xmom_eu[k];
    double ymom_new = ymom_c0 + timestep * ymom_eu[k];
    double num;

    num = stage_c0 - timestep * stage_siu[k];
    if (stage_c0 != 0.0 && num * stage_c0 > 0.0)
        stage_new = stage_new * stage_c0 / num;
    num = xmom_c0 - timestep * xmom_siu[k];
    if (xmom_c0 != 0.0 && num * xmom_c0 > 0.0)
        xmom_new = xmom_new * xmom_c0 / num;
    num = ymom_c0 - timestep * ymom_siu[k];
    if (ymom_c0 != 0.0 && num * ymom_c0 > 0.0)
        ymom_new = ymom_new * ymom_c0 / num;

    stage_c[k] = stage_new;
    xmom_c[k] = xmom_new;
    ymom_c[k] = ymom_new;
    stage_siu[k] = 0.0;
    xmom_siu[k] = 0.0;
    ymom_siu[k] = 0.0;
}

KINLINE void sw_backup(anuga_idx k, GLOBAL const double *stage_c,
                       GLOBAL const double *xmom_c, GLOBAL const double *ymom_c,
                       GLOBAL double *stage_bk, GLOBAL double *xmom_bk,
                       GLOBAL double *ymom_bk)
{
    stage_bk[k] = stage_c[k];
    xmom_bk[k] = xmom_c[k];
    ymom_bk[k] = ymom_c[k];
}

/* core_saxpy_conserved_quantities (Q = a Q + b Q_backup, then Q *= 1/c
 * when c is neither 0 nor 1), followed by the height update of
 * gpu_saxpy_*. c_inv is 1.0 / c, computed once by the caller as ANUGA
 * does; scale is non-zero when the 1/c pass applies. */
KINLINE void sw_saxpy(anuga_idx k, double a, double b, int scale, double c_inv,
                      GLOBAL const anuga_zq *zq, anuga_z0 z0,
                      GLOBAL double *stage_c, GLOBAL double *xmom_c,
                      GLOBAL double *ymom_c, GLOBAL double *height_c,
                      GLOBAL const double *stage_bk,
                      GLOBAL const double *xmom_bk,
                      GLOBAL const double *ymom_bk)
{
    double s = a * stage_c[k] + b * stage_bk[k];
    double x = a * xmom_c[k] + b * xmom_bk[k];
    double y = a * ymom_c[k] + b * ymom_bk[k];

    if (scale) {
        s *= c_inv;
        x *= c_inv;
        y *= c_inv;
    }
    stage_c[k] = s;
    xmom_c[k] = x;
    ymom_c[k] = y;
    height_c[k] = fmax(s - anuga_bed(zq, z0, k), 0.0);
}

/* Water volume of triangle k (gpu_compute_water_volume). */
KINLINE double sw_volume(anuga_idx k, GLOBAL const anuga_zq *zq, anuga_z0 z0,
                         GLOBAL const double *stage_c,
                         GLOBAL const double *areas)
{
    double h = stage_c[k] - anuga_bed(zq, z0, k);

    return h > 0.0 ? h * areas[k] : 0.0;
}

#endif /* R_HYDRO_ANUGA_CL_SW_H */
