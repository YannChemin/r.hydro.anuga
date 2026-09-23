/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Definitions shared by the C (OpenMP) and OpenCL C 1.1
 *               compilations of the solver kernels (PLAN.md section 3.1).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_CL_COMMON_H
#define R_HYDRO_ANUGA_CL_COMMON_H

#ifdef __OPENCL_VERSION__

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
/* No fused multiply-add: results must not depend on contraction choices
 * (PLAN.md section 4.7). */
#pragma OPENCL FP_CONTRACT OFF

#define GLOBAL   __global
#define CONSTANT __constant
#define KINLINE  inline
typedef int anuga_idx;
typedef uint anuga_zq;
typedef long anuga_z0;
typedef uchar anuga_u8;

#else /* C */

#include <math.h>
#include <stdint.h>

#define GLOBAL
#define CONSTANT
#define KINLINE static inline
typedef int32_t anuga_idx;
typedef uint32_t anuga_zq;
typedef int64_t anuga_z0;
typedef uint8_t anuga_u8;

#endif

/* Scaled bed (PLAN.md section 4.7): one integer addition, then one
 * correctly rounded multiplication; identical on host and device. */
#define ANUGA_BED_INV_SCALE 1.0e-4

KINLINE double anuga_bed(GLOBAL const anuga_zq *zq, anuga_z0 z0, anuga_idx k)
{
    return (double)((anuga_z0)zq[k] + z0) * ANUGA_BED_INV_SCALE;
}

/* Scalar solver parameters. All members are doubles so the layout is the
 * same for the host compiler and the OpenCL compiler when the struct is
 * passed by value to a kernel. */
struct sw_params {
    double g;
    double epsilon;
    double minimum_allowed_height;
    double beta_w, beta_w_dry;
    double beta_uh, beta_uh_dry;
    double beta_vh, beta_vh_dry;
    double low_froude;                        /* 0, 1 or 2 */
    double extrapolate_velocity_second_order; /* 0 or 1 */
};

/* Boundary condition types, one per boundary edge. */
#define BC_REFLECTIVE   0
#define BC_TRANSMISSIVE 1
#define BC_DIRICHLET    2

#endif /* R_HYDRO_ANUGA_CL_COMMON_H */
