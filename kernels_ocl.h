/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL tier of the solver (PLAN.md sections 3.1 and 5).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_KERNELS_OCL_H
#define R_HYDRO_ANUGA_KERNELS_OCL_H

#include "evolve.h"
#include "ocl_backend.h"

extern const struct solver_ops ocl_ops;

/* Build the program, create the device buffers and upload the whole host
 * state (call after all setup_*()). Fails loudly on any OpenCL error. */
void ocl_solver_init(struct sw_state *s, const struct ocl_backend *backend);

/* Release the device state. */
void ocl_solver_free(struct sw_state *s);

#endif /* R_HYDRO_ANUGA_KERNELS_OCL_H */
