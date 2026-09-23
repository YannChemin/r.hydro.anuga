/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL device selection for r.hydro.anuga: GPU, then CPU
 *               (e.g. PoCL), then plain OpenMP, requiring double precision.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_OCL_BACKEND_H
#define R_HYDRO_ANUGA_OCL_BACKEND_H

/* Target the OpenCL 1.1 API: the GPU test host only exposes Mesa Clover,
 * which is OpenCL 1.1 (PLAN.md section 2). */
#define CL_TARGET_OPENCL_VERSION 110
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <stddef.h>
#include <stdint.h>

#include <CL/cl.h>

/* Compute tier actually selected. DEV_OMP means no OpenCL context was
 * created and the caller runs the OpenMP code path. */
enum ocl_device_tier { DEV_GPU, DEV_CPU, DEV_OMP };

struct ocl_backend {
    enum ocl_device_tier tier;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    char platform_name[256];
    char device_name[256];
    char device_version[128];
    cl_ulong global_mem_size;
    cl_ulong max_alloc_size;
    cl_uint compute_units;
    size_t max_work_group_size;
    int ok;
};

/* Select and initialize a compute backend.
 *
 * device_opt is "auto", "gpu", "cpu" or "omp". Only devices that support
 * double precision (cl_khr_fp64) and successfully compile a small fp64
 * OpenCL C 1.1 probe kernel are eligible; others are skipped with a
 * verbose message. "gpu"/"cpu" fail (return 0) if no eligible device of
 * that type exists. "auto" tries GPU, then CPU, then falls back to OpenMP.
 * "omp" skips OpenCL entirely.
 *
 * Returns 1 on success, 0 on failure. */
int ocl_backend_init(struct ocl_backend *backend, const char *device_opt);

/* Release the OpenCL context/queue held by backend, if any. */
void ocl_backend_free(struct ocl_backend *backend);

/* Dequantise the scaled bed zq on the OpenCL device with the kernel
 * expression of PLAN.md section 4.7 and count values that are not
 * bitwise equal to expected (the host's result). Returns 0 for the
 * OpenMP tier. */
long ocl_check_dequantize(const struct ocl_backend *backend, const uint32_t *zq,
                          long n, int64_t z0, const double *expected);

/* Human-readable name of the selected tier. */
const char *ocl_backend_tier_name(const struct ocl_backend *backend);

#endif /* R_HYDRO_ANUGA_OCL_BACKEND_H */
