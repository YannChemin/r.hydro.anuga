/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL device selection for r.hydro.anuga: GPU, then CPU
 *               (e.g. PoCL), then plain OpenMP, requiring double precision.
 *               Tiering follows r.watershed.opencl; selection is by
 *               capability rather than platform order, because on the
 *               GPU test host Mesa rusticl exposes the same GPU without
 *               fp64 (PLAN.md section 2).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "ocl_backend.h"

#define MAX_PLATFORMS 16
#define MAX_DEVICES   16

/* Smallest kernel exercising what the solver needs from the compiler:
 * the fp64 extension pragma, double arithmetic and a 64-bit integer to
 * double conversion (elevation dequantisation, PLAN.md section 4.7). */
static const char *fp64_probe_source =
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "__kernel void probe(__global const uint *zq, __global double *z,\n"
    "                    const long z0, const int n)\n"
    "{\n"
    "    int k = get_global_id(0);\n"
    "    if (k < n)\n"
    "        z[k] = (double)((long)zq[k] + z0) * 1.0e-4;\n"
    "}\n";

static int device_has_extension(cl_device_id device, const char *name)
{
    size_t size = 0;
    char *extensions;
    int found;

    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &size) !=
            CL_SUCCESS ||
        size == 0)
        return 0;
    extensions = G_malloc(size + 1);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, size, extensions, NULL);
    extensions[size] = '\0';
    found = strstr(extensions, name) != NULL;
    G_free(extensions);

    return found;
}

/* Build the fp64 probe on a context for the device. Returns 1 if the
 * program builds, 0 otherwise (the build log is shown verbosely). */
static int device_compiles_fp64(cl_context context, cl_device_id device)
{
    cl_int err;
    cl_program program;
    int ok;

    program =
        clCreateProgramWithSource(context, 1, &fp64_probe_source, NULL, &err);
    if (err != CL_SUCCESS)
        return 0;

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.1", NULL, NULL);
    ok = (err == CL_SUCCESS);
    if (!ok) {
        char log[4096];

        log[0] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log) - 1, log, NULL);
        log[sizeof(log) - 1] = '\0';
        G_verbose_message(_("OpenCL: fp64 probe kernel failed to build "
                            "(error %d): %s"),
                          err, log);
    }
    clReleaseProgram(program);

    return ok;
}

static void fill_device_info(struct ocl_backend *backend)
{
    clGetPlatformInfo(backend->platform, CL_PLATFORM_NAME,
                      sizeof(backend->platform_name), backend->platform_name,
                      NULL);
    clGetDeviceInfo(backend->device, CL_DEVICE_NAME,
                    sizeof(backend->device_name), backend->device_name, NULL);
    clGetDeviceInfo(backend->device, CL_DEVICE_VERSION,
                    sizeof(backend->device_version), backend->device_version,
                    NULL);
    clGetDeviceInfo(backend->device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(cl_ulong), &backend->global_mem_size, NULL);
    clGetDeviceInfo(backend->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(cl_ulong), &backend->max_alloc_size, NULL);
    clGetDeviceInfo(backend->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(cl_uint), &backend->compute_units, NULL);
    clGetDeviceInfo(backend->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(size_t), &backend->max_work_group_size, NULL);
}

/* Try one device: it must support fp64, get a context and queue, and
 * build the probe kernel. On success the backend keeps the context and
 * queue. */
static int try_device(struct ocl_backend *backend, cl_platform_id platform,
                      cl_device_id device)
{
    cl_int err;
    char name[256];

    name[0] = '\0';
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);

    if (!device_has_extension(device, "cl_khr_fp64")) {
        G_verbose_message(_("OpenCL: skipping device '%s' (no cl_khr_fp64)"),
                          name);
        return 0;
    }

    backend->context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        G_verbose_message(_("OpenCL: skipping device '%s' (context "
                            "creation failed, error %d)"),
                          name, err);
        backend->context = NULL;
        return 0;
    }

    if (!device_compiles_fp64(backend->context, device)) {
        G_verbose_message(_("OpenCL: skipping device '%s' (fp64 probe "
                            "kernel does not build)"),
                          name);
        clReleaseContext(backend->context);
        backend->context = NULL;
        return 0;
    }

    backend->queue = clCreateCommandQueue(backend->context, device, 0, &err);
    if (err != CL_SUCCESS) {
        G_verbose_message(_("OpenCL: skipping device '%s' (command queue "
                            "creation failed, error %d)"),
                          name, err);
        clReleaseContext(backend->context);
        backend->context = NULL;
        backend->queue = NULL;
        return 0;
    }

    backend->platform = platform;
    backend->device = device;
    fill_device_info(backend);

    return 1;
}

/* Scan all platforms for the first eligible device of the given type. */
static int try_tier(struct ocl_backend *backend, enum ocl_device_tier tier,
                    cl_device_type type)
{
    cl_platform_id platforms[MAX_PLATFORMS];
    cl_device_id devices[MAX_DEVICES];
    cl_uint n_platforms = 0, n_devices, p, d;

    if (clGetPlatformIDs(MAX_PLATFORMS, platforms, &n_platforms) !=
            CL_SUCCESS ||
        n_platforms == 0)
        return 0;
    if (n_platforms > MAX_PLATFORMS)
        n_platforms = MAX_PLATFORMS;

    for (p = 0; p < n_platforms; p++) {
        if (clGetDeviceIDs(platforms[p], type, MAX_DEVICES, devices,
                           &n_devices) != CL_SUCCESS)
            continue;
        if (n_devices > MAX_DEVICES)
            n_devices = MAX_DEVICES;
        for (d = 0; d < n_devices; d++) {
            if (try_device(backend, platforms[p], devices[d])) {
                backend->tier = tier;
                backend->ok = 1;
                return 1;
            }
        }
    }

    return 0;
}

int ocl_backend_init(struct ocl_backend *backend, const char *device_opt)
{
    memset(backend, 0, sizeof(*backend));

    if (!device_opt || strcmp(device_opt, "omp") == 0) {
        backend->tier = DEV_OMP;
        backend->ok = 1;
    }
    else if (strcmp(device_opt, "gpu") == 0) {
        if (!try_tier(backend, DEV_GPU, CL_DEVICE_TYPE_GPU)) {
            G_warning(_("OpenCL: no GPU device with double precision "
                        "support found"));
            return 0;
        }
    }
    else if (strcmp(device_opt, "cpu") == 0) {
        if (!try_tier(backend, DEV_CPU, CL_DEVICE_TYPE_CPU)) {
            G_warning(_("OpenCL: no CPU device with double precision "
                        "support found"));
            return 0;
        }
    }
    else if (strcmp(device_opt, "auto") == 0) {
        if (!try_tier(backend, DEV_GPU, CL_DEVICE_TYPE_GPU) &&
            !try_tier(backend, DEV_CPU, CL_DEVICE_TYPE_CPU)) {
            G_message(_("OpenCL: no device with double precision support "
                        "found, falling back to OpenMP"));
            backend->tier = DEV_OMP;
            backend->ok = 1;
        }
    }
    else {
        G_fatal_error(_("Invalid device option '%s'"), device_opt);
    }

    if (backend->tier != DEV_OMP)
        G_verbose_message(_("OpenCL: using %s device '%s' on platform '%s' "
                            "(%s)"),
                          ocl_backend_tier_name(backend), backend->device_name,
                          backend->platform_name, backend->device_version);
    else
        G_verbose_message(_("Using the OpenMP backend"));

    return backend->ok;
}

void ocl_backend_free(struct ocl_backend *backend)
{
    if (!backend)
        return;

    if (backend->queue) {
        clReleaseCommandQueue(backend->queue);
        backend->queue = NULL;
    }
    if (backend->context) {
        clReleaseContext(backend->context);
        backend->context = NULL;
    }
}

const char *ocl_backend_tier_name(const struct ocl_backend *backend)
{
    switch (backend->tier) {
    case DEV_GPU:
        return "GPU";
    case DEV_CPU:
        return "CPU";
    default:
        return "OpenMP";
    }
}

long ocl_check_dequantize(const struct ocl_backend *backend, const uint32_t *zq,
                          long n, int64_t z0, const double *expected)
{
    cl_int err;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_zq, d_z;
    double *z;
    cl_long z0_arg = z0;
    cl_int n_arg = (cl_int)n;
    size_t global = (size_t)n;
    long k, mismatches = 0;

    if (backend->tier == DEV_OMP || n == 0)
        return 0;
    if (n > 2147483647L)
        G_fatal_error(_("Too many triangles for the dequantisation check"));

    program = clCreateProgramWithSource(backend->context, 1, &fp64_probe_source,
                                        NULL, &err);
    if (err == CL_SUCCESS)
        err = clBuildProgram(program, 1, &backend->device, "-cl-std=CL1.1",
                             NULL, NULL);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: dequantisation kernel failed to build "
                        "(error %d)"),
                      err);
    kernel = clCreateKernel(program, "probe", &err);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: clCreateKernel failed (error %d)"), err);

    d_zq = clCreateBuffer(backend->context,
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          n * sizeof(uint32_t), (void *)zq, &err);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: buffer allocation failed (error %d)"), err);
    d_z = clCreateBuffer(backend->context, CL_MEM_WRITE_ONLY,
                         n * sizeof(double), NULL, &err);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: buffer allocation failed (error %d)"), err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_zq);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_z);
    clSetKernelArg(kernel, 2, sizeof(cl_long), &z0_arg);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &n_arg);
    err = clEnqueueNDRangeKernel(backend->queue, kernel, 1, NULL, &global, NULL,
                                 0, NULL, NULL);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: kernel launch failed (error %d)"), err);

    z = G_malloc(n * sizeof(double));
    err = clEnqueueReadBuffer(backend->queue, d_z, CL_TRUE, 0,
                              n * sizeof(double), z, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: reading results failed (error %d)"), err);

    for (k = 0; k < n; k++)
        if (memcmp(&z[k], &expected[k], sizeof(double)) != 0)
            mismatches++;

    G_free(z);
    clReleaseMemObject(d_zq);
    clReleaseMemObject(d_z);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return mismatches;
}
