/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL tier of the solver: device buffers, program build and
 *               the solver_ops table over the kernels of cl/anuga_kernels.cl
 *               (PLAN.md sections 3.1 and 5). Reductions produce one
 *               partial per work-group, combined on the host in group
 *               order so results are deterministic.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "kernels_ocl.h"
#include "ocl_kernels_src.h"

#define MAX_WG 256

enum {
    K_PROTECT,
    K_EXTRAP1,
    K_EXTRAP2,
    K_EXTRAP3,
    K_BOUNDARIES,
    K_FLUXES,
    K_FRICTION_FLAT,
    K_FRICTION_SLOPED,
    K_UPDATE,
    K_BACKUP,
    K_SAXPY,
    K_VOLUME,
    N_KERNELS
};

static const char *kernel_names[N_KERNELS] = {
    "k_protect",       "k_extrapolate_pass1", "k_extrapolate_pass2",
    "k_extrapolate_pass3", "k_boundaries",   "k_fluxes",
    "k_friction_flat", "k_friction_sloped",   "k_update",
    "k_backup",        "k_saxpy",             "k_volume"};

struct ocl_dev {
    const struct ocl_backend *be;
    cl_program program;
    cl_kernel k[N_KERNELS];
    size_t wg, global_n, global_nb, n_groups;
    cl_int n, nb;

    /* Mesh and bed. */
    cl_mem zq, centroid, edge_coords, normals, edgelengths, radii, areas;
    cl_mem neighbours, neighbour_edges, surrogate, nbounds;
    cl_mem bnd_tri, bnd_edge, bnd_type, bnd_value;
    /* State. */
    cl_mem stage_c, xmom_c, ymom_c, height_c, friction;
    cl_mem stage_e, xmom_e, ymom_e, height_e;
    cl_mem stage_bv, xmom_bv, ymom_bv;
    cl_mem stage_eu, xmom_eu, ymom_eu, stage_siu, xmom_siu, ymom_siu;
    cl_mem stage_bk, xmom_bk, ymom_bk, xwork, ywork, max_speed;
    /* Reduction partials. */
    cl_mem partial_a, partial_b;
    double *host_partial;
};

static void check(cl_int err, const char *what)
{
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: %s failed (error %d)"), what, err);
}

static cl_mem buffer(const struct ocl_dev *d, const void *host, size_t bytes,
                     int read_only)
{
    cl_int err;
    cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
    cl_mem m;

    if (bytes == 0)
        bytes = 8; /* Buffers of size 0 are invalid. */
    m = clCreateBuffer(d->be->context,
                       flags | (host ? CL_MEM_COPY_HOST_PTR : 0), bytes,
                       (void *)host, &err);
    check(err, "clCreateBuffer");

    return m;
}

/* Build the program for work-group size wg; returns the smallest maximal
 * work-group size over all kernels. */
static size_t build(struct ocl_dev *d, size_t wg)
{
    const size_t n_lines = sizeof(ocl_kernel_lines) / sizeof(ocl_kernel_lines[0]);
    char options[128];
    size_t min_wg = (size_t)-1;
    cl_int err;
    int i;

    d->program = clCreateProgramWithSource(d->be->context, (cl_uint)n_lines,
                                           ocl_kernel_lines, NULL, &err);
    check(err, "clCreateProgramWithSource");
    snprintf(options, sizeof(options), "-cl-std=CL1.1 -DWG=%lu",
             (unsigned long)wg);
    G_verbose_message(_("OpenCL: building solver kernels with '%s'..."),
                      options);
    err = clBuildProgram(d->program, 1, &d->be->device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t size = 0;
        char *log;

        clGetProgramBuildInfo(d->program, d->be->device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &size);
        log = G_malloc(size + 1);
        clGetProgramBuildInfo(d->program, d->be->device, CL_PROGRAM_BUILD_LOG,
                              size, log, NULL);
        log[size] = '\0';
        G_fatal_error(_("OpenCL: building the solver kernels failed (error "
                        "%d):\n%s"),
                      err, log);
    }
    for (i = 0; i < N_KERNELS; i++) {
        size_t kwg = 0;

        d->k[i] = clCreateKernel(d->program, kernel_names[i], &err);
        check(err, kernel_names[i]);
        clGetKernelWorkGroupInfo(d->k[i], d->be->device,
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(kwg), &kwg,
                                 NULL);
        if (kwg < min_wg)
            min_wg = kwg;
    }

    return min_wg;
}

static void release_program(struct ocl_dev *d)
{
    int i;

    for (i = 0; i < N_KERNELS; i++)
        if (d->k[i])
            clReleaseKernel(d->k[i]);
    if (d->program)
        clReleaseProgram(d->program);
    memset(d->k, 0, sizeof(d->k));
    d->program = NULL;
}

void ocl_solver_init(struct sw_state *s, const struct ocl_backend *backend)
{
    struct ocl_dev *d = G_calloc(1, sizeof(struct ocl_dev));
    size_t n = s->n, nb = s->nb, wg, kmax;
    size_t dn = n * sizeof(double), dn3 = 3 * dn;

    d->be = backend;
    d->n = (cl_int)n;
    d->nb = (cl_int)nb;
    if (n > 2147483647UL)
        G_fatal_error(_("Too many triangles for the OpenCL tier"));

    /* Largest power of two not above the device and kernel limits. */
    wg = MAX_WG;
    while (wg > backend->max_work_group_size)
        wg >>= 1;
    kmax = build(d, wg);
    if (kmax < wg) {
        while (wg > kmax)
            wg >>= 1;
        release_program(d);
        if (build(d, wg) < wg)
            G_fatal_error(_("OpenCL: kernels cannot run with a work-group "
                            "size of %lu"),
                          (unsigned long)wg);
    }
    d->wg = wg;
    d->global_n = (n + wg - 1) / wg * wg;
    d->global_nb = (nb + wg - 1) / wg * wg;
    d->n_groups = d->global_n / wg;
    G_verbose_message(_("OpenCL: work-group size %lu, %lu groups"),
                      (unsigned long)wg, (unsigned long)d->n_groups);

    d->zq = buffer(d, s->zq, n * sizeof(anuga_zq), 1);
    d->centroid = buffer(d, s->centroid_coords, 2 * dn, 1);
    d->edge_coords = buffer(d, s->edge_coords, 6 * dn, 1);
    d->normals = buffer(d, s->normals, 6 * dn, 1);
    d->edgelengths = buffer(d, s->edgelengths, dn3, 1);
    d->radii = buffer(d, s->radii, dn, 1);
    d->areas = buffer(d, s->areas, dn, 1);
    d->neighbours = buffer(d, s->neighbours, 3 * n * sizeof(anuga_idx), 1);
    d->neighbour_edges =
        buffer(d, s->neighbour_edges, 3 * n * sizeof(anuga_idx), 1);
    d->surrogate = buffer(d, s->surrogate, 3 * n * sizeof(anuga_idx), 1);
    d->nbounds = buffer(d, s->number_of_boundaries, n * sizeof(anuga_idx), 1);
    d->bnd_tri = buffer(d, nb ? s->bnd_tri : NULL, nb * sizeof(anuga_idx), 1);
    d->bnd_edge = buffer(d, nb ? s->bnd_edge : NULL, nb * sizeof(anuga_idx), 1);
    d->bnd_type = buffer(d, nb ? s->bnd_type : NULL, nb * sizeof(anuga_u8), 1);
    d->bnd_value = buffer(d, nb ? s->bnd_value : NULL, 3 * nb * sizeof(double),
                          1);

    d->stage_c = buffer(d, s->stage_c, dn, 0);
    d->xmom_c = buffer(d, s->xmom_c, dn, 0);
    d->ymom_c = buffer(d, s->ymom_c, dn, 0);
    d->height_c = buffer(d, s->height_c, dn, 0);
    d->friction = buffer(d, s->friction, dn, 1);
    d->stage_e = buffer(d, s->stage_e, dn3, 0);
    d->xmom_e = buffer(d, s->xmom_e, dn3, 0);
    d->ymom_e = buffer(d, s->ymom_e, dn3, 0);
    d->height_e = buffer(d, s->height_e, dn3, 0);
    d->stage_bv = buffer(d, nb ? s->stage_bv : NULL, nb * sizeof(double), 0);
    d->xmom_bv = buffer(d, nb ? s->xmom_bv : NULL, nb * sizeof(double), 0);
    d->ymom_bv = buffer(d, nb ? s->ymom_bv : NULL, nb * sizeof(double), 0);
    d->stage_eu = buffer(d, s->stage_eu, dn, 0);
    d->xmom_eu = buffer(d, s->xmom_eu, dn, 0);
    d->ymom_eu = buffer(d, s->ymom_eu, dn, 0);
    d->stage_siu = buffer(d, s->stage_siu, dn, 0);
    d->xmom_siu = buffer(d, s->xmom_siu, dn, 0);
    d->ymom_siu = buffer(d, s->ymom_siu, dn, 0);
    d->stage_bk = buffer(d, s->stage_bk, dn, 0);
    d->xmom_bk = buffer(d, s->xmom_bk, dn, 0);
    d->ymom_bk = buffer(d, s->ymom_bk, dn, 0);
    d->xwork = buffer(d, s->xwork, dn, 0);
    d->ywork = buffer(d, s->ywork, dn, 0);
    d->max_speed = buffer(d, s->max_speed, dn, 0);
    d->partial_a = buffer(d, NULL, d->n_groups * sizeof(double), 0);
    d->partial_b = buffer(d, NULL, d->n_groups * sizeof(double), 0);
    d->host_partial = G_malloc(d->n_groups * sizeof(double));

    s->device = d;
}

void ocl_solver_free(struct sw_state *s)
{
    struct ocl_dev *d = s->device;
    cl_mem *mems[] = {
        &d->zq,        &d->centroid,  &d->edge_coords, &d->normals,
        &d->edgelengths, &d->radii,   &d->areas,       &d->neighbours,
        &d->neighbour_edges, &d->surrogate, &d->nbounds, &d->bnd_tri,
        &d->bnd_edge,  &d->bnd_type,  &d->bnd_value,   &d->stage_c,
        &d->xmom_c,    &d->ymom_c,    &d->height_c,    &d->friction,
        &d->stage_e,   &d->xmom_e,    &d->ymom_e,      &d->height_e,
        &d->stage_bv,  &d->xmom_bv,   &d->ymom_bv,     &d->stage_eu,
        &d->xmom_eu,   &d->ymom_eu,   &d->stage_siu,   &d->xmom_siu,
        &d->ymom_siu,  &d->stage_bk,  &d->xmom_bk,     &d->ymom_bk,
        &d->xwork,     &d->ywork,     &d->max_speed,   &d->partial_a,
        &d->partial_b};
    size_t i;

    if (!d)
        return;
    for (i = 0; i < sizeof(mems) / sizeof(mems[0]); i++)
        if (*mems[i])
            clReleaseMemObject(*mems[i]);
    release_program(d);
    G_free(d->host_partial);
    G_free(d);
    s->device = NULL;
}

/* Kernel argument helpers. */
#define ARG(kernel, i, value)                                                  \
    check(clSetKernelArg((kernel), (i), sizeof(value), &(value)),              \
          "clSetKernelArg")

static void run(const struct ocl_dev *d, cl_kernel kernel, size_t global)
{
    size_t local = d->wg;

    if (global == 0)
        return;
    check(clEnqueueNDRangeKernel(d->be->queue, kernel, 1, NULL, &global,
                                 &local, 0, NULL, NULL),
          "clEnqueueNDRangeKernel");
}

static void read_partials(const struct ocl_dev *d, cl_mem partial)
{
    check(clEnqueueReadBuffer(d->be->queue, partial, CL_TRUE, 0,
                              d->n_groups * sizeof(double), d->host_partial, 0,
                              NULL, NULL),
          "clEnqueueReadBuffer");
}

static double sum_partials(const struct ocl_dev *d)
{
    double sum = 0.0;
    size_t i;

    for (i = 0; i < d->n_groups; i++)
        sum += d->host_partial[i];

    return sum;
}

static double ocl_protect(struct sw_state *s)
{
    struct ocl_dev *d = s->device;
    cl_kernel k = d->k[K_PROTECT];
    cl_long z0 = s->z0;

    ARG(k, 0, d->n);
    ARG(k, 1, s->cfg.P);
    ARG(k, 2, d->zq);
    ARG(k, 3, z0);
    ARG(k, 4, d->stage_c);
    ARG(k, 5, d->xmom_c);
    ARG(k, 6, d->ymom_c);
    ARG(k, 7, d->height_c);
    ARG(k, 8, d->areas);
    ARG(k, 9, d->partial_b);
    run(d, k, d->global_n);
    read_partials(d, d->partial_b);

    return sum_partials(d);
}

static void ocl_extrapolate(struct sw_state *s)
{
    struct ocl_dev *d = s->device;
    cl_kernel k1 = d->k[K_EXTRAP1], k2 = d->k[K_EXTRAP2], k3 = d->k[K_EXTRAP3];
    cl_long z0 = s->z0;

    ARG(k1, 0, d->n);
    ARG(k1, 1, s->cfg.P);
    ARG(k1, 2, d->zq);
    ARG(k1, 3, z0);
    ARG(k1, 4, d->stage_c);
    ARG(k1, 5, d->xmom_c);
    ARG(k1, 6, d->ymom_c);
    ARG(k1, 7, d->height_c);
    ARG(k1, 8, d->xwork);
    ARG(k1, 9, d->ywork);
    run(d, k1, d->global_n);

    ARG(k2, 0, d->n);
    ARG(k2, 1, s->cfg.P);
    ARG(k2, 2, d->stage_c);
    ARG(k2, 3, d->xmom_c);
    ARG(k2, 4, d->ymom_c);
    ARG(k2, 5, d->height_c);
    ARG(k2, 6, d->xwork);
    ARG(k2, 7, d->ywork);
    ARG(k2, 8, d->stage_e);
    ARG(k2, 9, d->xmom_e);
    ARG(k2, 10, d->ymom_e);
    ARG(k2, 11, d->height_e);
    ARG(k2, 12, d->centroid);
    ARG(k2, 13, d->edge_coords);
    ARG(k2, 14, d->surrogate);
    ARG(k2, 15, d->nbounds);
    run(d, k2, d->global_n);

    if (s->cfg.P.extrapolate_velocity_second_order == 1.0) {
        ARG(k3, 0, d->n);
        ARG(k3, 1, d->xmom_c);
        ARG(k3, 2, d->ymom_c);
        ARG(k3, 3, d->xwork);
        ARG(k3, 4, d->ywork);
        run(d, k3, d->global_n);
    }
}

static void ocl_boundaries(struct sw_state *s)
{
    struct ocl_dev *d = s->device;
    cl_kernel k = d->k[K_BOUNDARIES];
    cl_int use_centroid = s->cfg.transmissive_use_centroid;

    if (d->nb == 0)
        return;
    ARG(k, 0, d->nb);
    ARG(k, 1, d->bnd_tri);
    ARG(k, 2, d->bnd_edge);
    ARG(k, 3, d->bnd_type);
    ARG(k, 4, d->bnd_value);
    ARG(k, 5, use_centroid);
    ARG(k, 6, d->stage_c);
    ARG(k, 7, d->xmom_c);
    ARG(k, 8, d->ymom_c);
    ARG(k, 9, d->stage_e);
    ARG(k, 10, d->xmom_e);
    ARG(k, 11, d->ymom_e);
    ARG(k, 12, d->normals);
    ARG(k, 13, d->stage_bv);
    ARG(k, 14, d->xmom_bv);
    ARG(k, 15, d->ymom_bv);
    run(d, k, d->global_nb);
}

static double ocl_fluxes(struct sw_state *s, int first_substep, double *bflux)
{
    struct ocl_dev *d = s->device;
    cl_kernel k = d->k[K_FLUXES];
    cl_int first = first_substep;
    cl_long z0 = s->z0;
    double dt_min = 1.0e+100;
    size_t i;

    ARG(k, 0, d->n);
    ARG(k, 1, s->cfg.P);
    ARG(k, 2, first);
    ARG(k, 3, d->zq);
    ARG(k, 4, z0);
    ARG(k, 5, d->height_c);
    ARG(k, 6, d->stage_e);
    ARG(k, 7, d->xmom_e);
    ARG(k, 8, d->ymom_e);
    ARG(k, 9, d->height_e);
    ARG(k, 10, d->stage_bv);
    ARG(k, 11, d->xmom_bv);
    ARG(k, 12, d->ymom_bv);
    ARG(k, 13, d->stage_eu);
    ARG(k, 14, d->xmom_eu);
    ARG(k, 15, d->ymom_eu);
    ARG(k, 16, d->neighbours);
    ARG(k, 17, d->neighbour_edges);
    ARG(k, 18, d->normals);
    ARG(k, 19, d->edgelengths);
    ARG(k, 20, d->radii);
    ARG(k, 21, d->areas);
    ARG(k, 22, d->max_speed);
    ARG(k, 23, d->partial_a);
    ARG(k, 24, d->partial_b);
    run(d, k, d->global_n);

    read_partials(d, d->partial_b);
    *bflux = sum_partials(d);
    read_partials(d, d->partial_a);
    for (i = 0; i < d->n_groups; i++)
        dt_min = fmin(dt_min, d->host_partial[i]);

    return dt_min;
}

static void ocl_friction(struct sw_state *s)
{
    struct ocl_dev *d = s->device;
    cl_long z0 = s->z0;

    if (s->cfg.sloped_friction) {
        cl_kernel k = d->k[K_FRICTION_SLOPED];

        ARG(k, 0, d->n);
        ARG(k, 1, s->cfg.P);
        ARG(k, 2, d->stage_c);
        ARG(k, 3, d->stage_e);
        ARG(k, 4, d->height_e);
        ARG(k, 5, d->xmom_c);
        ARG(k, 6, d->ymom_c);
        ARG(k, 7, d->friction);
        ARG(k, 8, d->edge_coords);
        ARG(k, 9, d->xmom_siu);
        ARG(k, 10, d->ymom_siu);
        run(d, k, d->global_n);
    }
    else {
        cl_kernel k = d->k[K_FRICTION_FLAT];

        ARG(k, 0, d->n);
        ARG(k, 1, s->cfg.P);
        ARG(k, 2, d->zq);
        ARG(k, 3, z0);
        ARG(k, 4, d->stage_c);
        ARG(k, 5, d->xmom_c);
        ARG(k, 6, d->ymom_c);
        ARG(k, 7, d->friction);
        ARG(k, 8, d->xmom_siu);
        ARG(k, 9, d->ymom_siu);
        run(d, k, d->global_n);
    }
}

static void ocl_update(struct sw_state *s, double dt)
{
    struct ocl_dev *d = s->device;
    cl_kernel k = d->k[K_UPDATE];
    cl_double cdt = dt;

    ARG(k, 0, d->n);
    ARG(k, 1, cdt);
    ARG(k, 2, d->stage_c);
    ARG(k, 3, d->xmom_c);
    ARG(k, 4, d->ymom_c);
    ARG(k, 5, d->stage_eu);
    ARG(k, 6, d->xmom_eu);
    ARG(k, 7, d->ymom_eu);
    ARG(k, 8, d->stage_siu);
    ARG(k, 9, d->xmom_siu);
    ARG(k, 10, d->ymom_siu);
    run(d, k, d->global_n);
}

static void ocl_backup(struct sw_state *s)
{
    struct ocl_dev *d = s->device;
    cl_kernel k = d->k[K_BACKUP];

    ARG(k, 0, d->n);
    ARG(k, 1, d->stage_c);
    ARG(k, 2, d->xmom_c);
    ARG(k, 3, d->ymom_c);
    ARG(k, 4, d->stage_bk);
    ARG(k, 5, d->xmom_bk);
    ARG(k, 6, d->ymom_bk);
    run(d, k, d->global_n);
}

static void ocl_saxpy(struct sw_state *s, double a, double b, double c)
{
    struct ocl_dev *d = s->device;
    cl_kernel k = d->k[K_SAXPY];
    cl_int scale = (c != 1.0 && c != 0.0);
    cl_double ca = a, cb = b, c_inv = scale ? 1.0 / c : 1.0;
    cl_long z0 = s->z0;

    ARG(k, 0, d->n);
    ARG(k, 1, ca);
    ARG(k, 2, cb);
    ARG(k, 3, scale);
    ARG(k, 4, c_inv);
    ARG(k, 5, d->zq);
    ARG(k, 6, z0);
    ARG(k, 7, d->stage_c);
    ARG(k, 8, d->xmom_c);
    ARG(k, 9, d->ymom_c);
    ARG(k, 10, d->height_c);
    ARG(k, 11, d->stage_bk);
    ARG(k, 12, d->xmom_bk);
    ARG(k, 13, d->ymom_bk);
    run(d, k, d->global_n);
}

static double ocl_volume(struct sw_state *s)
{
    struct ocl_dev *d = s->device;
    cl_kernel k = d->k[K_VOLUME];
    cl_long z0 = s->z0;

    ARG(k, 0, d->n);
    ARG(k, 1, d->zq);
    ARG(k, 2, z0);
    ARG(k, 3, d->stage_c);
    ARG(k, 4, d->areas);
    ARG(k, 5, d->partial_b);
    run(d, k, d->global_n);
    read_partials(d, d->partial_b);

    return sum_partials(d);
}

static void read_array(const struct ocl_dev *d, cl_mem m, double *host,
                       size_t count)
{
    check(clEnqueueReadBuffer(d->be->queue, m, CL_TRUE, 0,
                              count * sizeof(double), host, 0, NULL, NULL),
          "clEnqueueReadBuffer");
}

static void ocl_sync_to_host(struct sw_state *s)
{
    struct ocl_dev *d = s->device;

    read_array(d, d->stage_c, s->stage_c, s->n);
    read_array(d, d->xmom_c, s->xmom_c, s->n);
    read_array(d, d->ymom_c, s->ymom_c, s->n);
    read_array(d, d->height_c, s->height_c, s->n);
    read_array(d, d->max_speed, s->max_speed, s->n);
}

const struct solver_ops ocl_ops = {
    "OpenCL",   ocl_protect,  ocl_extrapolate, ocl_boundaries,
    ocl_fluxes, ocl_friction, ocl_update,      ocl_backup,
    ocl_saxpy,  ocl_volume,   ocl_sync_to_host};
