/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Stack of DEMs at their native resolutions: elevation from
 *               the finest DEM available, vertical bias check between
 *               overlapping DEMs, and smooth blending across the seams
 *               (PLAN.md section 4.2).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "dem_stack.h"

#define CHAMFER_INF 1.0e30f

/* Cell index range [c0, c1) x [r0, r1) of the cells whose centres lie in
 * the box, clamped to the grid. */
static void box_cells(const struct Cell_head *w, double x0, double y0,
                      double x1, double y1, long *c0, long *c1, long *r0,
                      long *r1)
{
    *c0 = (long)ceil((x0 - w->west) / w->ew_res - 0.5);
    *c1 = (long)ceil((x1 - w->west) / w->ew_res - 0.5);
    *r0 = (long)floor((w->north - y1) / w->ns_res - 0.5) + 1;
    *r1 = (long)floor((w->north - y0) / w->ns_res - 0.5) + 1;
    if (*c0 < 0)
        *c0 = 0;
    if (*r0 < 0)
        *r0 = 0;
    if (*c1 > w->cols)
        *c1 = w->cols;
    if (*r1 > w->rows)
        *r1 = w->rows;
}

long long dem_layer_count(const struct dem_layer *l, double x0, double y0,
                          double x1, double y1)
{
    const struct Cell_head *w = &l->grid.win;
    long c0, c1, r0, r1, stride = w->cols + 1;

    box_cells(w, x0, y0, x1, y1, &c0, &c1, &r0, &r1);
    if (c1 <= c0 || r1 <= r0)
        return 0;

    return l->count_sat[r1 * stride + c1] - l->count_sat[r0 * stride + c1] -
           l->count_sat[r1 * stride + c0] + l->count_sat[r0 * stride + c0];
}

static void build_count_sat(struct dem_layer *l)
{
    const struct Cell_head *w = &l->grid.win;
    long stride = w->cols + 1, r, c;

    l->count_sat = G_calloc((size_t)(w->rows + 1) * stride, sizeof(long long));
    for (r = 0; r < w->rows; r++) {
        long long run = 0;

        for (c = 0; c < w->cols; c++) {
            run += !isnan(l->grid.data[(size_t)r * w->cols + c]);
            l->count_sat[(r + 1) * stride + c + 1] =
                l->count_sat[r * stride + c + 1] + run;
        }
    }
}

/* Chamfer (3-4) distance of each valid cell to the nearest invalid cell,
 * in map units from the cell centre to the footprint edge. Beyond a grid
 * side that coincides with the region edge the footprint is taken to
 * continue: the region edge is not a seam between DEMs. */
static void build_edge_distance(struct dem_layer *l,
                                const struct Cell_head *region)
{
    const struct Cell_head *w = &l->grid.win;
    long rows = w->rows, cols = w->cols, r, c;
    double eps = 1e-6 * w->ew_res;
    /* The window is aligned outward to the DEM grid, so it may reach
     * slightly beyond the region. */
    int open_n = w->north >= region->north - eps;
    int open_s = w->south <= region->south + eps;
    int open_e = w->east >= region->east - eps;
    int open_w = w->west <= region->west + eps;
    float *d = G_malloc((size_t)rows * cols * sizeof(float));

#define D(rr, cc)                                                              \
    (((rr) < 0)       ? (open_n ? CHAMFER_INF : 0.0f)                          \
     : ((rr) >= rows) ? (open_s ? CHAMFER_INF : 0.0f)                          \
     : ((cc) < 0)     ? (open_w ? CHAMFER_INF : 0.0f)                          \
     : ((cc) >= cols) ? (open_e ? CHAMFER_INF : 0.0f)                          \
                      : d[(size_t)(rr) * cols + (cc)])

    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++)
            d[(size_t)r * cols + c] =
                isnan(l->grid.data[(size_t)r * cols + c]) ? 0.0f : CHAMFER_INF;

    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++) {
            float *p = &d[(size_t)r * cols + c];

            if (*p == 0.0f)
                continue;
            *p = fminf(*p, D(r - 1, c) + 3.0f);
            *p = fminf(*p, D(r, c - 1) + 3.0f);
            *p = fminf(*p, D(r - 1, c - 1) + 4.0f);
            *p = fminf(*p, D(r - 1, c + 1) + 4.0f);
        }
    for (r = rows - 1; r >= 0; r--)
        for (c = cols - 1; c >= 0; c--) {
            float *p = &d[(size_t)r * cols + c];

            if (*p == 0.0f)
                continue;
            *p = fminf(*p, D(r + 1, c) + 3.0f);
            *p = fminf(*p, D(r, c + 1) + 3.0f);
            *p = fminf(*p, D(r + 1, c + 1) + 4.0f);
            *p = fminf(*p, D(r + 1, c - 1) + 4.0f);
        }
#undef D

    /* Chamfer units (3 per cell) to map units, measured from the cell
     * centre to the footprint edge (half a cell inside it). */
    for (r = 0; r < rows * cols; r++)
        if (d[r] > 0.0f && d[r] < CHAMFER_INF)
            d[r] = (float)((d[r] / 3.0 - 0.5) * w->ew_res);
    l->edge_distance = d;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;

    return (x > y) - (x < y);
}

/* Median and MAD of (coarse - mean of fine) over coarse cells at least 90%
 * covered by the fine DEM. Returns the number of cells compared. */
static long bias(const struct dem_layer *fine, const struct dem_layer *coarse,
                 double *median, double *mad)
{
    const struct Cell_head *cw = &coarse->grid.win;
    const struct Cell_head *fw = &fine->grid.win;
    double expected = (cw->ew_res / fw->ew_res) * (cw->ns_res / fw->ns_res);
    double *diff = NULL;
    long n = 0, cap = 0, r, c, i;

    for (r = 0; r < cw->rows; r++) {
        double y1 = cw->north - r * cw->ns_res, y0 = y1 - cw->ns_res;

        if (y0 >= fw->north || y1 <= fw->south)
            continue;
        for (c = 0; c < cw->cols; c++) {
            double x0 = cw->west + c * cw->ew_res, x1 = x0 + cw->ew_res;
            double zc = coarse->grid.data[(size_t)r * cw->cols + c], zf;
            long count;

            if (isnan(zc) || x0 >= fw->east || x1 <= fw->west)
                continue;
            zf = raster_grid_box_mean(&fine->grid, x0, y0, x1, y1, &count);
            if (count < 0.9 * expected)
                continue;
            if (n == cap) {
                cap = cap ? 2 * cap : 1024;
                diff = G_realloc(diff, cap * sizeof(double));
            }
            diff[n++] = (zc + coarse->offset) - (zf + fine->offset);
        }
    }
    if (n == 0) {
        G_free(diff);
        return 0;
    }
    qsort(diff, n, sizeof(double), cmp_double);
    *median = n % 2 ? diff[n / 2] : 0.5 * (diff[n / 2 - 1] + diff[n / 2]);
    for (i = 0; i < n; i++)
        diff[i] = fabs(diff[i] - *median);
    qsort(diff, n, sizeof(double), cmp_double);
    *mad = n % 2 ? diff[n / 2] : 0.5 * (diff[n / 2 - 1] + diff[n / 2]);
    G_free(diff);

    return n;
}

void dem_stack_load(struct dem_stack *st, const struct preflight *pf,
                    char **names, char **offset_answers, double tolerance,
                    double blend_width)
{
    struct Cell_head region;
    int i, j, n_names = 0, n_offsets = 0;
    int has_offset[MAX_DEMS] = {0};

    memset(st, 0, sizeof(*st));
    G_get_window(&region);
    while (names[n_names])
        n_names++;
    while (offset_answers && offset_answers[n_offsets])
        n_offsets++;
    if (n_offsets && n_offsets != n_names)
        G_fatal_error(_("dem_offset= needs one value per elevation map (%d "
                        "given for %d maps)"),
                      n_offsets, n_names);

    st->n = pf->n_dems;
    for (i = 0; i < st->n; i++) {
        struct dem_layer *l = &st->layers[i];

        G_message(_("Reading elevation map <%s> at %g m..."),
                  pf->dems[i].name, pf->dems[i].res);
        raster_grid_load(&l->grid, pf->dems[i].name);
        l->res = pf->dems[i].res;
        l->level = pf->dems[i].level;
        for (j = 0; j < n_offsets; j++)
            if (strcmp(names[j], pf->dems[i].name) == 0 ||
                strcmp(names[j], G_fully_qualified_name(
                                     pf->dems[i].name, pf->dems[i].mapset)) ==
                    0) {
                l->offset = atof(offset_answers[j]);
                has_offset[i] = 1;
            }
        build_count_sat(l);
    }

    for (i = 0; i + 1 < st->n; i++) {
        struct dem_layer *fine = &st->layers[i], *coarse = &st->layers[i + 1];
        double median = 0.0, mad = 0.0;
        long n = bias(fine, coarse, &median, &mad);

        fine->blend_width = blend_width * coarse->res;
        build_edge_distance(fine, &region);
        if (n == 0) {
            G_verbose_message(_("<%s> and <%s> do not overlap enough for a "
                                "bias check"),
                              fine->grid.name, coarse->grid.name);
            continue;
        }
        G_message(_("Vertical difference <%s> - <%s>: median %.3f m, MAD "
                    "%.3f m over %ld cells"),
                  coarse->grid.name, fine->grid.name, median, mad, n);
        if (fabs(median) > tolerance && !has_offset[i])
            G_fatal_error(_("The median vertical difference between <%s> and "
                            "<%s> is %.3f m (tolerance %.3f m, "
                            "dem_bias_tolerance=). Check their vertical "
                            "datums, or give dem_offset= explicitly (e.g. "
                            "%.3f for <%s>)"),
                          coarse->grid.name, fine->grid.name, median,
                          tolerance, median, fine->grid.name);
    }
}

void dem_stack_free(struct dem_stack *st)
{
    int i;

    for (i = 0; i < st->n; i++) {
        raster_grid_free(&st->layers[i].grid);
        G_free(st->layers[i].count_sat);
        G_free(st->layers[i].edge_distance);
    }
    memset(st, 0, sizeof(*st));
}

static double smoothstep(double t)
{
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);

    return t * t * (3.0 - 2.0 * t);
}

/* Blend weight of layer i at (x, y): 1 inside its footprint beyond the
 * blend width, falling smoothly to 0 at the footprint edge. */
static double blend_weight(const struct dem_layer *l, double x, double y)
{
    const struct Cell_head *w = &l->grid.win;
    long col = (long)floor((x - w->west) / w->ew_res);
    long row = (long)floor((w->north - y) / w->ns_res);
    double d;

    if (!l->edge_distance || l->blend_width <= 0.0)
        return 1.0;
    if (col < 0 || row < 0 || col >= w->cols || row >= w->rows)
        return 0.0;
    d = l->edge_distance[(size_t)row * w->cols + col];

    return d >= l->blend_width ? 1.0 : smoothstep(d / l->blend_width);
}

static double point_from(const struct dem_stack *st, int i, double x,
                         double y)
{
    for (; i < st->n; i++) {
        const struct dem_layer *l = &st->layers[i];
        double z, w, zc;

        if (i < st->n - 1 && isnan(raster_grid_nearest(&l->grid, x, y)))
            continue;
        z = raster_grid_bilinear(&l->grid, x, y);
        if (isnan(z))
            continue;
        z += l->offset;
        if (i == st->n - 1 || (w = blend_weight(l, x, y)) >= 1.0)
            return z;
        zc = point_from(st, i + 1, x, y);

        return isnan(zc) ? z : w * z + (1.0 - w) * zc;
    }

    return NAN;
}

double dem_stack_point(const struct dem_stack *st, double x, double y)
{
    return point_from(st, 0, x, y);
}

static double box_from(const struct dem_stack *st, int i, double x0,
                       double y0, double x1, double y1)
{
    for (; i < st->n; i++) {
        const struct dem_layer *l = &st->layers[i];
        double z, w, zc, xc = 0.5 * (x0 + x1), yc = 0.5 * (y0 + y1);
        long count;

        if (dem_layer_count(l, x0, y0, x1, y1) > 0)
            z = raster_grid_box_mean(&l->grid, x0, y0, x1, y1, &count);
        else if (!isnan(raster_grid_nearest(&l->grid, xc, yc)))
            /* Box smaller than the DEM cells and containing none of their
             * centres (e.g. a fine fringe leaf over a coarse DEM). */
            z = raster_grid_bilinear(&l->grid, xc, yc);
        else
            continue;
        if (isnan(z))
            continue;
        z += l->offset;
        if (i == st->n - 1 || (w = blend_weight(l, xc, yc)) >= 1.0)
            return z;
        zc = box_from(st, i + 1, x0, y0, x1, y1);

        return isnan(zc) ? z : w * z + (1.0 - w) * zc;
    }

    return NAN;
}

double dem_stack_box_mean(const struct dem_stack *st, double x0, double y0,
                          double x1, double y1)
{
    return box_from(st, 0, x0, y0, x1, y1);
}
