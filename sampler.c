/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Raster maps loaded at their native resolution (clipped to
 *               the current region) and sampled at arbitrary points or
 *               boxes (PLAN.md sections 4.2 and 7.1).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "sampler.h"

void raster_grid_load(struct raster_grid *grid, const char *name)
{
    struct Cell_head region, cellhd;
    const char *mapset;
    DCELL *buf;
    int fd, row, col;

    memset(grid, 0, sizeof(*grid));
    G_strlcpy(grid->name, name, sizeof(grid->name));
    mapset = G_find_raster2(name, "");
    if (!mapset)
        G_fatal_error(_("Raster map <%s> not found"), name);
    Rast_get_cellhd(name, mapset, &cellhd);

    /* The region as last set by Rast_set_input_window() may differ from
     * the process region; always clip against the process region. */
    G_get_window(&region);
    grid->win = region;
    grid->win.north = fmin(region.north, cellhd.north);
    grid->win.south = fmax(region.south, cellhd.south);
    grid->win.east = fmin(region.east, cellhd.east);
    grid->win.west = fmax(region.west, cellhd.west);
    if (grid->win.north <= grid->win.south || grid->win.east <= grid->win.west)
        G_fatal_error(_("Raster map <%s> does not overlap the current region"),
                      name);
    Rast_align_window(&grid->win, &cellhd);

    Rast_set_input_window(&grid->win);
    fd = Rast_open_old(name, mapset);
    buf = Rast_allocate_d_input_buf();
    grid->data =
        G_malloc((size_t)grid->win.rows * grid->win.cols * sizeof(double));
    for (row = 0; row < grid->win.rows; row++) {
        double *out = grid->data + (size_t)row * grid->win.cols;

        G_percent(row, grid->win.rows, 5);
        Rast_get_d_row(fd, buf, row);
        for (col = 0; col < grid->win.cols; col++) {
            if (Rast_is_d_null_value(&buf[col])) {
                out[col] = NAN;
                continue;
            }
            if (!isfinite(buf[col]))
                G_fatal_error(_("Raster map <%s> contains a non-finite value "
                                "at row %d, col %d"),
                              name, row, col);
            out[col] = buf[col];
        }
    }
    G_percent(1, 1, 1);
    G_free(buf);
    Rast_close(fd);
    Rast_set_input_window(&region);
}

void raster_grid_free(struct raster_grid *grid)
{
    G_free(grid->data);
    grid->data = NULL;
}

static double cell(const struct raster_grid *grid, long row, long col)
{
    if (row < 0 || col < 0 || row >= grid->win.rows || col >= grid->win.cols)
        return NAN;

    return grid->data[(size_t)row * grid->win.cols + col];
}

double raster_grid_nearest(const struct raster_grid *grid, double x, double y)
{
    long col = (long)floor((x - grid->win.west) / grid->win.ew_res);
    long row = (long)floor((grid->win.north - y) / grid->win.ns_res);

    return cell(grid, row, col);
}

double raster_grid_bilinear(const struct raster_grid *grid, double x, double y)
{
    /* Continuous cell-centre coordinates: centre of cell (r, c) is at
     * (c, r). */
    double fc = (x - grid->win.west) / grid->win.ew_res - 0.5;
    double fr = (grid->win.north - y) / grid->win.ns_res - 0.5;
    long c0 = (long)floor(fc), r0 = (long)floor(fr);
    double tc = fc - c0, tr = fr - r0;
    double w[4], v[4], sum = 0.0, wsum = 0.0;
    int k;

    w[0] = (1.0 - tr) * (1.0 - tc);
    w[1] = (1.0 - tr) * tc;
    w[2] = tr * (1.0 - tc);
    w[3] = tr * tc;
    v[0] = cell(grid, r0, c0);
    v[1] = cell(grid, r0, c0 + 1);
    v[2] = cell(grid, r0 + 1, c0);
    v[3] = cell(grid, r0 + 1, c0 + 1);
    for (k = 0; k < 4; k++) {
        if (w[k] == 0.0 || isnan(v[k]))
            continue;
        sum += w[k] * v[k];
        wsum += w[k];
    }

    return wsum > 0.0 ? sum / wsum : NAN;
}

double raster_grid_box_mean(const struct raster_grid *grid, double x0,
                            double y0, double x1, double y1, long *count)
{
    /* Cells whose centres lie in [x0, x1) x [y0, y1). */
    long c0 = (long)ceil((x0 - grid->win.west) / grid->win.ew_res - 0.5);
    long c1 = (long)ceil((x1 - grid->win.west) / grid->win.ew_res - 0.5);
    long r0 = (long)floor((grid->win.north - y1) / grid->win.ns_res - 0.5) + 1;
    long r1 = (long)floor((grid->win.north - y0) / grid->win.ns_res - 0.5) + 1;
    long r, c, n = 0;
    double sum = 0.0;

    for (r = r0; r < r1; r++)
        for (c = c0; c < c1; c++) {
            double v = cell(grid, r, c);

            if (isnan(v))
                continue;
            sum += v;
            n++;
        }
    if (count)
        *count = n;

    return n > 0 ? sum / n : NAN;
}
