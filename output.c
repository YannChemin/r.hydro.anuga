/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Raster outputs on the current region: mesh level, time
 *               series registered as space-time raster datasets, summary
 *               rasters and the mass balance table (PLAN.md section 8).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#define _GNU_SOURCE /* timegm() */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "hashmap.h"
#include "output.h"

#define G_ACCEL 9.8 /* anuga.config.g, as the solver */

static const char *q_names[N_QUANTITIES] = {
    "depth",     "stage",     "xvelocity", "yvelocity", "speed", "direction",
    "discharge", "xmomentum", "ymomentum", "froude",    "hazard"};

static const char *q_units[N_QUANTITIES] = {"m",    "m",       "m/s",  "m/s",
                                            "m/s",  "degrees", "m2/s", "m2/s",
                                            "m2/s", "",        "m2/s"};

static const char *q_titles[N_QUANTITIES] = {
    "Water depth",
    "Water surface elevation (stage)",
    "Velocity, x component",
    "Velocity, y component",
    "Flow speed",
    "Flow direction (degrees counter-clockwise from east)",
    "Unit discharge |h u|",
    "Momentum, x component (h u)",
    "Momentum, y component (h v)",
    "Froude number",
    "Hazard rating h (v + 0.5)"};

static const char *q_colors[N_QUANTITIES] = {
    "water", "elevation",   "differences", "differences", "bcyr", "aspectcolr",
    "bcyr",  "differences", "differences", "bcyr",        "bcyr"};

/* ---- Mesh level --------------------------------------------------------- */

static uint64_t leaf_key(int level, int64_t ix, int64_t iy)
{
    return ((uint64_t)level << 58) | ((uint64_t)ix << 29) | (uint64_t)iy;
}

static void write_history(const char *name)
{
    struct History history;

    Rast_short_history(name, "raster", &history);
    Rast_command_history(&history);
    Rast_write_history(name, &history);
}

void output_mesh_level(const struct quadtree *qt, const char *name)
{
    struct Cell_head region;
    struct hashmap leaves;
    CELL *buf;
    int fd, row, col;
    long k;

    hashmap_init(&leaves, qt->n_leaves);
    for (k = 0; k < qt->n_leaves; k++)
        hashmap_put_new(
            &leaves,
            leaf_key(qt->leaves[k].level, qt->leaves[k].ix, qt->leaves[k].iy),
            k);

    G_get_window(&region);
    Rast_set_input_window(&region);
    fd = Rast_open_c_new(name);
    buf = Rast_allocate_c_output_buf();
    for (row = 0; row < region.rows; row++) {
        double y = Rast_row_to_northing(row + 0.5, &region) - qt->south;

        G_percent(row, region.rows, 5);
        for (col = 0; col < region.cols; col++) {
            double x = Rast_col_to_easting(col + 0.5, &region) - qt->west;
            int level, found = -1;

            /* Finest level first: at most one leaf contains the point. */
            for (level = qt->n_levels - 1; level >= 0 && found < 0; level--) {
                double size = ldexp(qt->res_max, -level);
                int64_t ix = (int64_t)floor(x / size);
                int64_t iy = (int64_t)floor(y / size);

                if (ix >= 0 && iy >= 0 &&
                    hashmap_get(&leaves, leaf_key(level, ix, iy)) >= 0)
                    found = level;
            }
            if (found < 0)
                Rast_set_c_null_value(&buf[col], 1);
            else
                buf[col] = found;
        }
        Rast_put_c_row(fd, buf);
    }
    G_percent(1, 1, 1);
    G_free(buf);
    Rast_close(fd);
    hashmap_free(&leaves);

    write_history(name);
    Rast_put_cell_title(name, _("Mesh quadtree level (0 = coarsest)"));
}

/* ---- Triangle to raster transfer --------------------------------------- */

/* 1 if (x, y) lies in triangle k (counter-clockwise), with a tolerance so
 * points on shared edges are found. */
static int in_triangle(const struct mesh *m, long k, double x, double y)
{
    const double *v = m->vertex_coordinates + 6 * k;
    double tol = -1e-9 * (fabs(v[2] - v[0]) + fabs(v[3] - v[1]));
    int i;

    for (i = 0; i < 3; i++) {
        double ax = v[2 * i], ay = v[2 * i + 1];
        double bx = v[2 * ((i + 1) % 3)], by = v[2 * ((i + 1) % 3) + 1];

        if ((bx - ax) * (y - ay) - (by - ay) * (x - ax) < tol)
            return 0;
    }

    return 1;
}

/* Visit the (cell, triangle, weight) contributions; pass 0 counts, pass 1
 * fills. */
static void grid_pass(struct out_grid *g, const struct quadtree *qt,
                      const struct mesh *m, const long *first, int pass,
                      long *fill)
{
    const struct Cell_head *w = &g->win;
    long l;

    for (l = 0; l < qt->n_leaves; l++) {
        const struct leaf *lf = &qt->leaves[l];
        double size = quadtree_leaf_size(qt, lf), lx, ly;
        long t0 = first[l], t1 = first[l + 1], k;

        quadtree_leaf_origin(qt, lf, &lx, &ly);
        lx += m->origin_x;
        ly += m->origin_y;

        if (w->ew_res >= size * (1.0 - 1e-9) &&
            w->ns_res >= size * (1.0 - 1e-9)) {
            /* Cell at least as large as the leaf: whole leaf, by area. */
            long col = (long)floor((lx + 0.5 * size - w->west) / w->ew_res);
            long row = (long)floor((w->north - (ly + 0.5 * size)) / w->ns_res);
            long cell;

            if (col < 0 || row < 0 || col >= w->cols || row >= w->rows)
                continue;
            cell = row * w->cols + col;
            for (k = t0; k < t1; k++) {
                if (pass == 0)
                    g->start[cell + 1]++;
                else {
                    g->tri[fill[cell]] = (int32_t)k;
                    g->weight[fill[cell]] = m->areas[k];
                    fill[cell]++;
                }
            }
        }
        else {
            /* Finer cells: the triangle containing each cell centre. */
            long c0 = (long)ceil((lx - w->west) / w->ew_res - 0.5);
            long c1 = (long)ceil((lx + size - w->west) / w->ew_res - 0.5);
            long r0 =
                (long)floor((w->north - (ly + size)) / w->ns_res - 0.5) + 1;
            long r1 = (long)floor((w->north - ly) / w->ns_res - 0.5) + 1;
            long row, col;

            for (row = (r0 > 0 ? r0 : 0); row < r1 && row < w->rows; row++)
                for (col = (c0 > 0 ? c0 : 0); col < c1 && col < w->cols;
                     col++) {
                    double x = w->west + (col + 0.5) * w->ew_res - m->origin_x;
                    double y = w->north - (row + 0.5) * w->ns_res - m->origin_y;
                    long cell = row * w->cols + col;

                    for (k = t0; k < t1; k++)
                        if (in_triangle(m, k, x, y))
                            break;
                    if (k == t1)
                        continue;
                    if (pass == 0)
                        g->start[cell + 1]++;
                    else {
                        g->tri[fill[cell]] = (int32_t)k;
                        g->weight[fill[cell]] = 1.0;
                        fill[cell]++;
                    }
                }
        }
    }
}

void out_grid_build(struct out_grid *g, const struct quadtree *qt,
                    const struct mesh *m, const struct Cell_head *win,
                    const char *suffix)
{
    long *first, *fill, cell, k, n_entries;

    memset(g, 0, sizeof(*g));
    if (win)
        g->win = *win;
    else
        G_get_window(&g->win);
    G_strlcpy(g->suffix, suffix ? suffix : "", sizeof(g->suffix));
    g->n_cells = (long)g->win.rows * g->win.cols;

    /* Triangles of each leaf are contiguous, in leaf order. */
    first = G_calloc(qt->n_leaves + 1, sizeof(long));
    for (k = 0; k < m->n_tri; k++)
        first[m->tri_leaf[k] + 1]++;
    for (k = 0; k < qt->n_leaves; k++)
        first[k + 1] += first[k];

    g->start = G_calloc(g->n_cells + 1, sizeof(long));
    grid_pass(g, qt, m, first, 0, NULL);
    for (cell = 0; cell < g->n_cells; cell++)
        g->start[cell + 1] += g->start[cell];
    n_entries = g->start[g->n_cells];
    g->tri = G_malloc((n_entries > 0 ? n_entries : 1) * sizeof(int32_t));
    g->weight = G_malloc((n_entries > 0 ? n_entries : 1) * sizeof(double));
    fill = G_malloc((g->n_cells > 0 ? g->n_cells : 1) * sizeof(long));
    memcpy(fill, g->start, g->n_cells * sizeof(long));
    grid_pass(g, qt, m, first, 1, fill);

    for (cell = 0; cell < g->n_cells; cell++) {
        double sum = 0.0;
        long e;

        for (e = g->start[cell]; e < g->start[cell + 1]; e++)
            sum += g->weight[e];
        for (e = g->start[cell]; e < g->start[cell + 1]; e++)
            g->weight[e] /= sum;
    }
    G_free(first);
    G_free(fill);
    G_verbose_message(_("Output grid: %ld cells, %ld triangle contributions"),
                      g->n_cells, n_entries);
}

void out_grid_free(struct out_grid *g)
{
    G_free(g->start);
    G_free(g->tri);
    G_free(g->weight);
    memset(g, 0, sizeof(*g));
}

/* ---- Options ------------------------------------------------------------- */

void output_parse_quantities(struct output_options *opt, char **answers)
{
    int i, q;

    memset(opt->quantity, 0, sizeof(opt->quantity));
    opt->quantity[Q_DEPTH] = 1;
    for (i = 0; answers && answers[i]; i++) {
        if (strcmp(answers[i], "infiltration") == 0)
            G_fatal_error(_("The infiltration output is not implemented yet "
                            "(phase 6 of PLAN.md)"));
        for (q = 0; q < N_QUANTITIES; q++)
            if (strcmp(answers[i], q_names[q]) == 0)
                break;
        if (q == N_QUANTITIES)
            G_fatal_error(_("Unknown output quantity '%s'"), answers[i]);
        opt->quantity[q] = 1;
    }
}

double output_parse_start(const char *text)
{
    struct tm tm;
    int n;

    memset(&tm, 0, sizeof(tm));
    n = sscanf(text, "%d-%d-%d%*[ T]%d:%d:%d", &tm.tm_year, &tm.tm_mon,
               &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec);
    if (n != 3 && n != 5 && n != 6)
        G_fatal_error(
            _("Invalid start time '%s' (use YYYY-MM-DD[ HH:MM[:SS]])"), text);
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;

    return (double)timegm(&tm);
}

static long n_outputs(const struct output_options *opt)
{
    return 1 + (long)ceil(opt->duration / opt->output_step - 1e-9);
}

static int index_width(long n)
{
    int w = 1;

    while (n > 9) {
        n /= 10;
        w++;
    }

    return w < 3 ? 3 : w;
}

static void map_name(char *buf, size_t size, const struct output_options *opt,
                     const char *suffix, int q, long index, int width)
{
    char digits[24];

    /* Width is at most 19 digits for a long. */
    snprintf(digits, sizeof(digits), "%0*ld", width > 19 ? 19 : width, index);
    snprintf(buf, size, "%s%s_%s_%s", opt->basename, suffix, q_names[q],
             digits);
}

static void check_raster(const char *name)
{
    if (G_legal_filename(name) < 0)
        G_fatal_error(_("<%s> is not a legal raster map name"), name);
    if (G_find_raster2(name, G_mapset()) && !G_get_overwrite())
        G_fatal_error(_("Raster map <%s> already exists (use --overwrite)"),
                      name);
}

void output_check_names(const struct output_options *opt, int n_detail)
{
    char name[GNAME_MAX * 2], suffix[32];
    const char *summaries[] = {opt->max_depth,    opt->max_speed,
                               opt->max_stage,    opt->max_hazard,
                               opt->arrival_time, opt->inundation_duration};
    const char *final[] = {"stage", "xmom", "ymom"};
    long n, i;
    int q, width, g;
    size_t j;

    if (opt->basename && opt->output_step != floor(opt->output_step))
        G_fatal_error(_("output_step= must be a whole number of seconds "
                        "for time series outputs"));
    n = n_outputs(opt);
    width = index_width(n - 1);
    for (g = 0; g <= n_detail; g++) {
        if (g == 0)
            suffix[0] = '\0';
        else
            snprintf(suffix, sizeof(suffix), "_detail%d", g);
        if (opt->basename)
            for (q = 0; q < N_QUANTITIES; q++) {
                if (!opt->quantity[q])
                    continue;
                for (i = 0; i < n; i++) {
                    map_name(name, sizeof(name), opt, suffix, q, i, width);
                    check_raster(name);
                }
            }
        /* The main-grid summary names are checked by the parser. */
        if (g > 0)
            for (j = 0; j < sizeof(summaries) / sizeof(summaries[0]); j++)
                if (summaries[j]) {
                    snprintf(name, sizeof(name), "%s%s", summaries[j], suffix);
                    check_raster(name);
                }
        if (opt->final_prefix)
            for (j = 0; j < 3; j++) {
                snprintf(name, sizeof(name), "%s%s_%s", opt->final_prefix,
                         suffix, final[j]);
                check_raster(name);
            }
    }
}

/* ---- Time series ------------------------------------------------------- */

static void prepare_arrays(struct output_context *ctx, const struct sw_state *s)
{
    long k;

    for (k = 0; k < s->n; k++) {
        double h = s->stage_c[k] - anuga_bed(s->zq, s->z0, (anuga_idx)k);

        ctx->h[k] = h > 0.0 ? h : 0.0;
    }
}

void output_begin(struct output_context *ctx, const struct output_options *opt,
                  const struct out_grid *grids, int n_grids,
                  const struct evolve_log *log, const struct sw_state *s)
{
    int q, g;

    memset(ctx, 0, sizeof(*ctx));
    ctx->opt = opt;
    ctx->grids = grids;
    ctx->n_grids = grids ? n_grids : 0;
    G_get_window(&ctx->region);
    ctx->log = log;
    ctx->n_outputs = n_outputs(opt);
    ctx->index_width = index_width(ctx->n_outputs - 1);
    ctx->h = G_malloc((s->n > 0 ? s->n : 1) * sizeof(double));

    if (opt->basename)
        for (g = 0; g < ctx->n_grids; g++)
            for (q = 0; q < N_QUANTITIES; q++) {
                if (!opt->quantity[q])
                    continue;
                ctx->register_path[g][q] = G_tempfile();
                ctx->register_file[g][q] = fopen(ctx->register_path[g][q], "w");
                if (!ctx->register_file[g][q])
                    G_fatal_error(_("Unable to write <%s>: %s"),
                                  ctx->register_path[g][q], strerror(errno));
            }

    if (opt->massbalance) {
        ctx->massbal = fopen(opt->massbalance, "w");
        if (!ctx->massbal)
            G_fatal_error(_("Unable to write <%s>: %s"), opt->massbalance,
                          strerror(errno));
        fprintf(ctx->massbal, "time,volume,signed_volume,boundary_inflow,"
                              "clamping_added,mass_error,relative_error\n");
    }
}

static double signed_volume(const struct sw_state *s)
{
    double v = 0.0;
    long k;

    for (k = 0; k < s->n; k++)
        v += (s->stage_c[k] - anuga_bed(s->zq, s->z0, (anuga_idx)k)) *
             s->areas[k];

    return v;
}

static void write_massbalance(struct output_context *ctx,
                              const struct sw_state *s, double t)
{
    double volume = 0.0, sv = signed_volume(s), error, ref;
    long k;

    for (k = 0; k < s->n; k++)
        volume += ctx->h[k] * s->areas[k];
    if (t == 0.0)
        ctx->initial_signed_volume = sv;
    error = sv - (ctx->initial_signed_volume + ctx->log->boundary_mass +
                  ctx->log->protect_mass);
    ref = fabs(ctx->initial_signed_volume) > 0.0
              ? fabs(ctx->initial_signed_volume)
              : 1.0;
    if (ctx->massbal)
        fprintf(ctx->massbal, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", t,
                volume, sv, ctx->log->boundary_mass, ctx->log->protect_mass,
                error, error / ref);
}

/* Cell values of the time series quantities. Returns 0 outside the mesh. */
static int cell_values(const struct output_context *ctx,
                       const struct out_grid *g, const struct sw_state *s,
                       long cell, double *v)
{
    double H = 0.0, S = 0.0, UH = 0.0, VH = 0.0, u = 0.0, w = 0.0, sp;
    long e;

    if (g->start[cell] == g->start[cell + 1])
        return 0;
    for (e = g->start[cell]; e < g->start[cell + 1]; e++) {
        long k = g->tri[e];
        double wt = g->weight[e];

        H += wt * ctx->h[k];
        S += wt * s->stage_c[k];
        UH += wt * s->xmom_c[k];
        VH += wt * s->ymom_c[k];
    }
    if (H > ctx->opt->min_depth) {
        u = UH / H;
        w = VH / H;
    }
    sp = sqrt(u * u + w * w);

    v[Q_DEPTH] = H;
    v[Q_STAGE] = S;
    v[Q_XVELOCITY] = u;
    v[Q_YVELOCITY] = w;
    v[Q_SPEED] = sp;
    v[Q_DIRECTION] =
        sp > 0.0 ? fmod(atan2(w, u) * 180.0 / M_PI + 360.0, 360.0) : NAN;
    v[Q_DISCHARGE] = H > ctx->opt->min_depth ? sqrt(UH * UH + VH * VH) : 0.0;
    v[Q_XMOMENTUM] = UH;
    v[Q_YMOMENTUM] = VH;
    v[Q_FROUDE] = H > ctx->opt->min_depth ? sp / sqrt(G_ACCEL * H) : 0.0;
    v[Q_HAZARD] = H > ctx->opt->min_depth ? H * (sp + 0.5) : 0.0;

    return 1;
}

static void put_value(void *buf, int col, double value, int dcell)
{
    if (isnan(value)) {
        if (dcell)
            Rast_set_d_null_value((DCELL *)buf + col, 1);
        else
            Rast_set_f_null_value((FCELL *)buf + col, 1);
    }
    else if (dcell)
        ((DCELL *)buf)[col] = value;
    else
        ((FCELL *)buf)[col] = (FCELL)value;
}

static void set_metadata(const char *name, const char *title, const char *units)
{
    write_history(name);
    Rast_put_cell_title(name, title);
    if (units && *units)
        Rast_write_units(name, units);
}

/* Write the time series maps of one output time on grid gi. */
static void write_step(struct output_context *ctx, int gi,
                       const struct sw_state *s, double t, long index)
{
    const struct output_options *opt = ctx->opt;
    const struct out_grid *g = &ctx->grids[gi];
    int fd[N_QUANTITIES], q, row, col;
    void *buf[N_QUANTITIES];
    char name[N_QUANTITIES][GNAME_MAX * 2];
    RASTER_MAP_TYPE type = opt->dcell ? DCELL_TYPE : FCELL_TYPE;

    Rast_set_output_window((struct Cell_head *)&g->win);
    for (q = 0; q < N_QUANTITIES; q++) {
        if (!opt->quantity[q])
            continue;
        map_name(name[q], sizeof(name[q]), opt, g->suffix, q, index,
                 ctx->index_width);
        fd[q] = Rast_open_new(name[q], type);
        buf[q] = Rast_allocate_output_buf(type);
    }

    for (row = 0; row < g->win.rows; row++) {
        for (col = 0; col < g->win.cols; col++) {
            double v[N_QUANTITIES];
            long cell = (long)row * g->win.cols + col;
            int covered = cell_values(ctx, g, s, cell, v);
            int dry = covered && v[Q_DEPTH] <= opt->min_depth;

            for (q = 0; q < N_QUANTITIES; q++) {
                double value;

                if (!opt->quantity[q])
                    continue;
                value = covered ? v[q] : NAN;
                if (dry && q != Q_STAGE)
                    value = (q == Q_DIRECTION) ? NAN : 0.0;
                if (dry && opt->null_dry)
                    value = NAN;
                put_value(buf[q], col, value, opt->dcell);
            }
        }
        for (q = 0; q < N_QUANTITIES; q++)
            if (opt->quantity[q])
                Rast_put_row(fd[q], buf[q], type);
    }

    for (q = 0; q < N_QUANTITIES; q++) {
        char title[256];

        if (!opt->quantity[q])
            continue;
        Rast_close(fd[q]);
        G_free(buf[q]);
        snprintf(title, sizeof(title), _("%s at t = %g s"), q_titles[q], t);
        set_metadata(name[q], title, q_units[q]);

        if (opt->absolute) {
            time_t tt = (time_t)llround(opt->start_epoch + t);
            struct tm tm;
            char when[64];

            gmtime_r(&tt, &tm);
            strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &tm);
            fprintf(ctx->register_file[gi][q], "%s|%s\n", name[q], when);
        }
        else {
            fprintf(ctx->register_file[gi][q], "%s|%lld\n", name[q],
                    llround(t));
        }
    }
    Rast_set_output_window(&ctx->region);
}

void output_step_fn(struct sw_state *s, double t, void *data)
{
    struct output_context *ctx = data;
    long index = ctx->next_index++;
    int g;

    prepare_arrays(ctx, s);
    write_massbalance(ctx, s, t);
    if (!ctx->opt->basename)
        return;
    G_verbose_message(_("Writing outputs at t = %g s..."), t);
    for (g = 0; g < ctx->n_grids; g++)
        write_step(ctx, g, s, t, index);
}

/* ---- Summary rasters and registration ---------------------------------- */

enum combine { COMBINE_MEAN, COMBINE_MAX, COMBINE_MIN_POSITIVE };

static void write_summary_grid(const struct output_context *ctx,
                               const struct out_grid *g, const char *name,
                               const double *values, enum combine how,
                               const char *title, const char *units,
                               const char *color, const struct sw_state *s,
                               int null_if_dry)
{
    const struct output_options *opt = ctx->opt;
    struct Colors colors;
    struct FPRange range;
    DCELL min, max;
    void *buf;
    int fd, row, col;
    RASTER_MAP_TYPE type = opt->dcell ? DCELL_TYPE : FCELL_TYPE;

    Rast_set_output_window((struct Cell_head *)&g->win);
    fd = Rast_open_new(name, type);
    buf = Rast_allocate_output_buf(type);
    for (row = 0; row < g->win.rows; row++) {
        for (col = 0; col < g->win.cols; col++) {
            long cell = (long)row * g->win.cols + col, e;
            double v = NAN, maxd = 0.0;

            for (e = g->start[cell]; e < g->start[cell + 1]; e++) {
                double x = values[g->tri[e]];

                if (s->stats)
                    maxd = fmax(maxd, s->stat_max_depth[g->tri[e]]);
                if (how == COMBINE_MEAN)
                    v = (isnan(v) ? 0.0 : v) + g->weight[e] * x;
                else if (how == COMBINE_MAX)
                    v = isnan(v) ? x : fmax(v, x);
                else if (x >= 0.0)
                    v = isnan(v) ? x : fmin(v, x);
            }
            if (null_if_dry && s->stats && opt->null_dry &&
                maxd <= opt->min_depth)
                v = NAN;
            put_value(buf, col, v, opt->dcell);
        }
        Rast_put_row(fd, buf, type);
    }
    Rast_close(fd);
    G_free(buf);
    set_metadata(name, title, units);

    Rast_read_fp_range(name, G_mapset(), &range);
    Rast_get_fp_range_min_max(&range, &min, &max);
    if (!Rast_is_d_null_value(&min)) {
        Rast_make_fp_colors(&colors, color, min, max > min ? max : min + 1.0);
        Rast_write_colors(name, G_mapset(), &colors);
    }
    Rast_set_output_window((struct Cell_head *)&ctx->region);
}

/* A summary raster on every grid: name on the main grid, name followed by
 * the grid's suffix on the detail grids. */
static void write_summary(const struct output_context *ctx, const char *name,
                          const double *values, enum combine how,
                          const char *title, const char *units,
                          const char *color, const struct sw_state *s,
                          int null_if_dry)
{
    char full[GNAME_MAX * 2];
    int g;

    for (g = 0; g < ctx->n_grids; g++) {
        snprintf(full, sizeof(full), "%s%s", name, ctx->grids[g].suffix);
        write_summary_grid(ctx, &ctx->grids[g], full, values, how, title, units,
                           color, s, null_if_dry);
    }
}

static void run_command(const char *cmd)
{
    G_debug(1, "%s", cmd);
    if (system(cmd) != 0)
        G_fatal_error(_("Command failed: %s"), cmd);
}

static void register_strds(const struct output_context *ctx, int g, int q)
{
    const struct output_options *opt = ctx->opt;
    char strds[GNAME_MAX * 2], cmd[4 * GPATH_MAX];
    const char *ow = G_get_overwrite() ? " --overwrite" : "";

    snprintf(strds, sizeof(strds), "%s%s_%s", opt->basename,
             ctx->grids[g].suffix, q_names[q]);
    snprintf(cmd, sizeof(cmd),
             "t.create --quiet%s output=%s type=strds temporaltype=%s "
             "semantictype=mean title=\"r.hydro.anuga %s\" "
             "description=\"%s simulated by r.hydro.anuga\"",
             ow, strds, opt->absolute ? "absolute" : "relative", q_names[q],
             q_titles[q]);
    run_command(cmd);
    snprintf(cmd, sizeof(cmd), "t.register --quiet%s input=%s file=%s%s", ow,
             strds, ctx->register_path[g][q],
             opt->absolute ? "" : " unit=seconds");
    run_command(cmd);
    snprintf(cmd, sizeof(cmd), "t.rast.colors --quiet input=%s color=%s", strds,
             q_colors[q]);
    run_command(cmd);
    G_message(_("Space-time raster dataset <%s> registered"), strds);
}

void output_end(struct output_context *ctx, const struct sw_state *s)
{
    const struct output_options *opt = ctx->opt;
    int q, g;

    if (s->stats) {
        if (opt->max_depth)
            write_summary(ctx, opt->max_depth, s->stat_max_depth, COMBINE_MAX,
                          _("Maximum water depth"), "m", "water", s, 1);
        if (opt->max_speed)
            write_summary(ctx, opt->max_speed, s->stat_max_speed, COMBINE_MAX,
                          _("Maximum flow speed"), "m/s", "bcyr", s, 1);
        if (opt->max_stage)
            write_summary(ctx, opt->max_stage, s->stat_max_stage, COMBINE_MAX,
                          _("Maximum water surface elevation"), "m",
                          "elevation", s, 1);
        if (opt->max_hazard)
            write_summary(ctx, opt->max_hazard, s->stat_max_hazard, COMBINE_MAX,
                          _("Maximum hazard rating h (v + 0.5)"), "m2/s",
                          "bcyr", s, 1);
        if (opt->arrival_time)
            write_summary(ctx, opt->arrival_time, s->stat_arrival,
                          COMBINE_MIN_POSITIVE,
                          _("Time water first exceeds the arrival depth"), "s",
                          "bcyr", s, 0);
        if (opt->inundation_duration)
            write_summary(ctx, opt->inundation_duration, s->stat_duration,
                          COMBINE_MAX,
                          _("Time with water above the arrival depth"), "s",
                          "bcyr", s, 1);
    }
    if (opt->final_prefix)
        for (g = 0; g < ctx->n_grids; g++) {
            const struct out_grid *grid = &ctx->grids[g];
            char name[GNAME_MAX * 2];

            snprintf(name, sizeof(name), "%s%s_stage", opt->final_prefix,
                     grid->suffix);
            write_summary_grid(ctx, grid, name, s->stage_c, COMBINE_MEAN,
                               _("Final water surface elevation"), "m",
                               "elevation", s, 0);
            snprintf(name, sizeof(name), "%s%s_xmom", opt->final_prefix,
                     grid->suffix);
            write_summary_grid(ctx, grid, name, s->xmom_c, COMBINE_MEAN,
                               _("Final momentum, x component"), "m2/s",
                               "differences", s, 0);
            snprintf(name, sizeof(name), "%s%s_ymom", opt->final_prefix,
                     grid->suffix);
            write_summary_grid(ctx, grid, name, s->ymom_c, COMBINE_MEAN,
                               _("Final momentum, y component"), "m2/s",
                               "differences", s, 0);
        }

    if (ctx->massbal) {
        fclose(ctx->massbal);
        ctx->massbal = NULL;
    }
    for (g = 0; g < ctx->n_grids; g++)
        for (q = 0; q < N_QUANTITIES; q++) {
            if (!ctx->register_file[g][q])
                continue;
            fclose(ctx->register_file[g][q]);
            register_strds(ctx, g, q);
            remove(ctx->register_path[g][q]);
            G_free(ctx->register_path[g][q]);
        }
    G_free(ctx->h);
}
