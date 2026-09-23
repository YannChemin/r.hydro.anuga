/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Solver setup from module options: configuration, boundary
 *               conditions, initial conditions, friction, and the state
 *               export used for validation (PLAN.md sections 5 and 7).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "sampler.h"
#include "setup.h"

void setup_config(struct solver_config *cfg, const char *algorithm,
                  const char *cfl, const char *friction_method)
{
    enum flow_algorithm alg = ALG_DE1;

    if (strcmp(algorithm, "DE0") == 0)
        alg = ALG_DE0;
    else if (strcmp(algorithm, "DE2") == 0)
        alg = ALG_DE2;
    solver_config_init(cfg, alg);
    if (cfl) {
        cfg->cfl = atof(cfl);
        if (!(cfg->cfl > 0.0) || cfg->cfl > 1.0)
            G_fatal_error(_("cfl= must be in (0, 1], got '%s'"), cfl);
    }
    cfg->sloped_friction =
        friction_method && strcmp(friction_method, "sloped") == 0;
}

static int side_tag(const char *side)
{
    const char *names[] = {"north", "south", "east", "west", "null"};
    int i;

    for (i = 0; i < 5; i++)
        if (strcmp(side, names[i]) == 0)
            return i;
    G_fatal_error(_("Unknown boundary side '%s' (use north, south, east, "
                    "west or null)"),
                  side);

    return -1;
}

void setup_boundaries(struct sw_state *s, const struct mesh *m, char **answers)
{
    int type[5] = {BC_REFLECTIVE, BC_REFLECTIVE, BC_REFLECTIVE, BC_REFLECTIVE,
                   BC_REFLECTIVE};
    double value[5][3] = {{0}};
    long j;
    int i;

    for (i = 0; answers && answers[i]; i++) {
        char **tokens = G_tokenize(answers[i], ":");
        int n = G_number_of_tokens(tokens), tag;

        if (n < 2)
            G_fatal_error(_("Invalid boundary '%s' (use side:type[:value])"),
                          answers[i]);
        tag = side_tag(tokens[0]);
        if (strcmp(tokens[1], "reflective") == 0 && n == 2)
            type[tag] = BC_REFLECTIVE;
        else if (strcmp(tokens[1], "transmissive") == 0 && n == 2)
            type[tag] = BC_TRANSMISSIVE;
        else if (strcmp(tokens[1], "dirichlet") == 0 && (n == 3 || n == 5)) {
            type[tag] = BC_DIRICHLET;
            value[tag][0] = atof(tokens[2]);
            value[tag][1] = n == 5 ? atof(tokens[3]) : 0.0;
            value[tag][2] = n == 5 ? atof(tokens[4]) : 0.0;
        }
        else if (strcmp(tokens[1], "stage") == 0 ||
                 strcmp(tokens[1], "flather") == 0)
            G_fatal_error(_("Boundary type '%s' is not implemented yet "
                            "(phase 6 of PLAN.md)"),
                          tokens[1]);
        else
            G_fatal_error(_("Invalid boundary '%s': types are reflective, "
                            "transmissive, dirichlet:stage[:xmom:ymom]"),
                          answers[i]);
        G_free_tokens(tokens);
    }

    for (j = 0; j < s->nb; j++) {
        int tag = m->boundary_tag[j];

        s->bnd_type[j] = (anuga_u8)type[tag];
        s->bnd_value[3 * j] = value[tag][0];
        s->bnd_value[3 * j + 1] = value[tag][1];
        s->bnd_value[3 * j + 2] = value[tag][2];
    }
}

void setup_initial(struct sw_state *s, const struct mesh *m,
                   const char *initial_depth, const char *initial_stage)
{
    struct raster_grid grid;
    const char *name = initial_depth ? initial_depth : initial_stage;
    long k, n_wet = 0;

    if (!name)
        return;
    raster_grid_load(&grid, name);
    for (k = 0; k < s->n; k++) {
        double x = m->origin_x + m->centroid_coordinates[2 * k];
        double y = m->origin_y + m->centroid_coordinates[2 * k + 1];
        double v = raster_grid_nearest(&grid, x, y);
        double bed = anuga_bed(s->zq, s->z0, (anuga_idx)k);

        if (isnan(v))
            continue;
        if (initial_depth) {
            if (v < 0.0)
                G_fatal_error(_("Raster map <%s> has a negative depth (%g) at "
                                "(%.3f, %.3f)"),
                              name, v, x, y);
            s->stage_c[k] = bed + v;
        }
        else {
            s->stage_c[k] = v;
        }
        if (s->stage_c[k] > bed)
            n_wet++;
    }
    raster_grid_free(&grid);
    G_verbose_message(_("Initial water in %ld of %ld triangles"), n_wet, s->n);
}

void setup_friction(struct sw_state *s, const struct mesh *m,
                    const char *manning, double manning_value)
{
    struct raster_grid grid;
    long k;

    if (!manning) {
        for (k = 0; k < s->n; k++)
            s->friction[k] = manning_value;
        return;
    }
    raster_grid_load(&grid, manning);
    for (k = 0; k < s->n; k++) {
        double v = raster_grid_nearest(
            &grid, m->origin_x + m->centroid_coordinates[2 * k],
            m->origin_y + m->centroid_coordinates[2 * k + 1]);

        s->friction[k] = isnan(v) ? manning_value : v;
    }
    raster_grid_free(&grid);
}

static double *copy(const double *src, long n)
{
    double *dst = G_malloc((n > 0 ? n : 1) * sizeof(double));

    memcpy(dst, src, n * sizeof(double));

    return dst;
}

void snapshot_take(struct state_snapshot *snap, struct sw_state *s,
                   const struct solver_ops *ops)
{
    long k;

    ops->sync_to_host(s);
    snap->stage = copy(s->stage_c, s->n);
    snap->xmom = copy(s->xmom_c, s->n);
    snap->ymom = copy(s->ymom_c, s->n);
    snap->volume = ops->volume(s);
    /* Signed volume: includes the small negative depths an update can
     * leave before the next protect() clamps them. Mass conservation
     * holds for this sum, not for the positive-only volume. */
    snap->signed_volume = 0.0;
    for (k = 0; k < s->n; k++)
        snap->signed_volume +=
            (s->stage_c[k] - anuga_bed(s->zq, s->z0, (anuga_idx)k)) *
            s->areas[k];
}

void snapshot_free(struct state_snapshot *snap)
{
    G_free(snap->stage);
    G_free(snap->xmom);
    G_free(snap->ymom);
    snap->stage = snap->xmom = snap->ymom = NULL;
}

static void write_raw(FILE *manifest, const char *dir, const char *name,
                      const void *data, size_t elem, long rows, int cols,
                      const char *dtype, int *first)
{
    char path[GPATH_MAX];
    FILE *fp;

    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    fp = fopen(path, "wb");
    if (!fp)
        G_fatal_error(_("Unable to write <%s>: %s"), path, strerror(errno));
    if (rows > 0 && fwrite(data, elem * cols, rows, fp) != (size_t)rows)
        G_fatal_error(_("Error writing <%s>"), path);
    fclose(fp);
    fprintf(manifest,
            "%s    \"%s\": {\"file\": \"%s.bin\", \"dtype\": \"%s\", "
            "\"shape\": [%ld, %d]}",
            *first ? "" : ",\n", name, name, dtype, rows, cols);
    *first = 0;
}

void state_export(const char *dir, const struct sw_state *s,
                  const struct state_snapshot *initial,
                  const struct state_snapshot *final,
                  const struct evolve_log *log, double duration,
                  double yieldstep)
{
    const char *alg[] = {"DE0", "DE1", "DE2"};
    char path[GPATH_MAX];
    FILE *mf;
    double *bed;
    long k;
    int first = 1;
    union {
        uint16_t u;
        uint8_t b[2];
    } endian = {1};

    if (G_mkdir(dir) != 0 && errno != EEXIST)
        G_fatal_error(_("Unable to create directory <%s>: %s"), dir,
                      strerror(errno));
    snprintf(path, sizeof(path), "%s/manifest.json", dir);
    mf = fopen(path, "w");
    if (!mf)
        G_fatal_error(_("Unable to write <%s>: %s"), path, strerror(errno));

    fprintf(mf,
            "{\n  \"format\": \"r.hydro.anuga state 1\",\n"
            "  \"byteorder\": \"%s\",\n"
            "  \"algorithm\": \"%s\",\n  \"cfl\": %.17g,\n"
            "  \"sloped_friction\": %d,\n"
            "  \"duration\": %.17g,\n  \"yieldstep\": %.17g,\n"
            "  \"n_steps\": %ld,\n  \"boundary_mass\": %.17g,\n"
            "  \"protect_mass\": %.17g,\n"
            "  \"initial_volume\": %.17g,\n  \"final_volume\": %.17g,\n"
            "  \"initial_signed_volume\": %.17g,\n"
            "  \"final_signed_volume\": %.17g,\n"
            "  \"boundary_types\": [\"reflective\", \"transmissive\", "
            "\"dirichlet\"],\n  \"arrays\": {\n",
            endian.b[0] ? "little" : "big", alg[s->cfg.algorithm], s->cfg.cfl,
            s->cfg.sloped_friction, duration, yieldstep, log->n_steps,
            log->boundary_mass, log->protect_mass, initial->volume,
            final->volume, initial->signed_volume, final->signed_volume);

    bed = G_malloc((s->n > 0 ? s->n : 1) * sizeof(double));
    for (k = 0; k < s->n; k++)
        bed[k] = anuga_bed(s->zq, s->z0, (anuga_idx)k);

    write_raw(mf, dir, "bed", bed, sizeof(double), s->n, 1, "float64", &first);
    write_raw(mf, dir, "friction", s->friction, sizeof(double), s->n, 1,
              "float64", &first);
    write_raw(mf, dir, "initial_stage", initial->stage, sizeof(double), s->n, 1,
              "float64", &first);
    write_raw(mf, dir, "initial_xmom", initial->xmom, sizeof(double), s->n, 1,
              "float64", &first);
    write_raw(mf, dir, "initial_ymom", initial->ymom, sizeof(double), s->n, 1,
              "float64", &first);
    write_raw(mf, dir, "stage", final->stage, sizeof(double), s->n, 1,
              "float64", &first);
    write_raw(mf, dir, "xmom", final->xmom, sizeof(double), s->n, 1, "float64",
              &first);
    write_raw(mf, dir, "ymom", final->ymom, sizeof(double), s->n, 1, "float64",
              &first);
    write_raw(mf, dir, "max_speed", s->max_speed, sizeof(double), s->n, 1,
              "float64", &first);
    write_raw(mf, dir, "boundary_type", s->bnd_type, sizeof(anuga_u8), s->nb, 1,
              "uint8", &first);
    write_raw(mf, dir, "boundary_value", s->bnd_value, sizeof(double), s->nb, 3,
              "float64", &first);
    write_raw(mf, dir, "step_time", log->t, sizeof(double), log->n_steps, 1,
              "float64", &first);
    write_raw(mf, dir, "step_dt", log->dt, sizeof(double), log->n_steps, 1,
              "float64", &first);
    G_free(bed);

    fprintf(mf, "\n  }\n}\n");
    fclose(mf);
    G_message(_("State exported to <%s>"), dir);
}
