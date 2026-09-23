/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Raster outputs on the current region (PLAN.md section 8).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "hashmap.h"
#include "output.h"

static uint64_t leaf_key(int level, int64_t ix, int64_t iy)
{
    return ((uint64_t)level << 58) | ((uint64_t)ix << 29) | (uint64_t)iy;
}

void output_mesh_level(const struct quadtree *qt, const char *name)
{
    struct Cell_head region;
    struct hashmap leaves;
    struct History history;
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

    Rast_short_history(name, "raster", &history);
    Rast_command_history(&history);
    Rast_write_history(name, &history);
    Rast_put_cell_title(name, _("Mesh quadtree level (0 = coarsest)"));
}
