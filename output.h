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

#ifndef R_HYDRO_ANUGA_OUTPUT_H
#define R_HYDRO_ANUGA_OUTPUT_H

#include "quadtree.h"

/* Write the quadtree level of the leaf containing each region cell centre
 * (NULL outside the domain). */
void output_mesh_level(const struct quadtree *qt, const char *name);

#endif /* R_HYDRO_ANUGA_OUTPUT_H */
