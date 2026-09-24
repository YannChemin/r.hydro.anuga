/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Conforming triangle mesh from quadtree leaves, with ANUGA's
 *               geometry and connectivity conventions and the scaled
 *               integer bed elevation (PLAN.md sections 4.5 and 4.7).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_MESH_H
#define R_HYDRO_ANUGA_MESH_H

#include <stdint.h>

#include "dem_stack.h"
#include "quadtree.h"

/* Boundary tags of edges without a neighbouring triangle. */
enum boundary_tag { TAG_NORTH, TAG_SOUTH, TAG_EAST, TAG_WEST, TAG_NULL };

#define BED_SCALE     10000.0 /* Scaled units per metre (0.1 mm). */
#define BED_INV_SCALE 1.0e-4
#define BED_MARGIN    10000 /* Datum margin below the lowest bed (1 m). */

/* Exact dequantisation of the scaled bed (PLAN.md section 4.7): one
 * integer addition, then one correctly rounded multiplication. The same
 * expression is used by the OpenCL kernels. */
static inline double bed_dequantize(uint32_t zq, int64_t z0)
{
    return (double)((int64_t)zq + z0) * BED_INV_SCALE;
}

/* All arrays follow ANUGA's layout: per-triangle arrays of 3 hold values
 * for vertices (vertex i) or edges (edge i, opposite vertex i). */
struct mesh {
    long n_tri, n_nodes, n_boundary;
    double origin_x, origin_y; /* Absolute coordinates of the local origin. */

    double *node_xy;        /* n_nodes * 2, relative to the origin. */
    int64_t *node_key;      /* n_nodes, doubled finest-grid coordinates. */
    double *node_elevation; /* n_nodes, fp64 from the DEM. */
    int32_t *triangles;     /* n_tri * 3 node indices. */
    int32_t *tri_leaf;      /* n_tri, index into the quadtree leaves. */

    double *vertex_coordinates;   /* n_tri * 6 */
    double *edge_coordinates;     /* n_tri * 6, edge midpoints */
    double *centroid_coordinates; /* n_tri * 2 */
    double *normals;              /* n_tri * 6 */
    double *edgelengths;          /* n_tri * 3 */
    double *areas;                /* n_tri */
    double *radii;                /* n_tri */

    int32_t *neighbours;           /* n_tri * 3; boundary j stored as -(j+1) */
    int32_t *neighbour_edges;      /* n_tri * 3; -1 on boundary edges */
    int32_t *surrogate_neighbours; /* n_tri * 3 */
    int32_t *number_of_boundaries; /* n_tri */
    int32_t *tri_full_flag;        /* n_tri */

    int32_t *boundary_tri;  /* n_boundary, sorted by (triangle, edge) */
    int32_t *boundary_edge; /* n_boundary */
    uint8_t *boundary_tag;  /* n_boundary, enum boundary_tag */

    double *bed_centroid_f64; /* n_tri, (v0 + v1 + v2) / 3 before scaling */
    uint32_t *zq;             /* n_tri, scaled bed above z0 */
    int64_t z0;               /* Datum, in scaled units. */
    double max_quantization_error; /* max |dequantized - fp64| (m) */
};

/* Build the mesh from the quadtree leaves (with their hanging-node
 * masks), taking node elevations from the DEM stack (bilinear between
 * cell centres at corners and hanging nodes, mean of the governing DEM's
 * cells inside the leaf at leaf centres), then quantize the centroid bed. */
void mesh_build(struct mesh *m, const struct quadtree *qt,
                const struct dem_stack *dem);

void mesh_free(struct mesh *m);

/* Write every mesh array as raw little-endian binary files plus a
 * manifest.json into directory dir (created if needed). */
void mesh_export(const struct mesh *m, const struct quadtree *qt,
                 const char *dir);

#endif /* R_HYDRO_ANUGA_MESH_H */
