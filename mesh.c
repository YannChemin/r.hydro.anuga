/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Conforming triangle mesh from quadtree leaves, with ANUGA's
 *               geometry and connectivity conventions and the scaled
 *               integer bed elevation (PLAN.md sections 4.5 and 4.7).
 *               Geometry expressions follow ANUGA's general_mesh.py and
 *               neighbour_mesh.py operation by operation, so that a
 *               single-level mesh is bitwise identical to
 *               anuga.rectangular_cross (the build uses
 *               -ffp-contract=off).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *               Geometry and connectivity conventions derived from ANUGA,
 *               (C) 2004-2015 Australian National University and
 *               Geoscience Australia, Apache License 2.0 (LICENSE.ANUGA).
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "hashmap.h"
#include "mesh.h"

/* Sides of a leaf in ANUGA's rectangular_cross triangle order. */
enum { SIDE_W, SIDE_S, SIDE_E, SIDE_N };

struct builder {
    struct mesh *m;
    const struct quadtree *qt;
    const struct raster_grid *dem;
    struct hashmap nodes; /* doubled finest-grid key -> node index */
    long node_capacity;
    int64_t unit; /* Doubled finest-grid units per level-0 cell. */
};

static uint64_t node_key(int64_t gx, int64_t gy)
{
    return ((uint64_t)gx << 32) | (uint64_t)gy;
}

/* Find or create the node at doubled finest-grid coordinates (gx, gy),
 * with relative coordinates (x, y) and elevation from the DEM (bilinear)
 * or the given value when finite. */
static int32_t get_node(struct builder *b, int64_t gx, int64_t gy, double x,
                        double y, double elevation)
{
    struct mesh *m = b->m;
    uint64_t key = node_key(gx, gy);
    int64_t idx = hashmap_get(&b->nodes, key);

    if (idx >= 0)
        return (int32_t)idx;

    idx = m->n_nodes++;
    if (idx >= b->node_capacity)
        G_fatal_error("Internal error: node capacity exceeded");
    hashmap_put_new(&b->nodes, key, idx);
    m->node_xy[2 * idx] = x;
    m->node_xy[2 * idx + 1] = y;
    m->node_key[idx] = (int64_t)key;
    if (!isfinite(elevation))
        elevation =
            raster_grid_bilinear(b->dem, m->origin_x + x, m->origin_y + y);
    if (!isfinite(elevation))
        G_fatal_error(_("No elevation available at mesh node (%.3f, %.3f)"),
                      m->origin_x + x, m->origin_y + y);
    m->node_elevation[idx] = elevation;

    return (int32_t)idx;
}

/* Emit the fan of triangles of one leaf: walk the perimeter clockwise, W
 * side (south to north), S side (east to west), E side (north to south),
 * N side (west to east), and emit [segment start, centre, segment end] per
 * segment. Without hanging nodes this is exactly rectangular_cross's
 * left, bottom, right and top triangles; edge 1 (opposite the centre) is
 * always the leaf's outer edge. hanging is a bit mask (1 << side) of sides
 * split by a finer neighbour's corner. */
static void emit_leaf(struct builder *b, long leaf_index, int hanging)
{
    struct mesh *m = b->m;
    const struct leaf *lf = &b->qt->leaves[leaf_index];
    double size = quadtree_leaf_size(b->qt, lf);
    int64_t span = b->unit >> lf->level; /* Doubled units per leaf side. */
    int64_t gx0 = (int64_t)lf->ix * span, gy0 = (int64_t)lf->iy * span;
    double x0 = lf->ix * size, x1 = (lf->ix + 1) * size;
    double y0 = lf->iy * size, y1 = (lf->iy + 1) * size;
    int32_t bl, tl, tr, br, c, corners[4][2];
    double cx, cy, zc;
    long count;
    int side;

    bl = get_node(b, gx0, gy0, x0, y0, NAN);
    tl = get_node(b, gx0, gy0 + span, x0, y1, NAN);
    tr = get_node(b, gx0 + span, gy0 + span, x1, y1, NAN);
    br = get_node(b, gx0 + span, gy0, x1, y0, NAN);

    /* Centre as in rectangular_cross_construct (mesh_factory_ext.pyx):
     * (i + 0.5) * delta, not the mean of the corners, which can differ in
     * the last bit. */
    cx = (lf->ix + 0.5) * size;
    cy = (lf->iy + 0.5) * size;
    zc = raster_grid_box_mean(b->dem, m->origin_x + x0, m->origin_y + y0,
                              m->origin_x + x1, m->origin_y + y1, &count);
    c = get_node(b, gx0 + span / 2, gy0 + span / 2, cx, cy,
                 count > 0 ? zc : NAN);

    /* Clockwise perimeter segments per side. */
    corners[SIDE_W][0] = bl;
    corners[SIDE_W][1] = tl;
    corners[SIDE_S][0] = br;
    corners[SIDE_S][1] = bl;
    corners[SIDE_E][0] = tr;
    corners[SIDE_E][1] = br;
    corners[SIDE_N][0] = tl;
    corners[SIDE_N][1] = tr;

    for (side = SIDE_W; side <= SIDE_N; side++) {
        int32_t seg[3];
        int n_seg = 1, s;

        seg[0] = corners[side][0];
        if (hanging & (1 << side)) {
            int64_t mx, my;
            double hx, hy;

            switch (side) {
            case SIDE_W:
                mx = gx0, my = gy0 + span / 2, hx = x0, hy = (y0 + y1) / 2;
                break;
            case SIDE_S:
                mx = gx0 + span / 2, my = gy0, hx = (x0 + x1) / 2, hy = y0;
                break;
            case SIDE_E:
                mx = gx0 + span, my = gy0 + span / 2, hx = x1,
                hy = (y0 + y1) / 2;
                break;
            default:
                mx = gx0 + span / 2, my = gy0 + span, hx = (x0 + x1) / 2,
                hy = y1;
                break;
            }
            seg[1] = get_node(b, mx, my, hx, hy, NAN);
            n_seg = 2;
        }
        seg[n_seg] = corners[side][1];

        for (s = 0; s < n_seg; s++) {
            long k = m->n_tri++;

            m->triangles[3 * k] = seg[s];
            m->triangles[3 * k + 1] = c;
            m->triangles[3 * k + 2] = seg[s + 1];
            m->tri_leaf[k] = (int32_t)leaf_index;
        }
    }
}

/* Per-triangle geometry, following General_mesh.__init__. */
static void compute_geometry(struct mesh *m)
{
    long k;

    for (k = 0; k < m->n_tri; k++) {
        const int32_t *t = m->triangles + 3 * k;
        double x0 = m->node_xy[2 * t[0]], y0 = m->node_xy[2 * t[0] + 1];
        double x1 = m->node_xy[2 * t[1]], y1 = m->node_xy[2 * t[1] + 1];
        double x2 = m->node_xy[2 * t[2]], y2 = m->node_xy[2 * t[2] + 1];
        double xn0, yn0, l0, xn1, yn1, l1, xn2, yn2, l2;
        double xc, yc, xm0, ym0, xm1, ym1, xm2, ym2, d0, d1, d2;
        double *v = m->vertex_coordinates + 6 * k;
        double *e = m->edge_coordinates + 6 * k;
        double *n = m->normals + 6 * k;

        v[0] = x0, v[1] = y0, v[2] = x1, v[3] = y1, v[4] = x2, v[5] = y2;

        m->areas[k] =
            -((x1 * y0 - x0 * y1) + (x2 * y1 - x1 * y2) + (x0 * y2 - x2 * y0)) /
            2.0;
        if (!(m->areas[k] > 0.0))
            G_fatal_error("Internal error: degenerate or clockwise triangle "
                          "%ld",
                          k);

        xn0 = x2 - x1;
        yn0 = y2 - y1;
        l0 = sqrt(xn0 * xn0 + yn0 * yn0);
        xn0 /= l0;
        yn0 /= l0;
        xn1 = x0 - x2;
        yn1 = y0 - y2;
        l1 = sqrt(xn1 * xn1 + yn1 * yn1);
        xn1 /= l1;
        yn1 /= l1;
        xn2 = x1 - x0;
        yn2 = y1 - y0;
        l2 = sqrt(xn2 * xn2 + yn2 * yn2);
        xn2 /= l2;
        yn2 /= l2;

        n[0] = yn0, n[1] = -xn0;
        n[2] = yn1, n[3] = -xn1;
        n[4] = yn2, n[5] = -xn2;
        m->edgelengths[3 * k] = l0;
        m->edgelengths[3 * k + 1] = l1;
        m->edgelengths[3 * k + 2] = l2;

        xc = (x0 + x1 + x2) / 3;
        yc = (y0 + y1 + y2) / 3;
        m->centroid_coordinates[2 * k] = xc;
        m->centroid_coordinates[2 * k + 1] = yc;

        /* compute_edge_midpoint_coordinates: 0.5 * (V1 + V2) etc. */
        e[0] = 0.5 * (x1 + x2), e[1] = 0.5 * (y1 + y2);
        e[2] = 0.5 * (x2 + x0), e[3] = 0.5 * (y2 + y0);
        e[4] = 0.5 * (x0 + x1), e[5] = 0.5 * (y0 + y1);

        /* Radius: distance from the centroid to the nearest edge
         * midpoint (use_inscribed_circle=False, ANUGA's default). */
        xm0 = (x1 + x2) / 2, ym0 = (y1 + y2) / 2;
        xm1 = (x2 + x0) / 2, ym1 = (y2 + y0) / 2;
        xm2 = (x0 + x1) / 2, ym2 = (y0 + y1) / 2;
        d0 = sqrt((xc - xm0) * (xc - xm0) + (yc - ym0) * (yc - ym0));
        d1 = sqrt((xc - xm1) * (xc - xm1) + (yc - ym1) * (yc - ym1));
        d2 = sqrt((xc - xm2) * (xc - xm2) + (yc - ym2) * (yc - ym2));
        m->radii[k] = fmin(fmin(d0, d1), d2);
    }
}

/* Edge i of a triangle joins its vertices (i+1)%3 and (i+2)%3. */
static uint64_t edge_key(const struct mesh *m, long k, int i)
{
    uint64_t a = (uint64_t)m->triangles[3 * k + (i + 1) % 3];
    uint64_t b = (uint64_t)m->triangles[3 * k + (i + 2) % 3];

    return a < b ? (a << 32) | b : (b << 32) | a;
}

static enum boundary_tag edge_tag(const struct mesh *m, const struct builder *b,
                                  long k, int i)
{
    int64_t ka = m->node_key[m->triangles[3 * k + (i + 1) % 3]];
    int64_t kb = m->node_key[m->triangles[3 * k + (i + 2) % 3]];
    int64_t xa = (uint64_t)ka >> 32, ya = ka & 0xFFFFFFFF;
    int64_t xb = (uint64_t)kb >> 32, yb = kb & 0xFFFFFFFF;
    int64_t xmax = (int64_t)b->qt->nx0 * b->unit;
    int64_t ymax = (int64_t)b->qt->ny0 * b->unit;

    if (xa == 0 && xb == 0)
        return TAG_WEST;
    if (xa == xmax && xb == xmax)
        return TAG_EAST;
    if (ya == 0 && yb == 0)
        return TAG_SOUTH;
    if (ya == ymax && yb == ymax)
        return TAG_NORTH;

    return TAG_NULL;
}

/* Neighbours by matching shared edges (as build_neighbour_structure),
 * then ANUGA's boundary enumeration in sorted (triangle, edge) order and
 * the surrogate neighbours. */
static void compute_connectivity(struct mesh *m, const struct builder *b)
{
    struct hashmap edges;
    long k, n_bnd = 0;
    int i;

    for (k = 0; k < 3 * m->n_tri; k++) {
        m->neighbours[k] = -1;
        m->neighbour_edges[k] = -1;
    }

    hashmap_init(&edges, 3 * m->n_tri);
    for (k = 0; k < m->n_tri; k++)
        for (i = 0; i < 3; i++) {
            uint64_t key = edge_key(m, k, i);
            int64_t other = hashmap_put_new(&edges, key, 3 * k + i);

            if (other != 3 * k + i) {
                long ko = other / 3;
                int io = other % 3;

                if (m->neighbours[3 * ko + io] >= 0)
                    G_fatal_error("Internal error: edge shared by more than "
                                  "two triangles");
                m->neighbours[3 * k + i] = (int32_t)ko;
                m->neighbour_edges[3 * k + i] = io;
                m->neighbours[3 * ko + io] = (int32_t)k;
                m->neighbour_edges[3 * ko + io] = i;
            }
        }
    hashmap_free(&edges);

    for (k = 0; k < 3 * m->n_tri; k++)
        if (m->neighbours[k] < 0)
            n_bnd++;
    m->n_boundary = n_bnd;
    m->boundary_tri = G_malloc((n_bnd ? n_bnd : 1) * sizeof(int32_t));
    m->boundary_edge = G_malloc((n_bnd ? n_bnd : 1) * sizeof(int32_t));
    m->boundary_tag = G_malloc((n_bnd ? n_bnd : 1) * sizeof(uint8_t));

    n_bnd = 0;
    for (k = 0; k < m->n_tri; k++) {
        m->number_of_boundaries[k] = 0;
        m->tri_full_flag[k] = 1;
        for (i = 0; i < 3; i++) {
            if (m->neighbours[3 * k + i] >= 0) {
                m->surrogate_neighbours[3 * k + i] = m->neighbours[3 * k + i];
                continue;
            }
            m->boundary_tri[n_bnd] = (int32_t)k;
            m->boundary_edge[n_bnd] = i;
            m->boundary_tag[n_bnd] = (uint8_t)edge_tag(m, b, k, i);
            m->neighbours[3 * k + i] = (int32_t)(-(n_bnd + 1));
            m->surrogate_neighbours[3 * k + i] = (int32_t)k;
            m->number_of_boundaries[k]++;
            n_bnd++;
        }
    }
}

/* Centroid bed as ANUGA's _interpolate: (q0 + q1 + q2) / 3.0 from the
 * vertex values, then the scaled integer encoding of section 4.7. */
static void compute_bed(struct mesh *m)
{
    double z_min = INFINITY, z_max = -INFINITY;
    long k;

    for (k = 0; k < m->n_tri; k++) {
        const int32_t *t = m->triangles + 3 * k;
        double q0 = m->node_elevation[t[0]];
        double q1 = m->node_elevation[t[1]];
        double q2 = m->node_elevation[t[2]];
        double zc = (q0 + q1 + q2) / 3.0;

        m->bed_centroid_f64[k] = zc;
        z_min = fmin(z_min, zc);
        z_max = fmax(z_max, zc);
    }

    m->z0 = (int64_t)floor(z_min * BED_SCALE) - BED_MARGIN;
    if ((z_max * BED_SCALE - (double)m->z0) >= 4294967295.0)
        G_fatal_error(_("Elevation range %.4f .. %.4f m exceeds the scaled "
                        "integer range (429496 m)"),
                      z_min, z_max);

    m->max_quantization_error = 0.0;
    for (k = 0; k < m->n_tri; k++) {
        int64_t q =
            (int64_t)llround(m->bed_centroid_f64[k] * BED_SCALE) - m->z0;
        double err;

        if (q < 0 || q > (int64_t)UINT32_MAX)
            G_fatal_error("Internal error: scaled bed out of range at "
                          "triangle %ld",
                          k);
        m->zq[k] = (uint32_t)q;
        err = fabs(bed_dequantize(m->zq[k], m->z0) - m->bed_centroid_f64[k]);
        if (err > m->max_quantization_error)
            m->max_quantization_error = err;
    }
}

void mesh_build(struct mesh *m, const struct quadtree *qt,
                const struct raster_grid *dem)
{
    struct builder b;
    int64_t gmax;
    long max_tri, k;

    memset(m, 0, sizeof(*m));
    m->origin_x = qt->west;
    m->origin_y = qt->south;

    b.m = m;
    b.qt = qt;
    b.dem = dem;
    b.unit = (int64_t)2 << (qt->n_levels - 1);
    gmax = (int64_t)(qt->nx0 > qt->ny0 ? qt->nx0 : qt->ny0) * b.unit;
    if (gmax >= ((int64_t)1 << 31))
        G_fatal_error(_("Mesh too large: %lld finest cells across"),
                      (long long)(gmax / 2));

    /* Phase 1: no hanging nodes, 4 triangles and 1 centre per leaf; the
     * corners are shared, so nodes <= 4 per leaf plus the boundary row. */
    max_tri = 4 * qt->n_leaves;
    b.node_capacity =
        2 * qt->n_leaves + qt->nx0 + qt->ny0 + 2 + 2 * qt->n_leaves;
    hashmap_init(&b.nodes, b.node_capacity);

    m->node_xy = G_malloc(2 * b.node_capacity * sizeof(double));
    m->node_key = G_malloc(b.node_capacity * sizeof(int64_t));
    m->node_elevation = G_malloc(b.node_capacity * sizeof(double));
    m->triangles = G_malloc(3 * max_tri * sizeof(int32_t));
    m->tri_leaf = G_malloc(max_tri * sizeof(int32_t));

    G_message(_("Building mesh triangles..."));
    for (k = 0; k < qt->n_leaves; k++) {
        G_percent(k, qt->n_leaves, 5);
        emit_leaf(&b, k, 0);
    }
    G_percent(1, 1, 1);
    hashmap_free(&b.nodes);

    m->vertex_coordinates = G_malloc(6 * m->n_tri * sizeof(double));
    m->edge_coordinates = G_malloc(6 * m->n_tri * sizeof(double));
    m->centroid_coordinates = G_malloc(2 * m->n_tri * sizeof(double));
    m->normals = G_malloc(6 * m->n_tri * sizeof(double));
    m->edgelengths = G_malloc(3 * m->n_tri * sizeof(double));
    m->areas = G_malloc(m->n_tri * sizeof(double));
    m->radii = G_malloc(m->n_tri * sizeof(double));
    m->neighbours = G_malloc(3 * m->n_tri * sizeof(int32_t));
    m->neighbour_edges = G_malloc(3 * m->n_tri * sizeof(int32_t));
    m->surrogate_neighbours = G_malloc(3 * m->n_tri * sizeof(int32_t));
    m->number_of_boundaries = G_malloc(m->n_tri * sizeof(int32_t));
    m->tri_full_flag = G_malloc(m->n_tri * sizeof(int32_t));
    m->bed_centroid_f64 = G_malloc(m->n_tri * sizeof(double));
    m->zq = G_malloc(m->n_tri * sizeof(uint32_t));

    G_message(_("Computing mesh geometry and connectivity..."));
    compute_geometry(m);
    compute_connectivity(m, &b);
    compute_bed(m);
}

void mesh_free(struct mesh *m)
{
    void *arrays[] = {m->node_xy,
                      m->node_key,
                      m->node_elevation,
                      m->triangles,
                      m->tri_leaf,
                      m->vertex_coordinates,
                      m->edge_coordinates,
                      m->centroid_coordinates,
                      m->normals,
                      m->edgelengths,
                      m->areas,
                      m->radii,
                      m->neighbours,
                      m->neighbour_edges,
                      m->surrogate_neighbours,
                      m->number_of_boundaries,
                      m->tri_full_flag,
                      m->boundary_tri,
                      m->boundary_edge,
                      m->boundary_tag,
                      m->bed_centroid_f64,
                      m->zq};
    size_t i;

    for (i = 0; i < sizeof(arrays) / sizeof(arrays[0]); i++)
        G_free(arrays[i]);
    memset(m, 0, sizeof(*m));
}

/* Export: raw arrays plus a JSON manifest. */

struct manifest {
    FILE *fp;
    const char *dir;
    int first;
};

static void write_array(struct manifest *mf, const char *name, const void *data,
                        size_t elem_size, long rows, int cols,
                        const char *dtype)
{
    char path[GPATH_MAX];
    FILE *fp;

    snprintf(path, sizeof(path), "%s/%s.bin", mf->dir, name);
    fp = fopen(path, "wb");
    if (!fp)
        G_fatal_error(_("Unable to write <%s>: %s"), path, strerror(errno));
    if (rows > 0 && fwrite(data, elem_size * cols, rows, fp) != (size_t)rows)
        G_fatal_error(_("Error writing <%s>"), path);
    fclose(fp);

    fprintf(mf->fp,
            "%s    \"%s\": {\"file\": \"%s.bin\", \"dtype\": \"%s\", "
            "\"shape\": [%ld, %d]}",
            mf->first ? "" : ",\n", name, name, dtype, rows, cols);
    mf->first = 0;
}

void mesh_export(const struct mesh *m, const struct quadtree *qt,
                 const char *dir)
{
    char path[GPATH_MAX];
    struct manifest mf;
    int32_t *leaf_data;
    long k;
    union {
        uint16_t u;
        uint8_t b[2];
    } endian = {1};

    if (G_mkdir(dir) != 0 && errno != EEXIST)
        G_fatal_error(_("Unable to create directory <%s>: %s"), dir,
                      strerror(errno));

    snprintf(path, sizeof(path), "%s/manifest.json", dir);
    mf.fp = fopen(path, "w");
    if (!mf.fp)
        G_fatal_error(_("Unable to write <%s>: %s"), path, strerror(errno));
    mf.dir = dir;
    mf.first = 1;

    fprintf(mf.fp,
            "{\n  \"format\": \"r.hydro.anuga mesh 1\",\n"
            "  \"byteorder\": \"%s\",\n"
            "  \"origin\": [%.17g, %.17g],\n"
            "  \"res_max\": %.17g,\n  \"n_levels\": %d,\n"
            "  \"nx0\": %d,\n  \"ny0\": %d,\n"
            "  \"bed_z0\": %lld,\n  \"bed_scale\": %.17g,\n"
            "  \"max_quantization_error\": %.17g,\n"
            "  \"boundary_tags\": [\"north\", \"south\", \"east\", \"west\", "
            "\"null\"],\n  \"arrays\": {\n",
            endian.b[0] ? "little" : "big", m->origin_x, m->origin_y,
            qt->res_max, qt->n_levels, qt->nx0, qt->ny0, (long long)m->z0,
            BED_SCALE, m->max_quantization_error);

    write_array(&mf, "nodes", m->node_xy, sizeof(double), m->n_nodes, 2,
                "float64");
    write_array(&mf, "node_elevation", m->node_elevation, sizeof(double),
                m->n_nodes, 1, "float64");
    write_array(&mf, "triangles", m->triangles, sizeof(int32_t), m->n_tri, 3,
                "int32");
    write_array(&mf, "tri_leaf", m->tri_leaf, sizeof(int32_t), m->n_tri, 1,
                "int32");
    write_array(&mf, "vertex_coordinates", m->vertex_coordinates,
                sizeof(double), m->n_tri, 6, "float64");
    write_array(&mf, "edge_coordinates", m->edge_coordinates, sizeof(double),
                m->n_tri, 6, "float64");
    write_array(&mf, "centroid_coordinates", m->centroid_coordinates,
                sizeof(double), m->n_tri, 2, "float64");
    write_array(&mf, "normals", m->normals, sizeof(double), m->n_tri, 6,
                "float64");
    write_array(&mf, "edgelengths", m->edgelengths, sizeof(double), m->n_tri, 3,
                "float64");
    write_array(&mf, "areas", m->areas, sizeof(double), m->n_tri, 1, "float64");
    write_array(&mf, "radii", m->radii, sizeof(double), m->n_tri, 1, "float64");
    write_array(&mf, "neighbours", m->neighbours, sizeof(int32_t), m->n_tri, 3,
                "int32");
    write_array(&mf, "neighbour_edges", m->neighbour_edges, sizeof(int32_t),
                m->n_tri, 3, "int32");
    write_array(&mf, "surrogate_neighbours", m->surrogate_neighbours,
                sizeof(int32_t), m->n_tri, 3, "int32");
    write_array(&mf, "number_of_boundaries", m->number_of_boundaries,
                sizeof(int32_t), m->n_tri, 1, "int32");
    write_array(&mf, "boundary_tri", m->boundary_tri, sizeof(int32_t),
                m->n_boundary, 1, "int32");
    write_array(&mf, "boundary_edge", m->boundary_edge, sizeof(int32_t),
                m->n_boundary, 1, "int32");
    write_array(&mf, "boundary_tag", m->boundary_tag, sizeof(uint8_t),
                m->n_boundary, 1, "uint8");
    write_array(&mf, "bed_centroid_f64", m->bed_centroid_f64, sizeof(double),
                m->n_tri, 1, "float64");
    write_array(&mf, "bed_zq", m->zq, sizeof(uint32_t), m->n_tri, 1, "uint32");

    leaf_data = G_malloc(3 * qt->n_leaves * sizeof(int32_t));
    for (k = 0; k < qt->n_leaves; k++) {
        leaf_data[3 * k] = qt->leaves[k].level;
        leaf_data[3 * k + 1] = qt->leaves[k].ix;
        leaf_data[3 * k + 2] = qt->leaves[k].iy;
    }
    write_array(&mf, "leaves", leaf_data, sizeof(int32_t), qt->n_leaves, 3,
                "int32");
    G_free(leaf_data);

    fprintf(mf.fp, "\n  }\n}\n");
    fclose(mf.fp);
    G_message(_("Mesh exported to <%s>"), dir);
}
