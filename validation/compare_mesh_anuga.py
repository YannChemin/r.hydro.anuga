"""Compare an r.hydro.anuga mesh export with ANUGA's own rectangular_cross.

Run with a Python environment where ANUGA is importable:

    .venv-anuga/bin/python validation/compare_mesh_anuga.py MESH_DIR

The export must be a single-level mesh over a full rectangle (no NULL
cells). ANUGA builds the same mesh with anuga.rectangular_cross and
anuga.Domain; triangles are matched by their centroid coordinates, and
every geometry and connectivity array must be bitwise identical. The bed
is checked by giving ANUGA r.hydro.anuga's vertex elevations and requiring
ANUGA's centroid values to equal the exported fp64 centroid bed. Prints
a JSON summary and exits 1 on any mismatch.
"""

import json
import sys
from pathlib import Path

import numpy as np


def load_export(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["byteorder"] != sys.byteorder:
        msg = "Mesh export byte order differs from this machine"
        raise SystemExit(msg)
    arrays = {}
    for name, info in manifest["arrays"].items():
        data = np.fromfile(directory / info["file"], dtype=info["dtype"])
        rows, cols = info["shape"]
        arrays[name] = data.reshape(rows, cols) if cols > 1 else data
    return manifest, arrays


def main(directory):
    import anuga

    manifest, ours = load_export(directory)
    if manifest["n_levels"] != 1:
        msg = "Only single-level meshes can be compared with rectangular_cross"
        raise SystemExit(msg)
    nx, ny, res = manifest["nx0"], manifest["ny0"], manifest["res_max"]
    n_tri = ours["triangles"].shape[0]
    if n_tri != 4 * nx * ny:
        msg = "The export does not cover the full rectangle (NULL cells?)"
        raise SystemExit(msg)

    points, elements, boundary = anuga.rectangular_cross(
        nx, ny, len1=nx * res, len2=ny * res
    )
    domain = anuga.Domain(points, elements, boundary)
    domain.set_flow_algorithm("DE1")

    # Match triangles by exact centroid coordinates.
    theirs_c = domain.centroid_coordinates
    index = {tuple(c.tobytes() for c in row): k for k, row in enumerate(theirs_c)}
    perm = np.array(
        [
            index.get(tuple(c.tobytes() for c in row), -1)
            for row in ours["centroid_coordinates"]
        ]
    )
    results = {"triangles": int(n_tri), "unmatched_centroids": int((perm < 0).sum())}
    if results["unmatched_centroids"]:
        print(json.dumps(results, indent=2))
        return 1

    def same(name, a, b):
        equal = bool(np.array_equal(a, b))
        results[name] = "identical" if equal else "DIFFERENT"
        if not equal:
            diff = np.abs(np.asarray(a, float) - np.asarray(b, float))
            results[name + "_max_abs_diff"] = float(np.nanmax(diff))
        return equal

    ok = True
    ok &= same(
        "triangle_vertices",
        points[elements[perm]],
        ours["nodes"][ours["triangles"]],
    )
    ok &= same(
        "vertex_coordinates",
        domain.get_vertex_coordinates().reshape(-1, 6)[perm],
        ours["vertex_coordinates"],
    )
    ok &= same(
        "edge_coordinates",
        domain.get_edge_midpoint_coordinates().reshape(-1, 6)[perm],
        ours["edge_coordinates"],
    )
    ok &= same("normals", domain.normals[perm], ours["normals"])
    ok &= same("edgelengths", domain.edgelengths[perm], ours["edgelengths"])
    ok &= same("areas", domain.areas[perm], ours["areas"])
    ok &= same("radii", domain.radii[perm], ours["radii"])

    # Connectivity, mapped through the permutation. Boundary edges are
    # negative in both; their enumeration order depends on triangle order.
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n_tri)
    their_nb = domain.neighbours[perm]
    our_nb = ours["neighbours"]
    interior = our_nb >= 0
    ok &= same("boundary_edge_positions", their_nb < 0, our_nb < 0)
    ok &= same(
        "neighbours",
        inv[their_nb[interior]],
        our_nb[interior],
    )
    ok &= same(
        "neighbour_edges",
        domain.neighbour_edges[perm][interior],
        ours["neighbour_edges"][interior],
    )
    ok &= same(
        "surrogate_neighbours",
        inv[domain.surrogate_neighbours[perm]],
        ours["surrogate_neighbours"],
    )
    ok &= same(
        "number_of_boundaries",
        domain.number_of_boundaries[perm],
        ours["number_of_boundaries"],
    )

    tag_names = {"left": "west", "right": "east", "bottom": "south", "top": "north"}
    their_tags = sorted(
        (int(inv[k]), int(e), tag_names[t]) for (k, e), t in domain.boundary.items()
    )
    our_tags = sorted(
        (int(k), int(e), manifest["boundary_tags"][int(t)])
        for k, e, t in zip(
            ours["boundary_tri"], ours["boundary_edge"], ours["boundary_tag"]
        )
    )
    results["boundary_tags"] = "identical" if their_tags == our_tags else "DIFFERENT"
    ok &= their_tags == our_tags

    # Bed: ANUGA computes centroids from our vertex values.
    vertex_z = np.empty((n_tri, 3))
    vertex_z[perm] = ours["node_elevation"][ours["triangles"]]
    domain.set_quantity("elevation", vertex_z, location="vertices")
    ok &= same(
        "bed_centroid",
        domain.quantities["elevation"].centroid_values[perm],
        ours["bed_centroid_f64"],
    )

    # Scaled integer bed: exact dequantisation within half a unit.
    z0 = manifest["bed_z0"]
    dequantized = (ours["bed_zq"].astype(np.int64) + z0).astype(np.float64) * 1.0e-4
    err = float(np.max(np.abs(dequantized - ours["bed_centroid_f64"])))
    results["bed_quantization_max_error_m"] = err
    results["bed_quantization_within_half_unit"] = err <= 0.5e-4 * (1 + 1e-9)
    ok &= results["bed_quantization_within_half_unit"]

    results["all_identical"] = bool(ok)
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
