"""Compare an r.hydro.anuga solver run with ANUGA's "unified" compute mode.

Run with a Python environment where ANUGA is importable:

    .venv-anuga/bin/python validation/compare_state_anuga.py MESH_DIR STATE_DIR

MESH_DIR and STATE_DIR are the mesh_output= and state_output= exports of
the same r.hydro.anuga run on a single-level mesh over a full rectangle.
ANUGA rebuilds the mesh with anuga.rectangular_cross, receives exactly the
same centroid bed (the dequantised scaled bed), initial state, friction,
boundary conditions and CFL. Its compute mode 'unified' step functions
(the C kernels this module ports) are then called with the same
algorithm, output step and duration, using the time capping of
Generic_Domain.evolve() (see the comment in main() on why evolve() itself
is not used). The final centroid stage, xmomentum and ymomentum must be bitwise
identical, and so must the number of time steps. Prints a JSON summary and
exits 1 on any mismatch.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mesh_anuga import load_export  # noqa: E402

TAG_OF_SIDE = {"west": "left", "east": "right", "south": "bottom", "north": "top"}


def main(mesh_dir, state_dir):
    import anuga

    mesh_manifest, mesh = load_export(mesh_dir)
    state_manifest, state = load_export(state_dir)
    nx, ny = mesh_manifest["nx0"], mesh_manifest["ny0"]
    res = mesh_manifest["res_max"]
    n_tri = mesh["triangles"].shape[0]
    if mesh_manifest["n_levels"] != 1 or n_tri != 4 * nx * ny:
        msg = "Only single-level meshes over a full rectangle can be compared"
        raise SystemExit(msg)

    points, elements, boundary = anuga.rectangular_cross(
        nx, ny, len1=nx * res, len2=ny * res
    )
    domain = anuga.Domain(points, elements, boundary)
    domain.set_flow_algorithm(state_manifest["algorithm"])
    domain.set_cfl(state_manifest["cfl"])
    domain.set_name("compare_state")
    domain.set_store(False)
    domain.set_quantities_to_be_stored(None)

    theirs_c = domain.centroid_coordinates
    index = {tuple(c.tobytes() for c in row): k for k, row in enumerate(theirs_c)}
    perm = np.array(
        [
            index.get(tuple(c.tobytes() for c in row), -1)
            for row in mesh["centroid_coordinates"]
        ]
    )
    if (perm < 0).any():
        msg = "Triangle centroids do not match rectangular_cross"
        raise SystemExit(msg)

    def to_anuga(values):
        out = np.empty(n_tri)
        out[perm] = values
        return out

    domain.set_quantity("elevation", to_anuga(state["bed"]), location="centroids")
    domain.set_quantity("friction", to_anuga(state["friction"]), location="centroids")
    domain.set_quantity("stage", to_anuga(state["initial_stage"]), location="centroids")
    domain.set_quantity(
        "xmomentum", to_anuga(state["initial_xmom"]), location="centroids"
    )
    domain.set_quantity(
        "ymomentum", to_anuga(state["initial_ymom"]), location="centroids"
    )

    # One boundary object per side, from the exported per-edge setup.
    types = state_manifest["boundary_types"]
    tags = mesh_manifest["boundary_tags"]
    per_side = {}
    for j, tag_index in enumerate(mesh["boundary_tag"]):
        side = tags[int(tag_index)]
        kind = types[int(state["boundary_type"][j])]
        value = tuple(float(v) for v in state["boundary_value"][j])
        previous = per_side.setdefault(side, (kind, value))
        if previous != (kind, value):
            msg = f"Boundary side {side} mixes several conditions"
            raise SystemExit(msg)
    boundaries = {}
    for side, (kind, value) in per_side.items():
        if kind == "reflective":
            boundaries[TAG_OF_SIDE[side]] = anuga.Reflective_boundary(domain)
        elif kind == "transmissive":
            boundaries[TAG_OF_SIDE[side]] = anuga.Transmissive_boundary(domain)
        else:
            boundaries[TAG_OF_SIDE[side]] = anuga.Dirichlet_boundary(list(value))
    domain.set_boundary(boundaries)

    # Read by init_gpu_domain when the unified interface is built.
    domain.use_sloped_mannings = bool(state_manifest["sloped_friction"])
    domain.set_compute_mode("unified")
    if domain.gpu_interface is None:
        msg = "ANUGA did not build its unified-mode interface"
        raise SystemExit(msg)
    gpu_dom = domain.gpu_interface.gpu_dom
    from anuga.shallow_water import sw_domain_gpu_ext as ext

    step = {
        "DE0": ext.evolve_one_euler_step_gpu,
        "DE1": ext.evolve_one_rk2_step_gpu,
        "DE2": ext.evolve_one_rk3_step_gpu,
    }[state_manifest["algorithm"]]

    # The unified-mode step functions, driven with the time capping of
    # Generic_Domain.evolve() but without its yield-time host code. In a
    # CPU build that code (protect_new and the openmp extrapolation, run by
    # distribute_to_vertices_and_edges at t = 0 and at every yield) writes
    # into the same arrays the C steps use, so ANUGA's CPU trajectory
    # depends on yieldstep; on a GPU it does not. r.hydro.anuga follows the
    # device trajectory.
    duration = state_manifest["duration"]
    yieldstep = state_manifest["yieldstep"]
    t, yield_t, n_steps = 0.0, yieldstep, 0
    while True:
        max_dt = domain.evolve_max_timestep
        max_dt = min(max_dt, max(duration - t, 0.0))
        max_dt = min(max_dt, max(yield_t - t, 0.0))
        t = t + step(gpu_dom, max_dt, 1)
        n_steps += 1
        if t >= duration - 1.0e-12:
            break
        if t >= yield_t:
            yield_t += yieldstep
    ext.sync_from_device(gpu_dom)
    results = {
        "triangles": int(n_tri),
        "compute_mode": domain.compute_mode,
        "c_rk_loop": bool(getattr(domain, "use_c_rk_loop", False)),
        "anuga_steps": int(n_steps),
        "module_steps": int(state_manifest["n_steps"]),
    }

    ok = results["compute_mode"] == "unified"
    ok &= results["anuga_steps"] == results["module_steps"]
    for name, quantity in (
        ("stage", "stage"),
        ("xmom", "xmomentum"),
        ("ymom", "ymomentum"),
    ):
        theirs = domain.quantities[quantity].centroid_values[perm]
        ours = state[name]
        equal = bool(np.array_equal(theirs, ours))
        results[name] = "identical" if equal else "DIFFERENT"
        if not equal:
            diff = np.abs(theirs - ours)
            results[name + "_max_abs_diff"] = float(diff.max())
            results[name + "_n_different"] = int((theirs != ours).sum())
        ok &= equal

    results["all_identical"] = bool(ok)
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
