"""Phase 5 tests: DEM stack, graded multi-level meshes and detail outputs
(PLAN.md sections 4.2-4.5 and 9: V2m, V4m, V5m, V11)."""

import subprocess

import numpy as np
import pytest

from grass.tools import ToolError

from mesh_test import REPO, anuga_python
from solver_test import run

BUMPS = "0.4 * sin(x() / 37) * cos(y() / 29) - 0.002 * x()"


def dual_dem(tools, fine_offset=0.0, ripple=0.05):
    """An 8 m DEM over 512 m x 256 m and a 1 m DEM over its centre."""
    tools.g_region(n=256, s=0, e=512, w=0, res=8)
    tools.r_mapcalc(expression=f"coarse = {BUMPS}")
    tools.g_region(n=192, s=64, e=320, w=192, res=1)
    tools.r_mapcalc(
        expression=f"fine = {BUMPS} + {ripple} * sin(x() / 3) + {fine_offset}"
    )
    tools.g_region(n=256, s=0, e=512, w=0, res=8)


def test_levels_and_fans(tools, tmp_path):
    dual_dem(tools)
    tools.r_mapcalc(expression="w0 = 1")
    (mm, mesh), *_ = run(
        tools, tmp_path, elevation="coarse,fine", initial_stage="w0", duration=1
    )
    levels = np.bincount(mesh["leaves"][:, 0])
    assert mm["n_levels"] == 4 and len(levels) == 4 and levels.min() > 0
    per_leaf = np.bincount(mesh["tri_leaf"])
    assert per_leaf.min() == 4 and per_leaf.max() >= 5 and per_leaf.max() <= 8
    # The fine footprint (128 m x 128 m) is entirely at 1 m.
    assert levels[3] >= 128 * 128


def test_lake_at_rest_across_levels(tools, tmp_path):
    """V2m: well-balanced across hanging nodes and level changes."""
    dual_dem(tools)
    tools.r_mapcalc(expression="w0 = 1")
    _, (m, st), *_ = run(
        tools,
        tmp_path,
        elevation="coarse,fine",
        initial_stage="w0",
        duration=60,
        output_step=60,
        manning_value=0.03,
    )
    assert m["n_steps"] > 100
    assert np.max(np.abs(st["xmom"])) < 1e-12
    assert np.max(np.abs(st["ymom"])) < 1e-12


def test_bitwise_equal_to_anuga_on_multilevel_mesh(tools, tmp_path):
    """V4m: ANUGA built from the exported multi-level mesh gives the same
    state, for a dam break flowing through all levels."""
    python = anuga_python()
    dual_dem(tools)
    tools.r_mapcalc(expression="h0 = if(x() < 150, 1.5, 0)")
    *_, mesh_dir, state_dir = run(
        tools,
        tmp_path,
        elevation="coarse,fine",
        initial_depth="h0",
        duration=60,
        output_step=30,
        manning_value=0.025,
        boundary="east:transmissive",
    )
    result = subprocess.run(
        [
            python,
            str(REPO / "validation" / "compare_state_anuga.py"),
            str(mesh_dir),
            str(state_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exported multi-level" in result.stdout


def test_fringe_reflection(tools, tmp_path):
    """V5m: a wave entering the finer levels is not reflected (< 1%)."""
    tools.g_region(n=64, s=0, e=1024, w=0, res=8)
    tools.r_mapcalc(expression="coarse = 0")
    tools.g_region(n=64, s=0, e=640, w=384, res=1)
    tools.r_mapcalc(expression="fine = 0")
    tools.g_region(n=64, s=0, e=1024, w=0, res=8)
    tools.r_mapcalc(expression="w0 = 2 + 0.1 * exp(-min(((x() - 250) / 25)^2, 50))")
    common = dict(
        initial_stage="w0",
        duration=60,
        output_step=60,
        manning_value=0,
        boundary="west:transmissive,east:transmissive",
    )
    (_, mm), (_, sm), *_ = run(
        tools, tmp_path, name="multi", elevation="coarse,fine", **common
    )
    (_, mc), (_, sc), *_ = run(
        tools, tmp_path, name="coarse", elevation="coarse", **common
    )

    def profile(mesh, st):
        x, a = mesh["centroid_coordinates"][:, 0], mesh["areas"]
        eta = st["stage"] - 2.0
        edges = np.arange(0, 1025, 8)
        return edges, np.array(
            [
                np.sum((eta * a)[(x >= p) & (x < q)]) / np.sum(a[(x >= p) & (x < q)])
                for p, q in zip(edges[:-1], edges[1:])
            ]
        )

    edges, multi = profile(mm, sm)
    _, coarse = profile(mc, sc)
    upstream = (edges[:-1] >= 100) & (edges[1:] <= 330)
    assert np.max(np.abs(multi - coarse)[upstream]) < 0.01 * 0.1


def test_vertical_bias_check(tools, tmp_path):
    """V11: a 5 m datum offset is refused unless given explicitly."""
    dual_dem(tools, fine_offset=5.0, ripple=0.0)
    tools.r_mapcalc(expression="w0 = 1")
    with pytest.raises(ToolError, match=r"median vertical difference"):
        run(tools, tmp_path, elevation="coarse,fine", initial_stage="w0", duration=1)
    (_, mesh), *_ = run(
        tools,
        tmp_path,
        name="offset",
        elevation="coarse,fine",
        dem_offset="0,-5",
        initial_stage="w0",
        duration=1,
    )
    assert np.max(mesh["node_elevation"]) < 1.0


def test_seam_blending(tools, tmp_path):
    """Within tolerance, the seam between DEMs is blended, not a step."""
    dual_dem(tools, fine_offset=0.3, ripple=0.0)
    tools.r_mapcalc(expression="w0 = 2")
    steps = {}
    for width in (0, 4):
        (_, mesh), *_ = run(
            tools,
            tmp_path,
            name=f"b{width}",
            elevation="coarse,fine",
            blend_width=width,
            initial_stage="w0",
            duration=1,
        )
        z = mesh["node_elevation"]
        tri = mesh["triangles"]
        # Largest elevation difference along any triangle edge.
        steps[width] = max(
            np.max(np.abs(z[tri[:, i]] - z[tri[:, (i + 1) % 3]])) for i in range(3)
        )
    assert steps[4] < 0.5 * steps[0]


def test_refine_area(tools, tmp_path):
    tools.g_region(n=256, s=0, e=512, w=0, res=8)
    tools.r_mapcalc(expression=f"coarse = {BUMPS}")
    tools.r_mapcalc(expression="ref = if(x() > 400 && y() > 150, 1, null())")
    tools.r_mapcalc(expression="w0 = 1")
    (_, mesh), *_ = run(
        tools,
        tmp_path,
        elevation="coarse",
        res_min=2,
        refine="ref",
        refine_res=2,
        initial_stage="w0",
        duration=1,
    )
    levels = np.bincount(mesh["leaves"][:, 0])
    assert len(levels) == 3 and levels[2] > 0


def test_coarsen_not_implemented(tools, tmp_path):
    dual_dem(tools)
    with pytest.raises(ToolError, match="not implemented"):
        run(tools, tmp_path, elevation="coarse,fine", coarsen="coarse", duration=1)


def test_opencl_bitwise_on_multilevel_mesh(tools, tmp_path):
    from opencl_test import opencl_available

    if not opencl_available(tools):
        pytest.skip("No OpenCL device with double precision")
    dual_dem(tools)
    tools.r_mapcalc(expression="h0 = if(x() < 150, 1.5, 0)")
    common = dict(
        elevation="coarse,fine",
        initial_depth="h0",
        duration=30,
        output_step=30,
        manning_value=0,
        boundary="east:transmissive",
    )
    _, (_, a), *_ = run(tools, tmp_path, name="omp", **common)
    _, (_, b), *_ = run(tools, tmp_path, name="ocl", device="auto", **common)
    for q in ("stage", "xmom", "ymom"):
        assert np.array_equal(a[q], b[q]), q


def test_detail_outputs(tools):
    dual_dem(tools)
    tools.r_mapcalc(expression="h0 = if(x() < 150, 1.5, 0)")
    tools.r_hydro_anuga(
        elevation="coarse,fine",
        initial_depth="h0",
        duration=60,
        output_step=30,
        output="ml",
        max_depth="ml_maxd",
        flags="f",
        device="omp",
    )
    region = tools.r_info(map="ml_detail1_depth_002", flags="g").keyval
    assert float(region["nsres"]) == 1 and region["rows"] == 128
    assert region["cols"] == 128
    assert (
        tools.t_info(input="ml_detail1_depth", flags="g").keyval["number_of_maps"] == 3
    )
    main = tools.r_info(map="ml_depth_002", flags="g").keyval
    assert float(main["nsres"]) == 8
    # r.univar reads through the current region: use the detail grid's.
    tools.g_region(raster="ml_maxd_detail1")
    maxd = tools.r_univar(map="ml_maxd_detail1", format="json").json
    assert maxd["n"] == 128 * 128
    depth = tools.r_univar(map="ml_detail1_depth_002", format="json").json
    assert depth["n"] == 128 * 128
