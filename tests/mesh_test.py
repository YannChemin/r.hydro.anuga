"""Phase 1 tests: single-level mesh, elevation, scaled bed and mesh-only
outputs (PLAN.md sections 4.5, 4.7 and 10, validation V1q)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grass.tools import ToolError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "validation"))
from compare_mesh_anuga import load_export  # noqa: E402


def anuga_python():
    """Python interpreter with ANUGA, or skip."""
    candidate = os.environ.get("ANUGA_PYTHON") or str(
        REPO / ".venv-anuga" / "bin" / "python"
    )
    if not Path(candidate).exists():
        pytest.skip("No ANUGA Python environment (set ANUGA_PYTHON)")
    return candidate


def build_mesh(tools, tmp_path, **kwargs):
    out = tmp_path / "mesh"
    kwargs.setdefault("device", "omp")
    tools.r_hydro_anuga(mesh_output=str(out), **kwargs)
    return load_export(out)


@pytest.mark.parametrize(
    ("west", "south", "res", "cols", "rows"),
    [
        (0, 0, 30, 40, 30),
        (231000, 6729000, 30, 17, 11),
        (0, 0, 0.3, 24, 20),
    ],
)
def test_mesh_is_bitwise_anuga(tools, tmp_path, west, south, res, cols, rows):
    """Geometry, connectivity and centroid bed equal anuga.rectangular_cross."""
    python = anuga_python()
    tools.g_region(w=west, s=south, e=west + cols * res, n=south + rows * res, res=res)
    tools.r_mapcalc(
        expression="dem = 20 + 0.013 * x() + 0.021 * y()"
        " + 3 * sin(x() / 50) * cos(y() / 70)"
    )
    out = tmp_path / "mesh"
    tools.r_hydro_anuga(elevation="dem", mesh_output=str(out), device="omp")
    result = subprocess.run(
        [python, str(REPO / "validation" / "compare_mesh_anuga.py"), str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_node_elevation_rules(tools, tmp_path):
    """Corners: mean of the adjacent cells (bilinear between cell centres);
    centres: the cell value; centroid bed: mean of the triangle vertices."""
    tools.g_region(w=0, s=0, e=30, n=20, res=10)
    tools.r_mapcalc(expression="dem = row() * 10 + col()")
    manifest, mesh = build_mesh(tools, tmp_path, elevation="dem")
    values = np.array([[11, 12, 13], [21, 22, 23]], dtype=float)  # north first

    nodes, z = mesh["nodes"], mesh["node_elevation"]
    for (x, y), zn in zip(nodes, z):
        col_f, row_f = x / 10, (20 - y) / 10
        if col_f % 1 == 0.5:  # Leaf centre.
            assert zn == values[int(row_f), int(col_f)]
            continue
        cols = [c for c in (int(col_f) - 1, int(col_f)) if 0 <= c < 3]
        rows = [r for r in (int(row_f) - 1, int(row_f)) if 0 <= r < 2]
        expected = values[np.ix_(rows, cols)].mean()
        assert zn == pytest.approx(expected, abs=1e-12)

    tri_z = z[mesh["triangles"]]
    centroid = (tri_z[:, 0] + tri_z[:, 1] + tri_z[:, 2]) / 3.0
    assert np.array_equal(centroid, mesh["bed_centroid_f64"])


def test_null_cells_become_null_boundaries(tools, tmp_path):
    """A NULL hole removes its leaf; the edges facing it are tagged null."""
    tools.g_region(w=0, s=0, e=50, n=50, res=10)
    tools.r_mapcalc(expression="dem = if(row() == 3 && col() == 3, null(), 5)")
    manifest, mesh = build_mesh(tools, tmp_path, elevation="dem")
    tags = [manifest["boundary_tags"][t] for t in mesh["boundary_tag"]]

    assert mesh["triangles"].shape[0] == 4 * (25 - 1)
    assert tags.count("null") == 4
    for side in ("north", "south", "east", "west"):
        assert tags.count(side) == 5


def test_domain_mask_and_mesh_level(tools, tmp_path):
    """domain= restricts the active leaves; mesh_level= maps them."""
    tools.g_region(w=0, s=0, e=100, n=100, res=10)
    tools.r_mapcalc(expression="dem = 5")
    tools.r_mapcalc(expression="dom = if(col() <= 4, 1, null())")
    manifest, mesh = build_mesh(
        tools, tmp_path, elevation="dem", domain="dom", mesh_level="lvl"
    )
    assert mesh["triangles"].shape[0] == 4 * 40
    stats = tools.r_univar(map="lvl", format="json").json
    assert stats["n"] == 40
    assert stats["max"] == 0
    tags = [manifest["boundary_tags"][t] for t in mesh["boundary_tag"]]
    assert tags.count("null") == 10  # The eastern edge of the masked domain.


def test_negative_bed_and_datum(tools, tmp_path):
    """Negative elevations are represented through the datum."""
    tools.g_region(w=0, s=0, e=40, n=40, res=10)
    tools.r_mapcalc(expression="dem = -21.0803623 + 0.37 * col()")
    manifest, mesh = build_mesh(tools, tmp_path, elevation="dem")
    z0 = manifest["bed_z0"]
    assert z0 <= np.floor(mesh["bed_centroid_f64"].min() * 1e4) - 10000
    dequantized = (mesh["bed_zq"].astype(np.int64) + z0) * 1.0e-4
    assert np.max(np.abs(dequantized - mesh["bed_centroid_f64"])) <= 0.5e-4
    assert manifest["max_quantization_error"] <= 0.5e-4


def test_elevation_range_too_large_fails(tools, tmp_path):
    tools.g_region(w=0, s=0, e=20, n=10, res=10)
    tools.r_mapcalc(expression="dem = if(col() == 1, 0, 500000)")
    with pytest.raises(ToolError, match=r"scaled integer\s+range"):
        build_mesh(tools, tmp_path, elevation="dem")


def test_opencl_dequantization_is_bitwise(tools, tmp_path):
    """The OpenCL device reproduces the host's dequantised bed exactly."""
    tools.g_region(w=0, s=0, e=300, n=300, res=10)
    tools.r_mapcalc(expression="dem = 12.3456789 * sin(x() / 17) + y() / 7")
    probe = tools.r_hydro_anuga(
        flags="p", format="shell", elevation="dem", device="auto"
    ).keyval
    if probe["backend"] == "OpenMP":
        pytest.skip("No OpenCL device with double precision")
    # A mismatch is a fatal error, so a successful run is the check.
    tools.r_hydro_anuga(
        elevation="dem", mesh_output=str(tmp_path / "mesh"), device="auto"
    )


def test_nothing_to_do(tools):
    tools.g_region(w=0, s=0, e=20, n=20, res=10)
    tools.r_mapcalc(expression="dem = 1")
    with pytest.raises(ToolError, match="Nothing to do"):
        tools.r_hydro_anuga(elevation="dem", device="omp")
