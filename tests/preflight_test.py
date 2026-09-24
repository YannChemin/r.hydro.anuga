"""Phase 0 tests: option validation, device fallback and the -p pre-flight
report (PLAN.md sections 4.2-4.6 and 10)."""

import pytest

from grass.tools import ToolError


def dry_run(tools, **kwargs):
    kwargs.setdefault("device", "omp")
    return tools.r_hydro_anuga(flags="p", format="shell", **kwargs).keyval


def test_levels_snapping_and_estimate(dual_dem):
    """res_max snaps 30 m -> 32 m; DEM levels and the 1 m level count."""
    report = dry_run(dual_dem, elevation="coarse,fine")

    assert report["backend"] == "OpenMP"
    assert report["n_dems"] == 2
    # Sorted finest first regardless of the order given.
    assert report["dem0"].startswith("fine@")
    assert report["dem1"].startswith("coarse@")
    assert report["res_min"] == 1
    assert report["res_max"] == 32
    assert report["res_max_requested"] == 30
    assert report["res_max_snapped"] == 1
    assert report["n_levels"] == 6
    assert report["dem0_level"] == 5
    assert report["dem1_level"] == 0
    # 1024 x 1024 fine cells, 4 triangles each.
    assert report["dem0_valid_cells"] == 1024 * 1024
    assert report["level5_triangles"] == 4 * 1024 * 1024
    # Fringe bands exist at every intermediate level.
    for level in range(1, 5):
        assert report[f"level{level}_triangles"] > 0
    assert report["bytes_per_triangle"] == 440


def test_keep_order_flag(dual_dem):
    """-o keeps the command-line order instead of sorting finest first."""
    report = dual_dem.r_hydro_anuga(
        flags="po", format="shell", elevation="coarse,fine", device="omp"
    ).keyval
    assert report["dem0"].startswith("coarse@")


def test_explicit_res_min(dual_dem):
    """res_min=2 removes the 1 m level and quarters the fine triangles."""
    report = dry_run(dual_dem, elevation="coarse,fine", res_min=2)
    assert report["res_max"] == 32
    assert report["n_levels"] == 5
    assert report["dem0_level"] == 4
    assert report["level4_triangles"] >= 4 * 512 * 512


def test_memory_options_change_estimate(dual_dem):
    """Infiltration state and max outputs are included in the estimate."""
    dual_dem.r_mapcalc(expression="soil = 4")
    base = dry_run(dual_dem, elevation="coarse,fine")
    gar = dry_run(
        dual_dem,
        elevation="coarse,fine",
        infiltration="gar",
        soil_texture="soil",
        max_depth="md",
        max_speed="ms",
    )
    assert gar["bytes_per_triangle"] == base["bytes_per_triangle"] + 6 * 8 + 4 + 2 * 8


def test_region_not_multiple_of_base_cell_fails(dual_dem):
    dual_dem.g_region(n=3000, s=0, e=3000, w=0, res=30)
    with pytest.raises(ToolError, match="whole number of 32 base cells"):
        dry_run(dual_dem, elevation="coarse,fine")


def test_units_are_required(dual_dem):
    with pytest.raises(ToolError, match="rain_units"):
        dry_run(dual_dem, elevation="coarse", rain_value=10)


def test_infiltration_needs_soil(dual_dem):
    with pytest.raises(ToolError, match="soil_texture"):
        dry_run(dual_dem, elevation="coarse", infiltration="ga")


def test_non_square_cells_fail(tools):
    tools.g_region(n=960, s=0, e=960, w=0, nsres=30, ewres=32)
    tools.r_mapcalc(expression="rect = 1")
    with pytest.raises(ToolError, match="non-square"):
        dry_run(tools, elevation="rect")


def test_latlong_fails(latlong_tools):
    latlong_tools.g_region(n=1, s=0, e=1, w=0, res=0.01)
    latlong_tools.r_mapcalc(expression="dem = 1")
    with pytest.raises(ToolError, match="Latitude-longitude"):
        dry_run(latlong_tools, elevation="dem")


def test_duration_required(dual_dem):
    with pytest.raises(ToolError, match="duration= is required"):
        dual_dem.r_hydro_anuga(elevation="coarse", output="sim", device="omp")


@pytest.mark.parametrize(
    "option",
    [
        {"rain_value": 10, "rain_units": "mm/h"},
        {"initial_xmom": "coarse"},
        {"gauge_output": "g.csv"},
        {"stage_series": "s.csv"},
        {"min_timestep": 0.001},
        {
            "infiltration": "ga",
            "ks": "coarse",
            "suction": "coarse",
            "porosity": "coarse",
        },
    ],
)
def test_unimplemented_options_are_refused(dual_dem, option):
    """Options of the full design that are not implemented fail loudly
    instead of being ignored (-p still accepts them for its estimate)."""
    dry_run(dual_dem, elevation="coarse", **option)
    with pytest.raises(ToolError, match=r"not\s+implemented"):
        dual_dem.r_hydro_anuga(
            elevation="coarse", duration=1, output="sim", device="omp", **option
        )
