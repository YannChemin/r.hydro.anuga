"""Phase 4 tests: raster time series, space-time raster datasets, summary
rasters and the mass balance table (PLAN.md section 8)."""

import csv

import numpy as np
import pytest

from grass.tools import ToolError


def dam_break(tools):
    tools.g_region(n=200, s=0, e=1000, w=0, res=10)
    tools.r_mapcalc(expression="dem = -0.002 * x() + 0.3 * sin(y() / 40)")
    tools.r_mapcalc(expression="h0 = if(x() < 400, 2.0, 0.0)")


COMMON = {
    "elevation": "dem",
    "initial_depth": "h0",
    "duration": 120,
    "output_step": 30,
    "boundary": "west:dirichlet:2.5,east:transmissive",
    "device": "omp",
}


def raster_sum(tools, name):
    stats = tools.r_univar(map=name, format="json").json
    return stats["sum"], stats["n"]


def read_massbalance(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_absolute_time_series(tools, tmp_path):
    dam_break(tools)
    tools.r_hydro_anuga(
        output="db", outputs="depth,speed", start="2026-01-15 06:00", **COMMON
    )
    listing = tools.t_rast_list(
        input="db_depth", columns="name,start_time", format="json"
    ).json["data"]
    assert [row["name"] for row in listing] == [f"db_depth_{i:03d}" for i in range(5)]
    assert listing[0]["start_time"].startswith("2026-01-15 06:00:00")
    assert listing[-1]["start_time"].startswith("2026-01-15 06:02:00")
    info = tools.t_info(input="db_speed", flags="g").keyval
    assert info["temporal_type"] == "absolute"
    assert info["number_of_maps"] == 5


def test_relative_time_series(tools):
    dam_break(tools)
    tools.r_hydro_anuga(output="rel", **COMMON)
    info = tools.t_info(input="rel_depth", flags="g").keyval
    assert info["temporal_type"] == "relative"
    assert info["number_of_maps"] == 5
    assert str(info["start_time"]).strip("'") == "0"
    assert str(info["end_time"]).strip("'") == "120"


@pytest.mark.parametrize("res", [10, 20, 40])
def test_depth_raster_preserves_volume(tools, tmp_path, res):
    """Output cells at least as large as the mesh cells take the
    area-weighted mean depth: the raster volume equals the solver's."""
    dam_break(tools)
    tools.g_region(res=res)
    csv_path = tmp_path / "mb.csv"
    tools.r_hydro_anuga(
        output="v",
        massbalance=str(csv_path),
        min_depth=0,
        flags="d",
        **COMMON,
    )
    rows = read_massbalance(csv_path)
    total, _ = raster_sum(tools, "v_depth_004")
    volume = float(rows[-1]["volume"])
    assert total * res * res == pytest.approx(volume, rel=1e-12)


def test_finer_output_grid(tools):
    """Cells finer than the mesh take the triangle containing their centre."""
    dam_break(tools)
    tools.g_region(res=2.5)
    tools.r_hydro_anuga(output="fine", **COMMON)
    _, n = raster_sum(tools, "fine_depth_004")
    assert n == 400 * 80  # Every fine cell is covered.
    stats = tools.r_univar(map="fine_depth_004", format="json").json
    assert 0 <= stats["min"] and stats["max"] < 3.0


def test_summaries(tools):
    dam_break(tools)
    tools.r_hydro_anuga(
        output="s",
        max_depth="s_maxd",
        max_speed="s_maxv",
        max_stage="s_maxw",
        max_hazard="s_maxhz",
        arrival_time="s_arr",
        inundation_duration="s_dur",
        final_prefix="s_final",
        arrival_depth=0.05,
        **COMMON,
    )
    maxd = tools.r_univar(map="s_maxd", format="json").json
    last = tools.r_univar(map="s_depth_004", format="json").json
    assert maxd["max"] >= last["max"]
    # Initially wet cells arrive at t = 0; the front reaches the far end.
    arr = tools.r_univar(map="s_arr", format="json").json
    assert arr["min"] == 0 and 0 < arr["max"] <= 120
    dur = tools.r_univar(map="s_dur", format="json").json
    assert dur["max"] == pytest.approx(120, abs=1e-6)
    diff = tools.r_univar(map="s_final_stage", format="json").json
    assert diff["n"] == 2000


def test_dry_cells_null(tools):
    dam_break(tools)
    tools.r_hydro_anuga(output="n", flags="n", outputs="depth,speed", **COMMON)
    first = tools.r_univar(map="n_depth_000", format="json").json
    assert first["n"] == 800  # Only the initially wet 400 m x 200 m.
    speed = tools.r_univar(map="n_speed_000", format="json").json
    assert speed["n"] == 800


def test_mass_balance_table(tools, tmp_path):
    dam_break(tools)
    csv_path = tmp_path / "mb.csv"
    tools.r_hydro_anuga(massbalance=str(csv_path), **COMMON)
    rows = read_massbalance(csv_path)
    assert [float(r["time"]) for r in rows] == [0, 30, 60, 90, 120]
    v0 = float(rows[0]["signed_volume"])
    for r in rows:
        assert abs(float(r["relative_error"])) < 1e-12
        budget = v0 + float(r["boundary_inflow"]) + float(r["clamping_added"])
        assert float(r["signed_volume"]) == pytest.approx(budget, rel=1e-12)


def test_existing_outputs_fail_before_running(tools):
    dam_break(tools)
    tools.r_mapcalc(expression="taken_depth_002 = 1")
    with pytest.raises(ToolError, match="already exists"):
        tools.r_hydro_anuga(output="taken", **COMMON)
    tools.r_mapcalc(expression="taken_max = 1")
    # Checked by the GRASS parser itself for declared outputs.
    with pytest.raises(ToolError, match="exists"):
        tools.r_hydro_anuga(max_depth="taken_max", **COMMON)


def test_whole_seconds_required(tools):
    dam_break(tools)
    params = dict(COMMON, output_step=0.5)
    with pytest.raises(ToolError, match="whole number of seconds"):
        tools.r_hydro_anuga(output="x", **params)


def test_opencl_rasters_equal_openmp(tools):
    from opencl_test import opencl_available

    if not opencl_available(tools):
        pytest.skip("No OpenCL device with double precision")
    dam_break(tools)
    params = dict(COMMON, manning_value=0)
    tools.r_hydro_anuga(output="a", flags="d", outputs="depth,speed", **params)
    params["device"] = "auto"
    tools.r_hydro_anuga(output="b", flags="d", outputs="depth,speed", **params)
    for q in ("depth", "speed"):
        tools.r_mapcalc(expression=f"d_{q} = abs(a_{q}_004 - b_{q}_004)")
        assert tools.r_univar(map=f"d_{q}", format="json").json["max"] == 0
    assert np.isfinite(raster_sum(tools, "a_depth_004")[0])
