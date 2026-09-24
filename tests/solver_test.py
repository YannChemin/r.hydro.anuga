"""Phase 2 tests: the OpenMP solver against ANUGA and analytical solutions
(PLAN.md section 9: V2 lake at rest, V3 dam breaks, V4 bitwise
cross-check with ANUGA's unified compute mode, mass conservation)."""

import math
import subprocess

import numpy as np
import pytest

from conftest import TEST_DEVICE
from mesh_test import REPO, anuga_python, load_export

G = 9.8  # anuga.config.g


def run(tools, tmp_path, name="run", **kwargs):
    """Run the solver with mesh and state exports; return both. The device
    defaults to R_HYDRO_ANUGA_TEST_DEVICE."""
    mesh_dir, state_dir = tmp_path / f"{name}_mesh", tmp_path / f"{name}_state"
    kwargs.setdefault("device", TEST_DEVICE)
    tools.r_hydro_anuga(
        mesh_output=str(mesh_dir), state_output=str(state_dir), **kwargs
    )
    return load_export(mesh_dir), load_export(state_dir), mesh_dir, state_dir


@pytest.mark.parametrize(
    ("algorithm", "friction", "boundary", "duration", "step"),
    [
        ("DE0", "flat", "west:reflective", 60, 20),
        ("DE1", "flat", "west:reflective", 60, 10),
        ("DE2", "flat", "north:transmissive,south:transmissive", 120, 30),
        ("DE1", "sloped", "east:transmissive", 120, 30),
        ("DE1", "flat", "west:dirichlet:2.5,east:transmissive", 120, 30),
        ("DE0", "sloped", "west:dirichlet:2.5:0.1:0,east:transmissive", 120, 30),
    ],
)
def test_bitwise_equal_to_anuga(
    tools, tmp_path, algorithm, friction, boundary, duration, step
):
    """V4: same final state and number of steps as ANUGA's C kernels
    (OpenMP: OpenCL's pow() is only accurate to a few ulp)."""
    python = anuga_python()
    tools.g_region(n=200, s=0, e=1000, w=0, res=10)
    tools.r_mapcalc(expression="dem = -0.002 * x() + 0.3 * sin(y() / 40)")
    tools.r_mapcalc(expression="h0 = if(x() < 400, 2.0, 0.0)")
    *_, mesh_dir, state_dir = run(
        tools,
        tmp_path,
        device="omp",
        elevation="dem",
        initial_depth="h0",
        duration=duration,
        output_step=step,
        algorithm=algorithm,
        manning_value=0.03,
        friction_method=friction,
        boundary=boundary,
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


@pytest.mark.parametrize(
    ("bed", "stage"),
    [
        # Immersed bump.
        ("0.5 * exp(-((x() - 250)^2 + (y() - 100)^2) / 3000)", 1.0),
        # Steep island emerging above the lake (wet/dry edges).
        (
            "2.0 * exp(-((x() - 250)^2 + (y() - 100)^2) / 2000) + 0.1 * sin(x() / 17)",
            1.0,
        ),
    ],
)
def test_lake_at_rest(tools, tmp_path, bed, stage):
    """V2: a flat water surface over uneven terrain stays at rest."""
    tools.g_region(n=200, s=0, e=500, w=0, res=10)
    tools.r_mapcalc(expression=f"dem = {bed}")
    tools.r_mapcalc(expression=f"w0 = {stage}")
    _, (manifest, st), *_ = run(
        tools,
        tmp_path,
        elevation="dem",
        initial_stage="w0",
        duration=200,
        output_step=50,
        manning_value=0.03,
    )
    assert manifest["n_steps"] > 50
    assert np.max(np.abs(st["xmom"])) < 1e-10
    assert np.max(np.abs(st["ymom"])) < 1e-10
    wet = st["initial_stage"] > st["bed"]
    assert np.max(np.abs(st["stage"][wet] - stage)) < 1e-10


def stoker(x, t, h_right, h_left):
    """Wet dam break (Stoker): depth and velocity at x (dam at x = 0)."""

    def wu(h2):
        h21, h01 = h2 / h_left, h_right / h_left
        r = math.sqrt(h21)
        return h21 * (r * (r * (h21 - 9 * h01) + 16.0 * h01) - h01 * (h01 + 8.0)) + (
            h01**3
        )

    lo, hi = h_right, h_left
    for _ in range(200):  # Bisection, as ANUGA's brentq to full precision.
        mid = 0.5 * (lo + hi)
        if wu(lo) * wu(mid) <= 0:
            hi = mid
        else:
            lo = mid
    h2 = 0.5 * (lo + hi)
    u2 = 2.0 * (math.sqrt(G * h_left) - math.sqrt(G * h2))
    s = u2 * h2 / (h2 - h_right)
    c1, c2 = math.sqrt(G * h_left), math.sqrt(G * h2)
    h = np.select(
        [x < -t * c1, x < t * (u2 - c2), x < s * t],
        [h_left, (2.0 / 3.0 * c1 - x / (3.0 * t)) ** 2 / G, h2],
        default=h_right,
    )
    u = np.select(
        [x < -t * c1, x < t * (u2 - c2), x < s * t],
        [0.0, 2.0 / 3.0 * (c1 + x / t), u2],
        default=0.0,
    )
    return h, u


def ritter(x, t, h_left):
    """Dry dam break (Ritter): depth and velocity at x (dam at x = 0)."""
    c1 = math.sqrt(G * h_left)
    fan = (x >= -c1 * t) & (x <= 2 * c1 * t)
    h = np.where(x < -c1 * t, h_left, 0.0)
    h = np.where(fan, (2 * c1 - x / t) ** 2 / (9 * G), h)
    u = np.where(fan, 2.0 / 3.0 * (c1 + x / t), 0.0)
    return h, u


def dam_break_errors(tools, tmp_path, h_right, times):
    """Relative L1 errors of stage and xmomentum along a 1000 m x 5 m
    channel at 1 m, dam at x = 0, as ANUGA's dam_break validation."""
    tools.g_region(n=2.5, s=-2.5, e=500, w=-500, res=1)
    tools.r_mapcalc(expression="dem = 0")
    tools.r_mapcalc(expression=f"h0 = if(x() <= 0, 10.0, {h_right})")
    errors = []
    for t in times:
        (mm, mesh), (_, st), *_ = run(
            tools,
            tmp_path,
            name=f"t{t}",
            elevation="dem",
            initial_depth="h0",
            duration=t,
            output_step=0.5,
            manning_value=0,
            boundary="west:transmissive,east:transmissive",
        )
        x = mesh["centroid_coordinates"][:, 0] + mm["origin"][0]
        if h_right > 0:
            h, u = stoker(x, t, h_right, 10.0)
        else:
            h, u = ritter(x, t, 10.0)
        depth = st["stage"] - st["bed"]
        errors.append(
            (
                np.sum(np.abs(depth - h)) / np.sum(np.abs(h)),
                np.sum(np.abs(st["xmom"] - u * h)) / np.sum(np.abs(u * h)),
            )
        )
    print("dam break h_right", h_right, "L1 (stage, xmom):", errors)
    return errors


def test_dam_break_wet(tools, tmp_path):
    """V3: wet dam break within ANUGA's validation tolerances."""
    (s5, m5), (s25, m25), (s50, m50) = dam_break_errors(
        tools, tmp_path, 1.0, (5, 25, 50)
    )
    assert s5 < 0.01 and s25 < 0.01 and s50 < 0.01
    assert m5 < 0.02 and m25 < 0.01 and m50 < 0.01


def test_dam_break_dry(tools, tmp_path):
    """V3: dry dam break within ANUGA's validation tolerances."""
    (s5, m5), (s25, m25), (s50, m50) = dam_break_errors(
        tools, tmp_path, 0.0, (5, 25, 50)
    )
    assert s5 < 0.1 and s25 < 0.1 and s50 < 0.15
    assert m5 < 0.25 and m25 < 0.25 and m50 < 0.25


def test_closed_box_conserves_mass(tools, tmp_path):
    tools.g_region(n=200, s=0, e=500, w=0, res=10)
    tools.r_mapcalc(expression="dem = 0.001 * x() + 0.2 * cos(y() / 30)")
    tools.r_mapcalc(expression="h0 = if(x() < 150, 1.5, 0)")
    _, (manifest, _st), *_ = run(
        tools,
        tmp_path,
        elevation="dem",
        initial_depth="h0",
        duration=300,
        output_step=60,
        manning_value=0.03,
    )
    v0, v1 = manifest["initial_volume"], manifest["final_volume"]
    # Reflective walls: the boundary flux is round-off only.
    assert abs(manifest["boundary_mass"]) < 1e-9
    assert abs(v1 - v0) <= 1e-12 * v0


def test_open_boundaries_mass_balance(tools, tmp_path):
    """Volume change equals the integrated boundary flux."""
    tools.g_region(n=200, s=0, e=500, w=0, res=10)
    tools.r_mapcalc(expression="dem = -0.002 * x()")
    tools.r_mapcalc(expression="h0 = 0.5")
    _, (manifest, _st), *_ = run(
        tools,
        tmp_path,
        elevation="dem",
        initial_depth="h0",
        duration=200,
        output_step=50,
        manning_value=0.03,
        boundary="west:dirichlet:1.2,east:transmissive",
    )
    change = manifest["final_volume"] - manifest["initial_volume"]
    assert manifest["boundary_mass"] != 0
    assert abs(change - manifest["boundary_mass"]) <= 1e-9 * abs(change)


@pytest.mark.parametrize("algorithm", ["DE0", "DE1", "DE2"])
def test_mass_balance_identity_with_clamping(tools, tmp_path, algorithm):
    """Signed volume change = boundary flux + water added by clamping
    negative depths, weighted by the Runge-Kutta coefficients. A thin film
    on steep terrain makes DE1/DE2 (CFL 1) clamp a lot, as in ANUGA."""
    tools.g_region(n=900, s=0, e=900, w=0, res=30)
    tools.r_mapcalc(expression="dem = 40 * sin(x() / 60) * cos(y() / 50) + 0.1 * x()")
    tools.r_mapcalc(expression="h0 = 0.02")
    _, (m, _st), *_ = run(
        tools,
        tmp_path,
        elevation="dem",
        initial_depth="h0",
        duration=60,
        output_step=60,
        algorithm=algorithm,
        manning_value=0.03,
        boundary="east:transmissive",
    )
    change = m["final_signed_volume"] - m["initial_signed_volume"]
    budget = m["boundary_mass"] + m["protect_mass"]
    if algorithm != "DE0":
        assert m["protect_mass"] > 0  # The case really exercises clamping.
    assert abs(change - budget) <= 1e-9 * m["initial_signed_volume"]
