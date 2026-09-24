"""Phase 3 tests: the OpenCL tier against the OpenMP tier (PLAN.md V13).

The OpenMP tier is bitwise identical to ANUGA (solver_test.py). OpenCL
guarantees correctly rounded double +, -, *, / and sqrt, so without
friction the OpenCL tier must be bitwise identical too; Manning friction
uses pow(), which OpenCL only guarantees to a few ulp.
"""

import numpy as np
import pytest

from solver_test import run


def opencl_available(tools):
    tools.g_region(n=20, s=0, e=20, w=0, res=10)
    tools.r_mapcalc(expression="probe_dem = 0")
    report = tools.r_hydro_anuga(
        flags="p", format="shell", elevation="probe_dem", device="auto"
    ).keyval
    return report["backend"] != "OpenMP"


@pytest.fixture
def ocl_tools(tools):
    if not opencl_available(tools):
        pytest.skip("No OpenCL device with double precision")
    return tools


def run_both(tools, tmp_path, **kwargs):
    _, (m_omp, s_omp), *_ = run(tools, tmp_path, name="omp", device="omp", **kwargs)
    _, (m_ocl, s_ocl), *_ = run(tools, tmp_path, name="ocl", device="auto", **kwargs)
    return m_omp, s_omp, m_ocl, s_ocl


def channel(tools):
    tools.g_region(n=200, s=0, e=1000, w=0, res=10)
    tools.r_mapcalc(expression="dem = -0.002 * x() + 0.3 * sin(y() / 40)")
    tools.r_mapcalc(expression="h0 = if(x() < 400, 2.0, 0.0)")


@pytest.mark.parametrize("algorithm", ["DE0", "DE1", "DE2"])
def test_bitwise_equal_without_friction(ocl_tools, tmp_path, algorithm):
    channel(ocl_tools)
    m_omp, s_omp, m_ocl, s_ocl = run_both(
        ocl_tools,
        tmp_path,
        elevation="dem",
        initial_depth="h0",
        duration=60,
        output_step=20,
        algorithm=algorithm,
        manning_value=0,
        boundary="west:dirichlet:2.5,east:transmissive,north:transmissive",
    )
    assert m_omp["n_steps"] == m_ocl["n_steps"]
    for q in ("stage", "xmom", "ymom", "max_speed"):
        assert np.array_equal(s_omp[q], s_ocl[q]), q


@pytest.mark.parametrize("friction", ["flat", "sloped"])
def test_friction_within_ulps(ocl_tools, tmp_path, friction):
    channel(ocl_tools)
    m_omp, s_omp, m_ocl, s_ocl = run_both(
        ocl_tools,
        tmp_path,
        elevation="dem",
        initial_depth="h0",
        duration=60,
        output_step=20,
        manning_value=0.03,
        friction_method=friction,
        boundary="west:dirichlet:2.5,east:transmissive",
    )
    assert m_omp["n_steps"] == m_ocl["n_steps"]
    for q in ("stage", "xmom", "ymom"):
        scale = np.max(np.abs(s_omp[q]))
        assert np.max(np.abs(s_omp[q] - s_ocl[q])) <= 1e-12 * scale, q


def test_lake_at_rest(ocl_tools, tmp_path):
    ocl_tools.g_region(n=200, s=0, e=500, w=0, res=10)
    ocl_tools.r_mapcalc(
        expression="dem = 2.0 * exp(-((x() - 250)^2 + (y() - 100)^2) / 2000)"
    )
    ocl_tools.r_mapcalc(expression="w0 = 1.0")
    _, (_m, st), *_ = run(
        ocl_tools,
        tmp_path,
        elevation="dem",
        initial_stage="w0",
        duration=200,
        output_step=50,
        manning_value=0.03,
        device="auto",
    )
    assert np.max(np.abs(st["xmom"])) < 1e-10
    assert np.max(np.abs(st["ymom"])) < 1e-10


def test_mass_balance_identity(ocl_tools, tmp_path):
    ocl_tools.g_region(n=900, s=0, e=900, w=0, res=30)
    ocl_tools.r_mapcalc(
        expression="dem = 40 * sin(x() / 60) * cos(y() / 50) + 0.1 * x()"
    )
    ocl_tools.r_mapcalc(expression="h0 = 0.02")
    _, (m, _st), *_ = run(
        ocl_tools,
        tmp_path,
        elevation="dem",
        initial_depth="h0",
        duration=60,
        output_step=60,
        cfl=1.0,
        manning_value=0.03,
        boundary="east:transmissive",
        device="auto",
    )
    change = m["final_signed_volume"] - m["initial_signed_volume"]
    budget = m["boundary_mass"] + m["protect_mass"]
    assert m["protect_mass"] > 0
    assert abs(change - budget) <= 1e-9 * m["initial_signed_volume"]
