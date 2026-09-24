"""Fixtures for r.hydro.anuga tests.

Each fixture creates a throwaway project, so no sample dataset is needed.
The module is looked up on PATH first, then in R_HYDRO_ANUGA_BIN_DIR, then
in the GRASS source tree's dist directory ($HOME/dev/grass/dist.*/bin),
so tests run against a development build without installing it.

R_HYDRO_ANUGA_TEST_DEVICE (omp, auto, gpu or cpu; default omp) selects
the device of the tests that do not require bitwise equality with ANUGA.
CPU threads (OpenMP, including ANUGA's, and PoCL) are capped at half the
available cores unless OMP_NUM_THREADS, POCL_CPU_MAX_CU_COUNT or
POCL_MAX_PTHREAD_COUNT are already set.
"""

import glob
import os
import shutil
from pathlib import Path

import pytest

import grass.script as gs
from grass.tools import Tools

MODULE = "r.hydro.anuga"

TEST_DEVICE = os.environ.get("R_HYDRO_ANUGA_TEST_DEVICE", "omp")

# Some test machines are not stable under a sustained load on every core.
# Set before any session copies the environment, so the module and the
# ANUGA comparison scripts inherit it.
HALF_CORES = str(max(1, len(os.sched_getaffinity(0)) // 2))
for _var in ("OMP_NUM_THREADS", "POCL_CPU_MAX_CU_COUNT", "POCL_MAX_PTHREAD_COUNT"):
    os.environ.setdefault(_var, HALF_CORES)


def pytest_report_header(config):
    return (
        f"r.hydro.anuga: device={TEST_DEVICE}, "
        f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}, "
        f"POCL_CPU_MAX_CU_COUNT={os.environ['POCL_CPU_MAX_CU_COUNT']}"
    )


def _module_bin_dir():
    """Return the directory containing the module executable."""
    env_dir = os.environ.get("R_HYDRO_ANUGA_BIN_DIR")
    if env_dir and (Path(env_dir) / MODULE).exists():
        return env_dir
    found = shutil.which(MODULE)
    if found:
        return str(Path(found).parent)
    candidates = glob.glob(str(Path.home() / "dev/grass/dist.*/bin" / MODULE))
    if candidates:
        return str(Path(candidates[0]).parent)
    pytest.skip(f"{MODULE} executable not found; build it first")
    return None


def _session_tools(tmp_path, crs):
    project = tmp_path / "project"
    gs.create_project(project, crs=crs)
    session = gs.setup.init(project, env=os.environ.copy())
    env = session.env.copy()
    env["PATH"] = _module_bin_dir() + os.pathsep + env["PATH"]
    return session, Tools(env=env)


@pytest.fixture
def tools(tmp_path):
    """Tools in a fresh Lambert-93 (EPSG:2154) project."""
    session, tools = _session_tools(tmp_path, "EPSG:2154")
    with session:
        yield tools


@pytest.fixture
def latlong_tools(tmp_path):
    """Tools in a fresh WGS84 latitude-longitude project."""
    session, tools = _session_tools(tmp_path, "EPSG:4326")
    with session:
        yield tools


@pytest.fixture
def dual_dem(tools):
    """A 30 m coarse DEM over 3840 m and a 1 m fine DEM over its centre.

    3840 m is a whole number of both 30 m DEM cells and 32 m base cells,
    so the default res_max snaps from 30 m to 32 m (= 1 m * 2^5).
    """
    tools.g_region(n=3840, s=0, e=3840, w=0, res=30)
    tools.r_mapcalc(expression="coarse = 50 + 0.01 * x() - 0.005 * y()")
    tools.g_region(n=2432, s=1408, e=2432, w=1408, res=1)
    tools.r_mapcalc(
        expression="fine = 50 + 0.01 * x() - 0.005 * y() + 0.2 * sin(x() * 10)"
    )
    tools.g_region(n=3840, s=0, e=3840, w=0, res=32)
    return tools
