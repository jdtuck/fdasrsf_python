"""Shared pytest fixtures and helpers for the fdasrsf test suite.

Everything here is designed so that individual test modules stay small and
CWD-independent:

* ``DATA_DIR`` resolves the ``bin/`` sample-data directory from the repository
  root rather than from the current working directory (the legacy tests loaded
  ``"bin/simu_data.npz"`` relative to CWD and only passed when pytest ran from
  the repo root).
* Expensive objects (an aligned ``fdawarp``, a Karcher-mean ``fdacurve``) are
  built once per session and reused by the fPCA / boxplot / curve-stats tests.
* ``requires_ext`` gates tests that need a compiled C/Cython extension so the
  suite skips cleanly wherever an extension has not been built.
"""

import pathlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths / data
# ---------------------------------------------------------------------------

#: Repository root (the directory that contains ``bin/`` and ``fdasrsf/``).
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Directory holding the shipped sample-data ``.npz`` files.
DATA_DIR = REPO_ROOT / "bin"


def _data_path(name):
    """Return the absolute path to a sample-data file, skipping if absent."""
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"sample data file not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Extension gating
# ---------------------------------------------------------------------------
#
# Tests gate on a compiled extension with ``pytest.importorskip("<name>")``
# directly (the standard idiom); no cross-module import of a helper is needed,
# which keeps things working under ``--import-mode=importlib``.  The extension
# names are: optimum_reparamN2, optimum_reparam_N, crbfgs, cbayesian, cimage,
# mlogit_warp, ocmlogit_warp, oclogit_warp, fpls_warp, _DP.


# ---------------------------------------------------------------------------
# Synthetic signal fixtures (fast, deterministic)
# ---------------------------------------------------------------------------


@pytest.fixture
def M():
    """Default number of samples for synthetic 1-D signals."""
    return 101


@pytest.fixture
def timet(M):
    """A time vector on ``[0, 1]`` with ``M`` samples."""
    return np.linspace(0, 1, M)


@pytest.fixture
def sine_signal(M):
    """A single sinusoid, the shape most legacy tests build by hand."""
    return np.sin(np.linspace(0, 2 * np.pi, M))


@pytest.fixture
def identity_gamma(M):
    """The identity warping function on ``[0, 1]``."""
    return np.linspace(0, 1, M)


# ---------------------------------------------------------------------------
# Sample-data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def simu_data():
    """Load ``bin/simu_data.npz`` as ``(f, time)``.

    ``arr_0`` is the ``(M, N)`` matrix of functions, ``arr_1`` the length-``M``
    time vector.
    """
    data = np.load(_data_path("simu_data.npz"))
    return data["arr_0"], data["arr_1"]


@pytest.fixture(scope="session")
def mpeg7_beta():
    """Assemble the MPEG7 curves into a ``beta`` array of shape ``(n, M, K)``."""
    data = np.load(_data_path("MPEG7.npz"), allow_pickle=True)
    Xdata = data["Xdata"]
    n, M = Xdata[0, 1].shape
    K = Xdata.shape[1]
    beta = np.zeros((n, M, K))
    for i in range(K):
        beta[:, :, i] = Xdata[0, i]
    return beta


# ---------------------------------------------------------------------------
# Expensive, session-scoped model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def aligned_fdawarp(simu_data):
    """A single ``fdawarp`` with ``srsf_align`` already run (session-scoped).

    Reused by the fPCA, boxplot and fPNS tests so alignment happens once.
    Requires the ``optimum_reparamN2`` extension.
    """
    pytest.importorskip("optimum_reparamN2")
    import fdasrsf as fs

    f, time = simu_data
    obj = fs.fdawarp(f, time)
    obj.srsf_align()
    return obj


@pytest.fixture(scope="session")
def median_fdawarp(simu_data):
    """A single ``fdawarp`` aligned with ``method="median"`` (session-scoped).

    The amplitude/phase boxplots require a median alignment, so they use this
    fixture rather than the mean-aligned one.  Requires ``optimum_reparamN2``.
    """
    pytest.importorskip("optimum_reparamN2")
    import fdasrsf as fs

    f, time = simu_data
    obj = fs.fdawarp(f, time)
    obj.srsf_align(method="median")
    return obj


@pytest.fixture(scope="session")
def karcher_fdacurve(mpeg7_beta):
    """A ``fdacurve`` with a Karcher mean computed (session-scoped).

    Reused by the curve-statistics tests.  Requires ``optimum_reparam_N``.
    """
    pytest.importorskip("optimum_reparam_N")
    import fdasrsf as fs

    n, M, K = mpeg7_beta.shape
    obj = fs.fdacurve(mpeg7_beta, N=M)
    obj.karcher_mean()
    return obj
