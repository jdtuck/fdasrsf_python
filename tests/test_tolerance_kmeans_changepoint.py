"""Tests for :mod:`fdasrsf.tolerance`, :mod:`fdasrsf.kmeans` and
:mod:`fdasrsf.elastic_changepoint`.

All of these are iterative / stochastic and are marked ``slow`` (deselected by
default).  Inputs are kept small and, where a bootstrap count is exposed, it is
reduced so the tests remain quick when selected with ``pytest -m slow``.
"""

import numpy as np
import pytest

import fdasrsf as fs


@pytest.fixture
def small_functions():
    """A small ``(M, N)`` collection of shifted Gaussian bumps."""
    rng = np.random.default_rng(2)
    M, N = 50, 10
    time = np.linspace(0, 1, M)
    f = np.zeros((M, N))
    for i in range(N):
        c = 0.5 + rng.normal(scale=0.05)
        f[:, i] = np.exp(-((time - c) ** 2) / 0.01)
    return f, time


@pytest.mark.slow
def test_pcaTB_runs(small_functions):
    pytest.importorskip("optimum_reparamN2")
    f, time = small_functions
    out = fs.pcaTB(f, time, no=3, parallel=False)
    assert out is not None


@pytest.mark.slow
def test_bootTB_runs_small_B(small_functions):
    pytest.importorskip("optimum_reparamN2")
    f, time = small_functions
    out = fs.bootTB(f, time, B=5, no=3, parallel=False)
    assert out is not None


@pytest.fixture
def two_group_functions():
    """A ``(M, N)`` collection with two clearly separated groups of bumps.

    k-means needs each cluster to receive at least one member, so the two
    groups are placed far apart (peaks near 0.3 and 0.7).
    """
    rng = np.random.default_rng(3)
    M = 60
    time = np.linspace(0, 1, M)
    cols = []
    for c in (0.3, 0.7):
        for _ in range(5):
            cc = c + rng.normal(scale=0.02)
            cols.append(np.exp(-((time - cc) ** 2) / 0.005))
    f = np.column_stack(cols)
    return f, time


@pytest.mark.slow
def test_kmeans_align_runs(two_group_functions):
    pytest.importorskip("optimum_reparamN2")
    f, time = two_group_functions
    # seed one template from each group so neither cluster starts empty
    # (default seeding uses np.random.choice, which may pick duplicates)
    out = fs.kmeans_align(
        f, time, K=2, seeds=[0, 5], MaxItr=2, showplot=False, parallel=False
    )
    assert out is not None
    # both clusters received members
    assert set(np.unique(out["labels"])) == {0, 1}


@pytest.mark.slow
def test_elastic_change_compute_runs(small_functions):
    pytest.importorskip("optimum_reparamN2")
    f, time = small_functions
    obj = fs.elastic_change(f, time)
    obj.compute(d=50)
    # a change-point index and a p-value are produced
    assert hasattr(obj, "k_star")
    assert np.isfinite(obj.p)
