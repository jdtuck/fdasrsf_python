"""Tests for :mod:`fdasrsf.fPLS.pls_svd` (SVD-based partial least squares)."""

import numpy as np
import pytest

import fdasrsf as fs


def test_pls_svd_shapes_and_values():
    rng = np.random.default_rng(0)
    M, N, no = 40, 25, 3
    time = np.linspace(0, 1, M)
    # correlated pair of function collections
    base = rng.standard_normal((M, N))
    qf = base + 0.1 * rng.standard_normal((M, N))
    qg = base + 0.1 * rng.standard_normal((M, N))

    wqf, wqg, alpha, values, cost = fs.pls_svd(time, qf, qg, no)

    assert wqf.shape == (M, no)
    assert wqg.shape == (M, no)
    assert alpha == 0.0
    assert values.shape[0] >= no
    # weight functions are normalized to unit inner-product norm
    for ii in range(no):
        assert fs.innerprod_q(time, wqf[:, ii], wqf[:, ii]) == pytest.approx(1.0, abs=1e-6)
        assert fs.innerprod_q(time, wqg[:, ii], wqg[:, ii]) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(cost)
