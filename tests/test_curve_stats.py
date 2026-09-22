"""Tests for :mod:`fdasrsf.curve_stats.fdacurve` (shape statistics).

These use the MPEG7 sample curves via the session-scoped ``karcher_fdacurve``
fixture and reproduce the golden ``E[-1]`` value from the legacy suite.
"""

import numpy as np
import pytest

import fdasrsf as fs


def test_karcher_mean_populates_mean(karcher_fdacurve):
    obj = karcher_fdacurve
    assert obj.beta_mean is not None
    assert obj.q_mean is not None
    assert obj.E.ndim == 1 and obj.E.shape[0] >= 1


def test_karcher_mean_golden_energy(karcher_fdacurve):
    # regression value carried over from the legacy test/test_all.py suite
    assert karcher_fdacurve.E[-1] == pytest.approx(0.022668183569717587, rel=1e-6)


@pytest.mark.slow
def test_srvf_align_and_pca(mpeg7_beta):
    # build a dedicated object so the shared session fixture is not mutated
    n, M, K = mpeg7_beta.shape
    obj = fs.fdacurve(mpeg7_beta, N=M)
    obj.karcher_mean()
    obj.srvf_align()
    obj.karcher_cov()
    obj.shape_pca()
    # aligned curves keep the (n, T, K) layout of the input
    assert obj.betan.ndim == 3
    assert np.all(np.isfinite(obj.beta_mean))
