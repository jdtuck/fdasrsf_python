"""Tests for :mod:`fdasrsf.fPCA` (vertical / horizontal / joint fPCA).

All fPCA classes take an already-aligned ``fdawarp``; the session-scoped
``aligned_fdawarp`` fixture supplies one (skipping if the extension or data is
unavailable).
"""

import numpy as np
import pytest

import fdasrsf as fs


@pytest.mark.parametrize("cls_name", ["fdavpca", "fdahpca", "fdajpca"])
def test_calc_fpca_populates_components(aligned_fdawarp, cls_name):
    cls = getattr(fs, cls_name)
    obj = cls(aligned_fdawarp)
    obj.calc_fpca(no=3)
    assert obj.latent.shape[0] == 3
    assert obj.U.shape[1] == 3
    # eigenvalues are non-increasing and non-negative
    assert np.all(obj.latent >= -1e-9)
    assert np.all(np.diff(obj.latent) <= 1e-9)


def test_vpca_coef_rows_match_samples(aligned_fdawarp):
    obj = fs.fdavpca(aligned_fdawarp)
    obj.calc_fpca(no=2)
    # one coefficient row per input function
    assert obj.coef.shape[0] == aligned_fdawarp.f.shape[1]
    assert obj.coef.shape[1] == 2


@pytest.mark.parametrize("log_der", [False, True])
def test_hpca_log_der(aligned_fdawarp, log_der):
    obj = fs.fdahpca(aligned_fdawarp, log_der=log_der)
    obj.calc_fpca(no=2)
    N = aligned_fdawarp.f.shape[1]
    assert obj.log_der is log_der
    assert obj.coef.shape == (N, 2)
    # principal directions are valid warping functions on [0, 1]
    assert np.allclose(obj.gam_pca[:, 0, :], 0)
    assert np.allclose(obj.gam_pca[:, -1, :], 1)
    obj.project(aligned_fdawarp.f[:, 0:2])
    assert obj.new_coef.shape == (2, 2)
