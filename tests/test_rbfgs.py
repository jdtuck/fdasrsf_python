"""Tests for :mod:`fdasrsf.rbfgs` (pure-python Riemannian LBFGS solver)."""

import numpy as np
import pytest

from fdasrsf.rbfgs import rlbfgs


def test_rlbfgs_self_alignment_is_identity():
    M = 101
    q1 = np.sin(np.linspace(0, 2 * np.pi, M))
    t = np.linspace(0, 1, M)

    obj = rlbfgs(q1, q1, t)
    obj.solve()

    # aligning a signal to itself yields the identity warping function
    np.testing.assert_allclose(obj.gammaOpt, t, atol=1e-3)


def test_rlbfgs_via_optimum_reparam():
    # the public path (utility_functions.optimum_reparam with method="RBFGS")
    # uses rlbfgs under the hood and needs no compiled extension
    import fdasrsf as fs

    M = 101
    q1 = np.sin(np.linspace(0, 2 * np.pi, M))
    t = np.linspace(0, 1, M)
    gam = fs.optimum_reparam(q1, t, q1, method="RBFGS")
    assert sum(gam - t) == pytest.approx(0, abs=1e-6)


def _bumps(t, rng):
    # three random Gaussian bumps
    f = np.zeros_like(t)
    for _ in range(3):
        f += rng.standard_normal() * np.exp(
            -((t - rng.random()) ** 2) / (2 * (0.05 + 0.1 * rng.random()) ** 2)
        )
    return f


def _align_error(q1, q2, t, gam):
    from scipy.integrate import trapezoid
    from fdasrsf.utility_functions import warp_q_gamma

    qw = warp_q_gamma(t, q2, gam)
    return np.sqrt(trapezoid((q1 - qw) ** 2, t))


@pytest.mark.parametrize("method", ["RBFGS", "cRBFGS"])
def test_rbfgs_distance_matches_dp(method):
    # started from the identity, RBFGS stopped in local minima, up to 3x the
    # DP distance; it now refines a coarse DP warp on a finer grid
    import fdasrsf as fs

    if method == "cRBFGS":
        pytest.importorskip("crbfgs")
    M = 101
    t = np.linspace(0, 1, M)
    rng = np.random.default_rng(0)
    r = []
    for _ in range(10):
        q1 = fs.f_to_srsf(_bumps(t, rng), t)
        q2 = fs.f_to_srsf(_bumps(t, rng), t)
        d = _align_error(q1, q2, t, fs.optimum_reparam(q1, t, q2, method=method))
        d_dp = _align_error(q1, q2, t, fs.optimum_reparam(q1, t, q2, method="DP"))
        r.append(d / d_dp)
    assert np.median(r) < 1.02
    assert np.max(r) < 1.1


@pytest.mark.parametrize("method", ["RBFGS", "cRBFGS"])
def test_rbfgs_fits_known_warp(method):
    import fdasrsf as fs

    if method == "cRBFGS":
        pytest.importorskip("crbfgs")
    M = 101
    t = np.linspace(0, 1, M)
    f1 = np.exp(-((t - 0.3) ** 2) / 0.005) + 0.6 * np.exp(-((t - 0.7) ** 2) / 0.003)
    gam_true = (np.exp(1.5 * t) - 1) / (np.exp(1.5) - 1)
    f2 = np.interp(np.interp(t, gam_true, t), t, f1)
    q1 = fs.f_to_srsf(f1, t)
    q2 = fs.f_to_srsf(f2, t)
    gam = fs.optimum_reparam(q1, t, q2, method=method)
    # q2 o gam = q1 when gam is the inverse of the warp f2 was built with
    assert np.max(np.abs(gam - gam_true)) * (M - 1) < 1.0
    assert _align_error(q1, q2, t, gam) < 1.1 * _align_error(q1, q2, t, gam_true)


@pytest.mark.parametrize("method", ["RBFGS", "cRBFGS"])
@pytest.mark.parametrize("penalty", ["roughness", "l2gam", "l2psi", "geodesic"])
def test_rbfgs_penalty_reduces_warping(method, penalty):
    import fdasrsf as fs

    if method == "cRBFGS":
        pytest.importorskip("crbfgs")
    M = 101
    t = np.linspace(0, 1, M)
    rng = np.random.default_rng(1)
    q1 = fs.f_to_srsf(_bumps(t, rng), t)
    q2 = fs.f_to_srsf(_bumps(t, rng), t)
    gam0 = fs.optimum_reparam(q1, t, q2, method=method)
    gam1 = fs.optimum_reparam(q1, t, q2, method=method, lam=1.0, penalty=penalty)
    assert np.all(np.diff(gam1) >= 0)
    assert np.linalg.norm(gam1 - t) <= np.linalg.norm(gam0 - t) + 1e-8


def test_rlbfgs_starts_from_gam0():
    M = 101
    t = np.linspace(0, 1, M)
    f1 = np.exp(-((t - 0.3) ** 2) / 0.005) + 0.6 * np.exp(-((t - 0.7) ** 2) / 0.003)
    gam_true = (np.exp(1.5 * t) - 1) / (np.exp(1.5) - 1)
    f2 = np.interp(np.interp(t, gam_true, t), t, f1)
    import fdasrsf as fs

    q1 = fs.f_to_srsf(f1, t)
    q2 = fs.f_to_srsf(f2, t)
    obj = rlbfgs(q1, q2, t)
    obj.solve(gam0=gam_true)
    assert obj.stopReason in (1, 2, 3, 4)
    assert np.all(np.diff(obj.gammaOpt) >= 0)
    assert np.max(np.abs(obj.gammaOpt - gam_true)) * (M - 1) < 0.5

    cr = pytest.importorskip("crbfgs")
    g, info = cr.rlbfgs(q1, q2, t, 30, 0.0, 0, gam0=gam_true, return_info=True)
    assert info.shape == (4,)
    assert info[1] in (1, 2, 3)
    assert np.all(np.diff(g) >= 0)
    assert np.max(np.abs(g - gam_true)) * (M - 1) < 0.5
