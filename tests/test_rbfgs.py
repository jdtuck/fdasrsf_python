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


def _dp_cost(q1, q2, gam, t):
    # DP's objective at gam: the unit norm, linearly interpolated SRSFs, which
    # the DP solvers integrate exactly, here on a 20x finer grid
    from scipy.integrate import trapezoid
    from fdasrsf.utility_functions import warp_q_gamma

    M = t.shape[0]
    t1 = np.linspace(0, 1, M)
    tf = np.linspace(0, 1, (M - 1) * 20 + 1)
    a = np.interp(tf, t1, q1)
    a = a / np.sqrt(trapezoid(a**2, tf))
    b = np.interp(tf, t1, q2)
    b = b / np.sqrt(trapezoid(b**2, tf))
    qw = warp_q_gamma(tf, b, np.interp(tf, t1, gam))
    return trapezoid((a - qw) ** 2, tf)


@pytest.mark.parametrize("method", ["RBFGS", "cRBFGS"])
def test_rbfgs_cost_matches_dp(method):
    # started from the identity, RBFGS stopped in local minima, up to 3x the
    # DP distance.  Scored on DP's own objective, RBFGS is below DP on most
    # pairs; from the coarse DP start it can land in another minimum, and the
    # worst of these pairs is 1.29x
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
        gr = fs.optimum_reparam(q1, t, q2, method=method)
        gd = fs.optimum_reparam(q1, t, q2, method="DP")
        r.append(_dp_cost(q1, q2, gr, t) / _dp_cost(q1, q2, gd, t))
    assert np.median(r) < 1.0
    assert np.max(r) < 1.35


@pytest.mark.parametrize("method", ["RBFGS", "cRBFGS"])
def test_rbfgs_matches_dp_on_data(method, simu_data):
    # on consecutive pairs of the simulated data RBFGS is below DP's cost in
    # the median and on every pair
    import fdasrsf as fs

    if method == "cRBFGS":
        pytest.importorskip("crbfgs")
    f, t = simu_data
    q = fs.f_to_srsf(f, t)
    r = []
    for i in range(q.shape[1] - 1):
        gd = fs.optimum_reparam(q[:, i], t, q[:, i + 1], method="DP")
        gr = fs.optimum_reparam(q[:, i], t, q[:, i + 1], method=method)
        r.append(_dp_cost(q[:, i], q[:, i + 1], gr, t)
                 / _dp_cost(q[:, i], q[:, i + 1], gd, t))
    assert np.median(r) < 1.0
    assert np.max(r) < 1.25


def _solve(method, q1, q2, t, maxiter, gam0):
    if method == "cRBFGS":
        cr = pytest.importorskip("crbfgs")
        return cr.rlbfgs(q1, q2, t, maxiter, 0.0, 0, gam0=gam0, return_info=True)
    obj = rlbfgs(q1, q2, t)
    obj.solve(maxiter, gam0=gam0)
    return obj.gammaOpt, [obj.iterations, obj.stopReason, obj.cost, obj.gradNorm]


@pytest.mark.parametrize("method", ["RBFGS", "cRBFGS"])
def test_rlbfgs_starts_at_gam0(method, simu_data):
    # started from a DP warp, RBFGS starts there and then descends
    import fdasrsf as fs
    from scipy.integrate import trapezoid

    f, t = simu_data
    q = fs.f_to_srsf(f, t)
    t1 = np.linspace(0, 1, t.shape[0])
    q1 = q[:, 0] / np.sqrt(trapezoid(q[:, 0] ** 2, t1))
    q2 = q[:, 1] / np.sqrt(trapezoid(q[:, 1] ** 2, t1))
    gd = fs.optimum_reparam(q1, t1, q2, method="DP")
    g1, info1 = _solve(method, q1, q2, t1, 1, gd)
    _, info30 = _solve(method, q1, q2, t1, 30, gd)

    def psi(g):
        p = np.sqrt(np.gradient(g, t1))
        return p / np.sqrt(trapezoid(p**2, t1))

    dx = np.arccos(min(trapezoid(psi(g1) * psi(gd), t1), 1))
    assert dx < 0.15
    assert info30[2] < info1[2]


def test_rlbfgs_gradient_is_exact():
    # the gradient differentiates the discretised cost, penalty and barrier
    # exactly, so it matches central differences of the cost
    M = 41
    t = np.linspace(0, 1, M)
    rng = np.random.default_rng(2)
    q1 = _bumps(t, rng) + 0.5
    q2 = _bumps(t, rng) + 0.5
    obj = rlbfgs(q1, q2, t)
    l = 1 + 0.2 * np.sin(2 * np.pi * t) + 0.1 * t
    # gamma ends at 0.9, so the differences stay off the clamp at 1
    l = np.sqrt(0.9) * l / obj.norm(l)
    for penalty in ["roughness", "l2gam", "l2psi", "geodesic"]:
        c = obj.costgrad(l, 0.3, penalty)
        grad = c[2] + c[3]
        fd = np.zeros(M)
        for i in range(M):
            e = np.zeros(M)
            e[i] = 1e-6
            cp = obj.costgrad(l + e, 0.3, penalty)
            cm = obj.costgrad(l - e, 0.3, penalty)
            fd[i] = (cp[0] + cp[1] - cm[0] - cm[1]) / 2e-6
        np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "penalty, pen", [("roughness", 0), ("l2gam", 1), ("l2psi", 2), ("geodesic", 3)]
)
def test_rlbfgs_python_matches_compiled(penalty, pen):
    cr = pytest.importorskip("crbfgs")
    M = 101
    t = np.linspace(0, 1, M)
    rng = np.random.default_rng(3)
    import fdasrsf as fs

    q1 = fs.f_to_srsf(_bumps(t, rng), t)
    q2 = fs.f_to_srsf(_bumps(t, rng), t)
    lam = 0.0 if penalty == "roughness" else 0.5
    obj = rlbfgs(q1, q2, t)
    obj.solve(lam=lam, penalty=penalty)
    g, info = cr.rlbfgs(q1, q2, t, 30, lam, pen, return_info=True)
    assert info[0] == obj.iterations
    assert info[1] == obj.stopReason
    assert info[2] == pytest.approx(obj.cost, rel=1e-6)
    np.testing.assert_allclose(g, obj.gammaOpt, atol=1e-6)


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
    assert info[1] in (1, 2, 3, 4)
    assert np.all(np.diff(g) >= 0)
    assert np.max(np.abs(g - gam_true)) * (M - 1) < 0.5
