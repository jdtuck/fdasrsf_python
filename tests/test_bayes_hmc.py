"""Tests for the Bayesian HMC alignment (pairwise and group).

Mirrors ``tests/BayesHMCTest.m`` of fdasrvf_MATLAB: the exact likelihood
gradient used by the infinite-dimensional HMC sampler is checked against
finite differences for both warp representations, the batched geometry
against the single-warp one, and short pairwise and group runs are checked
to return valid warps close to the dynamic programming solution.
"""

import numpy as np
import pytest

import fdasrsf as fs
import fdasrsf.bayesian_functions as bf
import fdasrsf.utility_functions as uf


def small_problem():
    T = 101
    t = np.linspace(0, 1, T)
    Phi = bf.bayes_warp_basis(t, 10, "fourier")
    q1 = uf.f_to_srsf(np.exp(-((t - 0.4) ** 2) / 0.01), t)
    q2 = uf.f_to_srsf(np.exp(-((t - 0.6) ** 2) / 0.01), t)
    return Phi, q1, q2


def test_warp_basis_is_orthonormal_and_centered():
    T = 101
    t = np.linspace(0, 1, T)
    w, dt = bf._trapz_weights(T)
    # the bases are zero-mean and orthonormal in L2[0,1]; the trapezoid rule
    # is exact for the fourier basis up to round-off and O(h^2) for the
    # polynomials, so the legendre tolerance is the quadrature error
    for basis, ncoef, tol in (("fourier", 10, 1e-10), ("legendre", 6, 2e-2)):
        Phi = bf.bayes_warp_basis(t, ncoef, basis)
        assert Phi.shape == (T, ncoef)
        np.testing.assert_allclose(Phi.T @ w, np.zeros(ncoef), atol=tol)
        np.testing.assert_allclose(
            Phi.T @ (w[:, None] * Phi), np.eye(ncoef), atol=tol
        )


def test_gradient_matches_finite_difference():
    Phi, q1, q2 = small_problem()
    rng = np.random.default_rng(1)
    c = 0.2 * rng.standard_normal(Phi.shape[1])
    for rep in (0, 1):
        nll, g, SSE, gam, q2w = bf.bayes_warp_geom(c, Phi, rep, q1, q2, 0.1)
        fd = np.zeros(c.shape[0])
        e = 1e-7
        for i in range(0, c.shape[0]):
            cp = c.copy()
            cp[i] += e
            cm = c.copy()
            cm[i] -= e
            fd[i] = (
                bf.bayes_warp_geom(cp, Phi, rep, q1, q2, 0.1, grad=False)[0]
                - bf.bayes_warp_geom(cm, Phi, rep, q1, q2, 0.1, grad=False)[0]
            ) / (2 * e)
        assert np.abs(fd - g).max() / np.abs(fd).max() < 1e-5


def test_shooting_outside_injectivity_radius():
    Phi, q1, q2 = small_problem()
    c = np.zeros(Phi.shape[1])
    c[0] = 4  # |v| = 4 > pi
    assert bf.bayes_warp_geom(c, Phi, 0, q1, q2, 0.1)[0] == np.inf


def test_coef_gam_round_trip():
    Phi, q1, q2 = small_problem()
    rng = np.random.default_rng(2)
    c = 0.1 * rng.standard_normal((Phi.shape[1], 3))
    for rep in (0, 1):
        gam, psi = bf.bayes_coef_to_gam(c, Phi, rep)
        np.testing.assert_allclose(gam[0, :], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(gam[-1, :], np.ones(3), atol=1e-12)
        assert np.all(np.diff(gam, axis=0) >= 0)
        c2 = bf.bayes_gam_to_coef(gam, Phi, rep)
        assert np.abs(c2 - c).max() < 1e-2


def test_batched_geom_matches_single(simu_data):
    f, time = simu_data
    Phi, q1, _ = small_problem()
    tt = np.linspace(0, 1, Phi.shape[0])
    Q = uf.f_to_srsf(f[:, 0:5], tt)
    rng = np.random.default_rng(6)
    c = 0.2 * rng.standard_normal((Phi.shape[1], 5))
    for rep in (0, 1):
        n, g, s, gam, w = bf.bayes_warp_geom(c, Phi, rep, q1, Q, 0.1)
        for i in range(0, 5):
            n1, g1, s1, gam1, w1 = bf.bayes_warp_geom(
                c[:, i], Phi, rep, q1, Q[:, i], 0.1
            )
            np.testing.assert_allclose([n[i], s[i]], [n1, s1], atol=1e-10)
            np.testing.assert_allclose(g[:, i], g1, atol=1e-10)
            np.testing.assert_allclose(gam[:, i], gam1, atol=1e-13)
            np.testing.assert_allclose(w[:, i], w1, atol=1e-13)


def test_batched_hmc_samples_gaussian():
    # nll = 0: the target is the prior N(0, diag(C)), which the rotation
    # integrates exactly (every proposal is accepted)
    np.random.seed(8)
    C = np.array([1.0, 0.25, 0.04])

    def zero_geom(c):
        return np.zeros(c.shape[1]), np.zeros(c.shape), None

    c = np.zeros((3, 2000))
    nll, g, aux = zero_geom(c)
    for k in range(0, 20):
        c, nll, g, aux, acc, alpha = bf.infhmc_step_batch(
            c, nll, g, aux, zero_geom, C, 0.7, 5
        )
    assert np.all(acc)
    np.testing.assert_allclose(c.var(axis=1), C, rtol=0.1)


def test_dual_avg_step_reaches_target():
    # a step size whose acceptance probability decreases with epsilon: the
    # adaptation should settle where alpha equals the target
    state = bf.dual_avg_step(np.array(0.1))
    eps = 0.1
    for n in range(1, 400):
        alpha = np.exp(-eps / 0.05)
        eps, state = bf.dual_avg_step(state, alpha, 0.65)
    eps_bar = np.exp(state["log_eps_bar"])
    assert abs(np.exp(-eps_bar / 0.05) - 0.65) < 0.05


@pytest.mark.slow
@pytest.mark.parametrize("warp_rep", ["shooting", "clr"])
def test_pairwise_alignment(simu_data, warp_rep):
    pytest.importorskip("optimum_reparamN2")
    f, time = simu_data
    np.random.seed(4)
    y1 = f[:, 0] + 0.01 * np.random.randn(time.shape[0])
    y2 = f[:, 20] + 0.01 * np.random.randn(time.shape[0])
    f2n, gdp, q2n = fs.pairwise_align_functions(f[:, 0], f[:, 20], time)
    out = fs.pairwise_align_bayes_infHMC(
        y1, y2, time, {"iter": 1000, "thin": 1, "L": 10, "warp_rep": warp_rep}
    )
    assert out.theta_accept[500:].mean() > 0.3
    assert np.abs(out.gamma[:, 0] - gdp).max() < 0.08
    assert out.gamma_mat.shape == (time.shape[0], 501)
    assert np.all(np.diff(out.gamma[:, 0]) >= -1e-12)


@pytest.mark.slow
def test_group_alignment_infhmc(simu_data):
    pytest.importorskip("optimum_reparamN2")
    f, time = simu_data
    M, N = f.shape
    dp = fs.fdawarp(f, time)
    dp.srsf_align(MaxItr=10, verbose=False)
    np.random.seed(5)
    obj = fs.fdawarp(f, time)
    obj.time_warping_bayes_infHMC(iter=600, verbose=False)
    assert obj.gam.shape == (M, N)
    assert np.all(np.diff(obj.gam, axis=0) >= -1e-12)
    assert np.abs(obj.gam - dp.gam).mean() < 0.02
    assert obj.mcmc["accept"][:, 300:].mean() > 0.4
    fn = np.zeros((M, N))
    for k in range(0, N):
        fn[:, k] = uf.warp_f_gamma(time, f[:, k], obj.gam[:, k])
    assert np.abs(fn - obj.fn).max() < 1e-12
    assert obj.mcmc["gamma_stats"].shape == (M, 2, N)
    assert obj.mcmc["epsilon"].shape == (N, 600)


@pytest.mark.slow
def test_group_alignment_pcn(simu_data):
    pytest.importorskip("cbayesian")
    f, time = simu_data
    M, N = f.shape
    np.random.seed(1)
    obj = fs.fdawarp(f, time)
    obj.time_warping_bayes(iter=150, burnin=75, verbose=False)
    assert obj.gam.shape == (M, N)
    assert np.all(np.diff(obj.gam, axis=0) >= -1e-10)
    np.testing.assert_allclose(obj.gam[0, :], np.zeros(N), atol=1e-10)
    np.testing.assert_allclose(obj.gam[-1, :], np.ones(N), atol=1e-10)
    assert np.isfinite(obj.fmean).all()
    assert obj.mcmc["gamma_mat"].shape == (M, 76, N)
    assert obj.mcmc["g_coef"].shape == (150, 20, N)
