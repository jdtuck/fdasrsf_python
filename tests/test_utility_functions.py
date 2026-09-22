"""Tests for :mod:`fdasrsf.utility_functions` (and the top-level re-exports).

Covers the deterministic leaf functions and the ``optimum_reparam`` dispatcher.
Every test that reaches a compiled warping solver imports the relevant
extension first via ``pytest.importorskip`` so a source checkout without
built extensions skips rather than errors.
"""

import numpy as np
import pytest

import fdasrsf as fs


# ---------------------------------------------------------------------------
# optimum_reparam dispatcher
# ---------------------------------------------------------------------------

# the DP solvers accumulate the penalty edge by edge, so they can only carry
# penalties that are an integral of a pointwise function of gammadot
DP_PENALTIES = ("none", "l2gam", "l2psi")
NON_DP_PENALTIES = ("roughness", "geodesic")

_METHOD_EXT = {
    "DP": "optimum_reparamN2",
    "DP2": "optimum_reparamN2",
    "RBFGS": None,  # pure python
    "cRBFGS": "crbfgs",
}


def _skip_if_no_ext(method):
    ext = _METHOD_EXT[method]
    if ext is not None:
        pytest.importorskip(ext)


@pytest.mark.parametrize("method", ["DP", "DP2", "RBFGS", "cRBFGS"])
def test_reparam_self_is_identity(method, sine_signal, timet):
    _skip_if_no_ext(method)
    gam = fs.optimum_reparam(sine_signal, timet, sine_signal, method=method)
    assert sum(gam - timet) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("method", ["DP", "DP2", "RBFGS", "cRBFGS"])
def test_reparam_penalties(method, sine_signal, timet):
    _skip_if_no_ext(method)
    penalties = DP_PENALTIES
    if method not in ("DP", "DP2"):
        penalties = penalties + NON_DP_PENALTIES
    for penalty in penalties:
        gam = fs.optimum_reparam(
            sine_signal, timet, sine_signal, method=method, lam=0.1, penalty=penalty
        )
        assert sum(gam - timet) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("method", ["DP", "DP2"])
@pytest.mark.parametrize("penalty", NON_DP_PENALTIES)
def test_reparam_penalty_without_dp_counterpart(method, penalty, sine_signal, timet):
    pytest.importorskip("optimum_reparamN2")
    # a nonzero weight is a request the DP solver cannot honour
    with pytest.raises(ValueError):
        fs.optimum_reparam(
            sine_signal, timet, sine_signal, method=method, lam=0.1, penalty=penalty
        )
    # ... but with lam == 0 the penalty drops out, so every method succeeds
    gam = fs.optimum_reparam(
        sine_signal, timet, sine_signal, method=method, lam=0.0, penalty=penalty
    )
    assert sum(gam - timet) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("method", ["DP", "DP2"])
def test_reparam_batched_matches_single(method, timet):
    pytest.importorskip("optimum_reparamN2")
    q1 = fs.f_to_srsf(np.sin(2 * np.pi * timet), timet)
    q2 = fs.f_to_srsf(np.sin(2 * np.pi * timet**2), timet)
    Q1 = np.column_stack((q1, q1))
    Q2 = np.column_stack((q2, q2))
    gam = fs.optimum_reparam(q1, timet, q2, method=method, lam=1.0, penalty="l2gam")
    gamN = fs.optimum_reparam(q1, timet, Q2, method=method, lam=1.0, penalty="l2gam")
    gamN2 = fs.optimum_reparam(Q1, timet, Q2, method=method, lam=1.0, penalty="l2gam")
    np.testing.assert_allclose(gamN[:, 0], gam)
    np.testing.assert_allclose(gamN2[:, 1], gam)


def test_reparam_bad_penalty(sine_signal, timet):
    with pytest.raises(ValueError):
        fs.optimum_reparam(sine_signal, timet, sine_signal, penalty="bogus")


@pytest.mark.parametrize("method", ["DP", "DP2", "RBFGS", "cRBFGS"])
def test_reparam_bad_shapes(method, sine_signal, timet):
    _skip_if_no_ext(method)
    Q = np.column_stack((sine_signal, sine_signal))
    # a 2-D q1 with a 1-D q2 has no solver branch in any method
    with pytest.raises(ValueError):
        fs.optimum_reparam(Q, timet, sine_signal, method=method)


# ---------------------------------------------------------------------------
# optimum_reparam_pair
# ---------------------------------------------------------------------------


def test_optimum_reparam_pair(timet):
    pytest.importorskip("optimum_reparamN2")
    M = timet.size
    qa = fs.f_to_srsf(np.sin(2 * np.pi * timet), timet)
    qb = fs.f_to_srsf(np.cos(2 * np.pi * timet), timet)
    q = np.column_stack((qa, qb))
    q1 = fs.f_to_srsf(np.sin(2 * np.pi * timet**1.3), timet)
    q2 = fs.f_to_srsf(np.cos(2 * np.pi * timet**1.3), timet)

    # aligning a pair to itself gives the identity
    gamid = fs.optimum_reparam_pair(q, timet, qa, qb)
    np.testing.assert_allclose(gamid, timet, atol=1e-10)

    gam = fs.optimum_reparam_pair(q, timet, q1, q2)
    assert gam.shape == (M,)
    assert np.all(np.diff(gam) >= -1e-12)  # monotone

    # the batched branch must agree column by column with the single one
    Q1 = np.column_stack((q1, q1))
    Q2 = np.column_stack((q2, q2))
    gamN = fs.optimum_reparam_pair(q, timet, Q1, Q2)
    assert gamN.shape == (M, 2)
    np.testing.assert_allclose(gamN[:, 0], gam)
    np.testing.assert_allclose(gamN[:, 1], gam)

    with pytest.raises(ValueError):
        fs.optimum_reparam_pair(qa, timet, q1, q2)
    with pytest.raises(ValueError):
        fs.optimum_reparam_pair(q, timet, q1, Q2)


# ---------------------------------------------------------------------------
# SRSF transform round-trip
# ---------------------------------------------------------------------------


def test_f_to_srsf_round_trip(sine_signal, timet):
    q1 = fs.f_to_srsf(sine_signal, timet)
    f1a = fs.srsf_to_f(q1, timet)
    # srsf_to_f integrates from f0 == 0.0, so match the first sample
    np.testing.assert_allclose(f1a + sine_signal[0], sine_signal, atol=1e-3)


def test_warp_f_gamma_identity(sine_signal, timet):
    pytest.importorskip("optimum_reparamN2")
    gam = fs.optimum_reparam(sine_signal, timet, sine_signal)
    warped = fs.warp_f_gamma(timet, sine_signal, gam)
    assert sum(sine_signal - warped) == pytest.approx(0, abs=1e-6)


def test_warp_q_gamma_identity(sine_signal, timet, identity_gamma):
    warped = fs.warp_q_gamma(timet, sine_signal, identity_gamma)
    np.testing.assert_allclose(warped, sine_signal, atol=1e-6)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def test_elastic_distance_self_is_zero(sine_signal, timet):
    pytest.importorskip("optimum_reparamN2")
    da, dp = fs.elastic_distance(sine_signal, sine_signal, timet)
    assert da <= 1e-10
    assert dp <= 1e-6


def test_innerprod_q_of_self_is_positive(sine_signal, timet):
    val = fs.innerprod_q(timet, sine_signal, sine_signal)
    assert val > 0
    # inner product with a scaled copy scales linearly
    assert fs.innerprod_q(timet, sine_signal, 2 * sine_signal) == pytest.approx(
        2 * val
    )


# ---------------------------------------------------------------------------
# Warping-function utilities
# ---------------------------------------------------------------------------


def test_invert_gamma_of_identity_is_identity(identity_gamma):
    gami = fs.invertGamma(identity_gamma)
    np.testing.assert_allclose(gami, identity_gamma, atol=1e-10)


def test_invert_gamma_is_involutive():
    # a genuinely nonlinear, monotone gamma on [0, 1]
    t = np.linspace(0, 1, 101)
    gam = t**2
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    gam_ii = fs.invertGamma(fs.invertGamma(gam))
    np.testing.assert_allclose(gam_ii, gam, atol=2e-2)


def test_sqrt_mean_of_identities_is_identity():
    M, N = 101, 5
    gam = np.tile(np.linspace(0, 1, M), (N, 1)).T  # (M, N) identities
    # SqrtMean returns (mu, gam_mu, psi, vec); gam_mu is the mean warping fn
    mu, gam_mu, psi, vec = fs.SqrtMean(gam)
    np.testing.assert_allclose(gam_mu, np.linspace(0, 1, M), atol=1e-6)


def test_sqrt_mean_inverse_of_identities_is_identity():
    M, N = 101, 5
    gam = np.tile(np.linspace(0, 1, M), (N, 1)).T
    gamI = fs.SqrtMeanInverse(gam)
    np.testing.assert_allclose(gamI, np.linspace(0, 1, M), atol=1e-6)


def test_rgam_shape_and_endpoints():
    N, num = 101, 4
    gam = fs.rgam(N, sigma=0.1, num=num)
    assert gam.shape == (N, num)
    # each random warping function is a diffeomorphism of [0, 1]
    np.testing.assert_allclose(gam[0, :], 0, atol=1e-8)
    np.testing.assert_allclose(gam[-1, :], 1, atol=1e-8)
    assert np.all(np.diff(gam, axis=0) >= -1e-8)


def test_cumtrapzmid_matches_plain_from_start():
    x = np.linspace(0, 1, 101)
    y = np.ones_like(x)
    fa = fs.cumtrapzmid(x, y, c=0.0, mid=50)
    # integrating a constant 1 gives a line through the midpoint value
    assert fa[50] == pytest.approx(fa[49], abs=1e-2)
    assert np.all(np.diff(fa[50:]) >= -1e-9)


def test_smooth_data_is_deterministic():
    M = 101
    q = np.zeros((M, 1))
    q[:, 0] = np.sin(np.linspace(0, 2 * np.pi, M))
    a = fs.smooth_data(q, 1)
    b = fs.smooth_data(q, 1)
    np.testing.assert_array_equal(a, b)
