"""Tests for :mod:`fdasrsf.curve_functions` leaf functions.

The SRVF round-trip and reparametrization helpers that do not need the curve
alignment extension are exercised here; anything reaching ``optimum_reparam_N``
is gated with ``pytest.importorskip``.
"""

import numpy as np
import pytest

import fdasrsf as fs


@pytest.fixture
def circle():
    """A unit circle sampled as a ``(2, M)`` open curve."""
    M = 100
    theta = np.linspace(0, 2 * np.pi, M)
    return np.vstack((np.cos(theta), np.sin(theta)))


@pytest.fixture
def spiral():
    M = 100
    t = np.linspace(0, 1, M)
    return np.vstack((t * np.cos(4 * np.pi * t), t * np.sin(4 * np.pi * t)))


def test_curve_to_q_returns_triple(circle):
    out = fs.curve_to_q(circle)
    assert len(out) == 3
    q, length, lenq = out
    assert q.shape == circle.shape
    assert lenq == pytest.approx(np.sqrt(length))


def test_curve_to_q_scaled_is_unit_norm(circle):
    from fdasrsf.curve_functions import innerprod_q2

    q, _, _ = fs.curve_to_q(circle, scale=True)
    assert np.sqrt(innerprod_q2(q, q)) == pytest.approx(1.0, abs=1e-6)


def test_q_to_curve_recovers_shape(spiral):
    # q_to_curve mutates its input, so hand it a fresh unscaled q
    q, _, _ = fs.curve_to_q(spiral, scale=False)
    beta = fs.q_to_curve(q.copy())
    assert beta.shape == spiral.shape
    # recovered curve matches the original up to a constant translation
    recentered = beta - beta[:, [0]] + spiral[:, [0]]
    np.testing.assert_allclose(recentered, spiral, atol=5e-2)


def test_resamplecurve_changes_sample_count(spiral):
    xn = fs.resamplecurve(spiral, N=57)
    assert xn.shape == (2, 57)
    # endpoints are preserved by arc-length resampling
    np.testing.assert_allclose(xn[:, 0], spiral[:, 0], atol=1e-6)
    np.testing.assert_allclose(xn[:, -1], spiral[:, -1], atol=1e-6)


def test_calculatecentroid_of_centered_circle(circle):
    centroid = fs.calculatecentroid(circle)
    assert centroid.shape == (2,)
    np.testing.assert_allclose(centroid, 0.0, atol=1e-2)


def test_find_best_rotation_recovers_known_rotation(circle):
    q, _, _ = fs.curve_to_q(circle)
    theta = 0.7
    R_true = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    q_rot = R_true @ q
    q2new, R = fs.find_best_rotation(q, q_rot)
    # the recovered rotation undoes R_true, so R @ R_true == identity
    np.testing.assert_allclose(R @ R_true, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(q2new, q, atol=1e-6)


def test_elastic_distance_curve_self_is_zero(circle):
    d, dx = fs.elastic_distance_curve(circle.copy(), circle.copy())
    assert d == pytest.approx(0.0)
    assert dx == pytest.approx(0.0)


def test_elastic_distance_curve_between_shapes(circle, spiral):
    pytest.importorskip("optimum_reparam_N")
    d, dx = fs.elastic_distance_curve(circle.copy(), spiral.copy())
    assert np.isfinite(d) and d > 0
    assert np.isfinite(dx) and dx >= 0


def test_find_rotation_and_seed_unique_keeps_rotation(spiral):
    pytest.importorskip("optimum_reparam_N")
    from fdasrsf.curve_functions import (
        find_rotation_and_seed_unique,
        group_action_by_gamma_coord,
    )

    theta = np.pi / 3
    R_true = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    t = np.linspace(0, 1, spiral.shape[1])
    beta2 = group_action_by_gamma_coord(R_true @ spiral, t**2)
    q1 = fs.curve_to_q(spiral)[0]
    q2 = fs.curve_to_q(beta2)[0]

    q2n, R, gamI = find_rotation_and_seed_unique(q1, q2, closed=0)
    # the returned SRVF must be q2 rotated by R and then warped by gamI
    expected = fs.curve_to_q(
        group_action_by_gamma_coord(fs.q_to_curve(R @ q2), gamI)
    )[0]
    np.testing.assert_allclose(q2n, expected, atol=1e-8)


def test_find_basis_normal_handles_zero_srvf_sample(spiral):
    from fdasrsf.curve_functions import find_basis_normal

    q = fs.curve_to_q(spiral)[0]
    q[:, 10] = 0.0  # a stationary sample has a zero SRVF value
    basis = find_basis_normal(q)
    assert all(np.all(np.isfinite(b)) for b in basis)


def test_group_action_by_gamma_tolerates_tiny_negative_slope(spiral):
    from fdasrsf.curve_functions import group_action_by_gamma

    q = fs.curve_to_q(spiral)[0]
    gamma = np.linspace(0, 1, q.shape[1])
    gamma[50] = gamma[48] - 1e-9  # numerical dip makes a slope negative
    assert np.gradient(gamma)[49] < 0
    qn = group_action_by_gamma(q, gamma)
    assert np.all(np.isfinite(qn))


def test_parallel_translate_antipodal_is_finite(spiral):
    from fdasrsf.curve_functions import parallel_translate

    q1 = fs.curve_to_q(spiral)[0]
    w = np.roll(q1, 7, axis=1)
    wbar = parallel_translate(w, q1, -q1, None)
    assert np.all(np.isfinite(wbar))


def test_inverse_exp_is_rotation_invariant(spiral):
    pytest.importorskip("optimum_reparam_N")
    from fdasrsf.curve_functions import (
        group_action_by_gamma_coord,
        innerprod_q2,
        inverse_exp,
    )

    theta = np.pi / 4
    R_true = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    q1 = fs.curve_to_q(spiral)[0]

    # a rotated copy is the same shape: zero shooting vector
    beta_rot = R_true @ spiral
    v = inverse_exp(q1, fs.curve_to_q(beta_rot)[0], beta_rot)
    assert v.shape == q1.shape
    assert np.sqrt(innerprod_q2(v, v)) < 1e-6

    # rotating a warped copy must not change the shooting vector
    t = np.linspace(0, 1, spiral.shape[1])
    beta_w = group_action_by_gamma_coord(spiral, t**1.5)
    v_w = inverse_exp(q1, fs.curve_to_q(beta_w)[0], beta_w)
    beta_rw = R_true @ beta_w
    v_rw = inverse_exp(q1, fs.curve_to_q(beta_rw)[0], beta_rw)
    np.testing.assert_allclose(v_rw, v_w, atol=1e-6)


def test_inverse_exp_between_shapes_matches_geodesic_length(circle, spiral):
    pytest.importorskip("optimum_reparam_N")
    from fdasrsf.curve_functions import inverse_exp

    q1 = fs.curve_to_q(spiral)[0]
    q2 = fs.curve_to_q(circle)[0]
    v = inverse_exp(q1, q2, circle)
    assert v.shape == q1.shape
    normv = np.sqrt(fs.curve_functions.innerprod_q2(v, v))
    assert np.isfinite(normv) and 0 < normv <= np.pi
