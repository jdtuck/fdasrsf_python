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
