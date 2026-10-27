"""Tests for :mod:`fdasrsf.geometry` (pure python, no compiled extensions)."""

import numpy as np
import pytest

import fdasrsf as fs
import fdasrsf.geometry as geo


@pytest.fixture
def identity_psi():
    """The SRSF (psi) of the identity warping function: constant one."""
    M = 101
    gam = np.linspace(0, 1, M)
    binsize = np.mean(np.diff(gam))
    return np.sqrt(np.gradient(gam, binsize))


def test_l2norm_of_zero_is_zero():
    assert geo.L2norm(np.zeros(101)) == pytest.approx(0.0)


def test_l2norm_of_constant_one():
    # ||1|| over [0, 1] is 1
    assert geo.L2norm(np.ones(101)) == pytest.approx(1.0, abs=1e-6)


def test_inner_product_of_constant_one():
    assert geo.inner_product(np.ones(101), np.ones(101)) == pytest.approx(1.0, abs=1e-6)


def test_inv_exp_map_of_self_is_zero(identity_psi):
    out, theta = fs.inv_exp_map(identity_psi, identity_psi)
    assert sum(out) == pytest.approx(0.0, abs=1e-6)
    assert geo.L2norm(out) == pytest.approx(0.0, abs=1e-6)


def test_exp_inv_exp_round_trip(identity_psi):
    out, theta = fs.inv_exp_map(identity_psi, identity_psi)
    back = fs.exp_map(identity_psi, out)
    # exp of the zero tangent vector returns the base point
    np.testing.assert_allclose(back, identity_psi, atol=1e-6)


@pytest.mark.parametrize(
    "to_fn, from_fn",
    [
        ("gam_to_psi", "psi_to_gam"),
        ("gam_to_v", "v_to_gam"),
        ("gam_to_h", "h_to_gam"),
    ],
)
def test_gamma_representation_round_trip(to_fn, from_fn):
    # a smooth, monotone, nonlinear gamma on [0, 1]
    t = np.linspace(0, 1, 201)
    gam = (np.exp(2 * t) - 1) / (np.exp(2) - 1)
    rep = getattr(geo, to_fn)(gam, smooth=False)
    gam_back = getattr(geo, from_fn)(rep)
    np.testing.assert_allclose(gam_back, gam, atol=2e-2)


def test_inv_exp_map_small_angle(identity_psi):
    # the angle stays accurate for nearby points, where arccos of the inner
    # product would round to zero or ~1e-8
    M = identity_psi.shape[0]
    v0 = np.sin(2 * np.pi * np.linspace(0, 1, M))
    v = v0 - geo.inner_product(v0, identity_psi) * identity_psi
    v = 1e-7 * v / geo.L2norm(v)
    p2 = geo.exp_map(identity_psi, v)
    _, theta = geo.inv_exp_map(identity_psi, p2)
    assert theta == pytest.approx(1e-7, rel=1e-6)


def test_gam_to_psi_is_unit_norm():
    s = np.linspace(0, 1, 101)
    gam = np.column_stack((s, (np.exp(3 * s) - 1) / (np.exp(3) - 1)))
    psi = geo.gam_to_psi(gam)
    np.testing.assert_allclose(
        np.sqrt(np.trapezoid(psi**2, s, axis=0)), [1, 1], rtol=0, atol=1e-12
    )


def test_gam_to_h_flat_segment_is_finite():
    # a warp with a flat segment (gamdot = 0) gives a finite h
    s = np.linspace(0, 1, 101)
    gam = np.interp(s, [0, 0.3, 0.5, 1], [0, 0.4, 0.4, 1])
    assert np.all(np.isfinite(geo.gam_to_h(gam)))
