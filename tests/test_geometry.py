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
