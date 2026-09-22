"""Tests for :mod:`fdasrsf.image_functions` (pure-python image helpers)."""

import numpy as np

import fdasrsf as fs


def test_apply_gam_imag_identity_leaves_image_unchanged():
    # the identity diffeomorphism must leave the image alone
    m, n = 41, 33
    U = np.linspace(0, 1, m)
    V = np.linspace(0, 1, n)
    F = np.sin(2 * np.pi * U[:, None]) * np.cos(3 * np.pi * V[None, :])

    gamid = fs.makediffeoid(m, n)
    np.testing.assert_allclose(fs.apply_gam_imag(F, gamid), F, atol=1e-12)


def test_apply_gam_imag_samples_at_named_points():
    # a genuine 2-D diffeomorphism must sample the image at the points it names
    m, n = 41, 33
    U = np.linspace(0, 1, m)
    V = np.linspace(0, 1, n)
    F = np.sin(2 * np.pi * U[:, None]) * np.cos(3 * np.pi * V[None, :])

    UU, VV = np.meshgrid(U, V, indexing="ij")
    bump = 0.1 * np.sin(np.pi * UU) * np.sin(np.pi * VV)
    gam = np.zeros((m, n, 2))
    gam[:, :, 0] = VV - bump
    gam[:, :, 1] = UU + bump
    expected = np.sin(2 * np.pi * gam[:, :, 1]) * np.cos(3 * np.pi * gam[:, :, 0])
    # residual is the bilinear interpolation error on this grid
    np.testing.assert_allclose(fs.apply_gam_imag(F, gam), expected, atol=2e-2)


def test_makediffeoid_shape():
    gamid = fs.makediffeoid(20, 15)
    assert gamid.shape == (20, 15, 2)
