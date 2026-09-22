"""Tests for :mod:`fdasrsf.gp` (Gaussian-process helpers, pure python)."""

import numpy as np
import pytest

from fdasrsf.gp import kernel, gp_posterior


def test_kernel_diagonal_is_one():
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    K = kernel(X, X, l2=0.1)
    assert K.shape == (10, 10)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
    # RBF kernel is symmetric and bounded in (0, 1]
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.all(K > 0) and np.all(K <= 1 + 1e-12)


def test_gp_posterior_interpolates_training_points():
    # noiseless-ish GP posterior should reproduce the training targets when
    # evaluated at the training inputs
    X = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel()
    mu, sd = gp_posterior(X, y, X, l2=0.1, noise_var=1e-8)
    assert mu.shape == y.shape
    np.testing.assert_allclose(mu, y, atol=1e-3)
    # variance at observed points is (near) zero
    assert np.all(sd >= -1e-8)
    np.testing.assert_allclose(sd, 0.0, atol=1e-2)
