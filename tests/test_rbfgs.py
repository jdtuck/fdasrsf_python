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
