"""Tests for :mod:`fdasrsf.curve_stats.fdacurve` (shape statistics).

These use the MPEG7 sample curves via the session-scoped ``karcher_fdacurve``
fixture and reproduce the golden ``E[-1]`` value from the legacy suite.
"""

import numpy as np
import pytest

import fdasrsf as fs


def test_karcher_mean_populates_mean(karcher_fdacurve):
    obj = karcher_fdacurve
    assert obj.beta_mean is not None
    assert obj.q_mean is not None
    assert obj.E.ndim == 1 and obj.E.shape[0] >= 1


def test_karcher_mean_golden_energy(karcher_fdacurve):
    # regression value; updated after find_rotation_and_seed_unique was fixed to
    # keep the optimal rotation when applying the reparameterization (the old
    # value, 0.022668183569717587, reflected curves aligned without rotation)
    assert karcher_fdacurve.E[-1] == pytest.approx(0.029669729165135855, rel=1e-6)


@pytest.mark.slow
def test_srvf_align_and_pca(mpeg7_beta):
    # build a dedicated object so the shared session fixture is not mutated
    n, M, K = mpeg7_beta.shape
    obj = fs.fdacurve(mpeg7_beta, N=M)
    obj.karcher_mean()
    obj.srvf_align()
    obj.karcher_cov()
    obj.shape_pca()
    # aligned curves keep the (n, T, K) layout of the input
    assert obj.betan.ndim == 3
    assert np.all(np.isfinite(obj.beta_mean))


def test_karcher_calc_projects_onto_tangent_at_mean(mpeg7_beta):
    pytest.importorskip("optimum_reparam_N")
    from fdasrsf import curve_functions as cf
    from fdasrsf.curve_stats import karcher_calc

    mu = cf.project_curve(cf.curve_to_q(mpeg7_beta[:, :, 0])[0])
    q = cf.project_curve(cf.curve_to_q(mpeg7_beta[:, :, 1])[0])
    basis = cf.find_basis_normal(mu)
    v, _, _ = karcher_calc(mu, q, basis, 1, 0.0, True, "DP")
    # the shooting vector lives in the tangent space at mu
    assert cf.innerprod_q2(v, mu) == pytest.approx(0.0, abs=1e-4)
    bo = cf.gram_schmidt(basis)
    for b in bo:
        assert cf.innerprod_q2(v, b) == pytest.approx(0.0, abs=1e-4)
