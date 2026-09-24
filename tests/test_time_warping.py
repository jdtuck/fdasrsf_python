"""Tests for :mod:`fdasrsf.time_warping`.

``pairwise_align_functions`` is a fast DP path; the ``fdawarp.srsf_align``
integration test reproduces the golden amplitude variance from the legacy
suite and needs the ``optimum_reparamN2`` extension plus the sample data.
"""

import numpy as np
import pytest

import fdasrsf as fs


def test_pairwise_align_functions_self_is_identity():
    pytest.importorskip("optimum_reparamN2")
    M = 101
    t = np.linspace(0, 1, M)
    f1 = np.sin(2 * np.pi * t)
    f2n, gam, q2n = fs.pairwise_align_functions(f1, f1, t)
    np.testing.assert_allclose(gam, t, atol=1e-6)
    np.testing.assert_allclose(f2n, f1, atol=1e-6)


def test_pairwise_align_functions_aligns_warped_copy():
    pytest.importorskip("optimum_reparamN2")
    M = 201
    t = np.linspace(0, 1, M)
    f1 = np.exp(-((t - 0.4) ** 2) / 0.01)
    f2 = np.exp(-((t - 0.6) ** 2) / 0.01)  # same bump, shifted
    f2n, gam, q2n = fs.pairwise_align_functions(f1, f2, t)
    # alignment brings f2 closer to f1 than it started
    assert np.linalg.norm(f2n - f1) < np.linalg.norm(f2 - f1)
    assert np.all(np.diff(gam) >= -1e-9)  # monotone warp


def test_srsf_align_golden_amp_var(aligned_fdawarp):
    # regression value carried over from the legacy test/test_all.py suite
    assert aligned_fdawarp.amp_var == pytest.approx(0.018998691036349585, rel=1e-6)


def test_srsf_align_populates_outputs(aligned_fdawarp):
    obj = aligned_fdawarp
    assert obj.fn.shape == obj.f.shape
    assert obj.qn.shape == obj.fn.shape
    assert obj.gam.shape == obj.fn.shape
    assert np.isfinite(obj.phase_var)


@pytest.mark.slow
def test_align_fPCA_runs(simu_data):
    pytest.importorskip("optimum_reparamN2")
    f, time = simu_data
    obj = fs.align_fPCA(f, time, num_comp=2, showplot=False)
    assert obj.fn.shape == f.shape


def test_align_fPCA_gam_includes_final_warp(simu_data, monkeypatch):
    """After an early stop, gam must include the warp applied last."""
    pytest.importorskip("optimum_reparamN2")
    import fdasrsf.time_warping as tw

    f, time = simu_data
    f = f[:, :10]
    M, N = f.shape
    g = np.linspace(0, 1, M) ** 1.5

    # every matching step applies the same known warp g ...
    monkeypatch.setattr(
        tw.uf, "optimum_reparam", lambda *args, **kw: np.tile(g, (N, 1)).T
    )
    # ... and a constant cost stops the loop after the second step
    monkeypatch.setattr(tw, "norm", lambda x: 1.0)

    obj = fs.align_fPCA(f, time, num_comp=2, showplot=False)

    # fn is f warped by g twice (then centred), so gam must reproduce it
    for k in range(N):
        t0 = (time[-1] - time[0]) * obj.gam[:, k] + time[0]
        fk = np.interp(t0, time, f[:, k])
        np.testing.assert_allclose(fk, obj.fn[:, k], atol=0.02 * np.abs(f).max())


def test_gauss_model_samples(aligned_fdawarp):
    np.random.seed(0)
    aligned_fdawarp.gauss_model(n=3)
    assert aligned_fdawarp.fs.shape == (aligned_fdawarp.fn.shape[0], 3)
    assert np.all(np.isfinite(aligned_fdawarp.fs))
