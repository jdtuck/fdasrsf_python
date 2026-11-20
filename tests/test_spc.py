"""Tests for :mod:`fdasrsf.spc` (elastic functional SPC).

Checks the T^2/SPE statistics and limits for amplitude and phase, and that
amplitude and phase faults raise alarms (port of MATLAB SpcMonitorTest).
"""

import numpy as np
import pytest

import fdasrsf as fs


@pytest.fixture(scope="module")
def warp(simu_data):
    pytest.importorskip("optimum_reparamN2")
    f, time = simu_data
    obj = fs.fdawarp(f, time)
    obj.srsf_align(parallel=False)
    return obj


def test_in_sample_T2_matches_coef(warp):
    # without re-alignment T^2 is the scaled fPCA coefficient norm
    s = fs.fdaspc(warp)
    s.train(realign=False)
    np.testing.assert_allclose(
        s.amp["T2"], (s.vpca.coef**2 / s.vpca.latent).sum(axis=1), atol=1e-10
    )
    np.testing.assert_allclose(
        s.ph["T2"], (s.hpca.coef**2 / s.hpca.latent).sum(axis=1), atol=1e-10
    )


def test_contributions_sum(warp):
    s = fs.fdaspc(warp)
    s.train(no=[3, 2])
    assert s.amp["k"] == 3 and s.ph["k"] == 2
    out = s.monitor(warp.f[:, :4])
    for m in (out["amp"], out["ph"]):
        np.testing.assert_allclose(m["contT2"].sum(axis=1), m["T2"], rtol=1e-8)
        np.testing.assert_allclose(m["contSPE"].sum(axis=1), m["SPE"], rtol=1e-8)


def test_training_mostly_in_control(warp):
    s = fs.fdaspc(warp)
    s.train(alpha=0.05)
    out = s.monitor(warp.f)
    assert out["alarm"].mean() < 0.5


def test_faults_alarm(warp):
    s = fs.fdaspc(warp)
    s.train(alpha=0.05)
    f = warp.f[:, :3]
    t = warp.time
    g = np.clip(t + 0.08 * np.sin(2 * np.pi * t), 0, 1)
    fp = np.column_stack([fs.warp_f_gamma(t, f[:, i], g) for i in range(3)])
    out = s.monitor(np.hstack((1.5 * f, fp)))
    assert out["amp"]["alarm"][:3].all(), "amplitude fault missed"
    assert out["ph"]["alarm"][3:].all(), "phase fault missed"
    np.testing.assert_array_equal(s.mon["alarm"], out["alarm"])


def test_errors_and_plots(warp):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = fs.fdaspc(warp)
    with pytest.raises(Exception):
        s.monitor(warp.f[:, :2])
    with pytest.raises(Exception):
        s.train(alpha=1.5)
    s.train(var_exp=[0.9, 0.95], log_der=True)
    s.plot()
    s.monitor(warp.f[:, :2])
    s.plot()
    s.plot_contrib(0)
    plt.close("all")


def test_phase_features_match_hpca(warp):
    # monitored phase features use the same psi map as the phase basis
    from fdasrsf.spc import _ph_features

    s = fs.fdaspc(warp)
    s.train(realign=False)
    np.testing.assert_allclose(_ph_features(s.hpca, warp.gam), s.hpca.vec, atol=1e-10)
