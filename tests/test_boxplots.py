"""Tests for :mod:`fdasrsf.boxplots` (amplitude / phase boxplots).

Both boxplot classes require a *median*-aligned ``fdawarp`` (the session-scoped
``median_fdawarp`` fixture); constructing them from a mean alignment raises.
"""

import numpy as np
import pytest

import fdasrsf as fs


@pytest.mark.parametrize("cls_name", ["ampbox", "phbox"])
def test_construct_boxplot_runs(median_fdawarp, cls_name):
    cls = getattr(fs, cls_name)
    obj = cls(median_fdawarp)
    obj.construct_boxplot(alpha=0.05, k_a=1)
    # the median curve and the quartile envelopes are populated and finite
    assert np.all(np.isfinite(obj.Q1))
    assert np.all(np.isfinite(obj.Q1a))


@pytest.fixture(scope="module")
def boxplots(median_fdawarp):
    amp = fs.ampbox(median_fdawarp)
    amp.construct_boxplot(alpha=0.05, k_a=1)
    ph = fs.phbox(median_fdawarp)
    ph.construct_boxplot(alpha=0.05, k_a=1)
    return amp, ph


def test_region_edges(boxplots):
    # construct_boxplot stores the region edges it uses
    for b in boxplots:
        N = b.dist.shape[0]
        d = np.sort(b.dist)
        assert b.r50 == d[int(np.ceil(N / 2)) - 1]
        assert b.ralpha == d[int(np.round(N * (1 - b.alpha))) - 1]
        out = b.outlier_index.astype(int)
        assert np.all(b.dist[out] > b.whisker_dis)
        assert np.all(np.delete(b.dist, out) <= b.whisker_dis)


def test_boxplot_region_levels():
    # the whisker may fall inside ralpha; beyond it is always an outlier
    from fdasrsf.boxplots import boxplot_region

    level, region = boxplot_region([0.5, 1.5, 2.5, 4], [1, 3, 2], 0.05)
    np.testing.assert_array_equal(level, [1, 2, 4, 4])
    assert region[1] == "95% central region"
    assert region[0] == "50% central region"


def test_classify_outputs(boxplots, simu_data):
    f, _ = simu_data
    fnew = f[:, 0:3]
    for b in boxplots:
        c = b.classify(fnew)
        assert c.dist.shape == (3,)
        assert np.all(np.isfinite(c.dist))
        assert np.all(np.isin(c.level, [1, 2, 3, 4]))
        assert np.all(np.isin(c.side, ["Q1", "Q3"]))
        assert c.gam.shape == fnew.shape
        assert c.fn.shape == fnew.shape
        np.testing.assert_allclose(c.gam[0, :], 0, atol=1e-12)
        np.testing.assert_allclose(c.gam[-1, :], 1, atol=1e-12)


def test_classify_amplitude_median_and_outlier(boxplots, median_fdawarp, simu_data):
    amp = boxplots[0]
    c = amp.classify(median_fdawarp.fmean)
    assert c.level[0] == 1, "the template itself should be in the 50% region"
    f, _ = simu_data
    c = amp.classify(10 * f[:, 0])
    assert c.level[0] == 4, "a 10x scaled function should be an amplitude outlier"


@pytest.mark.parametrize("cls_name", ["ampbox", "phbox"])
def test_classify_requires_construct(median_fdawarp, cls_name):
    with pytest.raises(Exception, match="construct_boxplot"):
        getattr(fs, cls_name)(median_fdawarp).classify(median_fdawarp.f[:, 0])


def test_align_new_requires_alignment(simu_data):
    f, time = simu_data
    with pytest.raises(Exception, match="align"):
        fs.fdawarp(f, time).align_new(f[:, 0])


def test_align_new_matches_fit(median_fdawarp):
    # aligning the fitted data again gives warps that fit the template
    f = median_fdawarp.f[:, 0:2]
    fn, qn, gam = median_fdawarp.align_new(f)
    assert fn.shape == qn.shape == gam.shape == f.shape
    assert np.all(np.diff(gam, axis=0) >= 0)
    with pytest.raises(Exception):
        median_fdawarp.align_new(f[:-1, :])
