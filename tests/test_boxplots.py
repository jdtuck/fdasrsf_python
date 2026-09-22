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
