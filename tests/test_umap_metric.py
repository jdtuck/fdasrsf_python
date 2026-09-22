"""Tests for :mod:`fdasrsf.umap_metric` (needs the ``_DP`` cffi extension)."""

import numpy as np
import pytest


def test_efda_distance_self_is_zero():
    pytest.importorskip("_DP")
    from fdasrsf.umap_metric import efda_distance

    M = 101
    q1 = np.sin(np.linspace(0, 2 * np.pi, M))
    assert efda_distance(q1, q1) == 0.0 or abs(efda_distance(q1, q1)) < 1e-6


def test_efda_distance_between_signals_is_positive():
    pytest.importorskip("_DP")
    from fdasrsf.umap_metric import efda_distance

    M = 101
    q1 = np.sin(np.linspace(0, 2 * np.pi, M))
    q2 = np.cos(np.linspace(0, 2 * np.pi, M))
    d = efda_distance(q1, q2)
    assert np.isfinite(d)
    assert d > 0
