"""Tests for :mod:`fdasrsf.regression` (elastic function-on-scalar regression).

These fit iterative models and are marked ``slow`` (deselected by default);
they use small synthetic data and few iterations so they still finish quickly
when explicitly selected with ``pytest -m slow`` or ``pytest -m ''``.
"""

import numpy as np
import pytest

import fdasrsf as fs


@pytest.fixture
def scalar_regression_data():
    """A small function-on-scalar dataset: response = peak location."""
    rng = np.random.default_rng(1)
    M, N = 60, 12
    time = np.linspace(0, 1, M)
    centers = rng.uniform(0.3, 0.7, N)
    f = np.zeros((M, N))
    for i in range(N):
        f[:, i] = np.exp(-((time - centers[i]) ** 2) / 0.01)
    y = centers  # continuous response
    return f, y, time


@pytest.mark.slow
def test_elastic_regression_fits_and_predicts(scalar_regression_data):
    pytest.importorskip("optimum_reparamN2")
    f, y, time = scalar_regression_data
    model = fs.elastic_regression(f, y, time)
    model.calc_model(max_itr=2, df=10)
    model.predict()
    assert np.all(np.isfinite(model.y_pred))
    assert model.y_pred.shape[0] == y.shape[0]


@pytest.mark.slow
def test_elastic_logistic_fits(scalar_regression_data):
    pytest.importorskip("optimum_reparamN2")
    f, y_cont, time = scalar_regression_data
    y = np.where(y_cont > 0.5, 1, -1)  # binary labels in {-1, 1}
    model = fs.elastic_logistic(f, y, time)
    model.calc_model(max_itr=2, df=10)
    assert np.all(np.isfinite(np.atleast_1d(model.alpha)))
