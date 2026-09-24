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


# ---------------------------------------------------------------------------
# Multinomial logistic loss: softmax convention
# ---------------------------------------------------------------------------


@pytest.fixture
def mlogit_design():
    """Three well-separated classes in a two-feature design (plus intercept)."""
    rng = np.random.default_rng(0)
    m, N = 3, 300
    cls = rng.integers(0, m, N)
    X = np.column_stack(
        (np.ones(N), rng.normal(size=(N, 2)) + 3 * np.eye(m)[cls][:, :2])
    )
    Y = np.eye(m)[cls]
    return X, Y, cls


@pytest.mark.parametrize("module", ["regression", "curve_regression"])
def test_mlogit_gradient_matches_finite_differences(module, mlogit_design):
    pytest.importorskip("mlogit_warp")
    pytest.importorskip("ocmlogit_warp")
    import importlib

    rg = importlib.import_module("fdasrsf." + module)
    X, Y, _ = mlogit_design
    b = np.random.default_rng(1).normal(size=X.shape[1] * Y.shape[1])
    eps = 1e-6
    fd = np.array(
        [
            (rg.mlogit_loss(b + eps * e, X, Y) - rg.mlogit_loss(b - eps * e, X, Y))
            / (2 * eps)
            for e in np.eye(b.size)
        ]
    )
    np.testing.assert_allclose(rg.mlogit_gradient(b, X, Y), fd, atol=1e-6)


@pytest.mark.parametrize("module", ["regression", "curve_regression"])
def test_mlogit_fit_predicts_by_largest_score(module, mlogit_design):
    """The fitted model must rank classes the way predict() does (argmax)."""
    pytest.importorskip("mlogit_warp")
    pytest.importorskip("ocmlogit_warp")
    import importlib

    from scipy.optimize import fmin_l_bfgs_b

    rg = importlib.import_module("fdasrsf." + module)
    X, Y, cls = mlogit_design
    b = fmin_l_bfgs_b(
        rg.mlogit_loss,
        np.zeros(X.shape[1] * Y.shape[1]),
        fprime=rg.mlogit_gradient,
        args=(X, Y),
    )[0]
    scores = X @ b.reshape(X.shape[1], Y.shape[1])
    pred = rg.phi(scores.ravel()).reshape(scores.shape).argmax(axis=1)
    assert np.mean(pred == cls) > 0.85


@pytest.mark.slow
def test_elastic_mlpcr_regression_classifies_training_data():
    pytest.importorskip("optimum_reparamN2")
    rng = np.random.default_rng(2)
    M, per_class = 60, 8
    time = np.linspace(0, 1, M)
    f, y = [], []
    # classes differ in amplitude, which survives alignment
    for label, amp in enumerate([1.0, 2.0, 3.0], start=1):
        for _ in range(per_class):
            c = rng.uniform(0.4, 0.6)
            f.append(amp * np.exp(-((time - c) ** 2) / 0.01))
            y.append(label)
    f = np.array(f).T
    y = np.array(y)

    model = fs.elastic_mlpcr_regression(f, y, time)
    model.calc_model(pca_method="vert", no=2)
    assert np.isfinite(model.LL)
    model.predict()
    assert set(np.unique(model.y_labels)) <= {1, 2, 3}
    assert model.PCo > 0.9
