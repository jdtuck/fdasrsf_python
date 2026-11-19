import numpy as np


def kernel(x, y, l2):
    # RBF Kernel
    sqdist = np.sum(x**2, 1).reshape(-1, 1) + np.sum(y**2, 1) - 2 * np.dot(x, y.T)
    return np.exp(-0.5 * (1 / l2) * sqdist)


def gp_posterior(X, y, Xtest, l2=0.1, noise_var=1e-6):
    # compute the mean at our test points.
    N = len(X)
    K = kernel(X, X, l2)
    L = np.linalg.cholesky(K + noise_var * np.eye(N))
    Lk = np.linalg.solve(L, kernel(X, Xtest, l2))
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    # compute the variance at our test points.
    K_ = kernel(Xtest, Xtest, l2)
    sd = np.sqrt(np.diag(K_) - np.sum(Lk**2, axis=0))
    return (mu, sd)


def gp_fit_se(x, y, xtest=None):
    r"""
    Maximum likelihood fit of a squared-exponential GP with a constant mean
    and Gaussian noise, the model used by MATLAB's ``fitrgp`` with its default
    kernel and basis function

    .. math::
        y = \beta + z(x) + \epsilon, \quad
        \mathrm{cov}(z) = \sigma_f^2 \exp(-d^2/(2 L^2)), \quad
        \epsilon \sim N(0, \sigma_n^2)

    The constant mean is profiled out by generalized least squares and the
    three (log) covariance parameters are fit by maximizing the log marginal
    likelihood, started from ``fitrgp``'s initial values (L = std(x),
    sigma_f = sigma_n = std(y)/sqrt(2)).

    :param x: vector of sample points
    :param y: vector of observations
    :param xtest: points at which to evaluate the posterior mean
                  (default = x)

    :rtype: tuple
    :return L: fitted length scale
    :return sigmaF: fitted signal standard deviation
    :return sigmaN: fitted noise standard deviation
    :return mu: posterior mean at xtest
    """
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if xtest is None:
        xtest = x
    xtest = np.asarray(xtest, dtype=float).ravel()
    n = x.shape[0]
    d2 = (x.reshape((-1, 1)) - x.reshape((1, -1))) ** 2
    one = np.ones(n)

    def nlml(p):
        L, sf, sn = np.exp(p)
        K = sf**2 * np.exp(-d2 / (2 * L**2)) + (sn**2 + 1e-10) * np.eye(n)
        try:
            cf = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        Ki_y = np.linalg.solve(K, y)
        Ki_1 = np.linalg.solve(K, one)
        beta = (one @ Ki_y) / (one @ Ki_1)
        r = y - beta
        Ki_r = np.linalg.solve(K, r)
        out = 0.5 * (r @ Ki_r) + np.sum(np.log(np.diag(cf)))
        return out if np.isfinite(out) else 1e10

    sdx = x.std(ddof=1) if n > 1 else 1.0
    sdy = y.std(ddof=1) if n > 1 else 1.0
    sdx = sdx if sdx > 0 else 1.0
    sdy = sdy if sdy > 0 else 1.0
    p0 = np.log([sdx, sdy / np.sqrt(2), sdy / np.sqrt(2)])
    res = minimize(nlml, p0, method="L-BFGS-B")
    p = res.x if res.fun <= nlml(p0) else p0
    L, sf, sn = np.exp(p)

    K = sf**2 * np.exp(-d2 / (2 * L**2)) + (sn**2 + 1e-10) * np.eye(n)
    Ki_y = np.linalg.solve(K, y)
    Ki_1 = np.linalg.solve(K, one)
    beta = (one @ Ki_y) / (one @ Ki_1)
    Ki_r = np.linalg.solve(K, y - beta)
    Ks = sf**2 * np.exp(
        -((xtest.reshape((-1, 1)) - x.reshape((1, -1))) ** 2) / (2 * L**2)
    )
    mu = beta + Ks @ Ki_r

    return L, sf, sn, mu
