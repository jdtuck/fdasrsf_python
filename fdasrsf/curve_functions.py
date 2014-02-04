from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import trapz
from numpy import zeros, cumsum, linspace, gradient, sqrt
from scipy.linalg import norm


def resamplecurve(x, N):
    n, T = x.shape
    xn = zeros((n, N))

    delta = zeros(T)
    for r in range(1, T):
        delta[r] = norm(x[:, r] - x[:, r-1])

    cumdel = cumsum(delta)/delta.sum()
    newdel = linspace(0, 1, N)

    for r in range(0, n):
        s = InterpolatedUnivariateSpline(cumdel, x[r, :])
        xn[r, :] = s(newdel)

    return(xn)


def calculatecentroid(beta):
    n, T = beta.shape
    betadot = gradient(beta, 1./(T - 1))
    normbetadot = zeros(T)
    integrand = zeros((n, T))
    for i in range(0, T):
        normbetadot[i] = norm(betadot[:, i])
        integrand[:, i] = beta[:, i] * normbetadot[i]

    scale = trapz(normbetadot, linspace(0, 1, T))
    centroid = trapz(integrand, linspace(0, 1, T), axis=1)/scale

    return(centroid)


def curve_to_q(beta):
    n, T = beta.shape
    v = gradient(beta, 1./(T - 1))

    length = sum(sqrt(sum(v*v)))/T
    v = v/length
    q = zeros((n, T))
    for i in range(0, T):
        L = sqrt(norm(v[:, i]))
        if L > 0.0001:
            q[:, i] = v[:, i]/L
        else:
            q[:, i] = v[:, i]*0.0001

    return(q)
