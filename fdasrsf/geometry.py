"""
geometry functions for SRSF Manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

from numpy import arctan2, sin, cos, linspace, zeros, sqrt, newaxis
from numpy import ones, diff, gradient, log, exp, maximum, where, finfo, double
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapezoid, cumulative_trapezoid
import fdasrsf.utility_functions as uf


def inv_exp_map(Psi, psi):
    # The angle is computed with arctan2 of the tangent and normal components
    # rather than arccos of the inner product, which loses about half the
    # significant digits for nearby points
    ip = inner_product(Psi, psi)
    w = psi - ip * Psi
    wnorm = L2norm(w)
    theta = arctan2(wnorm, ip)

    if theta < 1e-10:
        exp_inv = zeros(psi.shape[0])
    else:
        exp_inv = theta / wnorm * w

    return exp_inv, theta


def exp_map(psi, v):
    v_norm = L2norm(v)
    if v_norm.sum() == 0:
        expgam = cos(v_norm) * psi
    else:
        expgam = cos(v_norm) * psi + sin(v_norm) * v / v_norm
    return expgam


def inner_product(psi1, psi2):
    M = psi1.shape[0]
    t = linspace(0, 1, M)
    ip = trapezoid(psi1 * psi2, t)
    return ip


def L2norm(psi):
    M = psi.shape[0]
    t = linspace(0, 1, M)
    l2norm = sqrt(trapezoid(psi * psi, t))
    return l2norm


def gam_to_h(gam, smooth=False):
    TT = gam.shape[0]
    time = linspace(0, 1, TT)
    binsize = diff(time)
    binsize = binsize.mean()
    eps = finfo(double).eps
    if gam.ndim == 1:
        if smooth:
            gamtmp = uf.smooth_data(gam[:, newaxis], 25)
            psi = log(maximum(gradient(gamtmp[:, 0], binsize), eps))
            h = psi - trapezoid(psi, time)
        else:
            psi = log(maximum(gradient(gam, binsize), eps))
            h = psi - trapezoid(psi, time)
    else:
        n = gam.shape[1]
        if smooth:
            gamtmp = uf.smooth_data(gam, 25)

        psi = zeros((TT, n))
        for i in range(0, n):
            if smooth:
                psi[:, i] = log(maximum(gradient(gamtmp[:, i], binsize), eps))
            else:
                psi[:, i] = log(maximum(gradient(gam[:, i], binsize), eps))

        h = zeros((TT, n))
        for i in range(0, n):
            h[:, i] = psi[:, i] - trapezoid(psi[:, i], time)

    return h


def gam_to_v(gam, smooth=False):
    TT = gam.shape[0]
    mu = ones(TT)
    psi = gam_to_psi(gam, smooth)
    if gam.ndim == 1:
        vec, theta = inv_exp_map(mu, psi)
    else:
        n = gam.shape[1]
        vec = zeros((TT, n))
        for i in range(0, n):
            out, theta = inv_exp_map(mu, psi[:, i])
            vec[:, i] = out

    return vec


def _gam_to_psi_1(gam, time, binsize, smooth):
    if smooth:
        tmp_spline = UnivariateSpline(time, gam, s=1e-4)
        g = tmp_spline(time, 1)
    else:
        g = gradient(gam, binsize)
    return sqrt(maximum(g, 0))


def gam_to_psi(gam, smooth=False):
    """
    psi = sqrt(gamdot), rescaled to unit L2 norm so it lies on the Hilbert
    sphere (smoothing and clipping negative slopes otherwise move it off)
    """
    TT = gam.shape[0]
    time = linspace(0, 1, TT)
    binsize = diff(time)
    binsize = binsize.mean()
    if gam.ndim == 1:
        psi = _gam_to_psi_1(gam, time, binsize, smooth)
    else:
        n = gam.shape[1]

        psi = zeros((TT, n))
        for i in range(0, n):
            psi[:, i] = _gam_to_psi_1(gam[:, i], time, binsize, smooth)

    nrm = sqrt(trapezoid(psi**2, time, axis=0))
    nrm = where(nrm == 0, 1, nrm)
    psi = psi / nrm

    return psi


def h_to_gam(h):
    TT = h.shape[0]
    time = linspace(0, 1, TT)
    if h.ndim == 1:
        gam0 = cumulative_trapezoid(exp(h), time, initial=0)
        gam0 /= trapezoid(exp(h), time)
        gam = (gam0 - gam0.min()) / (gam0.max() - gam0.min())
    else:
        n = h.shape[1]

        gam = zeros((TT, n))
        for i in range(0, n):
            gam0 = cumulative_trapezoid(exp(h[:, i]), time, initial=0)
            gam0 /= trapezoid(exp(h[:, i]), time)
            gam[:, i] = (gam0 - gam0.min()) / (gam0.max() - gam0.min())

    return gam


def v_to_gam(v):
    TT = v.shape[0]
    time = linspace(0, 1, TT)
    mu = ones(TT)
    if v.ndim == 1:
        psi = exp_map(mu, v)
        gam0 = cumulative_trapezoid(psi * psi, time, initial=0)
        gam = (gam0 - gam0.min()) / (gam0.max() - gam0.min())
    else:
        n = v.shape[1]

        gam = zeros((TT, n))
        for i in range(0, n):
            psi = exp_map(mu, v[:, i])
            gam0 = cumulative_trapezoid(psi * psi, time, initial=0)
            gam[:, i] = (gam0 - gam0.min()) / (gam0.max() - gam0.min())

    return gam


def psi_to_gam(psi):
    TT = psi.shape[0]
    time = linspace(0, 1, TT)
    if psi.ndim == 1:
        gam0 = cumulative_trapezoid(psi * psi, time, initial=0)
        gam = (gam0 - gam0.min()) / (gam0.max() - gam0.min())
    else:
        n = psi.shape[1]

        gam = zeros((TT, n))
        for i in range(0, n):
            gam0 = cumulative_trapezoid(psi[:, i] * psi[:, i], time, initial=0)
            gam[:, i] = (gam0 - gam0.min()) / (gam0.max() - gam0.min())

    return gam
