"""
geometry functions for SRSF Manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

from numpy import arccos, sin, cos, linspace, zeros, sqrt, newaxis
from numpy import ones, diff, gradient, log, exp, logspace, any
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapezoid, cumulative_trapezoid
import fdasrsf.utility_functions as uf

def inv_exp_map(Psi, psi):
    tmp = inner_product(Psi, psi)
    if tmp > 1:
        tmp = 1
    if tmp < -1:
        tmp = -1

    theta = arccos(tmp)

    if theta < 1e-10:
        exp_inv = zeros(psi.shape[0])
    else:
        exp_inv = theta / sin(theta) * (psi - cos(theta) * Psi)

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


def gam_to_h(gam, smooth=True):
    TT = gam.shape[0]
    time = linspace(0, 1, TT)
    binsize = diff(time)
    binsize = binsize.mean()
    if gam.ndim == 1:
        if smooth:
            gamtmp = uf.smooth_data(gam[:, newaxis], 25)
            psi = log(gradient(gamtmp[:, 0], binsize))
            h = psi - trapezoid(psi, time)
        else:
            psi = log(gradient(gam, binsize))
            h = psi - trapezoid(psi, time)
    else:
        n = gam.shape[1]
        if smooth:
            gamtmp = uf.smooth_data(gam[:, newaxis], 25)

        psi = zeros((TT, n))
        for i in range(0, n):
            if smooth:
                psi[:, i] = log(gradient(gamtmp[:, i], binsize))
            else:
                psi[:, i] = log(gradient(gam[:, i], binsize))

        h = zeros((TT, n))
        for i in range(0, n):
            h[:, i] = psi[:, i] - trapezoid(psi[:, i], time)

    return h


def gam_to_v(gam, smooth=True):
    TT = gam.shape[0]
    time = linspace(0, 1, TT)
    binsize = diff(time)
    binsize = binsize.mean()
    mu = ones(TT)
    if gam.ndim == 1:
        if smooth:
            tmp_spline = UnivariateSpline(time, gam, s=1e-4)
            g = tmp_spline(time, 1)
            idx = g <= 0
            g[idx] = 0
            psi = sqrt(g)
        else:
            psi = sqrt(gradient(gam, binsize))
        vec, theta = inv_exp_map(mu, psi)
    else:
        n = gam.shape[1]

        psi = zeros((TT, n))
        for i in range(0, n):
            if smooth:
                tmp_spline = UnivariateSpline(time, gam[:, i], s=1e-4)
                g = tmp_spline(time, 1)
                idx = g <= 0
                g[idx] = 0
                psi[:, i] = sqrt(g)
            else:
                psi[:, i] = sqrt(gradient(gam[:, i], binsize))

        vec = zeros((TT, n))
        for i in range(0, n):
            out, theta = inv_exp_map(mu, psi[:, i])
            vec[:, i] = out

    return vec


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
