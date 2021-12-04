"""
geometry functions for SRSF Manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

from numpy import arccos, sin, cos, linspace, zeros, sqrt, finfo, double
from numpy import ones, diff, gradient
from scipy.integrate import trapz, cumtrapz


def inv_exp_map(Psi, psi):
    tmp = inner_product(Psi,psi)
    if tmp > 1:
        tmp = 1
    if tmp < -1:
        tmp = -1

    theta = arccos(tmp)

    if (theta < 1e-10):
        exp_inv = zeros(psi.shape[0])
    else: 
        exp_inv = theta / sin(theta) * (psi - cos(theta)*Psi)
    
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
    t = linspace(0,1,M)
    ip = trapz(psi1*psi2, t)
    return ip


def L2norm(psi):
    M = psi.shape[0]
    t = linspace(0,1,M)
    l2norm = sqrt(trapz(psi*psi,t))
    return l2norm


def gam_to_v(gam):
    TT = gam.shape[0]
    time = linspace(0,1,TT)
    binsize = diff(time)
    binsize = binsize.mean()
    mu = ones(TT)
    if gam.ndim == 1:
        psi = sqrt(gradient(gam,binsize))
        vec, theta = inv_exp_map(mu,psi)
    else:
        n = gam.shape[1]

        psi = zeros((TT,n))
        for i in range(0,n):
            psi[:,i] = sqrt(gradient(gam[:, i],binsize))
        
        vec = zeros((TT,n))
        for i in range(0,n):
            out, theta = inv_exp_map(mu,psi[:,i])
            vec[:,i] = out
    
    return vec


def v_to_gam(v):
    TT = v.shape[0]
    time = linspace(0,1,TT)
    mu = ones(TT)
    if v.ndim == 1:
        psi = exp_map(mu,v)
        gam0 = cumtrapz(psi*psi, time, initial=0)
        gam = (gam0 - gam0.min()) / (gam0.max() - gam0.min())
    else:
        n = v.shape[1]

        gam = zeros((TT,n))
        for i in range(0,n):
            psi = exp_map(mu,v[:,i])
            gam0 = cumtrapz(psi*psi, time, initial=0)
            gam[:,i] = (gam0 - gam0.min()) / (gam0.max() - gam0.min())
    
    return(gam)
