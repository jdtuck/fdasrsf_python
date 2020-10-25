import numba
from numba.core.typing import cffi_utils
from _DP import ffi, lib
import _DP
from numpy import linspace, interp, zeros, diff, double, sqrt, arange, float64, int32, int64, trapz, ones
from numpy import zeros, frombuffer, ascontiguousarray, empty, load, roll, dot, eye, arccos, reshape, float32
from numpy import kron
from numpy.linalg import norm, svd, det


DP = lib.DP
cffi_utils.register_module(_DP)

@numba.jit()
def grad(f, binsize):
    n = f.shape[0]
    g = zeros(n)
    h = binsize*arange(1,n+1)
    g[0] = (f[1] - f[0])/(h[1]-h[0])
    g[-1] = (f[-1] - f[(-2)])/(h[-1]-h[-2])

    h = h[2:]-h[0:-2]
    g[1:-1] = (f[2:]-f[0:-2])/h[0]

    return g

@numba.njit()
def warp(q1, q2):
    M = q1.shape[0]
    disp = 0
    n1 = 1
    lam = 0.0
    gam = zeros(M)
    q1 = q1 / norm(q1)
    q2 = q2 / norm(q2)
    q1 = ascontiguousarray(q1)
    q2 = ascontiguousarray(q2)
    gam = ascontiguousarray(gam)
    q2i = ffi.from_buffer(q2)
    q1i = ffi.from_buffer(q1)
    gami = ffi.from_buffer(gam)
    DP(q2i,q1i,n1,M,lam,disp,gami)
       
    return gam

@numba.jit()
def freshape(f):
    n,M = f.shape
    out1 = f.reshape(M*n)
    out = zeros(M*n)
    out[::2] = out1[0:M]
    out[1::2] = out1[M:]
    
    return out

@numba.njit()
def warp_curve(q1, q2):
    n1, M = q1.shape
    disp = 0
    lam = 0.0
    gam = zeros(M)
    q1i = freshape(q1)
    q2i = freshape(q2)

    q1i = ascontiguousarray(q1i)
    q2i = ascontiguousarray(q2i)

    gam = ascontiguousarray(gam)
    q2ptr = ffi.from_buffer(q1i)
    q1ptr = ffi.from_buffer(q2i)
    gami = ffi.from_buffer(gam)
    DP(q2ptr,q1ptr,n1,M,lam,disp,gami)
       
    return gam

@numba.jit()
def shift_curve(f, tau):
    n, T = f.shape
    fn = zeros((n, T))
    for i in range(n):
        fn[i, 0:(T - 1)] = roll(f[i, 0:(T - 1)], tau)
        fn[i, T - 1] = fn[i, 0]
    return (fn)

@numba.jit()
def c_to_q(beta):
    n, T = beta.shape
    v = zeros((n,T))
    for i in range(n):
        v[i,:] = grad(beta[i,:], 1./(T-1))

    q = zeros((n,T))
    for i in range(T):
        L = sqrt(norm(v[:,i]))
        if L > 0.0001:
            q[:, i] = v[:, i] / L
        else:
            q[:, i] = v[:, i] * 0.0001
    
    tmp = q*q
    tmp.sum() / T
    tmp1 = tmp.sum() / T
    q = q / sqrt(tmp1)

    return(q)

@numba.jit()
def find_rot(q1, q2):
    q1 = ascontiguousarray(q1)
    q2 = ascontiguousarray(q2)
    n = q1.shape[0]
    A = q1.dot(q2.T)
    U, s, V = svd(A)

    tst = det(A)
    if tst > 0:
        S = eye(n)
    else:
        S = eye(n)
        S[:,-1] = -S[:,-1]
    
    R = U.dot(S).dot(V.T)
    return R

@numba.jit()
def find_seed_rot(beta1, beta2):
    beta1 = ascontiguousarray(beta1)
    beta2 = ascontiguousarray(beta2)
    q1 = c_to_q(beta1)
    n, T = beta1.shape
    Ltwo = zeros(T)
    Rlist = zeros((n, n, T))
    for ctr in range(0,T):
        beta2n = shift_curve(beta2, ctr)
        R = find_rot(beta1,beta2n)
        Rlist[:, :, ctr] = R
        beta2new = R.dot(beta2n)
        q2new = c_to_q(beta2new)
        cst = q1 - q2new
        tmp = cst * cst
        Ltwo[ctr] = tmp.sum() / T
    
    tau = Ltwo.argmin()
    O_hat = Rlist[:, :, tau]
    beta2new = shift_curve(beta2, tau)
    beta2new = O_hat.dot(beta2new)

    return beta2new

@numba.njit()
def curve_center(beta):
    n, T = beta.shape
    betadot = zeros((n,T))
    for i in range(n):
        betadot[i,:] = grad(beta[i,:], 1./(T-1))
    
    normbetadot = zeros(T)
    integrand = zeros((n,T))
    for i in range(T):
        normbetadot[i] = norm(betadot[:,i])
        integrand[:,i] = beta[:,i] * normbetadot[i]
    
    scale = trapz(normbetadot, linspace(0,1,T))
    centroid = zeros(n)
    for i in range(n):
        centroid[i] = trapz(integrand[i,:], linspace(0,1,T))/scale
    
    return centroid

@numba.njit()
def efda_distance(q1, q2):
    """"
    calculates the distances between two curves, where 
    q2 is aligned to q1. In other words calculates the elastic distances/
    This metric is set up for use with UMAP or t-sne from scikit-learn

    :param q1: vector of size N
    :param q2: vector of size N

    :rtype: scalar
    :return dist: amplitude distance
    """
    tst = q1-q2
    if tst.sum() == 0:
        dist = 0
    else:
        gam = warp(q1, q2)
        M = q1.shape[0]
        time = linspace(0,1,q1.shape[0])
        gam = (gam - gam[0]) / (gam[-1] - gam[0])
        gam_dev = grad(gam, 1 / double(M - 1))
        tmp = interp((time[-1] - time[0]) * gam + time[0], time, q2)

        qw = tmp * sqrt(gam_dev)

        y = (qw - q1) ** 2
        tmp = diff(time)*(y[0:-1]+y[1:])/2
        dist = sqrt(tmp.sum())
       
    return dist

@numba.njit()
def efda_distance_curve(beta1, beta2):
    """"
    calculates the distances between two curves, where 
    beta2 is aligned to beta1. In other words calculates the elastic distance.
    This metric is set up for use with UMAP or t-sne from scikit-learn

    :param beta1: vector of size n*M
    :param beta2: vector of size n*M

    :rtype: scalar
    :return dist: shape distance
    """
    tst = beta1-beta2
    if tst.sum() == 0:
        dist = 0
    else:
        n = int64(2)
        T = int64(beta1.shape[0]/n)
        beta1 = ascontiguousarray(beta1)
        beta2 = ascontiguousarray(beta2)
        beta1_i = beta1.reshape((n,T))
        beta2_i = beta2.reshape((n,T))
        beta1_i = beta1_i.astype(double)
        beta2_i = beta2_i.astype(double)

        centroid1 = curve_center(beta1_i)
        beta1_i = beta1_i - kron(ones((T,1)), centroid1).T
        centroid2 = curve_center(beta2_i)
        beta2_i = beta2_i - kron(ones((T,1)), centroid2).T

        q1 = c_to_q(beta1_i)

        q1 = ascontiguousarray(q1)
        # UMAP uses float32, need 64 for a lot of the computation
        
        x = linspace(0,1,T)

        # optimize over SO(n)
        beta2_i = find_seed_rot(beta1_i, beta2_i)
        q2 = c_to_q(beta2_i)

        # optimize over Gamma
        gam = warp_curve(q1, q2)
        gamI = interp(x,gam,x)

        # apply warp
        beta_warp = zeros((n,T))
        for j in range(0,n):
            beta_warp[j,:] = interp(gamI, x, beta2_i[j,:])

        beta2_in = find_seed_rot(beta1_i, beta_warp)

        q2 = c_to_q(beta2_in)
        tmp = q1 * q2
        q1dotq2 = tmp.sum() / T
        dist = arccos(q1dotq2)
       
    return dist
