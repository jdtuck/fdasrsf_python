import numba
from numba.core.typing import cffi_utils
from _DP import ffi, lib
import _DP
from numpy import linspace, interp, zeros, diff, double, sqrt, arange, float64, int32, int64
from numpy import zeros, frombuffer, ascontiguousarray, empty, roll, dot, eye, arccos, reshape
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
def find_rot(q1, q2):
    q1 = ascontiguousarray(q1)
    q2 = ascontiguousarray(q2)
    eps = 2.220446049250313e-16
    n = q1.shape[0]
    A = q1.dot(q2.T)
    U, s, V = svd(A)

    tst = abs(det(U)*det(V)-1)
    if tst < 10*eps:
        S = eye(n)
    else:
        S = eye(n)
        S[:,-1] = -S[:,-1]
    
    R = U.dot(S).dot(V.T)
    return R

@numba.jit()
def find_seed_rot(q1, q2):
    q1 = ascontiguousarray(q1)
    q2 = ascontiguousarray(q2)
    n, T = q1.shape
    Ltwo = zeros(T)
    Rlist = zeros((n, n, T))
    for ctr in range(0,T):
        q2n = shift_curve(q2, ctr)
        R = find_rot(q1,q2n)
        Rlist[:, :, ctr] = R
        q2new = R.dot(q2)
        cst = q1 - q2new
        tmp = cst * cst
        Ltwo[ctr] = tmp.sum() / T
    
    tau = Ltwo.argmin()
    O_hat = Rlist[:, :, tau]
    q2new = shift_curve(q2, tau)
    q2new = O_hat.dot(q2new)
    return q2new

@numba.njit()
def efda_distance(q1, q2):
    """"
    calculates the distances between square root slope functions, where 
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
def efda_distance_curve(q1, q2):
    """"
    calculates the distances between square root velocity functions, where 
    q2 is aligned to q1. In other words calculates the elastic distance.
    This metric is set up for use with UMAP or t-sne from scikit-learn

    :param q1: vector of size n*M
    :param q2: vector of size n*M

    :rtype: scalar
    :return dist: shape distance
    """
    tst = q1-q2
    if tst.sum() == 0:
        dist = 0
    else:
        n = int64(2)
        T = int64(q1.shape[0]/n)
        q1_i = reshape(q1, (n,T))
        q2_i = reshape(q2, (n,T))
        x = linspace(0,1,T)
        # optimize over SO(n)
        q2new = find_seed_rot(q1_i, q2_i)

        # optimize over Gamma
        gam = warp_curve(q1_i, q2new)
        gamI = interp(x,gam,x)
        gam_dev = grad(gamI, 1. / T)
        qwarp = zeros((n,T))
        for j in range(0,n):
            qwarp[j,:] = interp(gamI, x, q2new[j,:])
            qwarp[j,:] = qwarp[j,:] * sqrt(gam_dev)
        
        tmp = qwarp * qwarp
        Ltwo = tmp.sum() / T
        qwarp = qwarp / sqrt(Ltwo)

        q2n = find_seed_rot(q1_i, qwarp)

        tmp = q1_i * q2n
        q1dotq2 = tmp.sum() / T
        dist = arccos(q1dotq2)
       
    return dist
