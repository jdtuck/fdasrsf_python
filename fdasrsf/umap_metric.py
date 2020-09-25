import numba
from numba.core.typing import cffi_utils
from _DP import ffi, lib
import _DP
from numpy import linspace, interp, zeros, diff, double, sqrt, arange, float64, int32, zeros, frombuffer, ascontiguousarray, empty
from numpy.linalg import norm


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
def warp(q1, q2, n1):
    M = q1.shape[0]
    disp = 1
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

@numba.njit()
def efda_distance(q1, q2):
    tst = q1-q2
    if tst.sum() == 0:
        dist = 0
    else:
        gam = warp(q1, q2, 1)
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
