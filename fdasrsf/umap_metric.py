import numba
from numba.extending import get_cython_function_address
import ctypes
from numpy import ascontiguousarray, linspace, interp, zeros, diff, double, sqrt, arange


_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("optimum_reparam_N", "reparm_dp")
reparm_dp_functype = ctypes.CFUNCTYPE(_ptr_dble, _ptr_dble)
reparm_dp_numba = reparm_dp_functype(addr)

@numba.jit()
def grad(f, binsize):
    n = f.shape[0]
    g = zeros(n)
    h = binsize*arange(1,n+1)
    g[0] = (f[0] - f[0])/(h[1]-h[0])
    g[-1] = (f[-1] - f[(-2)])/(h[-1]-h[-2])

    h = h[2:-1]-h[0:-3]
    g[1:-2] = (f[2:-1]-f[0:-3])/h[0]

    return g

@numba.njit()
def efda_distance(q1, q2):
    if (q1 == q2).all():
        dist = 0
    else:
        time = linspace(0,1,q1.shape[0])
        q1 = ascontiguousarray(q1)
        q2 = ascontiguousarray(q2)
        
        gam = reparm_dp_numba(q1, q2)
        M = gam.size
        gam_dev = grad(gam, 1 / double(M - 1))
        tmp = interp((time[-1] - time[0]) * gam + time[0], time, q2)

        qw = tmp * sqrt(gam_dev)

        y = (qw - q1) ** 2
        tmp = diff(time)*(y[0:-1]+y[1:])/2
        dist = sqrt(tmp.sum())
       
    return dist


