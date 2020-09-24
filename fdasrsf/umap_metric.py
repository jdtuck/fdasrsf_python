import numba
from numba.core.typing import cffi_utils as cffi_support
import cffi
ffi = cffi.FFI()
import _DP
from numpy import linspace, interp, zeros, diff, double, sqrt, arange, float64, int32, zeros, frombuffer, ascontiguousarray, empty

DP = _DP.lib.DP
cffi_support.register_module(_DP)

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
        n1 = 1
        M = q1.shape[0]
        disp = 0
        lam = 0
        time = linspace(0,1,q1.shape[0])
        gam = zeros(M)
        q1 = ascontiguousarray(q1)
        q2 = ascontiguousarray(q2)

        n1_arr=empty(1,dtype=int32)
        n1_arr[0]=n1
        n1_arr_ptr=ffi.from_buffer(n1_arr)
        M_arr=empty(1,dtype=int32)
        M_arr[0]=M
        M_arr_ptr=ffi.from_buffer(M_arr)
        disp_arr=empty(1,dtype=int32)
        disp_arr[0]=disp
        disp_arr_ptr=ffi.from_buffer(disp_arr)
        lam_arr=empty(1,dtype=double)
        lam_arr[0]=lam
        lam_arr_ptr=ffi.from_buffer(lam_arr)
        q1_wrap = ffi.from_buffer(q1)
        q2_wrap = ffi.from_buffer(q2)
        gami = ffi.from_buffer(gam)
        DP(q2_wrap,q1_wrap,n1_arr_ptr,M_arr_ptr,lam_arr_ptr,disp_arr_ptr,gami)

        gam_dev = grad(gam, 1 / double(M - 1))
        tmp = interp((time[-1] - time[0]) * gam + time[0], time, q2)

        qw = tmp * sqrt(gam_dev)

        y = (qw - q1) ** 2
        tmp = diff(time)*(y[0:-1]+y[1:])/2
        dist = sqrt(tmp.sum())
       
    return dist
