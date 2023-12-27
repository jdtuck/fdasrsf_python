# cython: language_level=2
import numpy as np
cimport crbfgs
cimport cyarma
cimport numpy as np
from cython.parallel import prange

include "cyarma.pyx"

from libcpp cimport bool

def rlbfgs(np.ndarray[double, ndim=1, mode="c"] q1, np.ndarray[double, ndim=1, mode="c"] q2, 
           np.ndarray[double, ndim=1, mode="c"] time, maxiter=30, lam=0.0, penalty=0):
    q1 = np.ascontiguousarray(q1)
    q2 = np.ascontiguousarray(q2)
    time = np.ascontiguousarray(time)
    cdef vec aq1 = numpy_to_vec_d(q1)
    cdef vec aq2 = numpy_to_vec_d(q2)
    cdef vec atime = numpy_to_vec_d(time)
    cdef vec out = rlbfgs_optim(aq1, aq2, atime, maxiter, lam, penalty)

    cdef np.ndarray[np.double_t,ndim=1] gam = vec_to_numpy(out, None)

    return(gam)


def rlbfgs_dist(np.ndarray[double, ndim=2, mode="c"] q1, np.ndarray[double, ndim=2, mode="c"] q2,
                np.ndarray[double, ndim=2, mode="c"] idx):
    q1 = np.ascontiguousarray(q1)
    q2 = np.ascontiguousarray(q2)
    d = np.zeros((q2.shape[1], idx.shape[1]))
    M = q1.shape[0]
    N = q2.shape[1]
    alpha = 0.5
    time = np.linspace(0, 1, M)
    for i in prange(N, nogil=True):
       for j in range(idx.shape[1]):
              q1t = q1[:,idx[i,j]]
              q1t = np.ascontiguousarray(q1t)
              gam = rlbfgs(q1t, time, q2)
              # warp q
              gam_dev = np.gradient(gam, 1.0 / (M - 1))
              tmp = np.interp((time[-1] - time[0]) * gam + time[0], time, q2)

              qw = tmp * np.sqrt(gam_dev)
              Dy = np.sqrt(np.trapz((qw - q1t) ** 2, time))

              binsize = np.mean(np.diff(time))
              psi = np.sqrt(np.gradient(gam, binsize))
              q1dotq2 = np.trapz(psi, time)
              if q1dotq2 > 1:
              q1dotq2 = 1
              elif q1dotq2 < -1:
              q1dotq2 = -1

              Dx = np.real(np.arccos(q1dotq2))

              d[i,j] = alpha * Dy + (1-alpha) * Dx
       
    return d
        
    