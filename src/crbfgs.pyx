# cython: language_level=2
import numpy as np
cimport crbfgs
cimport cyarma
cimport numpy as np

include "cyarma.pyx"

from libcpp cimport bool

def rlbfs(np.ndarray[double, ndim=1, mode="c"] q1, np.ndarray[double, ndim=1, mode="c"] q2, 
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
