import numpy as np
cimport crbfgs
cimport cyarma
cimport numpy as np

include "cyarma.pyx"

from libcpp cimport bool

def rlbfgs(np.ndarray[double, ndim=1, mode="c"] q1, np.ndarray[double, ndim=1, mode="c"] q2, 
           np.ndarray[double, ndim=1, mode="c"] time, maxiter=30, lam=0.0, penalty=0,
           gam0=None, return_info=False):
    """
    Align q2 to q1 with the Riemannian LBFGS solver.

    gam0, when given, is the warping (sampled on time) to start from instead
    of the identity.  With return_info, also returns
    [iterations, stop reason, cost, gradient norm], the stop reason being 1
    for no acceptable step, 2 for the gradient norm, 3 for maxiter and 4 for
    the relative change of the cost.
    """
    q1 = np.ascontiguousarray(q1)
    q2 = np.ascontiguousarray(q2)
    time = np.ascontiguousarray(time)
    cdef vec aq1 = numpy_to_vec_d(q1)
    cdef vec aq2 = numpy_to_vec_d(q2)
    cdef vec atime = numpy_to_vec_d(time)
    cdef vec agam0
    cdef vec ainfo
    cdef np.ndarray[double, ndim=1, mode="c"] g0
    if gam0 is not None:
        g0 = np.ascontiguousarray(gam0, dtype=np.double)
        if g0.shape[0] != time.shape[0]:
            raise ValueError("gam0 must have as many elements as time")
        agam0 = numpy_to_vec_d(g0)
    cdef vec out = rlbfgs_optim(aq1, aq2, atime, maxiter, lam, penalty, agam0,
                                ainfo)

    cdef np.ndarray[np.double_t,ndim=1] gam = vec_to_numpy(out, None)

    if return_info:
        return gam, vec_to_numpy(ainfo, None)

    return(gam)


def rlbfgs_dist(np.ndarray[double, ndim=2, mode="c"] q1, np.ndarray[double, ndim=1, mode="c"] q2):
    q1 = np.ascontiguousarray(q1)
    q2 = np.ascontiguousarray(q2)
    d = np.zeros(q1.shape[1])
    M = q1.shape[0]
    alpha = 0.5
    time = np.linspace(0, 1, M)
    for i in range(q1.shape[1]):
        q1t = q1[:,i]
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

        d[i] = alpha * Dy + (1-alpha) * Dx
    
    return d
        
        
    
