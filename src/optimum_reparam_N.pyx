cimport cDP
import numpy as np
from numpy.linalg import norm

cimport numpy as np
from cpython cimport array


def _check_pen(int pen):
    if pen < 0 or pen > 2:
        raise ValueError("pen must be one of 0 (none), 1 (l2gam) or 2 (l2psi)")


cdef _check_status(int status):
    if status == -2:
        raise ValueError("unknown penalty type passed to the DP solver")
    if status != 0:
        raise MemoryError("the DP solver could not allocate its work buffers")


def coptimum_reparam_N(np.ndarray[double, ndim=1, mode="c"] mq, np.ndarray[double, ndim=1, mode="c"] time,
                      np.ndarray[double, ndim=2, mode="c"] q, lam1=0.0, int pen=2):
    """
    cython interface calculates the warping to align a set of SRSFS q to a single SRSF mq

    :param mq: vector of size N samples of first SRSF
    :param time: vector of size N describing the sample points
    :param q: numpy ndarray of shape (M,N) of N srsfs with M samples
    :param lam1: controls the amount of elasticity (default = 0.0)
    :param pen: penalty weighted by lam1, 0 = none, 1 = l2gam, 2 = l2psi
                (default = 2).  The roughness and geodesic penalties of the
                RBFGS solvers have no dynamic-programming counterpart.

    :rtype numpy ndarray
    :return gam: describing the warping functions used to align columns of q with mq

    """
    _check_pen(pen)
    cdef int M, N, n1, disp
    cdef double lam
    mq = mq / norm(mq)
    M, N = q.shape[0], q.shape[1]
    n1 = 1
    disp = 0
    lam = lam1
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] qi = np.zeros(M)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        qi = q[:, k] / norm(q[:, k])
        qi = np.ascontiguousarray(qi)

        _check_status(cDP.DP(&qi[0], &mq[0], n1, M, lam, pen, disp, &gami[0]))
        gam[:, k] = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_N2(np.ndarray[double, ndim=2, mode="c"] q1, np.ndarray[double, ndim=1, mode="c"] time,
                       np.ndarray[double, ndim=2, mode="c"] q2, lam1=0.0, int pen=2):
    """
    cython interface calculates the warping to align a set of SRSFs q1 to another set of SRSFs q2

    :param q1: numpy ndarray of shape (M,N) of M srsfs with N samples
    :param time: vector of size N describing the sample points
    :param q2: numpy ndarray of shape (M,N) of M srsfs with N samples
    :param lam1: controls the amount of elasticity (default = 0.0)
    :param pen: penalty weighted by lam1, 0 = none, 1 = l2gam, 2 = l2psi
                (default = 2).  The roughness and geodesic penalties of the
                RBFGS solvers have no dynamic-programming counterpart.

    :rtype numpy ndarray
    :return gam: describing the warping functions used to align columns of q with mq

    """
    _check_pen(pen)
    cdef int M, N, n1, disp
    cdef double lam

    M, N = q1.shape[0], q1.shape[1]
    n1 = 1
    lam = lam1
    disp = 0
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] q1i = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] q2i = np.zeros(M)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        q1i = q1[:, k] / norm(q1[:, k])
        q2i = q2[:, k] / norm(q2[:, k])
        q1i = np.ascontiguousarray(q1i)
        q2i = np.ascontiguousarray(q2i)

        _check_status(cDP.DP(&q2i[0], &q1i[0], n1, M, lam, pen, disp, &gami[0]))
        gam[:, k] = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam(np.ndarray[double, ndim=1, mode="c"] q1, np.ndarray[double, ndim=1, mode="c"] time,
                     np.ndarray[double, ndim=1, mode="c"] q2, lam1=0.0, int pen=2):
    """
    cython interface for calculates the warping to align SRSFs q2 to q1

    :param q1: vector of size N samples of first SRSF
    :param time: vector of size N describing the sample points
    :param q2: vector of size N samples of second SRSF
    :param lam1: controls the amount of elasticity (default = 0.0)
    :param pen: penalty weighted by lam1, 0 = none, 1 = l2gam, 2 = l2psi
                (default = 2).  The roughness and geodesic penalties of the
                RBFGS solvers have no dynamic-programming counterpart.

    :rtype vector
    :return gam: describing the warping function used to align q2 with q1
    """
    _check_pen(pen)
    cdef int M, n1, disp
    cdef double lam
    M = q1.shape[0]
    n1 = 1
    lam = lam1
    disp = 0
    q1 = q1 / norm(q1)
    q2 = q2 / norm(q2)
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)

    _check_status(cDP.DP(&q2[0], &q1[0], n1, M, lam, pen, disp, &gami[0]))
    gam = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_N2_pair(np.ndarray[double, ndim=2, mode="c"] q, np.ndarray[double, ndim=1, mode="c"] time,
                            np.ndarray[double, ndim=2, mode="c"] q1, np.ndarray[double, ndim=2, mode="c"] q2, lam1=0.0,
                            int pen=2):
    """
    cython interface for calculates the warping to align paired SRSF f1 and f2 to q

    :param q: vector of size N samples of first SRSF
    :param time: vector of size N describing the sample points
    :param q1: vector of size N samples of second SRSF
    :param q2: vector of size N samples of second SRSF
    :param lam1: controls the amount of elasticity (default = 0.0)
    :param pen: penalty weighted by lam1, 0 = none, 1 = l2gam, 2 = l2psi
                (default = 2).  The roughness and geodesic penalties of the
                RBFGS solvers have no dynamic-programming counterpart.

    :rtype vector
    :return gam: describing the warping function used to align q2 with q1
    """
    _check_pen(pen)
    cdef int M, N, n1, disp
    n1 = 2
    cdef double lam
    M, N = q1.shape[0], q1.shape[1]
    lam = lam1
    disp = 0
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] q1i = np.zeros(M * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] q2i = np.zeros(M * n1)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        q1i = q.reshape(M*n1)
        q2tmp = np.column_stack((q1[:, k], q2[:, k]))
        q2i = q2tmp.reshape(M*n1)

        q1i = np.ascontiguousarray(q1i)
        q2i = np.ascontiguousarray(q2i)

        _check_status(cDP.DP(&q2i[0], &q1i[0], n1, M, lam, pen, disp, &gami[0]))
        gam[:, k] = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_pair_q(np.ndarray[double, ndim=2, mode="c"] q1, np.ndarray[double, ndim=1, mode="c"] time,
                          np.ndarray[double, ndim=2, mode="c"] q2, lam1=0.0, int pen=2):
    """
    cython interface for calculates the warping to align paired srsf q2 to q1

    :param q1: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param q2: vector of size N samples of second function
    :param lam1: controls the amount of elasticity (default = 0.0)
    :param pen: penalty weighted by lam1, 0 = none, 1 = l2gam, 2 = l2psi
                (default = 2).  The roughness and geodesic penalties of the
                RBFGS solvers have no dynamic-programming counterpart.

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    _check_pen(pen)
    cdef int M, N, disp
    cdef double lam
    M, N = q1.shape[0], q1.shape[1]
    lam = lam1
    disp = 0
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] q1i = np.zeros(M * N)
    cdef np.ndarray[double, ndim=1, mode="c"] q2i = np.zeros(M * N)

    sizes = np.zeros(1, dtype=np.int32)
    q1i = q1.reshape(M*N)
    q2i = q2.reshape(M*N)

    q1i = np.ascontiguousarray(q1i)
    q2i = np.ascontiguousarray(q2i)

    _check_status(cDP.DP(&q2i[0], &q1i[0], N, M, lam, pen, disp, &gami[0]))
    gam = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_curve(np.ndarray[double, ndim=2, mode="c"] q1, np.ndarray[double, ndim=1, mode="c"] time,
                     np.ndarray[double, ndim=2, mode="c"] q2, lam1=0.0, int pen=2):
    """
    cython interface for calculates the warping to align curve q2 to q1

    :param q1: matrix of size nxN samples of first SRVF
    :param time: vector of size N describing the sample points
    :param q2: matrix of size nxN samples of second SRVF
    :param lam1: controls the amount of elasticity (default = 0.0)
    :param pen: penalty weighted by lam1, 0 = none, 1 = l2gam, 2 = l2psi
                (default = 2).  The roughness and geodesic penalties of the
                RBFGS solvers have no dynamic-programming counterpart.

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    _check_pen(pen)
    cdef int M, n1, disp
    cdef double lam
    n1 = q1.shape[0]
    M = q1.shape[1]
    lam = lam1
    disp = 0
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] q1i = np.zeros(M * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] q2i = np.zeros(M * n1)

    q1i = q1.reshape(M*n1, order='F')
    q2i = q2.reshape(M*n1, order='F')

    q1i = np.ascontiguousarray(q1i)
    q2i = np.ascontiguousarray(q2i)

    _check_status(cDP.DP(&q2i[0], &q1i[0], n1, M, lam, pen, disp, &gami[0]))
    gam = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam
