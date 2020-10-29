# cython: language_level=2
import numpy as np
cimport coclogit

cimport numpy as np
from cpython cimport array


def oclogit_warp(np.ndarray[double, ndim=1, mode="c"] alpha,
                  np.ndarray[double, ndim=2, mode="c"] nu,
                  np.ndarray[double, ndim=2, mode="c"] q,
                  np.ndarray[int, ndim=1, mode="c"] y,
                  max_iter=8000, tol=1e-6, deltaO=0.1, deltag=0.05, display=0):
    """
    cython interface perform warping calculation for logistic cost function for open curves

    :param alpha: scalar
    :param nu: array of size 2*N
    :param q: numpy vector of size M srsf
    :param y: scalar class label
    :param max_iter: maximal number of iterations (default = 400)
    :param tol: stopping tolerance (default = 1e-4)
    :param deltaO: step size (default = 0.1)
    :param deltag: step size (default = 0.05)
    :param display: show iterations (default = 0)

    :rtype numpy ndarray
    :return O_hat: rotation matrix
    :return gamo: describing the warping function

    """
    cdef int m1, TT, max_itri, displayi
    cdef double toli, deltagi, deltaOi
    toli = tol
    deltagi = deltag
    deltaOi = deltaO
    max_itri = max_iter
    displayi = display
    m1 = q.shape[0]
    TT = q.shape[1]

    cdef np.ndarray[double, ndim = 1, mode = "c"] nu1 = np.zeros(m1 * TT)
    cdef np.ndarray[double, ndim = 1, mode = "c"] q1 = np.zeros(m1 * TT)
    cdef np.ndarray[double, ndim = 1, mode = "c"] gamout = np.zeros(TT)
    cdef np.ndarray[double, ndim = 1, mode = "c"] O = np.zeros(m1*m1)

    nu1 = nu.reshape(m1*TT, order='F')
    q1 = q.reshape(m1*TT, order='F')

    nu1 = np.ascontiguousarray(nu1)
    q1 = np.ascontiguousarray(q1)
    gamout = np.ascontiguousarray(gamout)
    O = np.ascontiguousarray(O)

    coclogit.oclogit_warp_grad(&m1, &TT, &alpha[0], &nu1[0], &q1[0], &y[0],
                               &max_itri, &toli, &deltaOi, &deltagi,
                               &displayi, &gamout[0], &O[0])

    Oout = O.reshape((m1, m1), order='F')

    return gamout, Oout
