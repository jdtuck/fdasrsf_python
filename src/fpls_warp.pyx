# cython: language_level=2
cimport cfPLS
import numpy as np

cimport numpy as np
from cpython cimport array

def fpls_warp(np.ndarray[double, ndim=1, mode="c"] time, np.ndarray[double, ndim=2, mode="c"] gam,
              np.ndarray[double, ndim=2, mode="c"] qf, np.ndarray[double, ndim=2, mode="c"] qg,
              np.ndarray[double, ndim=1, mode="c"] wf, np.ndarray[double, ndim=1, mode="c"] wg,
              max_iter = 100, tol = 1e-4, delta = 0.1, display = 1):
    """
    cython interface perform warping calculation for PLS cost function

    :param time: vector of size N describing the sample points
    :param gam: numpy ndarray of shape (M,N) of N init warping functions with M samples
    :param qf: numpy ndarray of shape (M,N) of N srsfs with M samples
    :param qg: numpy ndarray of shape (M,N) of N srsfs with M samples
    :param wf: numpy ndarray of shape (M,1) weight function f
    :param wg: numpy ndarray of shape (M,1) weight function f
    :param max_iter: maximal number of iterations (default = 100)
    :param tol: stopping tolerance (default = 1e-4)
    :param delta: step size (default = .1)
    :param display: show iterations (default = 1)

    :rtype numpy ndarray
    :return gamo: describing the warping functions

    """
    cdef int m1, n1, max_itri, displayi
    cdef double toli, deltai
    toli = tol
    deltai = delta
    max_itri = max_iter
    displayi = display
    m1 = gam.shape[0]
    n1 = gam.shape[1]
    cdef np.ndarray[double, ndim=1, mode="c"] gam1 = np.zeros(m1 * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] qf1 = np.zeros(m1 * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] qg1 = np.zeros(m1 * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] gamout = np.zeros(m1 * n1)
    for ii in xrange(0, n1):
        gam1[ii * m1:ii * m1 + m1] = gam[:, ii]
        qf1[ii * m1:ii * m1 + m1] = qf[:, ii]
        qg1[ii * m1:ii * m1 + m1] = qg[:, ii]

    gam1 = np.ascontiguousarray(gam1)
    qf1 = np.ascontiguousarray(qf1)
    qg1 = np.ascontiguousarray(qg1)
    gamout = np.ascontiguousarray(gamout)

    cfPLS.fpls_warp_grad(&m1, &n1, &time[0], &gam1[0], &qf1[0], &qg1[0], &wf[0], &wg[0], &max_itri, &toli, &deltai,
                         &displayi, &gamout[0])

    gamo = np.zeros((m1, n1))
    for ii in xrange(0, n1):
        gamo[:, ii] = gamout[ii * m1:ii * m1 + m1]

    return gamo
