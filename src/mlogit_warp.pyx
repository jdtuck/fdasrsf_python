# cython: language_level=2
cimport cmlogit
import numpy as np
from scipy.linalg import norm

cimport numpy as np
from cpython cimport array


def mlogit_warp(np.ndarray[double, ndim=1, mode="c"] alpha,
                np.ndarray[double, ndim=2, mode="c"] beta,
                np.ndarray[double, ndim=1, mode="c"] time,
                np.ndarray[double, ndim=1, mode="c"] q,
                np.ndarray[int, ndim=1, mode="c"] y,
                max_iter=4000, tol=1e-10, delta=0.008, display=0):
    """
    cython interface perform warping calculation for multinomial cost function

    :param alpha: vector of size m:number of classes
    :param beta: matrix of size Nxm
    :param time: vector of size N describing the sample points
    :param q: numpy vector of size M srsf
    :param y: numpy ndarray of shape m class labels
    :param max_iter: maximal number of iterations (default = 400)
    :param tol: stopping tolerance (default = 1e-4)
    :param delta: step size (default = 0.008)
    :param display: show iterations (default = 0)

    :rtype numpy ndarray
    :return gamo: describing the warping function

    """
    cdef int m1, m, max_itri, displayi
    cdef double toli, deltai
    toli = tol
    deltai = delta
    max_itri = max_iter
    displayi = display
    m1 = time.size
    m = beta.shape[1]
    alpha = alpha/norm(alpha)
    q = q/norm(q)
    for i in range(0, m):
        beta[:, i] = beta[:, i]/norm(beta[:, i])

    cdef np.ndarray[double, ndim = 1, mode = "c"] gam1 = np.linspace(0, 1, m1)
    cdef np.ndarray[double, ndim = 1, mode = "c"] beta1 = np.zeros(m1 * m)
    cdef np.ndarray[double, ndim = 1, mode = "c"] gamout = np.zeros(m1)

    for ii in xrange(0, m):
        beta1[ii * m1:ii * m1 + m1] = beta[:, ii]


    gam1 = np.ascontiguousarray(gam1)
    beta1 = np.ascontiguousarray(beta1)
    gamout = np.ascontiguousarray(gamout)

    cmlogit.mlogit_warp_grad(&m1, &m, &alpha[0], &beta1[0], &time[0], &gam1[0], &q[0], &y[0], &max_itri, &toli, &deltai, &displayi, &gamout[0])

    return gamout
