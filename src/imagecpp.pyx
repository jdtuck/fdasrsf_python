# cython: language_level=2
cimport cUnitSquareImage
import numpy as np

cimport numpy as np
from cpython cimport array

from libcpp cimport bool

np.import_array()

def compgrad3D(np.ndarray[double, ndim=3, mode="c"] F):
    """
    cython interface calculates the 2D gradient of q-map image

    :param F: matrix of three dimension 

    :rtype numpy ndarray
    :return dfdu: derivative
    :return dfdv: derivative

    """
    cdef int n, t, d
    cdef double lam

    n = F.shape[0]
    t = F.shape[1]
    d = F.shape[2]

    cdef np.ndarray[double, ndim=1, mode="c"] Fi = np.zeros(n*t*d)
    cdef np.ndarray[double, ndim=1, mode="c"] dfdui = np.zeros(n*t*d)
    cdef np.ndarray[double, ndim=1, mode="c"] dfdvi = np.zeros(n*t*d)

    Fi = F.reshape(n*t*d, order='F')

    Fi = np.ascontiguousarray(Fi)

    cUnitSquareImage.findgrad2D(&dfdui[0], &dfdvi[0], &Fi[0], n, t, d)

    return dfdui, dfdvi


def compgrad2D(np.ndarray[double, ndim=2, mode="c"] F):
    """
    cython interface calculates the 2D gradient of q-map image

    :param F: matrix of three dimension 

    :rtype numpy ndarray
    :return dfdu: derivative
    :return dfdv: derivative

    """
    cdef int n, t, d
    cdef double lam

    n = F.shape[0]
    t = F.shape[1]
    d = 1

    cdef np.ndarray[double, ndim=1, mode="c"] Fi = np.zeros(n*t*d)
    cdef np.ndarray[double, ndim=1, mode="c"] dfdui = np.zeros(n*t*d)
    cdef np.ndarray[double, ndim=1, mode="c"] dfdvi = np.zeros(n*t*d)

    Fi = F.reshape(n*t*d, order='F')

    Fi = np.ascontiguousarray(Fi)

    cUnitSquareImage.findgrad2D(&dfdui[0], &dfdvi[0], &Fi[0], n, t, d)

    return dfdui, dfdvi


def check_crossing(np.ndarray[double, ndim=3, mode="c"] gam):
    cdef int n, t, D, is_diffeo

    n = gam.shape[0]
    t = gam.shape[1]
    D = gam.shape[2]

    is_diffeo = 1

    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(n*t*D)

    gami = gam.reshape(n*t*D, order='F')

    gami = np.ascontiguousarray(gami)

    is_diffeo = cUnitSquareImage.check_crossing(&gami[0], n, t, D)

    return is_diffeo
