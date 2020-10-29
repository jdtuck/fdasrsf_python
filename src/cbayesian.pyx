# cython: language_level=2
import numpy as np
cimport cbayesian
cimport cyarma
cimport numpy as np

include "cyarma.pyx"

from libcpp cimport bool

def bcalcY(area, np.ndarray[double, ndim=1, mode="c"] y):
    y = np.ascontiguousarray(y)
    cdef vec ay = numpy_to_vec_d(y)
    cdef vec out = calcY(area, ay)

    cdef np.ndarray[np.double_t,ndim=1] out1 = vec_to_numpy(out, None)

    return(out1)

def bcuL2norm2(np.ndarray[double, ndim=1, mode="c"] x, np.ndarray[double, ndim=1, mode="c"] y):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    cdef vec ax = numpy_to_vec_d(x)
    cdef vec ay = numpy_to_vec_d(y)
    cdef vec out = cuL2norm2(ax, ay)

    cdef np.ndarray[np.double_t,ndim=1] out1 = vec_to_numpy(out, None)

    return(out1)

def ctrapzCpp(np.ndarray[double, ndim=1, mode="c"] x, np.ndarray[double, ndim=1, mode="c"] y):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    cdef vec ax = numpy_to_vec_d(x)
    cdef vec ay = numpy_to_vec_d(y)

    cdef double out = trapzCpp(ax, ay)

    return(out)

def border_l2norm(np.ndarray[double, ndim=1, mode="c"] x, np.ndarray[double, ndim=1, mode="c"] y):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    cdef vec ax = numpy_to_vec_d(x)
    cdef vec ay = numpy_to_vec_d(y)

    cdef double out = order_l2norm(ax, ay)

    return(out)
