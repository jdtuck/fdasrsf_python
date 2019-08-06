# distutils: language = c++
cimport cDPg
import numpy as np
from numpy.linalg import norm

cimport numpy as np
from libcpp cimport bool
from cpython cimport array

def coptimum_reparam_N(np.ndarray[double, ndim=1, mode="c"] mf, np.ndarray[double, ndim=1, mode="c"] time,
                      np.ndarray[double, ndim=2, mode="c"] f, 
                      onlyDP=False, rotated=False, isclosed=False, skipm=0,
                      auto=0, w=0.0, lam1=0.0):
    """
    cython interface calculates the warping to align a set of functions f to a single function mf

    :param mf: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param onlyDP: use onlyDP (default = False)
    :param rotated: solve for rotation (default = False)
    :param isclosed: is a closed curve (default = False)
    :param skipm: (default 0)
    :param auto: (default 0)
    :param w: barrier weight (default 0.0)
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype numpy ndarray
    :return gam: describing the warping functions used to align columns of f with mf

    """
    cdef int M, N, n1, skipmi, autoi, swap
    cdef double lam, wi
    cdef bool onlyDP1, rotated1, isclosed1
    cdef np.ndarray[double, ndim=1, mode="c"] fopts = np.zeros(5)
    cdef np.ndarray[double, ndim=1, mode="c"] comtime = np.zeros(5)
    lam = lam1
    skipmi = skipm
    autoi = auto
    onlyDP1 = onlyDP
    rotated1 = rotated
    isclosed1 = isclosed
    wi = w
    n1 = 1

    mf = mf / norm(mf)
    M, N = f.shape[0], f.shape[1]

    cdef np.ndarray[double, ndim=1, mode="c"] opti = np.zeros(M+2)
    cdef np.ndarray[double, ndim=1, mode="c"] fi = np.zeros(M)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        fopts = np.zeros(5)
        comtime = np.zeros(5)
        swap = False
        fi = f[:, k] / norm(f[:, k])
        fi = np.ascontiguousarray(fi)

        cDPg.optimum_reparam(&fi[0], &mf[0], M, n1, wi, onlyDP1, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
        if fopts[0] == 1000:
            cDPg.optimum_reparam(&fi[0], &mf[0], M, n1, wi, True, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
        if swap:
            x = np.arange(1, M+1) / float(M)
            gam[:,k] = np.interp(x,opti[0:M],x)
        else:
            gam[:,k] = opti[0:M]
        
        gam[:,k] = (gam[:,k] - gam[0,k]) / (gam[-1,k] - gam[0,k])

    return gam


def coptimum_reparam_N2(np.ndarray[double, ndim=2, mode="c"] f1, np.ndarray[double, ndim=1, mode="c"] time,
                       np.ndarray[double, ndim=2, mode="c"] f2, 
                      onlyDP=False, rotated=False, isclosed=False, skipm=0,
                      auto=0, w=0.0, lam1=0.0):
    """
    cython interface calculates the warping to align a set of functions f1 to another set of functions f2

    :param f1: numpy ndarray of shape (M,N) of M functions with N samples
    :param time: vector of size N describing the sample points
    :param f2: numpy ndarray of shape (M,N) of M functions with N samples
    :param onlyDP: use onlyDP (default = False)
    :param rotated: solve for rotation (default = False)
    :param isclosed: is a closed curve (default = False)
    :param skipm: (default 0)
    :param auto: (default 0)
    :param w: barrier weight (default 0.0)
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype numpy ndarray
    :return gam: describing the warping functions used to align columns of f with mf

    """
    cdef int M, N, n1, skipmi, autoi
    cdef double lam, wi
    cdef bool onlyDP1, rotated1, isclosed1, swap
    cdef np.ndarray[double, ndim=1, mode="c"] fopts = np.zeros(5)
    cdef np.ndarray[double, ndim=1, mode="c"] comtime = np.zeros(5)
    lam = lam1
    skipmi = skipm
    autoi = auto
    onlyDP1 = onlyDP
    rotated1 = rotated
    isclosed1 = isclosed
    wi = w

    M, N = f1.shape[0], f1.shape[1]
    n1 = 1
    lam = lam1
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] opti = np.zeros(M+2)
    cdef np.ndarray[double, ndim=1, mode="c"] f1i = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] f2i = np.zeros(M)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        fopts = np.zeros(5)
        comtime = np.zeros(5)
        swap = False
        f1i = f1[:, k] / norm(f1[:, k])
        f2i = f2[:, k] / norm(f2[:, k])
        f1i = np.ascontiguousarray(f1i)
        f2i = np.ascontiguousarray(f2i)

        cDPg.optimum_reparam(&f1i[0], &f2i[0], M, n1, wi, onlyDP1, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
        if fopts[0] == 1000:
            cDPg.optimum_reparam(&f1i[0], &f2i[0], M, n1, wi, True, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
        if swap:
            x = np.arange(1, M+1) / float(M)
            gami = np.interp(x,opti[0:M],x)
        else:
            gami = opti[0:M]

        gam[:, k] = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam


def coptimum_reparam(np.ndarray[double, ndim=1, mode="c"] f1, np.ndarray[double, ndim=1, mode="c"] time,
                     np.ndarray[double, ndim=1, mode="c"] f2, 
                     onlyDP=False, rotated=False, isclosed=False, skipm=0,
                     auto=0, w=0.0, lam1=0.0):
    """
    cython interface for calculates the warping to align functions f2 to f1

    :param f1: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param f2: vector of size N samples of second function
    :param onlyDP: use onlyDP (default = False)
    :param rotated: solve for rotation (default = False)
    :param isclosed: is a closed curve (default = False)
    :param skipm: (default 0)
    :param auto: (default 0)
    :param w: barrier weight (default 0.0)
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    cdef int M, n1, skipmi, autoi
    cdef double lam, wi
    cdef bool onlyDP1, rotated1, isclosed1, swap
    cdef np.ndarray[double, ndim=1, mode="c"] fopts = np.zeros(5)
    cdef np.ndarray[double, ndim=1, mode="c"] comtime = np.zeros(5)
    lam = lam1
    skipmi = skipm
    autoi = auto
    onlyDP1 = onlyDP
    rotated1 = rotated
    isclosed1 = isclosed
    swap = False
    wi = w

    M = f1.shape[0]
    n1 = 1

    cdef np.ndarray[double, ndim=1, mode="c"] opti = np.zeros(M+2)
    gam = np.zeros(M)

    cDPg.optimum_reparam(&f1[0], &f2[0], M, n1, wi, onlyDP1, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
    if fopts[0] == 1000:
        cDPg.optimum_reparam(&f1[0], &f2[0], M, n1, wi, True, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
    
    if swap:
        x = np.linspace(0,1,M)
        gam = np.interp(x,opti[0:M],x)
    else:
        gam = opti[0:M]

    gam = (gam - gam[0]) / (gam[-1] - gam[0])

    return gam


def coptimum_reparam_N2_pair(np.ndarray[double, ndim=2, mode="c"] f, np.ndarray[double, ndim=1, mode="c"] time,
                            np.ndarray[double, ndim=2, mode="c"] f1, np.ndarray[double, ndim=2, mode="c"] f2, 
                            onlyDP=False, rotated=False, isclosed=False, skipm=0,
                            auto=0, w=0.0, lam1=0.0):
    """
    cython interface for calculates the warping to align paired function f1 and f2 to f

    :param f: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param f1: vector of size N samples of second function
    :param f2: vector of size N samples of second function
    :param onlyDP: use onlyDP (default = False)
    :param rotated: solve for rotation (default = False)
    :param isclosed: is a closed curve (default = False)
    :param skipm: (default 0)
    :param auto: (default 0)
    :param w: barrier weight (default 0.0)
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    cdef int M, N, n1, skipmi, autoi
    n1 = 2
    cdef double lam, wi
    cdef bool onlyDP1, rotated1, isclosed1, swap
    cdef np.ndarray[double, ndim=1, mode="c"] fopts = np.zeros(5)
    cdef np.ndarray[double, ndim=1, mode="c"] comtime = np.zeros(5)
    lam = lam1
    skipmi = skipm
    autoi = auto
    onlyDP1 = onlyDP
    rotated1 = rotated
    isclosed1 = isclosed
    wi = w
    M, N = f1.shape[0], f1.shape[1]
    lam = lam1

    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] opti = np.zeros(M+2)
    cdef np.ndarray[double, ndim=1, mode="c"] f1i = np.zeros(M * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] f2i = np.zeros(M * n1)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        fopts = np.zeros(5)
        comtime = np.zeros(5)
        swap = False
        f1i = f.reshape(M*n1)
        f2tmp = np.column_stack((f1[:, k], f2[:, k]))
        f2i = f2tmp.reshape(M*n1)

        f1i = np.ascontiguousarray(f1i)
        f2i = np.ascontiguousarray(f2i)

        cDPg.optimum_reparam(&f1i[0], &f2i[0], M, n1, wi, onlyDP1, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
        if fopts[0] == 1000:
            cDPg.optimum_reparam(&f1i[0], &f2i[0], M, n1, wi, True, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
        if swap:
            x = np.arange(1, M+1) / float(M)
            gami = np.interp(x,opti[0:M],x)
        else:
            gami = opti[0:M]

        gam[:, k] = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam


def coptimum_reparam_pair_f(np.ndarray[double, ndim=2, mode="c"] f1, np.ndarray[double, ndim=1, mode="c"] time,
                          np.ndarray[double, ndim=2, mode="c"] f2, 
                          onlyDP=False, rotated=False, isclosed=False, skipm=0,
                          auto=0, w=0.0, lam1=0.0):
    """
    cython interface for calculates the warping to align paired function f2 to f1

    :param f1: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param f2: vector of size N samples of second function
    :param onlyDP: use onlyDP (default = False)
    :param rotated: solve for rotation (default = False)
    :param isclosed: is a closed curve (default = False)
    :param skipm: (default 0)
    :param auto: (default 0)
    :param w: barrier weight (default 0.0)
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    cdef int M, N, skipmi, autoi
    cdef double lam, wi
    cdef bool onlyDP1, rotated1, isclosed1, swap
    cdef np.ndarray[double, ndim=1, mode="c"] fopts = np.zeros(5)
    cdef np.ndarray[double, ndim=1, mode="c"] comtime = np.zeros(5)
    M, N = f1.shape[0], f1.shape[1]
    lam = lam1
    skipmi = skipm
    autoi = auto
    onlyDP1 = onlyDP
    rotated1 = rotated
    isclosed1 = isclosed
    wi = w
    swap = False

    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] opti = np.zeros(M+2)
    cdef np.ndarray[double, ndim=1, mode="c"] f1i = np.zeros(M * N)
    cdef np.ndarray[double, ndim=1, mode="c"] f2i = np.zeros(M * N)

    sizes = np.zeros(1, dtype=np.int32)
    f1i = f1.reshape(M*N)
    f2i = f2.reshape(M*N)

    f1i = np.ascontiguousarray(f1i)
    f2i = np.ascontiguousarray(f2i)

    cDPg.optimum_reparam(&f2i[0], &f2i[0], M, N, wi, onlyDP1, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
    if fopts[0] == 1000:
        cDPg.optimum_reparam(&f2i[0], &f2i[0], M, N, wi, True, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
    
    if swap:
        x = np.arange(1, M+1) / float(M)
        gami = np.interp(x,opti[0:M],x)
    else:
        gami = opti[0:M]

    gam = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam


def coptimum_reparam_curve_f(np.ndarray[double, ndim=2, mode="c"] f1, np.ndarray[double, ndim=1, mode="c"] time,
                     np.ndarray[double, ndim=2, mode="c"] f2, 
                      onlyDP=False, rotated=False, isclosed=False, skipm=0,
                      auto=0, w=0.0, lam1=0.0):
    """
    cython interface for calculates the warping to align curve f2 to f1

    :param f1: matrix of size nxN samples of first SRVF
    :param time: vector of size N describing the sample points
    :param f2: matrix of size nxN samples of second SRVF
    :param onlyDP: use onlyDP (default = False)
    :param rotated: solve for rotation (default = False)
    :param isclosed: is a closed curve (default = False)
    :param skipm: (default 0)
    :param auto: (default 0)
    :param w: barrier weight (default 0.0)
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    cdef int M, n1, skipmi, autoi
    cdef double lam, wi
    cdef bool onlyDP1, rotated1, isclosed1, swap
    cdef np.ndarray[double, ndim=1, mode="c"] fopts = np.zeros(5)
    cdef np.ndarray[double, ndim=1, mode="c"] comtime = np.zeros(5)
    n1 = f1.shape[0]
    M = f1.shape[1]
    lam = lam1
    skipmi = skipm
    autoi = auto
    onlyDP1 = onlyDP
    rotated1 = rotated
    isclosed1 = isclosed
    wi = w
    swap = False

    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] opti = np.zeros(M+2)
    cdef np.ndarray[double, ndim=1, mode="c"] f1i = np.zeros(M * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] f2i = np.zeros(M * n1)

    f1i = f1.reshape(M*n1, order='F')
    f2i = f2.reshape(M*n1, order='F')

    f1i = np.ascontiguousarray(f1i)
    f2i = np.ascontiguousarray(f2i)

    cDPg.optimum_reparam(&f1i[0], &f2i[0], M, n1, wi, onlyDP1, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
        
    if fopts[0] == 1000:
        cDPg.optimum_reparam(&f1i[0], &f2i[0], M, n1, wi, True, rotated1, isclosed1, skipmi, autoi, &opti[0], swap, &fopts[0], &comtime[0])
    
    if swap:
        x = np.arange(1, M+1) / float(M)
        gami = np.interp(x,opti[0:M],x)
    else:
        gami = opti[0:M]

    gam = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam
