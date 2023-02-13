"""
functions for SRVF image manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
from scipy.interpolate import griddata


def apply_gam_to_gam(gamnew, gam):
    m = gam.shape[0]
    n = gam.shape[1]
    D = gam.shape[2]
    md = 8
    mt = md*m
    nt = md*n

    U = np.linspace(0,1,m)
    V = np.linspace(0,1,n)

    return


def apply_gam_imag(F, gam):
    (m,n,d) = F.shape

    Fnew = np.zeros((m,n,d))

    U = np.linspace(0,1,m)
    V = np.linspace(0,1,n)

    for j in range(d):
        Fnew[:,:,j] = griddata((U,V), F[:,:,j], (gam[:,:,0], gam[:,:,1]), method='cubic')
    
    return Fnew


def makediffeoid(nrow,ncol):
    D = 2
    gamid = np.zeros((nrow,ncol,D))

    U, V = np.meshgrid(np.linspace(0,1,ncol), np.linspace(0,1,nrow), indexing='xy')

    gamid[:,:,0] = U
    gamid[:,:,1] = V

    return gamid


def image_to_q(F):
    d = F.shape[2]

    if d < 2:
        raise NameError('Data dimension is wrong!')
    
    q = F.copy()

    sqrtmultfact = np.sqrt(Jacob_imag(F))
    for i in range(d):
        q[:,:,i] = sqrtmultfact*F[:,:,i]
    
    return q


def Jacob_imag(F):
    (m,n,d) = F.shape

    if d < 2:
        raise NameError('Data dimension is wrong!')
    
    dfdu, dfdv = compgrad2D(F)

    multFactor = np.zeros((m,n))

    if d==2:
        multFactor = dfdu[:,:,0]@dfdv[:,:,1] - dfdu[:,:,1]@dfdv[:,:,0]
        multFactor = np.abs(multFactor)
    elif d==3:
        multFactor = (dfdu[:,:,1]@dfdv[:,:,2] - dfdu[:,:,2]@dfdv[:,:,1])**2 + (dfdu[:,:,0]@dfdv[:,:,2] - dfdu[:,:,2]@dfdv[:,:,0])**2 + (dfdu[:,:,0]@dfdv[:,:,1] - dfdu[:,:,1]@dfdv[:,:,0])**2
        multFactor = np.sqrt(multFactor)
    
    return multFactor


