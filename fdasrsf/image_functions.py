"""
functions for SRVF image manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cimage as im


def apply_gam_imag(F, gam):
    """
    apply diffeomorphism to image

    :param F: numpy array of image
    :param gam: numpy array of diffeomorphism

    :rtype: np.array
    :return Fnew: warped image

    """
    ndim = F.ndim

    x2 = gam[:,:,1].ravel(order='F')
    gamtmp = gam[:,:,0].T
    y2 = np.sort(gamtmp.ravel(order='F'))

    if ndim == 3:
        (m,n,d) = F.shape
        Fnew = np.zeros((m,n,d))

        U = np.linspace(0,1,m)
        V = np.linspace(0,1,n)

        for j in range(d):
            interp = RegularGridInterpolator((U, V), F[:,:,j], method="linear")
            tmp = interp((x2,y2))
            Fnew[:,:,j] = tmp.reshape((m,n),order='F')
    elif ndim == 2:
        (m,n) = F.shape

        U = np.linspace(0,1,m)
        V = np.linspace(0,1,n)

        interp = RegularGridInterpolator((U, V), F, method="linear")
        tmp = interp((x2,y2))
        Fnew = tmp.reshape((m,n),order='F')
    
    return Fnew


def makediffeoid(nrow,ncol):
    """
    computes a 2-D identity diffeomorphism

    :param nrow: number of rows
    :param ncol: number of columns

    :rtype: np.array
    :return gamid: amplitude depth

    """
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
    
    F = np.ascontiguousarray(F)
    dfdu, dfdv = im.compgrad3D(F)
    dfdu = dfdu.reshape((m,n,d), order='F')
    dfdv = dfdv.reshape((m,n,d), order='F')

    multFactor = np.zeros((m,n))

    if d==2:
        multFactor = dfdu[:,:,0]*dfdv[:,:,1] - dfdu[:,:,1]*dfdv[:,:,0]
        multFactor = np.abs(multFactor)
    elif d==3:
        multFactor = (dfdu[:,:,1]*dfdv[:,:,2] - dfdu[:,:,2]*dfdv[:,:,1])**2 + (dfdu[:,:,0]*dfdv[:,:,2] - dfdu[:,:,2]*dfdv[:,:,0])**2 + (dfdu[:,:,0]*dfdv[:,:,1] - dfdu[:,:,1]*dfdv[:,:,0])**2
        multFactor = np.sqrt(multFactor)
    
    return multFactor


def compEnergy(q1,q2):
    (m,n,d) = q1.shape
    ds = 1/(m-1)/(n-1)

    tmp = q1-q2

    H = np.dot(tmp.ravel(),tmp.ravel())*ds

    return H


def updateGam(qt,qm,b):
    v = qt - qm
    w = findphistar(qt,b)
    gamupdate = findupdategam(v,w,b)

    return gamupdate


def findphistar(q,b):
    (m,n,D,K) = b.shape

    d = q.shape[2]
    dbxdu = np.zeros((m,n,K))
    dbydv = np.zeros((m,n,K))
    expr1 = np.zeros((m,n,d,K))
    expr2 = np.zeros((m,n,d,K))

    for k in range(K):
        btmp = b[:,:,0,k]
        btmp = np.ascontiguousarray(btmp)
        dbxdui, tmp = im.compgrad2D(btmp)
        dbxdu[:,:,k] = dbxdui.reshape((m,n), order='F')
        btmp = b[:,:,1,k]
        btmp = np.ascontiguousarray(btmp)
        tmp, dbydvi = im.compgrad2D(btmp)
        dbydv[:,:,k] = dbydvi.reshape((m,n), order='F')
    
    divb = dbxdu + dbydv

    dqdu, dqdv = im.compgrad3D(q)
    dqdu = dqdu.reshape(q.shape, order='F')
    dqdv = dqdv.reshape(q.shape, order='F')

    for k in range(K):
        for j in range(d):
            expr1[:,:,j,k] = divb[:,:,k]*q[:,:,j]
            expr2[:,:,j,k] = dqdu[:,:,j]*b[:,:,0,k] + dqdv[:,:,j]*b[:,:,1,k]
    
    w = 0.5 * expr1 + expr2

    return w


def findupdategam(v,w,b):
    (m,n,D,K) = b.shape
    ds = 1/((m-1)*(n-1))

    innp = np.zeros(K)

    gamupdate = np.zeros((m,n,D))

    for k in range(K):
        vt = w[:,:,:,k]
        innp[k] = np.dot(v.ravel(),vt.ravel())*ds

        gamupdate = gamupdate + innp[k]*b[:,:,:,k]
    
    return gamupdate


def apply_gam_gamid(gamid, gaminc):
    (m, n, d) = gamid.shape

    U = np.linspace(0,1,m)
    V = np.linspace(0,1,n)

    gamcum = np.zeros((m,n,d))
    x2 = gaminc[:,:,1].ravel(order='F')
    gamtmp = gaminc[:,:,0].T
    y2 = np.sort(gamtmp.ravel(order='F'))
    for j in range(d):
        interp = RegularGridInterpolator((U, V), gamid[:,:,j], method="linear")
        tmp = interp((x2,y2))
        gamcum[:,:,j] = tmp.reshape((m,n),order='F')
    
    return gamcum


def gen_basis(m, n, M=10, N=10, baseType = 't', ortho=False):

    b = formbasisTid(M, m, n, baseType)

    if ortho:
        b = GramSchmidt(b)
    
    return b


def formbasisTid(M, m, n, base_type):
    """
    Generate Basis on tangent space at identity

    :param M: number of basis functions
    :param m: number of rows
    :param n: number of colunns
    :param base_type: basis type ('t' only implemented)

    :rtype: np.array
    :return b: basis functions

    """
    U, V = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,m), indexing='xy')

    idx = 0

    if base_type == 't':
        b = np.zeros((m,n,2,2*M))
        for s in range(1,M+1):
            const = np.sqrt(2)*np.pi*s
            sPI2 = 2*np.pi*s

            b[:,:,0,idx] = np.zeros((m,n))
            b[:,:,1,idx] = (np.cos(sPI2*V)-1)/const

            b[:,:,0,idx+1] = (np.cos(sPI2*U)-1)/const
            b[:,:,1,idx+1] = np.zeros((m,n))

            idx += 2

    return b


def GramSchmidt(b):
    (m,n,D,N) = b.shape

    ds = 1/((m-1)*(n-1))

    cnt = 0
    G = np.zeros((m,n,D,N))
    G[:,:,:,cnt] = b[:,:,:,cnt]

    Gtmp = G[:,:,:,cnt]
    Gtmp = np.ascontiguousarray(Gtmp)
    dvx, dvy = im.compgrad3D(Gtmp)
    dvx = dvx.reshape(Gtmp.shape, order='F')
    dvy = dvy.reshape(Gtmp.shape, order='F')
    l = np.dot(dvx.ravel(),dvx.ravel())*ds + np.dot(dvy.ravel(),dvy.ravel())*ds
    G[:,:,:,cnt] = G[:,:,:,cnt]/np.sqrt(l)

    for i in range(1,N):
        G[:,:,:,i] = b[:,:,:,i]
        Gtmp = G[:,:,:,i]
        Gtmp = np.ascontiguousarray(Gtmp)
        dv1x, dv1y = im.compgrad3D(Gtmp)
        dv1x = dv1x.reshape(Gtmp.shape, order='F')
        dv1y = dv1y.reshape(Gtmp.shape, order='F')

        for j in range(0,i-1):
            Gtmp = G[:,:,:,j]
            Gtmp = np.ascontiguousarray(Gtmp)
            dv2x, dv2y = im.compgrad3D(Gtmp)
            dv2x = dv2x.reshape(Gtmp.shape, order='F')
            dv2y = dv2y.reshape(Gtmp.shape, order='F')
            t = np.dot(dv1x.ravel(),dv2x.ravel())*ds + np.dot(dv1y.ravel(),dv2y.ravel())*ds
            G[:,:,:,i] = G[:,:,:,i]-t*G[:,:,:,j]
        
        v = G[:,:,:,i]
        l = np.dot(v.ravel(),v.ravel())*ds

        if l>0:
            cnt += 1
            G[:,:,:,cnt] = G[:,:,:,i]/np.sqrt(l)
    
    return G
