"""
Partial Least Squares using SVD

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""
import numpy as np
from numpy.linalg import norm
from utility_functions import diffop, geigen, update_progress, innerprod_q


def pls_svd(time, qf, qg, no, alpha=0.0):
    """
    This function computes the partial least squares using SVD

    :param time: vector describing time samples
    :param qf: numpy ndarray of shape (M,N) of M functions with N samples
    :param qg: numpy ndarray of shape (M,N) of M functions with N samples
    :param no: number of components
    :param alpha: amount of smoothing (Default = 0.0 i.e., none)

    :rtype: numpy ndarray
    :return wqf: f weight function
    :return wqg: g weight function
    :return alpha: smoothing value
    :return values: singular values

    """
    binsize = np.diff(time)
    binsize = binsize.mean()

    Kfg = np.cov(qf, qg)
    Kfg = Kfg[0:qf.shape[0], qf.shape[0]:Kfg.shape[0]]
    nx = Kfg.shape[0]
    D4x = diffop(nx, binsize)
    values, Lmat, Mmat = geigen(Kfg, np.eye(nx) + alpha * D4x, np.eye(nx) + alpha * D4x)
    wf = Lmat[:, 0:no]
    wg = Mmat[:, 0:no]

    for ii in xrange(0, no):
        wf[:, ii] = wf[:, ii] / np.sqrt(innerprod_q(time, wf[:, ii], wf[:, ii]))
        wg[:, ii] = wg[:, ii] / np.sqrt(innerprod_q(time, wg[:, ii], wg[:, ii]))

    wqf = wf
    wqg = wg

    return wqf, wqg, alpha, values
