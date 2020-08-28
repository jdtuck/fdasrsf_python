"""
Partial Least Squares using SVD

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
from fdasrsf.utility_functions import diffop, geigen, innerprod_q


def pls_svd(time, qf, qg, no, alpha=0.0):
    """
    This function computes the partial least squares using SVD

    :param time: vector describing time samples
    :param qf: numpy ndarray of shape (M,N) of N functions with M samples
    :param qg: numpy ndarray of shape (M,N) of N functions with M samples
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

    for ii in range(0, no):
        wf[:, ii] = wf[:, ii] / np.sqrt(innerprod_q(time, wf[:, ii], wf[:, ii]))
        wg[:, ii] = wg[:, ii] / np.sqrt(innerprod_q(time, wg[:, ii], wg[:, ii]))

    wqf = wf
    wqg = wg

    N = qf.shape[1]
    rfi = np.zeros(N)
    rgi = np.zeros(N)

    for l in range(0, N):
        rfi[l] = innerprod_q(time, qf[:, l], wqf[:, 0])
        rgi[l] = innerprod_q(time, qg[:, l], wqg[:, 0])

    cost = np.cov(rfi, rgi)[1, 0]

    return wqf, wqg, alpha, values, cost
