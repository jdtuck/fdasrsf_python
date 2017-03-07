"""
Gaussian Model of functional data

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""
import numpy as np
import fdasrsf.utility_functions as uf
import collections


def gauss_model(fn, time, qn, gam, n=1, sort_samples=False):
    """
    This function models the functional data using a Gaussian model
    extracted from the principal components of the srvfs

    :param fn: numpy ndarray of shape (M,N) of N aligned functions with
     M samples
    :param time: vector of size M describing the sample points
    :param qn: numpy ndarray of shape (M,N) of N aligned srvfs with M samples
    :param gam: warping functions
    :param n: number of random samples
    :param sort_samples: sort samples (default = T)
    :type n: integer
    :type sort_samples: bool
    :type fn: np.ndarray
    :type qn: np.ndarray
    :type gam: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of numpy array
    :return fs: random aligned samples
    :return gams: random warping functions
    :return ft: random samples
    """

    # Parameters
    eps = np.finfo(np.double).eps
    binsize = np.diff(time)
    binsize = binsize.mean()
    M = time.size

    # compute mean and covariance in q-domain
    mq_new = qn.mean(axis=1)
    mididx = np.round(time.shape[0] / 2)
    m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
    mqn = np.append(mq_new, m_new.mean())
    qn2 = np.vstack((qn, m_new))
    C = np.cov(qn2)

    q_s = np.random.multivariate_normal(mqn, C, n)
    q_s = q_s.transpose()

    # compute the correspondence to the original function domain
    fs = np.zeros((M, n))
    for k in range(0, n):
        fs[:, k] = uf.cumtrapzmid(time, q_s[0:M, k] * np.abs(q_s[0:M, k]),
                                  np.sign(q_s[M, k]) * (q_s[M, k] ** 2),
                                  mididx)

    fbar = fn.mean(axis=1)
    fsbar = fs.mean(axis=1)
    err = np.transpose(np.tile(fbar-fsbar, (n,1)))
    fs += err

    # random warping generation
    rgam = uf.randomGamma(gam, n)
    gams = np.zeros((M, n))
    for k in range(0, n):
        gams[:, k] = uf.invertGamma(rgam[:, k])

    # sort functions and warping
    if sort_samples:
        mx = fs.max(axis=0)
        seq1 = mx.argsort()

        # compute the psi-function
        fy = np.gradient(rgam, binsize)
        psi = fy / np.sqrt(abs(fy) + eps)
        ip = np.zeros(n)
        len = np.zeros(n)
        for i in range(0, n):
            tmp = np.ones(M)
            ip[i] = tmp.dot(psi[:, i] / M)
            len[i] = np.acos(tmp.dot(psi[:, i] / M))

        seq2 = len.argsort()

        # combine x-variability and y-variability
        ft = np.zeros((M, n))
        for k in range(0, n):
            ft[:, k] = np.interp(gams[:, seq2[k]], np.arange(0, M) /
                                 np.double(M - 1), fs[:, seq1[k]])
            tmp = np.isnan(ft[:, k])
            while tmp.any():
                rgam2 = uf.randomGamma(gam, 1)
                ft[:, k] = np.interp(gams[:, seq2[k]], np.arange(0, M) /
                                     np.double(M - 1), uf.invertGamma(rgam2))
    else:
        # combine x-variability and y-variability
        ft = np.zeros((M, n))
        for k in range(0, n):
            ft[:, k] = np.interp(gams[:, k], np.arange(0, M) /
                                 np.double(M - 1), fs[:, k])
            tmp = np.isnan(ft[:, k])
            while tmp.any():
                rgam2 = uf.randomGamma(gam, 1)
                ft[:, k] = np.interp(gams[:, k], np.arange(0, M) /
                                     np.double(M - 1), uf.invertGamma(rgam2))

    samples = collections.namedtuple('samples', ['fs', 'gams', 'ft'])
    out = samples(fs, rgam, ft)
    return out
