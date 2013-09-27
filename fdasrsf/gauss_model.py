"""
Gaussian Model of functional data

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""
import numpy as np
import utility_functions as uf


def gauss_model(fn, time, qn, gam, n=1, sort_samples=False):
    """
    This function models the functional data using a Gaussian model extracted from the principal components of the srvfs

    :param fn: numpy ndarray of shape (M,N) of M aligned functions with N samples
    :param time: vector of size N describing the sample points
    :param qn: numpy ndarray of shape (M,N) of M aligned srvfs with N samples
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
    no = 3
    eps = np.finfo(np.double).eps
    M = time.size

    # compute mean and covariance in q-domain
    mq_new = qn.mean(axis=1)
    N = mq_new.shape[0]
    mididx = np.round(time.shape[0] / 2)
    m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
    mqn = np.append(mq_new, m_new.mean())
    qn2 = np.vstack((qn, m_new))
    C = np.cov(qn2)

    q_s = np.random.multivariate_normal(mqn, C, n)
    q_s = q_s.transpose()

    # compute the correspondence to the original function domain
    fs = np.zeros((M, n))
    for k in xrange(0, n):
        fs[:, k] = uf.cumtrapzmid(time, q_s[0:M, k] * np.abs(q_s[0:M, k]), np.sign(q_s[M, k]) * (q_s[M, k] ** 2))

    # random warping generation

    return(samples)