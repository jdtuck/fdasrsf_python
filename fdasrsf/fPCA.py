"""
Vertical and Horizontal Functional Principal Component Analysis using SRSF

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""
import numpy as np
from . import utility_functions as uf
from scipy.linalg import norm
import matplotlib.pyplot as plt
from . import plot_style as plot
import collections


def vertfPCA(fn, time, qn, no=1, showplot=True):
    """
    This function calculates vertical functional principal component analysis on aligned data

    :param fn: numpy ndarray of shape (M,N) of M aligned functions with N samples
    :param time: vector of size N describing the sample points
    :param qn: numpy ndarray of shape (M,N) of M aligned SRSF with N samples
    :param no: number of components to extract (default = 1)
    :param showplot: Shows plots of results using matplotlib (default = T)
    :type showplot: bool
    :type no: int

    :rtype: tuple of numpy ndarray
    :return q_pca: srsf principal directions
    :return f_pca: functional principal directions
    :return latent: latent values
    :return coef: coefficients
    :return U: eigenvectors

    """
    coef = np.arange(-2., 3.)
    Nstd = coef.shape[0]

    # FPCA
    mq_new = qn.mean(axis=1)
    N = mq_new.shape[0]
    mididx = np.round(time.shape[0] / 2)
    m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
    mqn = np.append(mq_new, m_new.mean())
    qn2 = np.vstack((qn, m_new))
    K = np.cov(qn2)

    U, s, V = np.linalg.svd(K)
    stdS = np.sqrt(s)

    # compute the PCA in the q domain
    q_pca = np.ndarray(shape=(N + 1, Nstd, no), dtype=float)
    for k in range(0, no):
        for l in range(0, Nstd):
            q_pca[:, l, k] = mqn + coef[l] * stdS[k] * U[:, k]

    # compute the correspondence in the f domain
    f_pca = np.ndarray(shape=(N, Nstd, no), dtype=float)
    for k in range(0, no):
        for l in range(0, Nstd):
            f_pca[:, l, k] = uf.cumtrapzmid(time, q_pca[0:N, l, k] * np.abs(q_pca[0:N, l, k]),
                                            np.sign(q_pca[N, l, k]) * (q_pca[N, l, k] ** 2))

    N2 = qn.shape[1]
    c = np.zeros((N2, no))
    for k in range(0, no):
        for l in range(0, N2):
            c[l, k] = sum((np.append(qn[:, l], m_new[l]) - mqn) * U[:, k])

    vfpca_results = collections.namedtuple('vfpca', ['q_pca', 'f_pca', 'latent', 'coef', 'U'])
    vfpca = vfpca_results(q_pca, f_pca, s, c, U)

    if showplot:
        CBcdict = {
            'Bl': (0, 0, 0),
            'Or': (.9, .6, 0),
            'SB': (.35, .7, .9),
            'bG': (0, .6, .5),
            'Ye': (.95, .9, .25),
            'Bu': (0, .45, .7),
            'Ve': (.8, .4, 0),
            'rP': (.8, .6, .7),
        }
        cl = sorted(CBcdict.keys())
        fig, ax = plt.subplots(2, no)
        for k in range(0, no):
            axt = ax[0, k]
            for l in range(0, Nstd):
                axt.plot(time, q_pca[0:N, l, k], color=CBcdict[cl[l]])
                axt.hold(True)

            axt.set_title('q domain: PD %d' % (k + 1))
            plot.rstyle(axt)
            axt = ax[1, k]
            for l in range(0, Nstd):
                axt.plot(time, f_pca[:, l, k], color=CBcdict[cl[l]])
                axt.hold(True)

            axt.set_title('f domain: PD %d' % (k + 1))
            plot.rstyle(axt)
        fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(s) / sum(s)
        idx = np.arange(0, N + 1) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.xlabel("Percentage")
        plt.ylabel("Index")
        plt.show()

    return vfpca


def horizfPCA(gam, time, no, showplot=True):
    """
    This function calculates horizontal functional principal component analysis on aligned data

    :param gam: numpy ndarray of shape (M,N) of M warping functions
    :param time: vector of size N describing the sample points
    :param no: number of components to extract (default = 1)
    :param showplot: Shows plots of results using matplotlib (default = T)
    :type showplot: bool
    :type no: int

    :rtype: tuple of numpy ndarray
    :return q_pca: srsf principal directions
    :return f_pca: functional principal directions
    :return latent: latent values
    :return coef: coefficients
    :return U: eigenvectors

    """
    # Calculate Shooting Vectors
    mu, gam_mu, psi, vec = uf.SqrtMean(gam)
    tau = np.arange(1, 6)
    TT = time.shape[0]

    # TFPCA
    K = np.cov(vec)

    U, s, V = np.linalg.svd(K)
    vm = vec.mean(axis=1)

    gam_pca = np.ndarray(shape=(tau.shape[0], mu.shape[0] + 1, no), dtype=float)
    psi_pca = np.ndarray(shape=(tau.shape[0], mu.shape[0], no), dtype=float)
    for j in range(0, no):
        for k in tau:
            v = (k - 3) * np.sqrt(s[j]) * U[:, j]
            vn = norm(v) / np.sqrt(TT)
            if vn < 0.0001:
                psi_pca[k-1, :, j] = mu
            else:
                psi_pca[k-1, :, j] = np.cos(vn) * mu + np.sin(vn) * v / vn

            tmp = np.zeros(TT)
            tmp[1:TT] = np.cumsum(psi_pca[k-1, :, j] * psi_pca[k-1, :, j])
            gam_pca[k-1, :, j] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])

    hfpca_results = collections.namedtuple('hfpca', ['gam_pca', 'psi_pca', 'latent', 'U', 'gam_mu'])
    hfpca = hfpca_results(gam_pca, psi_pca, s, U, gam_mu)

    if showplot:
        CBcdict = {
            'Bl': (0, 0, 0),
            'Or': (.9, .6, 0),
            'SB': (.35, .7, .9),
            'bG': (0, .6, .5),
            'Ye': (.95, .9, .25),
            'Bu': (0, .45, .7),
            'Ve': (.8, .4, 0),
            'rP': (.8, .6, .7),
        }
        fig, ax = plt.subplots(1, no)
        for k in range(0, no):
            axt = ax[k]
            axt.set_color_cycle(CBcdict[c] for c in sorted(CBcdict.keys()))
            tmp = gam_pca[:, :, k]
            axt.plot(np.linspace(0, 1, TT), tmp.transpose())
            axt.set_title('PD %d' % (k + 1))
            axt.set_aspect('equal')
            plot.rstyle(axt)

        fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(s) / sum(s)
        idx = np.arange(0, TT-1) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.xlabel("Percentage")
        plt.ylabel("Index")
        plt.show()

    return hfpca