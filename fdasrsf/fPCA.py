"""
Vertical and Horizontal Functional Principal Component Analysis using SRSF

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
import fdasrsf.utility_functions as uf
import fdasrsf.geometry as geo
from scipy import dot
from scipy.linalg import norm, svd
from scipy.integrate import trapz, cumtrapz
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
import fdasrsf.plot_style as plot
import collections


def vertfPCA(fn, time, qn, no=2, showplot=True):
    """
    This function calculates vertical functional principal component analysis
    on aligned data

    :param fn: numpy ndarray of shape (M,N) of N aligned functions with M
               samples
    :param time: vector of size N describing the sample points
    :param qn: numpy ndarray of shape (M,N) of N aligned SRSF with M samples
    :param no: number of components to extract (default = 2)
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
    mididx = int(np.round(time.shape[0] / 2))
    m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
    mqn = np.append(mq_new, m_new.mean())
    qn2 = np.vstack((qn, m_new))
    K = np.cov(qn2)

    U, s, V = svd(K)
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
                                            np.sign(q_pca[N, l, k]) * (q_pca[N, l, k] ** 2),
                                            mididx)
        fbar = fn.mean(axis=1)
        fsbar = f_pca[:, :, k].mean(axis=1)
        err = np.transpose(np.tile(fbar-fsbar, (Nstd,1)))
        f_pca[:, :, k] += err

    N2 = qn.shape[1]
    c = np.zeros((N2, no))
    for k in range(0, no):
        for l in range(0, N2):
            c[l, k] = sum((np.append(qn[:, l], m_new[l]) - mqn) * U[:, k])

    vfpca_results = collections.namedtuple('vfpca', ['q_pca', 'f_pca', 'latent', 'coef', 'U', 'id', 'mqn', 'time'])
    vfpca = vfpca_results(q_pca, f_pca, s, c, U, mididx, mqn, time)

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

            axt.set_title('q domain: PD %d' % (k + 1))
            axt = ax[1, k]
            for l in range(0, Nstd):
                axt.plot(time, f_pca[:, l, k], color=CBcdict[cl[l]])

            axt.set_title('f domain: PD %d' % (k + 1))
        fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(s) / sum(s)
        idx = np.arange(0, N + 1) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.xlabel("Percentage")
        plt.ylabel("Index")
        plt.show()

    return vfpca


def horizfPCA(gam, time, no=2, showplot=True):
    """
    This function calculates horizontal functional principal component analysis on aligned data

    :param gam: numpy ndarray of shape (M,N) of N warping functions
    :param time: vector of size M describing the sample points
    :param no: number of components to extract (default = 2)
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

    U, s, V = svd(K)
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
    
    N2 = gam.shape[1]
    c = np.zeros((N2,no))
    for k in range(0,no_pca):
        for i in range(0,N2):
            c[i,k] = np.sum(dot(vec[:,i]-vm,U[:,k]))

    hfpca_results = collections.namedtuple('hfpca', ['gam_pca', 'psi_pca', 'latent', 'U', 'gam_mu', 'coef', 'vec'])
    hfpca = hfpca_results(gam_pca, psi_pca, s, U, gam_mu, c, vec)

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
            tmp = gam_pca[:, :, k]
            axt.plot(np.linspace(0, 1, TT), tmp.transpose())
            axt.set_title('PD %d' % (k + 1))
            axt.set_aspect('equal')

        fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(s) / sum(s)
        idx = np.arange(0, TT-1) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.xlabel("Percentage")
        plt.ylabel("Index")
        plt.show()

    return hfpca


def jointfPCA(fn, time, qn, q0, gam, no=2, showplot=True):
    """
    This function calculates joint functional principal component analysis
    on aligned data

    :param fn: numpy ndarray of shape (M,N) of N aligned functions with M
               samples
    :param time: vector of size N describing the sample points
    :param qn: numpy ndarray of shape (M,N) of N aligned SRSF with M samples
    :param no: number of components to extract (default = 2)
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
    coef = np.arange(-1., 2.)
    Nstd = coef.shape[0]

    # set up for fPCA in q-space
    mq_new = qn.mean(axis=1)
    M = time.shape[0]
    mididx = int(np.round(M / 2))
    m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
    mqn = np.append(mq_new, m_new.mean())
    qn2 = np.vstack((qn, m_new))

    # calculate vector space of warping functions
    mu_psi, gam_mu, psi, vec = uf.SqrtMean(gam)

    # joint fPCA
    C = fminbound(find_C,0,1e4,(qn2,vec,q0,no,mu_psi))
    qhat, gamhat, a, U, s, mu_g, g, cov = jointfPCAd(qn2, vec, C, no, mu_psi)

    # geodesic paths
    q_pca = np.ndarray(shape=(M, Nstd, no), dtype=float)
    f_pca = np.ndarray(shape=(M, Nstd, no), dtype=float)
    
    for k in range(0, no):
        for l in range(0, Nstd):
            qhat = mqn + dot(U[0:(M+1),k],coef[l]*np.sqrt(s[k]))
            vechat = dot(U[(M+1):,k],(coef[l]*np.sqrt(s[k]))/C)
            psihat = geo.exp_map(mu_psi,vechat)
            gamhat = cumtrapz(psihat*psihat,np.linspace(0,1,M),initial=0)
            gamhat = (gamhat - gamhat.min()) / (gamhat.max() - gamhat.min())
            if (sum(vechat)==0):
                gamhat = np.linspace(0,1,M)
            
            fhat = uf.cumtrapzmid(time, qhat[0:M]*np.fabs(qhat[0:M]), np.sign(qhat[M])*(qhat[M]*qhat[M]), mididx)
            f_pca[:,l,k] = uf.warp_f_gamma(np.linspace(0,1,M), fhat, gamhat)
            q_pca[:,l,k] = uf.warp_q_gamma(np.linspace(0,1,M), qhat[0:M], gamhat)

    jfpca_results = collections.namedtuple('jfpca', ['q_pca', 'f_pca', 'latent', 'coef', 'U', 'mu_psi', 'mu_g', 'id', 'C', 'time', 'g', 'cov'])
    jfpca = jfpca_results(q_pca, f_pca, s, a, U, mu_psi, mu_g, mididx, C, time, g, cov)

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
                axt.plot(time, q_pca[0:M, l, k], color=CBcdict[cl[l]])

            axt.set_title('q domain: PD %d' % (k + 1))
            axt = ax[1, k]
            for l in range(0, Nstd):
                axt.plot(time, f_pca[:, l, k], color=CBcdict[cl[l]])

            axt.set_title('f domain: PD %d' % (k + 1))
        fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(s) / sum(s)
        idx = np.arange(0, s.shape[0]) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.xlabel("Percentage")
        plt.ylabel("Index")
        plt.show()

    return jfpca


def jointfPCAd(qn, vec, C, m, mu_psi):
    (M,N) = qn.shape
    g = np.vstack((qn, C*vec))

    mu_q = qn.mean(axis=1)
    mu_g = g.mean(axis=1)

    K = np.cov(g)
    U, s, V = svd(K)

    a = np.zeros((N,m))
    for i in range(0,N):
        for j in range(0,m):
            tmp = (g[:,i]-mu_g)
            a[i,j] = dot(tmp.T, U[:,j])

    qhat = np.tile(mu_q, (N,1))
    qhat = qhat.T
    qhat = qhat + dot(U[0:M,0:m],a.T)

    vechat = dot(U[M:,0:m], a.T/C)
    psihat = np.zeros((M-1,N))
    gamhat = np.zeros((M-1,N))
    for ii in range(0,N):
        psihat[:,ii] = geo.exp_map(mu_psi,vechat[:,ii])
        gam_tmp = cumtrapz(psihat[:,ii]*psihat[:,ii], np.linspace(0,1,M-1), initial=0)
        gamhat[:,ii] = (gam_tmp - gam_tmp.min()) / (gam_tmp.max() - gam_tmp.min())
    
    U = U[:,0:m]
    s = s[0:m]

    return qhat, gamhat, a, U, s, mu_g, g, K

def find_C(C, qn, vec, q0, m, mu_psi):
    qhat, gamhat, a, U, s, mu_g = jointfPCAd(qn, vec, C, m, mu_psi)
    (M,N) = qn.shape
    time = np.linspace(0,1,M-1)

    d = np.zeros(N)
    for i in range(0,N):
        tmp = uf.warp_q_gamma(time, qhat[0:(M-1),i], uf.invertGamma(gamhat[:,i]))
        d[i] = trapz((tmp-q0[:,i])*(tmp-q0[:,i]), time)

    out = sum(d*d)/N

    return out