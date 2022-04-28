"""
Vertical and Horizontal Functional Principal Component Analysis using SRSF

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
import fdasrsf.utility_functions as uf
import fdasrsf.geometry as geo
from scipy.linalg import norm, svd
from scipy.integrate import trapz, cumtrapz
from scipy.optimize import fminbound
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import fdasrsf.plot_style as plot
import collections


class fdavpca:
    """
    This class provides vertical fPCA using the
    SRVF framework

    Usage:  obj = fdavpca(warp_data)

    :param warp_data: fdawarp class with alignment data
    :param q_pca: srvf principal directions
    :param f_pca: f principal directions
    :param latent: latent values
    :param coef: principal coefficients
    :param id: point used for f(0)
    :param mqn: mean srvf
    :param U: eigenvectors
    :param stds: geodesic directions

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  15-Mar-2018
    """

    def __init__(self, fdawarp):
        """
        Construct an instance of the fdavpca class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size==0:
            raise Exception('Please align fdawarp class using srsf_align!')

        self.warp_data = fdawarp

    def calc_fpca(self, no=3, id=None, stds = np.arange(-1, 2)):
        """
        This function calculates vertical functional principal component analysis
        on aligned data

        :param no: number of components to extract (default = 3)
        :param id: point to use for f(0) (default = midpoint)
        :param stds: number of standard deviations along gedoesic to compute (default = -1,0,1)
        :type no: int
        :type id: int

        :rtype: fdavpca object containing
        :return q_pca: srsf principal directions
        :return f_pca: functional principal directions
        :return latent: latent values
        :return coef: coefficients
        :return U: eigenvectors

        """
        fn = self.warp_data.fn
        time = self.warp_data.time
        qn = self.warp_data.qn

        M = time.shape[0]
        if id is None:
            mididx = int(np.round(M / 2))
        else:
            mididx = id

        Nstd = stds.shape[0]

        # FPCA
        mq_new = qn.mean(axis=1)
        N = mq_new.shape[0]
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
                q_pca[:, l, k] = mqn + stds[l] * stdS[k] * U[:, k]

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

        self.q_pca = q_pca
        self.f_pca = f_pca
        self.latent = s[0:no]
        self.coef = c
        self.U = U[:,0:no]
        self.id = mididx
        self.mqn = mqn
        self.time = time
        self.stds = stds
        self.no = no

        return

    def plot(self):
        """
        plot plot elastic vertical fPCA result
        Usage: obj.plot()
        """

        no = self.no
        Nstd = self.stds.shape[0]
        N = self.time.shape[0]
        num_plot = int(np.ceil(no/3))
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
        k = 0
        for ii in range(0, num_plot):

            if k > (no-1):
                break

            fig, ax = plt.subplots(2, 3)
            
            for k1 in range(0,3):
                k = k1+(ii)*3
                axt = ax[0, k1]
                if k > (no-1):
                    break

                for l in range(0, Nstd):
                    axt.plot(self.time, self.q_pca[0:N, l, k], color=CBcdict[cl[l]])

                axt.set_title('q domain: PD %d' % (k + 1))

                axt = ax[1, k1]
                for l in range(0, Nstd):
                    axt.plot(self.time, self.f_pca[:, l, k], color=CBcdict[cl[l]])

                axt.set_title('f domain: PD %d' % (k + 1))

            fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(self.latent) / sum(self.latent)
        N = self.latent.shape[0]
        idx = np.arange(0, N) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.ylabel("Percentage")
        plt.xlabel("Index")
        plt.show()

        return

class fdahpca:
    """
    This class provides horizontal fPCA using the
    SRVF framework

    Usage:  obj = fdahpca(warp_data)

    :param warp_data: fdawarp class with alignment data
    :param gam_pca: warping functions principal directions
    :param psi_pca: srvf principal directions
    :param latent: latent values
    :param U: eigenvectors
    :param coef: coefficients
    :param vec: shooting vectors
    :param mu: Karcher Mean
    :param tau: principal directions

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  15-Mar-2018
    """

    def __init__(self, fdawarp):
        """
        Construct an instance of the fdavpca class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size==0:
            raise Exception('Please align fdawarp class using srsf_align!')

        self.warp_data = fdawarp

    def calc_fpca(self, no=3, stds = np.arange(-1, 2)):
        """
        This function calculates horizontal functional principal component analysis on aligned data

        :param no: number of components to extract (default = 3)
        :param stds: number of standard deviations along gedoesic to compute (default = -1,0,1)
        :type no: int

        :rtype: fdahpca object of numpy ndarray
        :return q_pca: srsf principal directions
        :return f_pca: functional principal directions
        :return latent: latent values
        :return coef: coefficients
        :return U: eigenvectors

        """
        # Calculate Shooting Vectors
        gam = self.warp_data.gam
        mu, gam_mu, psi, vec = uf.SqrtMean(gam)
        TT = self.warp_data.time.shape[0]

        # TFPCA
        K = np.cov(vec)

        U, s, V = svd(K)
        vm = vec.mean(axis=1)

        gam_pca = np.ndarray(shape=(stds.shape[0], mu.shape[0], no), dtype=float)
        psi_pca = np.ndarray(shape=(stds.shape[0], mu.shape[0], no), dtype=float)
        for j in range(0, no):
            cnt = 0
            for k in stds:
                v = k * np.sqrt(s[j]) * U[:, j]
                vn = norm(v) / np.sqrt(TT)
                if vn < 0.0001:
                    psi_pca[cnt, :, j] = mu
                else:
                    psi_pca[cnt, :, j] = np.cos(vn) * mu + np.sin(vn) * v / vn

                tmp = cumtrapz(psi_pca[cnt, :, j] * psi_pca[cnt, :, j], np.linspace(0,1,TT), initial=0)
                gam_pca[cnt, :, j] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])
                cnt += 1

        N2 = gam.shape[1]
        c = np.zeros((N2,no))
        for k in range(0,no):
            for i in range(0,N2):
                c[i,k] = np.dot(vec[:,i]-vm,U[:,k])

        self.gam_pca = gam_pca
        self.psi_pca = psi_pca
        self.U = U[:,0:no]
        self.coef = c
        self.latent = s[0:no]
        self.gam_mu = gam_mu
        self.psi_mu = mu
        self.vec = vec
        self.no = no
        self.stds = stds

        return

    def plot(self):
        """
        plot plot elastic horizontal fPCA results

        Usage: obj.plot()
        """

        no = self.no
        TT = self.warp_data.time.shape[0]
        num_plot = int(np.ceil(no/3))
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
        k = 0
        for ii in range(0, num_plot):

            if k > (no-1):
                break

            fig, ax = plt.subplots(1, 3)
            
            for k1 in range(0,3):
                k = k1+(ii)*3
                axt = ax[k1]
                if k > (no-1):
                    break

                tmp = self.gam_pca[:, :, k]
                axt.plot(np.linspace(0, 1, TT), tmp.transpose())
                axt.set_title('PD %d' % (k + 1))
                axt.set_aspect('equal')

            fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(self.latent[0:no]) / sum(self.latent[0:no])
        idx = np.arange(0, no) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.ylabel("Percentage")
        plt.xlabel("Index")
        plt.show()

        return

class fdajpca:
    """
    This class provides joint fPCA using the
    SRVF framework

    Usage:  obj = fdajpca(warp_data)

    :param warp_data: fdawarp class with alignment data
    :param q_pca: srvf principal directions
    :param f_pca: f principal directions
    :param latent: latent values
    :param coef: principal coefficients
    :param id: point used for f(0)
    :param mqn: mean srvf
    :param U: eigenvectors
    :param mu_psi: mean psi
    :param mu_g: mean g
    :param C: scaling value
    :param stds: geodesic directions

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  18-Mar-2018
    """

    def __init__(self, fdawarp):
        """
        Construct an instance of the fdavpca class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size==0:
            raise Exception('Please align fdawarp class using srsf_align!')

        self.warp_data = fdawarp

    def calc_fpca(self, no=3, stds = np.arange(-1., 2.), 
                  id=None, parallel=False, cores=-1):
        """
        This function calculates joint functional principal component analysis
        on aligned data

        :param no: number of components to extract (default = 3)
        :param id: point to use for f(0) (default = midpoint)
        :param stds: number of standard deviations along gedoesic to compute (default = -1,0,1)
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :type no: int
        :type id: int
        :type parallel: bool
        :type cores: int

        :rtype: fdajpca object of numpy ndarray
        :return q_pca: srsf principal directions
        :return f_pca: functional principal directions
        :return latent: latent values
        :return coef: coefficients
        :return U: eigenvectors

        """
        fn = self.warp_data.fn
        time = self.warp_data.time
        qn = self.warp_data.qn
        q0 = self.warp_data.q0
        gam = self.warp_data.gam

        M = time.shape[0]
        if id is None:
            mididx = int(np.round(M / 2))
        else:
            mididx = id

        Nstd = stds.shape[0]

        # set up for fPCA in q-space
        mq_new = qn.mean(axis=1)
        m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
        mqn = np.append(mq_new, m_new.mean())
        qn2 = np.vstack((qn, m_new))

        # calculate vector space of warping functions
        mu_psi, gam_mu, psi, vec = uf.SqrtMean(gam, parallel, cores)

        # joint fPCA
        C = fminbound(find_C, 0, 1e4, (qn2, vec, q0,no, mu_psi, parallel, cores))
        qhat, gamhat, a, U, s, mu_g, g, cov = jointfPCAd(qn2, vec, C, no, mu_psi, parallel, cores)

        # geodesic paths
        q_pca = np.ndarray(shape=(M, Nstd, no), dtype=float)
        f_pca = np.ndarray(shape=(M, Nstd, no), dtype=float)

        for k in range(0, no):
            for l in range(0, Nstd):
                qhat = mqn + np.dot(U[0:(M+1),k],stds[l]*np.sqrt(s[k]))
                vechat = np.dot(U[(M+1):,k],(stds[l]*np.sqrt(s[k]))/C)
                psihat = geo.exp_map(mu_psi,vechat)
                gamhat = cumtrapz(psihat*psihat,np.linspace(0,1,M),initial=0)
                gamhat = (gamhat - gamhat.min()) / (gamhat.max() - gamhat.min())
                if (sum(vechat)==0):
                    gamhat = np.linspace(0,1,M)

                fhat = uf.cumtrapzmid(time, qhat[0:M]*np.fabs(qhat[0:M]), np.sign(qhat[M])*(qhat[M]*qhat[M]), mididx)
                f_pca[:,l,k] = uf.warp_f_gamma(np.linspace(0,1,M), fhat, gamhat)
                q_pca[:,l,k] = uf.warp_q_gamma(np.linspace(0,1,M), qhat[0:M], gamhat)

        self.q_pca = q_pca
        self.f_pca = f_pca
        self.latent = s[0:no]
        self.coef = a
        self.U = U[:,0:no]
        self.mu_psi = mu_psi
        self.mu_g = mu_g
        self.id = mididx
        self.C = C
        self.time = time
        self.g = g
        self.cov = cov
        self.no = no
        self.stds = stds

        return

    def plot(self):
        """
        plot plot elastic vertical fPCA result

        Usage: obj.plot()
        """
        no = self.no
        M = self.time.shape[0]
        Nstd = self.stds.shape[0]
        num_plot = int(np.ceil(no/3))
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
        k = 0
        for ii in range(0, num_plot):

            if k > (no-1):
                break

            fig, ax = plt.subplots(2, 3)
            
            for k1 in range(0,3):
                k = k1+(ii)*3
                axt = ax[0, k1]
                if k > (no-1):
                    break

                for l in range(0, Nstd):
                    axt.plot(self.time, self.q_pca[0:M, l, k], color=CBcdict[cl[l]])

                axt.set_title('q domain: PD %d' % (k + 1))

                axt = ax[1, k1]
                for l in range(0, Nstd):
                    axt.plot(self.time, self.f_pca[:, l, k], color=CBcdict[cl[l]])

                axt.set_title('f domain: PD %d' % (k + 1))

            fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(self.latent) / sum(self.latent)
        idx = np.arange(0, self.latent.shape[0]) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.ylabel("Percentage")
        plt.xlabel("Index")
        plt.show()

        return

def jointfPCAd(qn, vec, C, m, mu_psi, parallel, cores):
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
            a[i,j] = np.dot(tmp.T, U[:,j])

    qhat = np.tile(mu_q, (N,1))
    qhat = qhat.T
    qhat = qhat + np.dot(U[0:M,0:m],a.T)

    vechat = np.dot(U[M:,0:m], a.T/C)
    psihat = np.zeros((M-1,N))
    gamhat = np.zeros((M-1,N))
    if parallel:
        out = Parallel(n_jobs=cores)(delayed(jfpca_sub)(mu_psi, vechat[:,n]) for n in range(N))
        gamhat = np.array(out)
        gamhat = gamhat.transpose()
    else:
        for ii in range(0,N):
            psihat[:,ii] = geo.exp_map(mu_psi,vechat[:,ii])
            gam_tmp = cumtrapz(psihat[:,ii]*psihat[:,ii], np.linspace(0,1,M-1), initial=0)
            gamhat[:,ii] = (gam_tmp - gam_tmp.min()) / (gam_tmp.max() - gam_tmp.min())

    U = U[:,0:m]
    s = s[0:m]

    return qhat, gamhat, a, U, s, mu_g, g, K

def jfpca_sub(mu_psi, vechat):
    M = mu_psi.shape[0]
    psihat = geo.exp_map(mu_psi, vechat)
    gam_tmp = cumtrapz(psihat*psihat, np.linspace(0,1,M), initial=0)
    gamhat = (gam_tmp - gam_tmp.min()) / (gam_tmp.max() - gam_tmp.min())

    return gamhat

def find_C(C, qn, vec, q0, m, mu_psi, parallel, cores):
    qhat, gamhat, a, U, s, mu_g, g, K = jointfPCAd(qn, vec, C, m, mu_psi, parallel, cores)
    (M,N) = qn.shape
    time = np.linspace(0,1,M-1)

    d = np.zeros(N)
    if parallel:
        out = Parallel(n_jobs=cores)(delayed(find_C_sub)(time, qhat[0:(M-1),n], gamhat[:,n], q0[:,n]) for n in range(N))
        d = np.array(out)
    else:
        for i in range(0,N):
            tmp = uf.warp_q_gamma(time, qhat[0:(M-1),i], uf.invertGamma(gamhat[:,i]))
            d[i] = trapz((tmp-q0[:,i])*(tmp-q0[:,i]), time)

    out = sum(d*d)/N

    return out

def find_C_sub(time, qhat, gamhat, q0):
    tmp = uf.warp_q_gamma(time, qhat, uf.invertGamma(gamhat))
    d = trapz((tmp-q0)*(tmp-q0), time)

    return d
