"""
Vertical and Horizontal Functional Principal Component Analysis using SRSF

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
import fdasrsf as fs
import fdasrsf.utility_functions as uf
import fdasrsf.geometry as geo
from scipy.linalg import norm, svd
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import fminbound
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import fdasrsf.plot_style as plot


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
    :param new_coef: principal coefficients of new data

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  15-Mar-2018
    """

    def __init__(self, fdawarp):
        """
        Construct an instance of the fdavpca class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size == 0:
            raise Exception("Please align fdawarp class using srsf_align!")

        self.warp_data = fdawarp

    def calc_fpca(self, no=3, var_exp=None, id=None, stds=np.arange(-1, 2)):
        """
        This function calculates vertical functional principal component
        analysis on aligned data

        :param no: number of components to extract (default = 3)
        :param var_exp: compute no based on value percent variance explained
                        (example: 0.95)
        :param id: point to use for f(0) (default = midpoint)
        :param stds: number of standard deviations along geodesic to compute
                     (default = -1,0,1)
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

        if 0 in stds:
            stds = stds
        else:
            raise Exception("stds needs to contain 0")

        M = time.shape[0]
        if var_exp is not None:
            if var_exp > 1:
                raise Exception("var_exp is greater than 1")
            no = M

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
        self.mqn2 = mqn
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
                f_pca[:, l, k] = uf.cumtrapzmid(
                    time,
                    q_pca[0:N, l, k] * np.abs(q_pca[0:N, l, k]),
                    np.sign(q_pca[N, l, k]) * (q_pca[N, l, k] ** 2),
                    mididx,
                )
            fbar = fn.mean(axis=1)
            fsbar = f_pca[:, :, k].mean(axis=1)
            err = np.transpose(np.tile(fbar - fsbar, (Nstd, 1)))
            f_pca[:, :, k] += err

        N2 = qn.shape[1]
        c = np.zeros((N2, no))
        for k in range(0, no):
            for l in range(0, N2):
                c[l, k] = sum((np.append(qn[:, l], m_new[l]) - mqn) * U[:, k])

        if var_exp is not None:
            cumm_coef = np.cumsum(s) / sum(s)
            no = int(np.argwhere(cumm_coef <= var_exp)[-1])

        self.q_pca = q_pca
        self.f_pca = f_pca
        self.latent = s[0:no]
        self.coef = c[:, 0:no]
        self.U = U[:, 0:no]
        self.id = mididx
        self.mqn = mqn
        self.time = time
        self.stds = stds
        self.no = no

        return

    def project(self, f):
        """
        project new data onto fPCA basis

        Usage: obj.project(f)

        :param f: numpy array (MxN) of N functions on M time

        """

        q1 = fs.f_to_srsf(f, self.time)
        M = self.time.shape[0]
        n = q1.shape[1]
        mq = self.warp_data.mqn
        fn = np.zeros((M, n))
        qn = np.zeros((M, n))
        gam = np.zeros((M, n))
        for ii in range(0, n):
            gam[:, ii] = fs.optimum_reparam(mq, self.time, q1[:, ii])
            fn[:, ii] = fs.warp_f_gamma(self.time, f[:, ii], gam[:, ii])
            qn[:, ii] = fs.f_to_srsf(fn[:, ii], self.time)

        U = self.U
        no = U.shape[1]

        m_new = np.sign(fn[self.id, :]) * np.sqrt(np.abs(fn[self.id, :]))
        qn1 = np.vstack((qn, m_new))

        a = np.zeros((n, no))
        for i in range(0, n):
            for j in range(0, no):
                a[i, j] = sum((qn1[:, i] - self.mqn2) * U[:, j])

        self.new_coef = a

        return

    def plot(self):
        """
        plot plot elastic vertical fPCA result
        Usage: obj.plot()
        """

        no = self.no
        Nstd = self.stds.shape[0]
        N = self.time.shape[0]
        num_plot = int(np.ceil(no / 3))
        colors = [
            "#66C2A5",
            "#FC8D62",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
            "#B3B3B3",
        ]
        k = 0
        for ii in range(0, num_plot):
            if k > (no - 1):
                break

            fig, ax = plt.subplots(2, 3)

            for k1 in range(0, 3):
                k = k1 + (ii) * 3
                axt = ax[0, k1]
                if k > (no - 1):
                    break

                for l in range(0, Nstd):
                    axt.plot(self.time, self.q_pca[0:N, l, k], color=colors[l])

                l0 = np.where(self.stds == 0)[0]
                axt.plot(self.time, self.q_pca[0:N, l0, k], 'k')
                axt.set_title("q domain: PD %d" % (k + 1))
                plt.style.use("seaborn-v0_8-colorblind")

                axt = ax[1, k1]
                for l in range(0, Nstd):
                    axt.plot(self.time, self.f_pca[:, l, k], color=colors[l])

                l0 = np.where(self.stds == 0)[0]
                axt.plot(self.time, self.f_pca[:, l0, k], 'k')
                axt.set_title("f domain: PD %d" % (k + 1))
                plt.style.use("seaborn-v0_8-colorblind")

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
        Construct an instance of the fdahpca class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size == 0:
            raise Exception("Please align fdawarp class using srsf_align!")

        self.warp_data = fdawarp

    def calc_fpca(self, no=3, var_exp=None, stds=np.arange(-1, 2)):
        """
        This function calculates horizontal functional principal component
        analysis on aligned data

        :param no: number of components to extract (default = 3)
        :param var_exp: compute no based on value percent variance explained
                        (example: 0.95)
        :param stds: number of standard deviations along geodesic to compute
                     (default = -1,0,1)
        :type no: int

        :rtype: fdahpca object of numpy ndarray
        :return gam_pca: srsf principal directions
        :return psi_pca: functional principal directions
        :return latent: latent values
        :return coef: coefficients
        :return U: eigenvectors

        """
        # Calculate Shooting Vectors
        gam = self.warp_data.gam
        mu, gam_mu, psi, vec = uf.SqrtMean(gam)
        TT = self.warp_data.time.shape[0]

        if 0 in stds:
            stds = stds
        else:
            raise Exception("stds needs to contain 0")

        if var_exp is not None:
            if var_exp > 1:
                raise Exception("var_exp is greater than 1")
            no = TT

        # TFPCA
        K = np.cov(vec)

        U, s, V = svd(K)
        vm = vec.mean(axis=1)
        self.vm = vm
        self.mu_psi = mu

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

                tmp = cumulative_trapezoid(
                    psi_pca[cnt, :, j] * psi_pca[cnt, :, j],
                    np.linspace(0, 1, TT),
                    initial=0,
                )
                gam_pca[cnt, :, j] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])
                cnt += 1

        N2 = gam.shape[1]
        c = np.zeros((N2, no))
        for k in range(0, no):
            for i in range(0, N2):
                c[i, k] = np.dot(vec[:, i] - vm, U[:, k])

        if var_exp is not None:
            cumm_coef = np.cumsum(s) / sum(s)
            no = int(np.argwhere(cumm_coef <= var_exp)[-1])

        self.gam_pca = gam_pca
        self.psi_pca = psi_pca
        self.U = U[:, 0:no]
        self.coef = c[:, 0:no]
        self.latent = s[0:no]
        self.gam_mu = gam_mu
        self.psi_mu = mu
        self.vec = vec
        self.no = no
        self.stds = stds

        return

    def project(self, f):
        """
        project new data onto fPCA basis

        Usage: obj.project(f)

        :param f: numpy array (MxN) of N functions on M time

        """

        q1 = fs.f_to_srsf(f, self.warp_data.time)
        M = self.warp_data.time.shape[0]
        n = q1.shape[1]
        mq = self.warp_data.mqn
        gam = np.zeros((M, n))
        for ii in range(0, n):
            gam[:, ii] = fs.optimum_reparam(mq, self.warp_data.time, q1[:, ii])

        U = self.U
        no = U.shape[1]

        mu_psi = self.mu_psi
        vec = np.zeros((M, n))
        psi = np.zeros((M, n))
        time = np.linspace(0, 1, M)
        binsize = np.mean(np.diff(time))
        for i in range(0, n):
            psi[:, i] = np.sqrt(np.gradient(gam[:, i], binsize))
            out, theta = fs.inv_exp_map(mu_psi, psi[:, i])
            vec[:, i] = out

        a = np.zeros((n, no))
        for i in range(0, n):
            for j in range(0, no):
                a[i, j] = np.dot(vec[:, i] - self.vm, U[:, j])

        self.new_coef = a

        return

    def plot(self):
        """
        plot plot elastic horizontal fPCA results

        Usage: obj.plot()
        """

        no = self.no
        Nstd = self.stds.shape[0]
        TT = self.warp_data.time.shape[0]
        num_plot = int(np.ceil(no / 3))
        colors = [
            "#66C2A5",
            "#FC8D62",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
            "#B3B3B3",
        ]

        k = 0
        for ii in range(0, num_plot):
            if k > (no - 1):
                break

            fig, ax = plt.subplots(1, 3)

            for k1 in range(0, 3):
                k = k1 + (ii) * 3
                axt = ax[k1]
                if k > (no - 1):
                    break

                for ll in range(0, Nstd):
                    axt.plot(np.linspace(0, 1, TT), np.squeeze(self.gam_pca[ll, :, k]), 
                             color=colors[ll])
                l0 = np.where(self.stds == 0)[0]
                axt.plot(np.linspace(0, 1, TT), np.squeeze(self.gam_pca[l0, :, k]), 'k')
                plt.style.use("seaborn-v0_8-colorblind")
                axt.set_title("PD %d" % (k + 1))
                axt.set_aspect("equal")

            fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(self.latent[0:no]) / sum(self.latent[0:no])
        idx = np.arange(0, no) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.ylabel("Percentage")
        plt.xlabel("Index")
        plt.show()

        return
    

class fdahpcah:
    """
    This class provides horizontal fPCA using the
    SRVF framework using the log derivative transform

    Usage:  obj = fdahpcah(warp_data)

    :param warp_data: fdawarp class with alignment data
    :param gam_pca: warping functions principal directions
    :param h_pca: srvf principal directions
    :param latent: latent values
    :param U: eigenvectors
    :param coef: coefficients
    :param vec: shooting vectors
    :param mu: Karcher Mean
    :param tau: principal directions

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  25-Apr-2025
    """

    def __init__(self, fdawarp):
        """
        Construct an instance of the fdahpca class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size == 0:
            raise Exception("Please align fdawarp class using srsf_align!")

        self.warp_data = fdawarp

    def calc_fpca(self, no=3, var_exp=None, stds=np.arange(-1, 2)):
        """
        This function calculates horizontal functional principal component
        analysis on aligned data

        :param no: number of components to extract (default = 3)
        :param var_exp: compute no based on value percent variance explained
                        (example: 0.95)
        :param stds: number of standard deviations along geodesic to compute
                     (default = -1,0,1)
        :type no: int

        :rtype: fdahpca object of numpy ndarray
        :return gam_pca: srsf principal directions
        :return h_pca: functional principal directions
        :return latent: latent values
        :return coef: coefficients
        :return U: eigenvectors

        """
        # Calculate Shooting Vectors
        gam = self.warp_data.gam
        h = geo.gam_to_h(gam)
        mu = h.mean(axis=1)
        TT = self.warp_data.time.shape[0]

        if 0 in stds:
            stds = stds
        else:
            raise Exception("stds needs to contain 0")

        if var_exp is not None:
            if var_exp > 1:
                raise Exception("var_exp is greater than 1")
            no = TT

        # TFPCA
        K = np.cov(h)

        U, s, V = svd(K)
        self.mu_h = mu

        gam_pca = np.ndarray(shape=(stds.shape[0], mu.shape[0], no), dtype=float)
        h_pca = np.ndarray(shape=(stds.shape[0], mu.shape[0], no), dtype=float)
        for j in range(0, no):
            cnt = 0
            for k in stds:
                h_pca[cnt, :, j] = mu + k * np.sqrt(s[j]) * U[:, j]

                tmp = geo.h_to_gam(h_pca[cnt, :, j])
                gam_pca[cnt, :, j] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])
                cnt += 1

        N2 = gam.shape[1]
        c = np.zeros((N2, no))
        for k in range(0, no):
            for i in range(0, N2):
                c[i, k] = np.dot(h[:, i] - mu, U[:, k])

        if var_exp is not None:
            cumm_coef = np.cumsum(s) / sum(s)
            no = int(np.argwhere(cumm_coef <= var_exp)[-1])

        self.gam_pca = gam_pca
        self.h_pca = h_pca
        self.U = U[:, 0:no]
        self.coef = c[:, 0:no]
        self.latent = s[0:no]
        self.gam_mu = geo.h_to_gam(mu)
        self.h_mu = mu
        self.h = h
        self.no = no
        self.stds = stds

        return

    def project(self, f):
        """
        project new data onto fPCA basis

        Usage: obj.project(f)

        :param f: numpy array (MxN) of N functions on M time

        """

        q1 = fs.f_to_srsf(f, self.warp_data.time)
        M = self.warp_data.time.shape[0]
        n = q1.shape[1]
        mq = self.warp_data.mqn
        gam = np.zeros((M, n))
        for ii in range(0, n):
            gam[:, ii] = fs.optimum_reparam(mq, self.warp_data.time, q1[:, ii])

        U = self.U
        no = U.shape[1]

        mu = self.h_mu
        h = geo.gam_to_h(gam)

        a = np.zeros((n, no))
        for i in range(0, n):
            for j in range(0, no):
                a[i, j] = np.dot(h[:, i] - mu, U[:, j])

        self.new_coef = a

        return

    def plot(self):
        """
        plot plot elastic horizontal fPCA results

        Usage: obj.plot()
        """

        no = self.no
        Nstd = self.stds.shape[0]
        TT = self.warp_data.time.shape[0]
        num_plot = int(np.ceil(no / 3))
        colors = [
            "#66C2A5",
            "#FC8D62",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
            "#B3B3B3",
        ]

        k = 0
        for ii in range(0, num_plot):
            if k > (no - 1):
                break

            fig, ax = plt.subplots(1, 3)

            for k1 in range(0, 3):
                k = k1 + (ii) * 3
                axt = ax[k1]
                if k > (no - 1):
                    break

                for ll in range(0, Nstd):
                    axt.plot(np.linspace(0, 1, TT), np.squeeze(self.gam_pca[ll, :, k]), 
                             color=colors[ll])
                l0 = np.where(self.stds == 0)[0]
                axt.plot(np.linspace(0, 1, TT), np.squeeze(self.gam_pca[l0, :, k]), 'k')
                plt.style.use("seaborn-v0_8-colorblind")
                axt.set_title("PD %d" % (k + 1))
                axt.set_aspect("equal")

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
        Construct an instance of the fdajpca class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size == 0:
            raise Exception("Please align fdawarp class using srsf_align!")

        self.warp_data = fdawarp
        self.M = fdawarp.time.shape[0]

    def calc_fpca(
        self,
        no=3,
        var_exp=None,
        stds=np.arange(-1.0, 2.0),
        id=None,
        parallel=False,
        cores=-1,
    ):
        """
        This function calculates joint functional principal component analysis
        on aligned data

        :param no: number of components to extract (default = 3)
        :param var_exp: compute no based on value percent variance explained
                        (example: 0.95)
        :param id: point to use for f(0) (default = midpoint)
        :param stds: number of standard deviations along gedoesic to compute
                     (default = -1,0,1)
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

        if 0 in stds:
            stds = stds
        else:
            raise Exception("stds needs to contain 0")

        M = time.shape[0]
        if var_exp is not None:
            if var_exp > 1:
                raise Exception("var_exp is greater than 1")
            no = M

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
        C = fminbound(find_C, 0, 1e4, (qn2, vec, q0, no, mu_psi, parallel, cores))
        qhat, gamhat, a, U, s, mu_g, g, cov = jointfPCAd(
            qn2, vec, C, no, mu_psi, parallel, cores
        )

        # geodesic paths
        q_pca = np.ndarray(shape=(M, Nstd, no), dtype=float)
        f_pca = np.ndarray(shape=(M, Nstd, no), dtype=float)

        for k in range(0, no):
            for l in range(0, Nstd):
                qhat = mqn + np.dot(U[0: (M + 1), k], stds[l] * np.sqrt(s[k]))
                vechat = np.dot(U[(M + 1):, k], (stds[l] * np.sqrt(s[k])) / C)
                psihat = geo.exp_map(mu_psi, vechat)
                gamhat = cumulative_trapezoid(psihat * psihat,
                                              np.linspace(0, 1, M),
                                              initial=0)
                gamhat = (gamhat - gamhat.min()) / (gamhat.max() - gamhat.min())
                if sum(vechat) == 0:
                    gamhat = np.linspace(0, 1, M)

                fhat = uf.cumtrapzmid(
                    time,
                    qhat[0:M] * np.fabs(qhat[0:M]),
                    np.sign(qhat[M]) * (qhat[M] * qhat[M]),
                    mididx,
                )
                f_pca[:, l, k] = uf.warp_f_gamma(np.linspace(0, 1, M), 
                                                 fhat, gamhat)
                q_pca[:, l, k] = uf.warp_q_gamma(
                    np.linspace(0, 1, M), qhat[0:M], gamhat
                )

        if var_exp is not None:
            cumm_coef = np.cumsum(s) / s.sum()
            no = int(np.argwhere(cumm_coef >= var_exp)[0][0])

        self.q_pca = q_pca
        self.f_pca = f_pca
        self.latent = s[0:no]
        self.coef = a[:, 0:no]
        self.U = U[:, 0:no]
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

    def project(self, f):
        """
        project new data onto fPCA basis

        Usage: obj.project(f)

        :param f: numpy array (MxN) of N functions on M time

        """

        q1 = fs.f_to_srsf(f, self.time)
        M = self.time.shape[0]
        n = q1.shape[1]
        mq = self.warp_data.mqn
        fn = np.zeros((M, n))
        qn = np.zeros((M, n))
        gam = np.zeros((M, n))
        for ii in range(0, n):
            gam[:, ii] = fs.optimum_reparam(mq, self.time, q1[:, ii])
            fn[:, ii] = fs.warp_f_gamma(self.time, f[:, ii], gam[:, ii])
            qn[:, ii] = fs.f_to_srsf(fn[:, ii], self.time)

        U = self.U
        no = U.shape[1]

        m_new = np.sign(fn[self.id, :]) * np.sqrt(np.abs(fn[self.id, :]))
        qn1 = np.vstack((qn, m_new))
        C = self.C
        TT = self.time.shape[0]
        mu_g = self.mu_g
        mu_psi = self.mu_psi
        vec = np.zeros((M, n))
        psi = np.zeros((TT, n))
        time = np.linspace(0, 1, TT)
        binsize = np.mean(np.diff(time))
        for i in range(0, n):
            psi[:, i] = np.sqrt(np.gradient(gam[:, i], binsize))
            out, theta = fs.inv_exp_map(mu_psi, psi[:, i])
            vec[:, i] = out

        g = np.vstack((qn1, C * vec))
        a = np.zeros((n, no))
        for i in range(0, n):
            for j in range(0, no):
                tmp = g[:, i] - mu_g
                a[i, j] = np.dot(tmp.T, U[:, j])

        self.new_g = g
        self.new_coef = a

        return

    def plot(self):
        """
        plot plot elastic joint fPCA result

        Usage: obj.plot()
        """
        no = self.no
        M = self.time.shape[0]
        Nstd = self.stds.shape[0]
        num_plot = int(np.ceil(no / 3))
        colors = [
            "#66C2A5",
            "#FC8D62",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
            "#B3B3B3",
        ]
        k = 0
        for ii in range(0, num_plot):
            if k > (no - 1):
                break

            fig, ax = plt.subplots(2, 3)

            for k1 in range(0, 3):
                k = k1 + (ii) * 3
                axt = ax[0, k1]
                if k > (no - 1):
                    break

                for l in range(0, Nstd):
                    axt.plot(self.time, self.q_pca[0:M, l, k], color=colors[l])

                l0 = np.where(self.stds == 0)[0]
                axt.plot(self.time, self.q_pca[0:M, l0, k], 'k')
                axt.set_title("q domain: PD %d" % (k + 1))
                plt.style.use("seaborn-v0_8-colorblind")

                axt = ax[1, k1]
                for l in range(0, Nstd):
                    axt.plot(self.time, self.f_pca[:, l, k], color=colors[l])

                l0 = np.where(self.stds == 0)[0]
                axt.plot(self.time, self.f_pca[:, l0, k], 'k')
                axt.set_title("f domain: PD %d" % (k + 1))
                plt.style.use("seaborn-v0_8-colorblind")

            fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(self.latent) / sum(self.latent)
        idx = np.arange(0, self.latent.shape[0]) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.ylabel("Percentage")
        plt.xlabel("Index")
        plt.show()

        return


class fdajpcah:
    """
    This class provides joint fPCA using the
    SRVF framework using MFPCA

    Usage:  obj = fdajpcah(warp_data)

    :param warp_data: fdawarp class with alignment data
    :param q_pca: srvf principal directions
    :param f_pca: f principal directions
    :param latent: latent values
    :param coef: principal coefficients
    :param id: point used for f(0)
    :param mqn: mean srvf
    :param U_q: eigenvectors for q
    :param U_h: eigenvectors for gam
    :param C: scaling value
    :param stds: geodesic directions

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  06-April-2024
    """

    def __init__(self, fdawarp):
        """
        Construct an instance of the fdajpcah class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size == 0:
            raise Exception("Please align fdawarp class using srsf_align!")

        self.warp_data = fdawarp
        self.M = fdawarp.time.shape[0]

    def calc_fpca(
        self,
        var_exp=0.99,
        stds=np.arange(-1.0, 2.0),
        id=None,
        parallel=False,
        cores=-1,
        srsf=True,
    ):
        """
        This function calculates joint functional principal component analysis
        on aligned data

        :param var_exp: compute no based on value percent variance explained
                        (default: None)
        :param id: point to use for f(0) (default = midpoint)
        :param stds: number of standard deviations along gedoesic to compute
                     (default = -1,0,1)
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :type id: int
        :type parallel: bool
        :type cores: int

        :rtype: fdajpcah object of numpy ndarray
        :return q_pca: srsf principal directions
        :return f_pca: functional principal directions
        :return latent: latent values
        :return coef: coefficients
        :param U_q: eigenvectors for q
        :param U_h: eigenvectors for gam

        """
        fn = self.warp_data.fn
        time = self.warp_data.time
        qn = self.warp_data.qn
        gam = self.warp_data.gam

        if 0 in stds:
            stds = stds
        else:
            raise Exception("stds needs to contain 0")

        M = time.shape[0]
        if var_exp is not None:
            if var_exp > 1:
                raise Exception("var_exp is greater than 1")
            no = M

        if id is None:
            mididx = int(np.round(M / 2))
        else:
            mididx = id

        Nstd = stds.shape[0]

        # set up for fPCA in q-space
        if srsf:
            mq_new = qn.mean(axis=1)
            m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
            mqn = np.append(mq_new, m_new.mean())
            qn2 = np.vstack((qn, m_new))
            q0 = self.warp_data.q0
        else:
            mqn = fn.mean(axis=1)
            q0 = self.warp_data.f
            qn2 = fn.copy()

        # calculate vector space of warping functions
        h = geo.gam_to_h(gam)

        # joint fPCA
        C = fminbound(find_C_h, 0, 200, (qn2, h, q0, 0.99, parallel, cores, srsf))
        qhat, gamhat, cz, Psi_q, Psi_h, sz, U, Uh, Uz = jointfPCAhd(qn2, h, C, var_exp)

        hc = C * h
        mh = np.mean(hc, axis=1)

        # geodesic paths
        no = cz.shape[1]
        q_pca = np.ndarray(shape=(M, Nstd, no), dtype=float)
        f_pca = np.ndarray(shape=(M, Nstd, no), dtype=float)

        for k in range(0, no):
            for l in range(0, Nstd):
                qhat = mqn + np.dot(Psi_q[:, k], stds[l] * np.sqrt(sz[k]))
                hhat = np.dot(Psi_h[:, k], (stds[l] * np.sqrt(sz[k])) / C)
                gamhat = fs.geometry.h_to_gam(hhat)

                if srsf:
                    fhat = fs.utility_functions.cumtrapzmid(
                        time,
                        qhat[0:M] * np.fabs(qhat[0:M]),
                        np.sign(qhat[M]) * (qhat[M] * qhat[M]),
                        mididx,
                    )
                    f_pca[:, l, k] = fs.utility_functions.warp_f_gamma(
                        np.linspace(0, 1, M), fhat, gamhat
                    )
                    q_pca[:, l, k] = fs.utility_functions.warp_q_gamma(
                        np.linspace(0, 1, M), qhat[0:M], gamhat
                    )
                else:
                    f_pca[:, l, k] = fs.utility_functions.warp_f_gamma(
                        np.linspace(0, 1, M), qhat, gamhat
                    )
                    q_pca[:, l, k] = fs.f_to_srsf(f_pca[:, l, k],
                                                  np.linspace(0, 1, M))

        self.q_pca = q_pca
        self.f_pca = f_pca
        self.eigs = sz
        self.latent = sz[0:no]
        self.coef = cz[:, 0:no]
        self.U_q = Psi_q
        self.U_h = Psi_h
        self.id = mididx
        self.C = C
        self.h = h
        self.qn1 = qn2
        self.time = time
        self.no = no
        self.stds = stds
        self.mqn = mqn
        self.U = U
        self.U1 = Uh
        self.Uz = Uz
        self.mh = mh
        self.srsf = srsf

        return

    def project(self, f):
        """
        project new data onto fPCA basis

        Usage: obj.project(f)

        :param f: numpy array (MxN) of N functions on M time

        """

        q1 = fs.f_to_srsf(f, self.time)
        M = self.time.shape[0]
        n = q1.shape[1]
        mq = self.warp_data.mqn
        fn = np.zeros((M, n))
        qn = np.zeros((M, n))
        gam = np.zeros((M, n))
        for ii in range(0, n):
            gam[:, ii] = fs.optimum_reparam(mq, self.time, q1[:, ii])
            fn[:, ii] = fs.warp_f_gamma(self.time, f[:, ii], gam[:, ii])
            qn[:, ii] = fs.f_to_srsf(fn[:, ii], self.time)

        if self.srsf:
            m_new = np.sign(fn[self.id, :]) * np.sqrt(np.abs(fn[self.id, :]))
            qn1 = np.vstack((qn, m_new))
        else:
            qn1 = fn.copy()
        C = self.C
        
        h = geo.gam_to_h(gam)

        c = (qn1 - self.mqn[:, np.newaxis]).T @ self.U
        ch = (C*h - self.mh[:, np.newaxis]).T @ self.U1

        Xi = np.hstack((c, ch))

        cz = Xi @ self.Uz

        self.new_coef = cz
        self.new_qn1 = qn1
        self.new_h = h

        return

    def plot(self):
        """
        plot plot elastic joint fPCA result

        Usage: obj.plot()
        """
        no = self.no
        M = self.time.shape[0]
        Nstd = self.stds.shape[0]
        num_plot = int(np.ceil(no / 3))
        colors = [
            "#66C2A5",
            "#FC8D62",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
            "#B3B3B3",
        ]
        k = 0
        for ii in range(0, num_plot):
            if k > (no - 1):
                break

            fig, ax = plt.subplots(2, 3)

            for k1 in range(0, 3):
                k = k1 + (ii) * 3
                axt = ax[0, k1]
                if k > (no - 1):
                    break

                for l in range(0, Nstd):
                    axt.plot(self.time, self.q_pca[0:M, l, k], color=colors[l])

                l0 = np.where(self.stds == 0)[0]
                axt.plot(self.time, self.q_pca[0:M, l0, k], 'k')
                axt.set_title("q domain: PD %d" % (k + 1))
                plt.style.use("seaborn-v0_8-colorblind")

                axt = ax[1, k1]
                for l in range(0, Nstd):
                    axt.plot(self.time, self.f_pca[:, l, k], color=colors[l])

                l0 = np.where(self.stds == 0)[0]
                axt.plot(self.time, self.f_pca[:, l0, k], 'k')
                axt.set_title("f domain: PD %d" % (k + 1))
                plt.style.use("seaborn-v0_8-colorblind")

            fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(self.latent) / sum(self.latent)
        idx = np.arange(0, self.latent.shape[0]) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.ylabel("Percentage")
        plt.xlabel("Index")
        plt.show()

        return


def jointfPCAd(qn, vec, C, m, mu_psi, parallel, cores):
    (M, N) = qn.shape
    g = np.vstack((qn, C * vec))

    mu_q = qn.mean(axis=1)
    mu_g = g.mean(axis=1)

    K = np.cov(g)
    U, s, V = svd(K)

    a = np.zeros((N, m))
    for i in range(0, N):
        for j in range(0, m):
            tmp = g[:, i] - mu_g
            a[i, j] = np.dot(tmp.T, U[:, j])

    qhat = np.tile(mu_q, (N, 1))
    qhat = qhat.T
    qhat = qhat + np.dot(U[0:M, 0:m], a.T)

    vechat = np.dot(U[M:, 0:m], a.T / C)
    psihat = np.zeros((M - 1, N))
    gamhat = np.zeros((M - 1, N))
    if parallel:
        out = Parallel(n_jobs=cores)(
            delayed(jfpca_sub)(mu_psi, vechat[:, n]) for n in range(N)
        )
        gamhat = np.array(out)
        gamhat = gamhat.transpose()
    else:
        for ii in range(0, N):
            psihat[:, ii] = geo.exp_map(mu_psi, vechat[:, ii])
            gam_tmp = cumulative_trapezoid(
                psihat[:, ii] * psihat[:, ii], np.linspace(0, 1, M - 1), 
                initial=0)
            gamhat[:, ii] = (gam_tmp - gam_tmp.min()) / (gam_tmp.max() - gam_tmp.min())

    U = U[:, 0:m]
    s = s[0:m]

    return qhat, gamhat, a, U, s, mu_g, g, K


def jointfPCAhd(qn, h, C, var_exp=None):
    (M, N) = qn.shape

    # Run Univariate fPCA
    # q space
    K = np.cov(qn)

    mqn = qn.mean(axis=1)

    U, s, V = svd(K)
    U, V = uf.svd_flip(U, V)

    cumm_coef = np.cumsum(s) / s.sum()
    no_q = int(np.argwhere(cumm_coef >= var_exp)[0][0])

    c = (qn - mqn[:, np.newaxis]).T @ U
    c = c[:, 0:no_q]
    U = U[:, 0:no_q]

    # h space
    hc = C * h
    mh = np.mean(hc, axis=1)
    Kh = np.cov(hc)

    Uh, sh, Vh = svd(Kh)
    Uh, Vh = uf.svd_flip(Uh, Vh)

    cumm_coef = np.cumsum(sh) / sh.sum()
    no_h = int(np.argwhere(cumm_coef >= var_exp)[0][0]) + 1

    ch = (hc - mh[:, np.newaxis]).T @ Uh
    ch = ch[:, 0:no_h]
    Uh = Uh[:, 0:no_h]

    # Run Multivariate fPCA
    Xi = np.hstack((c, ch))
    Z = 1 / (Xi.shape[0] - 1) * Xi.T @ Xi

    Uz, sz, Vz = svd(Z)
    Uz, Vz = uf.svd_flip(Uz, Vz)

    cz = Xi @ Uz

    Psi_q = U @ Uz[0:no_q, :]
    Psi_h = Uh @ Uz[no_q:, :]

    hhat = Psi_h @ (cz).T
    gamhat = fs.geometry.h_to_gam(hhat/C)

    qhat = Psi_q @ cz.T + mqn[:, np.newaxis]

    return qhat, gamhat, cz, Psi_q, Psi_h, sz, U, Uh, Uz


def find_C_h(C, qn, h, q0, var_exp, parallel, cores, srsf):
    qhat, gamhat, cz, Psi_q, Psi_h, sz, U, Uh, Uz = jointfPCAhd(qn, h, C, var_exp)
    (M, N) = qn.shape
    if srsf:
        time = np.linspace(0, 1, M - 1)
    else:
        time = np.linspace(0, 1, M)

    d = np.zeros(N)
    if parallel:
        out = Parallel(n_jobs=cores)(
            delayed(find_C_sub)(time, qhat[:, n], gamhat[:, n], q0[:, n], srsf)
            for n in range(N)
        )
        d = np.array(out)
    else:
        for i in range(0, N):
            if srsf:
                tmp = uf.warp_q_gamma(time, qhat[0: (M - 1), i],
                                      uf.invertGamma(gamhat[:, i]))
            else:
                tmp = uf.warp_f_gamma(time, qhat[:, i],
                                      uf.invertGamma(gamhat[:, i]))
            d[i] = trapezoid((tmp - q0[:, i]) * (tmp - q0[:, i]), time)

    dout = d.mean()

    return dout


def jfpca_sub(mu_psi, vechat):
    M = mu_psi.shape[0]
    psihat = geo.exp_map(mu_psi, vechat)
    gam_tmp = cumulative_trapezoid(psihat * psihat, np.linspace(0, 1, M), 
                                   initial=0)
    gamhat = (gam_tmp - gam_tmp.min()) / (gam_tmp.max() - gam_tmp.min())

    return gamhat


def find_C(C, qn, vec, q0, m, mu_psi, parallel, cores):
    qhat, gamhat, a, U, s, mu_g, g, K = jointfPCAd(
        qn, vec, C, m, mu_psi, parallel, cores
    )
    (M, N) = qn.shape
    time = np.linspace(0, 1, M - 1)

    d = np.zeros(N)
    if parallel:
        out = Parallel(n_jobs=cores)(
            delayed(find_C_sub)(time, qhat[:, n], gamhat[:, n], 
                                q0[:, n])
            for n in range(N)
        )
        d = np.array(out)
    else:
        for i in range(0, N):
            tmp = uf.warp_q_gamma(
                time, qhat[0: (M - 1), i], uf.invertGamma(gamhat[:, i])
            )
            d[i] = trapezoid((tmp - q0[:, i]) * (tmp - q0[:, i]), time)

    out = d.mean()

    return out


def find_C_sub(time, qhat, gamhat, q0, srsf=True):
    if srsf:
        M = qhat.shape[0]
        tmp = uf.warp_q_gamma(time, qhat[0: (M - 1)], uf.invertGamma(gamhat))
    else:
        tmp = uf.warp_f_gamma(time, qhat, uf.invertGamma(gamhat))
    d = trapezoid((tmp - q0) * (tmp - q0), time)

    return d
