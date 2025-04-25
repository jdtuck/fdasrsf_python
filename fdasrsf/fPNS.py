"""
Horizontal Functional Principal Nested Spheres Analysis using SRSF

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf as fs
import fdasrsf.pns as pns
import fdasrsf.utility_functions as uf
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import fdasrsf.plot_style as plot


class fdahpns:
    """
    This class provides horizontal fPNS using the
    SRVF framework

    Usage:  obj = fdapns(warp_data)

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
    Date   :  03-Apr-2025
    """

    def __init__(self, fdawarp):
        """
        Construct an instance of the fdahpns class
        :param fdawarp: fdawarp class
        """
        if fdawarp.fn.size == 0:
            raise Exception("Please align fdawarp class using srsf_align!")

        self.warp_data = fdawarp

    def calc_pns(self, var_exp=0.99):
        """
        This function calculates horizontal functional principal nested
        spheres on aligned data

        :param var_exp: compute no based on value percent variance explained
                        (example: 0.95)

        :rtype: fdapns object of numpy ndarray
        :return gam_pca: srsf principal directions
        :return psi_pca: functional principal directions
        :return latent: latent values
        :return coef: coefficients
        :return U: eigenvectors

        """

        gam = self.warp_data.gam
        d, n = gam.shape
        t = np.linspace(0, 1, d)
        mu, gam_mu, psi, vec = uf.SqrtMean(gam)

        radius = np.mean(np.sqrt((psi**2).sum(axis=0)))
        pnsdat = psi / np.tile(np.sqrt((psi**2).sum(axis=0)), (d, 1))

        resmat, PNS = pns.PNSmainHDLSS(pnsdat)

        # Proportion of variance explained
        varPNS = np.sum(np.abs(resmat) ** 2, axis=1) / n
        cumvarPNS = np.cumsum(varPNS)
        propcumPNS = cumvarPNS / cumvarPNS[-1]
        propPNS = varPNS / cumvarPNS[-1] * 100

        # Projection of PCs
        no = int(np.argwhere(propcumPNS <= var_exp)[-1])+1
        if (no == 1):
            no += 1
        udir = np.eye(resmat.shape[0])
        projPsi = np.zeros((d, n, no))
        projGam = np.zeros((d, n, no))
        for PCnum in range(no):
            PCvec = pns.PNSe2s(np.outer(udir[:, PCnum], resmat[PCnum, :]), PNS)
            projPsi[:, :, PCnum] = PCvec * radius
            gamt = cumulative_trapezoid(projPsi[:, :, PCnum] ** 2, t, axis=0, 
                                        initial=0)
            for j in range(n):
                gamt[:, j] = (gamt[:, j] - gamt[:, j].min()) / (
                    gamt[:, j].max() - gamt[:, j].min()
                )
            projGam[:, :, PCnum] = gamt

        self.gam_pns = projGam
        self.psi_pns = projPsi
        self.cumvar = propcumPNS
        self.no = no
        self.psi = psi
        self.PNS = PNS
        self.coef = resmat
        self.radius = radius

        return

    def project(self, f):
        """
        project new data onto fPNS basis

        Usage: obj.project(f)

        :param f: numpy array (MxN) of N functions on M time

        """

        q1 = fs.f_to_srsf(f, self.time)
        M = self.time.shape[0]
        n = q1.shape[1]
        mq = self.warp_data.mqn
        gam = np.zeros((M, n))
        for ii in range(0, n):
            gam[:, ii] = fs.optimum_reparam(mq, self.time, q1[:, ii])

        psi = np.zeros((M, n))
        time = np.linspace(0, 1, M)
        binsize = np.mean(np.diff(time))
        for i in range(0, n):
            psi[:, i] = np.sqrt(np.gradient(gam[:, i], binsize))

        pnsdat = psi / np.tile(np.sqrt((psi**2).sum(axis=0)), (n, 1))

        resmat = pns.PNSs2e(pnsdat, self.PNS)

        self.new_coef = resmat

        return
    
    def plot(self):
        """
        plot plot elastic horizontal fPNS results

        Usage: obj.plot()
        """

        no = self.no
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

                axt.plot(np.linspace(0, 1, TT), np.squeeze(self.gam_pns[:, :, k]))
                plt.style.use("seaborn-v0_8-colorblind")
                axt.set_title("PD %d" % (k + 1))
                axt.set_aspect("equal")

            fig.set_tight_layout(True)

        cumm_coef = 100 * self.cumvar
        idx = np.arange(0, no) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.ylabel("Percentage")
        plt.xlabel("Index")
        plt.show()

        return


def project_pns_gam(resmat, PNS, radius, time):
   
    n = resmat.shape[1]
    d = time.shape[0]
    udir = np.eye(resmat.shape[0])
    PCvec = pns.PNSe2s(udir@resmat, PNS) * radius
    gam_hat = np.zeros((d, n))
    for i in range(n):
        gamt = cumulative_trapezoid(PCvec[:, i]**2, time, initial=0)
        gam_hat[:, i] = (gamt - gamt.min()) / (gamt.max() - gamt.min())
   
    return gam_hat
