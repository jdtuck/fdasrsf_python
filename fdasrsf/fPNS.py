"""
Horizontal Functional Principal Nested Spheres Analysis using SRSF

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf.pns as pns
import fdasrsf.utility_functions as uf
from scipy.integrate import cumulative_trapezoid

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
        pnsdat = psi / np.tile(np.sqrt((psi**2).sum(axis=0)), (d,1))

        resmat, PNS = pns.PNSmainHDLSS(pnsdat)

        # Proportion of variance explained
        varPNS = np.sum(np.abs(resmat)**2, axis=1) / n
        cumvarPNS = np.cumsum(varPNS)
        propcumPNS =cumvarPNS / cumvarPNS[-1]
        propPNS = varPNS / cumvarPNS[-1] * 100

        # Projection of PCs
        no = int(np.argwhere(propcumPNS <= var_exp)[-1])
        udir = np.eye(resmat.shape[0])
        projPsi = np.zeros((d, n, no))
        projGam = np.zeros((d, n, no))
        for PCnum in range(no):
            PCvec = pns.PNSe2s(np.outer(udir[:,PCnum],resmat[PCnum,:]), PNS)
            projPsi[:,:,PCnum] = PCvec*radius
            gamt = cumulative_trapezoid(projPsi[:,:,PCnum]**2, t, axis=0, initial=0)
            for j in range(n):
                gamt[:,j] = (gamt[:,j] - gamt[:,j].min()) / (gamt[:,j].max() - gamt[:,j].min())
            projGam[:,:,PCnum] = gamt
        
        self.gam_pns = projGam
        self.psi_pns = projPsi
        self.cumvar = propcumPNS
        self.no = no
        self.psi = psi
        self.PNS = PNS
        self.coef = resmat