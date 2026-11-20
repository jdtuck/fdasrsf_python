"""
Group-wise function alignment using SRSF framework and Dynamic Programming

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
import matplotlib.pyplot as plt
import fdasrsf.utility_functions as uf
import fdasrsf.bayesian_functions as bf
import fdasrsf.fPCA as fpca
import fdasrsf.geometry as geo
from fdasrsf.gp import gp_posterior, gp_fit_se
from fdasrsf.PPD import getPPDinfo, getPersistentPeaks, drawPPDBarChart, drawPPDSurface
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import svd, cholesky, solve_triangular
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.signal import find_peaks
from scipy.special import erfc
from numpy.linalg import norm, inv
from numpy.random import rand, normal, randn
from joblib import Parallel, delayed
from fdasrsf.fPLS import pls_svd
from tqdm import tqdm
import fdasrsf.plot_style as plot
import fpls_warp as fpls
import collections


class fdawarp:
    """
    This class provides alignment methods for functional data using the SRVF framework

    Usage:  obj = fdawarp(f,t)

    :param f: (M,N): matrix defining N functions of M samples
    :param time: time vector of length M
    :param fn: aligned functions
    :param qn: aligned srvfs
    :param q0: initial srvfs
    :param fmean: Karcher mean
    :param mqn: mean srvf
    :param gam: warping functions
    :param psi: srvf of warping functions
    :param stats: alignment statistics
    :param qun: cost function
    :param lambda: lambda
    :param method: optimization method
    :param gamI: inverse warping function
    :param rsamps: random samples
    :param fs: random aligned functions
    :param gams: random warping functions
    :param ft: random warped functions
    :param qs: random aligned srvfs
    :param type: alignment type
    :param mcmc: mcmc output if bayesian

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  15-Mar-2018
    """

    def __init__(self, f, time):
        """
        Construct an instance of the fdawarp class
        :param f: numpy ndarray of shape (M,N) of N functions with M samples
        :param time: vector of size M describing the sample points
        """
        a = time.shape[0]

        if f.shape[0] != a:
            raise Exception("Columns of f and time must be equal")

        self.f = f
        self.time = time
        self.rsamps = False
        self.mcmc = None

    def srsf_align(
        self,
        method="mean",
        omethod="DP2",
        center=True,
        smoothdata=False,
        MaxItr=20,
        parallel=False,
        lam=0.0,
        pen="l2psi",
        cores=-1,
        grid_dim=10,
        verbose=True,
	thresh = .01
    ):
        """
        This function aligns a collection of functions using the elastic
        square-root slope (srsf) framework.

        :param method: (string) warp calculate Karcher Mean or Median
                       (options = "mean" or "median") (default="mean")
        :param omethod: optimization method (DP, DP2, RBFGS, cRBFGS)
                        (default = DP2)
        :param center: center warping functions (default = T)
        :param smoothdata: Smooth the data using a box filter (default = F)
        :param MaxItr: Maximum number of iterations (default = 20)
        :param parallel: run in parallel (default = F)
        :param lam: controls the elasticity (default = 0)
        :param penalty: penalty type (default="l2psi") options are "roughness",
                "l2gam", "l2psi", "geodesic" and "none". The penalty is
                weighted by lam, so it has no effect when lam is 0.
                "roughness" and "geodesic" have no dynamic programming
                counterpart, so with a nonzero lam they need omethod
                "RBFGS" or "cRBFGS"
        :param cores: number of cores for parallel (default = -1 (all))
        :param grid_dim: size of the grid, for the DP2 method only
                         (default = 10)
        :param verbose: print status output (default = T)
        :param thresh: cost function threshold (default = .01)
        :type lam: double
        :type smoothdata: bool

        Examples
        >>> import tables
        >>> fun=tables.open_file("../Data/simu_data.h5")
        >>> f = fun.root.f[:]
        >>> f = f.transpose()
        >>> time = fun.root.time[:]
        >>> obj = fs.fdawarp(f,time)
        >>> obj.srsf_align()

        """
        M = self.f.shape[0]
        N = self.f.shape[1]
        self.lam = lam
        self.penalty = pen
        self.grid_dim = grid_dim
        self.smoothdata = smoothdata

        if M > 500:
            parallel = True
        elif N > 100:
            parallel = True

        eps = np.finfo(np.double).eps
        f0 = self.f
        self.method = omethod

        methods = ["mean", "median"]
        self.type = method
        self.mcmc = None

        # 0 mean, 1-median
        method = [i for i, x in enumerate(methods) if x == method]
        if len(method) == 0:
            method = 0
        else:
            method = method[0]

        # Compute SRSF function from data
        f, g, g2 = uf.gradient_spline(self.time, self.f, smoothdata)
        q = g / np.sqrt(abs(g) + eps)

        if verbose:
            print("Initializing...")
        mnq = q.mean(axis=1)
        a = mnq.repeat(N)
        d1 = a.reshape(M, N)
        d = (q - d1) ** 2
        dqq = np.sqrt(d.sum(axis=0))
        min_ind = dqq.argmin()
        mq = q[:, min_ind]
        mf = f[:, min_ind]

        if parallel:
            out = Parallel(n_jobs=cores)(
                delayed(uf.optimum_reparam)(
                    mq, self.time, q[:, n], omethod, lam, pen, grid_dim
                )
                for n in range(N)
            )
            gam = np.array(out)
            gam = gam.transpose()
        else:
            gam = np.zeros((M, N))
            for k in range(0, N):
                gam[:, k] = uf.optimum_reparam(
                    mq, self.time, q[:, k], omethod, lam, pen, grid_dim
                )

        gamI = uf.SqrtMeanInverse(gam)
        mf = np.interp(
            (self.time[-1] - self.time[0]) * gamI + self.time[0], self.time, mf
        )
        mq = uf.f_to_srsf(mf, self.time)

        # Compute Karcher Mean
        if verbose:
            if method == 0:
                print(
                    "Compute Karcher Mean of %d function in SRSF space with lam=%02f"
                    % (N, lam)
                )
            if method == 1:
                print(
                    "Compute Karcher Median of %d function in SRSF space with lam=%02f"
                    % (N, lam)
                )

        ds = np.repeat(0.0, MaxItr + 2)
        ds[0] = np.inf
        qun = np.repeat(0.0, MaxItr + 1)
        tmp = np.zeros((M, MaxItr + 2))
        tmp[:, 0] = mq
        mq = tmp
        tmp = np.zeros((M, MaxItr + 2))
        tmp[:, 0] = mf
        mf = tmp
        tmp = np.zeros((M, N, MaxItr + 2))
        tmp[:, :, 0] = self.f
        f = tmp
        tmp = np.zeros((M, N, MaxItr + 2))
        tmp[:, :, 0] = q
        q = tmp

        for r in range(0, MaxItr):
            if verbose:
                print("updating step: r=%d" % (r + 1))
                if r == (MaxItr - 1):
                    print("maximal number of iterations is reached")

            # Matching Step
            if parallel:
                out = Parallel(n_jobs=cores)(
                    delayed(uf.optimum_reparam)(
                        mq[:, r], self.time, q[:, n, 0], omethod, lam, pen, grid_dim
                    )
                    for n in range(N)
                )
                gam = np.array(out)
                gam = gam.transpose()
            else:
                for k in range(0, N):
                    gam[:, k] = uf.optimum_reparam(
                        mq[:, r], self.time, q[:, k, 0], omethod, lam, pen, grid_dim
                    )

            gam_dev = np.zeros((M, N))
            vtil = np.zeros((M, N))
            dtil = np.zeros(N)
            for k in range(0, N):
                f[:, k, r + 1] = np.interp(
                    (self.time[-1] - self.time[0]) * gam[:, k] + self.time[0],
                    self.time,
                    f[:, k, 0],
                )
                q[:, k, r + 1] = uf.f_to_srsf(f[:, k, r + 1], self.time)
                gam_dev[:, k] = np.gradient(gam[:, k], 1 / float(M - 1))
                v = q[:, k, r + 1] - mq[:, r]
                d = np.sqrt(trapezoid(v * v, self.time))
                vtil[:, k] = v / d
                dtil[k] = 1.0 / d

            mqt = mq[:, r]
            a = mqt.repeat(N)
            d1 = a.reshape(M, N)
            d = (q[:, :, r + 1] - d1) ** 2
            if method == 0:
                d1 = sum(trapezoid(d, self.time, axis=0))
                d2 = sum(trapezoid((1 - np.sqrt(gam_dev)) ** 2, self.time, axis=0))
                ds_tmp = d1 + lam * d2
                ds[r + 1] = ds_tmp

                # Minimization Step
                # compute the mean of the matched function
                qtemp = q[:, :, r + 1]
                ftemp = f[:, :, r + 1]
                mq[:, r + 1] = qtemp.mean(axis=1)
                mf[:, r + 1] = ftemp.mean(axis=1)

                qun[r] = norm(mq[:, r + 1] - mq[:, r]) / norm(mq[:, r])

            if method == 1:
                d1 = np.sqrt(sum(trapezoid(d, self.time, axis=0)))
                d2 = sum(trapezoid((1 - np.sqrt(gam_dev)) ** 2, self.time, axis=0))
                ds_tmp = d1 + lam * d2
                ds[r + 1] = ds_tmp

                # Minimization Step
                # compute the mean of the matched function
                stp = 0.3
                vbar = vtil.sum(axis=1) * (1 / dtil.sum())
                qtemp = q[:, :, r + 1]
                ftemp = f[:, :, r + 1]
                mq[:, r + 1] = mq[:, r] + stp * vbar
                mf[:, r + 1] = uf.srsf_to_f(mq[:, r + 1], self.time, np.median(f0[0, :]))

                qun[r] = norm(mq[:, r + 1] - mq[:, r]) / norm(mq[:, r])

            if qun[r] < thresh or r >= MaxItr:
                break

        # Last Step with centering of gam
        r += 1
        if parallel:
            out = Parallel(n_jobs=cores)(
                delayed(uf.optimum_reparam)(
                    mq[:, r], self.time, q[:, n, 0], omethod, lam, pen, grid_dim
                )
                for n in range(N)
            )
            gam = np.array(out)
            gam = gam.transpose()
        else:
            for k in range(0, N):
                gam[:, k] = uf.optimum_reparam(
                    mq[:, r], self.time, q[:, k, 0], omethod, lam, pen, grid_dim
                )

        gamI = uf.SqrtMeanInverse(gam)
        if center:
            # center gam (and, for the median, the template) by gamI
            time0 = (self.time[-1] - self.time[0]) * gamI + self.time[0]
            for k in range(0, N):
                gam[:, k] = np.interp(time0, self.time, gam[:, k])
            if method == 1:
                gamI_dev = np.gradient(gamI, 1 / float(M - 1))
                mq[:, r + 1] = np.interp(time0, self.time, mq[:, r]) * np.sqrt(
                    np.maximum(gamI_dev, 0)
                )
        elif method == 1:
            mq[:, r + 1] = mq[:, r]

        # warp the original data once by the final gam, so fn = f o gam holds
        # exactly (previously fn/qn kept the previous iteration's alignment
        # while gam came from this final matching step)
        for k in range(0, N):
            f[:, k, r + 1] = np.interp(
                (self.time[-1] - self.time[0]) * gam[:, k] + self.time[0],
                self.time,
                f[:, k, 0],
            )
        q[:, :, r + 1] = uf.f_to_srsf(f[:, :, r + 1], self.time)
        if method == 0:
            mq[:, r + 1] = q[:, :, r + 1].mean(axis=1)

        # Aligned data & stats
        self.center = center
        self.fn = f[:, :, r + 1]
        self.qn = q[:, :, r + 1]
        self.q0 = q[:, :, 0]
        self.gamI = gamI
        std_f0 = f0.std(axis=1, ddof=1)
        std_fn = self.fn.std(axis=1, ddof=1)
        self.gam = gam
        self.mqn = mq[:, r + 1]
        if method == 1:
            self.fmean = uf.srsf_to_f(self.mqn, self.time, np.median(f0[0, :]))
        else:
            self.fmean = uf.srsf_to_f(self.mqn, self.time, np.mean(f0[0, :]))

        fgam = np.zeros((M, N))
        for k in range(0, N):
            time0 = (self.time[-1] - self.time[0]) * gam[:, k] + self.time[0]
            fgam[:, k] = np.interp(time0, self.time, self.fmean)

        var_fgam = fgam.var(axis=1, ddof=1)
        self.orig_var = trapezoid(std_f0**2, self.time)
        self.amp_var = trapezoid(std_fn**2, self.time)
        self.phase_var = trapezoid(var_fgam, self.time)
        self.qun = qun[0:r]

        return
    
    def time_warping_bayes(
        self,
        iter=2000,
        burnin=None,
        alpha0=0.1,
        beta0=0.1,
        zpcn=None,
        propvar=4,
        initcoef=None,
        nbasis=30,
        npoints=200,
        extrainfo=True,
        ncenter=100,
        verbose=True,
    ):
        r"""
        Align multiple functions using a Bayesian method

        This function aligns a collection of functions using the elastic
        square-root slope (srsf) framework. This method uses a Bayesian
        approach.  It is based on mapping warping functions to a hypersphere,
        and a subsequent exponential mapping to a tangent space. In the
        tangent space, the Z-mixture pCN algorithm is used to explore both
        local and global structure in the posterior distribution.

        The Z-mixture pCN algorithm uses a mixture distribution for the
        proposal distribution, controlled by the input parameter zpcn. The
        zpcn['betas'] must be between 0 and 1, and are the coefficients of
        the mixture components, with larger coefficients corresponding to
        larger shifts in parameter space. The zpcn['probs'] give the
        probability of each shift size.

        Usage:  obj.time_warping_bayes()

        :param iter: number of iterations (default = 2000)
        :param burnin: number of burn-in iterations
                       (default = min(5e3, iter/2))
        :param alpha0: inverse gamma prior parameter (default = 0.1)
        :param beta0: inverse gamma prior parameter (default = 0.1)
        :param zpcn: dict of pCN parameters, default
                     {"betas":np.array([0.5,0.5,0.005,0.0001]),
                      "probs":np.array([0.1,0.1,0.7,0.1])}
        :param propvar: proposal variance (default = 4)
        :param initcoef: (D,N) initial warping function coefficients
                         (default = zeros((20,N)))
        :param nbasis: number of basis elements for q (default = 30)
        :param npoints: number of sample interpolation points (default = 200)
        :param extrainfo: return extra info about the mcmc (default = T)
        :param ncenter: number of iterations between centering gam
                        (default = 100)
        :param verbose: print status output (default = T)

        fills in the fdawarp object (fn, qn, q0, fmean, mqn, gam, psi, gamI,
        orig_var, amp_var, phase_var, qun) and self.mcmc containing

        mcmc['g_coef']: (iter,D,N) posterior warping coefficients
        mcmc['sigma1']: posterior sigma1
        mcmc['logl']: log-likelihood trace
        mcmc['burnin'], mcmc['iter']: burn-in and number of iterations
        mcmc['extrainfo']: whether the extra information was stored

        if extrainfo

        mcmc['accept']: acceptance of q_star samples
        mcmc['betas_ind']: pCN mixture component of each sample
        mcmc['zpcn']: pCN parameters used
        mcmc['gamma_mat']: (M,nsamp,N) posterior gammas
        mcmc['gamma_stats']: posterior gamma stats
        mcmc['q_star_mat']: (M,nsamp) posterior q_star
        mcmc['q_star_stats']: posterior q_star stats
        mcmc['fmean_mat']: (M,nsamp) posterior fmean
        mcmc['fmean_stats']: posterior fmean stats
        """
        M, N = self.f.shape
        iter = int(iter)
        if burnin is None:
            burnin = int(np.minimum(5 * (10**3), iter // 2))
        burnin = int(burnin)
        if burnin < 1 or burnin >= iter:
            raise Exception("burnin must be between 1 and iter-1")

        if zpcn is None:
            zpcn = {
                "betas": np.array([0.5, 0.5, 0.005, 0.0001]),
                "probs": np.array([0.1, 0.1, 0.7, 0.1]),
            }
        zpcn = {
            "betas": np.asarray(zpcn["betas"], dtype=float),
            "probs": np.asarray(zpcn["probs"], dtype=float),
        }
        if zpcn["betas"].shape[0] != zpcn["probs"].shape[0]:
            raise Exception("In zpcn, betas must equal length of probs")

        if initcoef is None:
            initcoef = np.zeros((20, N))
        initcoef = np.asarray(initcoef, dtype=float)
        if initcoef.ndim == 1:
            initcoef = np.tile(initcoef.reshape((-1, 1)), (1, N))
        if np.mod(initcoef.shape[0], 2) != 0:
            raise Exception("length of initcoef must be even")
        if np.mod(nbasis, 2) != 0:
            raise Exception("nbasis must be even")

        # Number of sig figs to report in gamma_mat
        SIG_GAM = 13

        # normalize time to [0,1]
        time0 = self.time.ravel()
        tn = (time0 - time0.min()) / (time0.max() - time0.min())

        # parameter settings
        valid_index = np.arange(burnin - 1, iter)
        pw_sim_global_Mg = initcoef.shape[0] // 2
        g_coef_ini = initcoef
        numSimPoints = int(npoints)
        pw_sim_global_domain_par = np.linspace(0, 1, numSimPoints)
        g_basis = uf.basis_fourier(pw_sim_global_domain_par, pw_sim_global_Mg, 1)
        pw_sim_global_Mq = int(nbasis) // 2
        q_basis = uf.basis_bspline(
            pw_sim_global_domain_par, 2 * pw_sim_global_Mq, 3
        )
        sigma1_ini = 1

        # srsf transformation
        f0 = np.zeros((numSimPoints, N))
        for ii in range(0, N):
            f0[:, ii] = np.interp(g_basis["x"], tn, self.f[:, ii])
        qo = uf.f_to_srsf(f0, g_basis["x"])

        pw_sim_global_sigma_g = propvar
        pw_sim_global_sigma_q = (
            np.percentile(qo, 95) - np.percentile(qo, 5)
        ) * 5

        def propose_g_coef(g_coef_curr):
            pCN_beta = zpcn["betas"]
            pCN_prob = zpcn["probs"]
            probm = np.insert(np.cumsum(pCN_prob), 0, 0)
            z = rand()
            result = {"prop": g_coef_curr, "ind": 1}
            for i in range(0, pCN_beta.shape[0]):
                if z <= probm[i + 1] and z > probm[i]:
                    g_coef_new = normal(
                        0,
                        pw_sim_global_sigma_g
                        / np.repeat(np.arange(1, pw_sim_global_Mg + 1), 2),
                    )
                    result["prop"] = (
                        np.sqrt(1 - pCN_beta[i] ** 2) * g_coef_curr
                        + pCN_beta[i] * g_coef_new
                    )
                    result["ind"] = i

            return result

        def propose_q_star(q_star_coef_curr):
            pCN_beta = zpcn["betas"]
            pCN_prob = zpcn["probs"]
            probm = np.insert(np.cumsum(pCN_prob), 0, 0)
            z = rand()
            sdvec = pw_sim_global_sigma_q / np.repeat(
                np.arange(1, pw_sim_global_Mq + 1), 2
            )
            result = {"prop": q_star_coef_curr, "ind": 1}
            for i in range(0, pCN_beta.shape[0]):
                if z <= probm[i + 1] and z > probm[i]:
                    q_star_new = normal(0, sdvec)
                    result["prop"] = (
                        np.sqrt(1 - pCN_beta[i] ** 2) * q_star_coef_curr
                        + pCN_beta[i] * q_star_new
                    )
                    result["ind"] = i

            return result

        for ii in range(0, N):
            tmp = uf.f_exp1(
                uf.f_basistofunction(g_basis["x"], 0, g_coef_ini[:, ii], g_basis)
            )
            if tmp.min() < 0:
                raise Exception("Invalid initial value of g for sample %d" % ii)

        # result objects
        result_g_coef = np.zeros((iter, g_coef_ini.shape[0], N))
        result_q_star = np.zeros((iter, numSimPoints))
        result_sigma1 = np.zeros(iter)
        result_logl = np.zeros(iter)
        result_SSE = np.zeros((iter, N))
        result_accept = np.zeros(iter, dtype=bool)
        result_accept_betas = np.zeros(iter)

        # init
        g_coef_curr = g_coef_ini.copy()
        sigma1_curr = sigma1_ini
        mnq = qo.mean(axis=1)
        dqq = np.sqrt(np.sum((qo - mnq.reshape((numSimPoints, 1))) ** 2, axis=0))
        q_star_curr = qo[:, dqq.argmin()]
        q_star_ini = q_star_curr.copy()
        SSE_curr = bf.f_SSEg_group(
            bf._g_coef_to_g(g_coef_ini, g_basis), qo, q_star_ini
        )
        logl_curr = bf.f_logl_group(
            bf._g_coef_to_g(g_coef_ini, g_basis),
            qo,
            q_star_ini,
            sigma1_ini**2,
            SSE_curr,
        )

        result_g_coef[0, :, :] = g_coef_ini
        result_sigma1[0] = sigma1_ini
        result_q_star[0, :] = q_star_curr
        result_SSE[0, :] = SSE_curr
        result_logl[0] = logl_curr

        q_star_coef_curr = np.linalg.lstsq(
            q_basis["matrix"], q_star_curr, rcond=None
        )[0]
        q_star_curr = q_star_ini.copy()
        gamI_o = pw_sim_global_domain_par.copy()

        # update the chain for iter-1 times
        for m in tqdm(range(1, iter), disable=not verbose):
            # update g
            for ii in range(0, N):
                g_coef_curr[:, ii], tmp_a, tmp_i = bf.f_updateg_group(
                    g_coef_curr[:, ii],
                    g_basis,
                    sigma1_curr**2,
                    qo[:, ii],
                    q_star_curr,
                    propose_g_coef,
                )
            g_curr = bf._g_coef_to_g(g_coef_curr, g_basis)
            SSE_curr = bf.f_SSEg_group(g_curr, qo, q_star_curr)

            # update q_star
            q_star_coef_curr, accept, zpcnInd = bf.f_updateq(
                g_coef_curr,
                g_basis,
                sigma1_curr**2,
                qo,
                q_star_coef_curr,
                q_basis,
                SSE_curr,
                propose_q_star,
            )
            q_star_curr = uf.f_basistofunction(
                q_basis["x"], 0, q_star_coef_curr, q_basis
            )

            # update sigma1
            SSE_curr = bf.f_SSEg_group(g_curr, qo, q_star_curr)
            newshape = N * M / 2 + alpha0
            newscale = 1 / 2 * SSE_curr.sum() + beta0
            sigma1_curr = np.sqrt(1 / np.random.gamma(newshape, 1 / newscale))
            logl_curr = bf.f_logl_group(
                g_curr, qo, q_star_curr, sigma1_curr**2, SSE_curr
            )

            # center
            if ncenter != 0 and np.mod(m + 1, ncenter) == 0:
                result_posterior_gamma = np.zeros((numSimPoints, N))
                for ii in range(0, N):
                    g_temp = uf.f_basistofunction(
                        pw_sim_global_domain_par, 0, g_coef_curr[:, ii], g_basis
                    )
                    psi_temp = uf.f_exp1(g_temp)
                    # transform posterior mean of psi to gamma
                    result_posterior_gamma[:, ii] = uf.norm_gam(
                        uf.f_phiinv(psi_temp)
                    )
                gamI_o = uf.SqrtMeanInverse(result_posterior_gamma)
                q_star_curr = uf.warp_q_gamma(
                    pw_sim_global_domain_par, q_star_curr, gamI_o
                )
                for k in range(0, N):
                    result_posterior_gamma[:, k] = np.interp(
                        gamI_o,
                        pw_sim_global_domain_par,
                        result_posterior_gamma[:, k],
                    )
                    psi1 = uf.f_phi(result_posterior_gamma[:, k])
                    yy = uf.f_exp1inv(psi1)
                    g_coef_curr[:, k] = np.linalg.lstsq(
                        g_basis["matrix"], yy, rcond=None
                    )[0]
                g_curr = bf._g_coef_to_g(g_coef_curr, g_basis)
                SSE_curr = bf.f_SSEg_group(g_curr, qo, q_star_curr)
                logl_curr = bf.f_logl_group(
                    g_curr, qo, q_star_curr, sigma1_curr**2, SSE_curr
                )
                q_star_coef_curr = np.linalg.lstsq(
                    q_basis["matrix"], q_star_curr, rcond=None
                )[0]

            # save update to results
            result_g_coef[m, :, :] = g_coef_curr
            result_q_star[m, :] = q_star_curr
            result_sigma1[m] = sigma1_curr
            result_SSE[m, :] = SSE_curr
            if extrainfo:
                result_logl[m] = logl_curr
                result_accept[m] = accept
                result_accept_betas[m] = zpcnInd

        # calculate posterior mean of psi
        nv = valid_index.shape[0]
        pw_sim_est_psi_matrix = np.zeros((numSimPoints, nv, N))
        for k in range(0, nv):
            for ii in range(0, N):
                g_temp = uf.f_basistofunction(
                    pw_sim_global_domain_par,
                    0,
                    result_g_coef[valid_index[k], :, ii],
                    g_basis,
                )
                pw_sim_est_psi_matrix[:, k, ii] = uf.f_exp1(g_temp)

        result_posterior_psi = np.zeros((M, N))
        result_posterior_gamma = np.zeros((M, N))
        f_warped = np.zeros((M, N))
        psi_sim = np.zeros((numSimPoints, N))
        for ii in range(0, N):
            psi_sim[:, ii] = uf.f_psimean(
                pw_sim_global_domain_par, pw_sim_est_psi_matrix[:, :, ii]
            )
            # resample to same number of points as the input f
            result_posterior_psi[:, ii] = np.interp(
                tn, pw_sim_global_domain_par, psi_sim[:, ii]
            )
            # transform posterior mean of psi to gamma
            result_posterior_gamma[:, ii] = uf.norm_gam(
                uf.f_phiinv(result_posterior_psi[:, ii])
            )
            f_warped[:, ii] = uf.warp_f_gamma(
                time0, self.f[:, ii], result_posterior_gamma[:, ii]
            )

        if extrainfo:
            # matrix of posterior draws from gamma
            gamma_mat = np.zeros((M, nv, N))
            gamma_stats = np.zeros((M, 2, N))
            for jj in range(0, N):
                for ii in range(0, nv):
                    result_i = np.interp(
                        tn,
                        pw_sim_global_domain_par,
                        pw_sim_est_psi_matrix[:, ii, jj],
                    )
                    gamma_mat[:, ii, jj] = np.round(
                        uf.norm_gam(uf.f_phiinv(result_i)), SIG_GAM
                    )
                gamma_stats[:, :, jj] = uf.statsFun(gamma_mat[:, :, jj])

        # return object
        self.fn = f_warped
        self.gam = result_posterior_gamma
        self.psi = result_posterior_psi
        self.qn = uf.f_to_srsf(f_warped, time0)
        self.q0 = uf.f_to_srsf(self.f, time0)
        # template on the original time scale (the srsf of the simulation
        # grid is on [0,1], and the srsf scales by 1/sqrt(T))
        mqn_temp = result_q_star[valid_index, :].mean(axis=0)
        self.mqn = np.interp(tn, g_basis["x"], mqn_temp) / np.sqrt(
            time0[-1] - time0[0]
        )
        self.fmean = self.f[0, :].mean() + cumulative_trapezoid(
            self.mqn * np.abs(self.mqn), time0, initial=0
        )

        std_f0 = self.f.std(axis=1, ddof=1)
        std_fn = self.fn.std(axis=1, ddof=1)
        fgam = np.zeros((M, N))
        for ii in range(0, N):
            fgam[:, ii] = uf.warp_f_gamma(time0, self.fmean, self.gam[:, ii])

        self.orig_var = trapezoid(std_f0**2, time0)
        self.amp_var = trapezoid(std_fn**2, time0)
        self.phase_var = trapezoid(fgam.var(axis=1, ddof=1), time0)

        self.qun = result_logl
        self.method = "bayesian"
        self.gamI = np.interp(tn, g_basis["x"], gamI_o)
        self.rsamps = False
        self.type = "mean"

        self.mcmc = {
            "g_coef": result_g_coef,
            "sigma1": result_sigma1,
            "logl": result_logl,
            "burnin": burnin,
            "iter": iter,
            "extrainfo": extrainfo,
        }

        if extrainfo:
            self.mcmc["accept"] = result_accept[1:]
            self.mcmc["betas_ind"] = result_accept_betas[1:]
            self.mcmc["zpcn"] = zpcn
            self.mcmc["gamma_mat"] = gamma_mat
            self.mcmc["gamma_stats"] = gamma_stats
            # posterior templates and mean functions on the original grid
            q_star_mat = np.zeros((M, nv))
            for k in range(0, nv):
                q_star_mat[:, k] = np.interp(
                    tn, g_basis["x"], result_q_star[valid_index[k], :]
                ) / np.sqrt(time0[-1] - time0[0])
            self.mcmc["q_star_mat"] = q_star_mat
            self.mcmc["q_star_stats"] = uf.statsFun(q_star_mat)
            fmean_mat = np.zeros((M, nv))
            f0_mean = self.f[0, :].mean()
            for k in range(0, nv):
                fmean_mat[:, k] = f0_mean + cumulative_trapezoid(
                    q_star_mat[:, k] * np.abs(q_star_mat[:, k]), time0, initial=0
                )
            self.mcmc["fmean_mat"] = fmean_mat
            self.mcmc["fmean_stats"] = uf.statsFun(fmean_mat)

        return

    def time_warping_bayes_infHMC(
        self,
        iter=2000,
        burnin=None,
        thin=1,
        alpha0=0.1,
        beta0=0.1,
        h=0.01,
        L=10,
        adapt=True,
        target_accept=0.65,
        ncoef=10,
        basis="fourier",
        warp_rep="shooting",
        vpriorvar=1,
        vpriordecay=1,
        nbasis=40,
        qpriorvar=1e3,
        init="dp",
        ncenter=20,
        center_all=True,
        extrainfo=True,
        verbose=True,
    ):
        r"""
        Bayesian group-wise alignment using infinite-dimensional Hamiltonian
        Monte Carlo

        Hierarchical Bayesian model in SRVF space. For i = 1, ..., N

        .. math::
            q^*(t_k) = (q_i \circ \gamma_i)(t_k)\sqrt{\gamma_i'(t_k)} +
            e_{ik}, \quad e_{ik} \sim N(0, \sigma^2)

        with template :math:`q^* = Bb` (B-spline basis,
        :math:`b \sim N(0, qpriorvar\, I)`), warping coefficients
        :math:`c_i \sim N(0, \mathrm{diag}(C))`,
        :math:`C_j = vpriorvar/j^{vpriordecay}`, and
        :math:`\sigma^2 \sim IG(alpha0, beta0)`. The warping functions use
        either the shooting vector (v-transform, :math:`|v| < \pi`) or the
        centered log-derivative (h-transform, unconstrained) representation.

        Each sweep updates every :math:`c_i` by preconditioned
        (infinite-dimensional) HMC with the exact likelihood gradient, then
        draws b and :math:`\sigma^2` from their conjugate full conditionals.
        Step sizes are adapted per curve by dual averaging during burn-in.
        The amplitude-phase split is identified only up to a common warp (a
        ridge that per-curve updates move along slowly), so the centering
        constraint is imposed by recentering the warps every ncenter
        iterations so that their Karcher mean is the identity (the template
        is warped accordingly). The chain is initialized from an elastic (DP)
        Karcher mean alignment unless init = "identity".

        Usage:  obj.time_warping_bayes_infHMC()

        :param iter: number of iterations (default = 2000)
        :param burnin: number of burn-in iterations (default = iter/2)
        :param thin: thinning of stored posterior draws (default = 1)
        :param alpha0: inverse gamma prior parameter for sigma^2
        :param beta0: inverse gamma prior parameter for sigma^2
        :param h: initial HMC step size is sqrt(h) (default = 0.01)
        :param L: maximum number of leapfrog steps (default = 10)
        :param adapt: adapt step sizes during burn-in (default = T)
        :param target_accept: target acceptance when adapting (default = .65)
        :param ncoef: number of warping basis coefficients (default = 10)
        :param basis: warping basis, 'fourier' or 'legendre'
        :param warp_rep: 'shooting' (v-transform) or 'clr' (h-transform)
        :param vpriorvar: prior variance scale for warping coefficients
        :param vpriordecay: prior variance decays as 1/j^vpriordecay
        :param nbasis: number of B-spline basis functions for q* (default=40)
        :param qpriorvar: prior variance of the template coefficients
        :param init: "dp" (elastic alignment) or "identity"
        :param ncenter: recenter every ncenter iterations (0: never)
        :param center_all: recenter after burn-in too (F: burn-in only)
        :param extrainfo: return posterior draws and summaries (default = T)
        :param verbose: print status output (default = T)

        fills in the fdawarp object (fn, qn, q0, fmean, mqn, gam, psi, gamI,
        orig_var, amp_var, phase_var, qun) with gam the posterior
        (extrinsic) mean warp of each curve, and self.mcmc containing

        mcmc['v_coef']: (D,N,nsamp) posterior draws of the warping coefficients
        mcmc['sigma']: sigma draws (all iterations)
        mcmc['accept']: (N,iter-1) HMC acceptance indicators
        mcmc['epsilon']: (N,iter) step sizes
        mcmc['logl']: log-likelihood trace
        mcmc['burnin'], mcmc['iter']: burn-in and number of iterations

        if extrainfo

        mcmc['gamma_mat']: (M,nsamp,N) posterior gammas
        mcmc['gamma_stats']: (M,2,N) pointwise 95% bands of gammas
        mcmc['q_star_mat']: (M,nsamp) posterior templates
        mcmc['q_star_stats']: pointwise 95% bands of the template
        mcmc['fmean_mat']: (M,nsamp) posterior mean functions
        mcmc['fmean_stats']: pointwise 95% bands of the mean function
        """
        M, N = self.f.shape
        iter = int(iter)
        if burnin is None:
            burnin = int(np.floor(iter / 2))
        burnin = int(burnin)
        if burnin < 1 or burnin >= iter:
            raise Exception("burnin must be between 1 and iter-1")

        ncoef = int(ncoef)
        if str(basis).lower() not in ("fourier", "legendre"):
            raise Exception("basis must be 'fourier' or 'legendre'")
        if str(basis).lower() == "fourier" and np.mod(ncoef, 2) != 0:
            raise Exception("ncoef must be even for the fourier basis")
        if str(warp_rep).lower() not in ("shooting", "clr"):
            raise Exception("warp_rep must be 'shooting' or 'clr'")
        if str(init).lower() not in ("dp", "identity"):
            raise Exception("init must be 'dp' or 'identity'")
        rep = 1 if str(warp_rep).lower() == "clr" else 0

        # work on a uniform grid on [0,1]
        time0 = self.time.ravel()
        tn = (time0 - time0[0]) / (time0[-1] - time0[0])
        t01 = np.linspace(0, 1, M)
        f0 = np.zeros((M, N))
        for ii in range(0, N):
            f0[:, ii] = np.interp(t01, tn, self.f[:, ii])
        q = uf.f_to_srsf(f0, t01)

        Phi = bf.bayes_warp_basis(t01, ncoef, basis)
        C = vpriorvar / np.ceil(np.arange(1, ncoef + 1) / 2) ** vpriordecay
        nbasis = int(nbasis)
        B = uf.create_basismatrix(t01, nbasis, 4)
        BtB = B.T @ B
        Bpinv = np.linalg.pinv(B)

        # initialization
        if str(init).lower() == "dp":
            tmp = fdawarp(f0, t01)
            tmp.srsf_align(MaxItr=1, verbose=False)
            c_curr = bf.bayes_gam_to_coef(tmp.gam, Phi, rep)
            gam0, psi0 = bf.bayes_coef_to_gam(c_curr, Phi, rep)
            qtil = np.zeros((M, N))
            for ii in range(0, N):
                qtil[:, ii] = uf.warp_q_gamma(t01, q[:, ii], gam0[:, ii])
            qstar = qtil.mean(axis=1)
        else:
            c_curr = np.zeros((ncoef, N))
            dqq = np.sqrt(
                np.sum((q - q.mean(axis=1).reshape((M, 1))) ** 2, axis=0)
            )
            qstar = q[:, dqq.argmin()]

        b_curr = Bpinv @ qstar
        qstar = B @ b_curr
        sigma2 = np.mean(np.sum((q - qstar.reshape((M, 1))) ** 2, axis=0)) / M

        # storage
        keep = np.arange(burnin - 1, iter, int(thin))
        nsamp = keep.shape[0]
        v_coef = np.zeros((ncoef, N, nsamp))
        qstar_mat = np.zeros((M, nsamp))
        sigma_tr = np.zeros(iter)
        logl_tr = np.zeros(iter)
        accept = np.zeros((N, iter))
        eps_tr = np.zeros((N, iter))
        epsilon = np.sqrt(h) * np.ones(N)
        da = bf.dual_avg_step(epsilon)
        sigma_tr[0] = np.sqrt(sigma2)
        eps_tr[:, 0] = epsilon
        ks = 0

        itr = tqdm(range(1, iter), disable=not verbose)
        for m in itr:
            # update warping coefficients: the curves are conditionally
            # independent given q* and sigma^2, so all are updated in one
            # batched HMC step
            c_curr, aux, accept[:, m], alpha = _group_hmc_update(
                c_curr, Phi, rep, qstar, q, sigma2, C, epsilon, L
            )
            qtil = aux[1:, :]

            # dual-averaging step-size adaptation during burn-in; keep the
            # averaged step sizes afterwards
            if adapt and (m + 1) <= burnin:
                epsilon, da = bf.dual_avg_step(da, alpha, target_accept)
                if (m + 1) == burnin:
                    epsilon = np.exp(da["log_eps_bar"])

            # template: conjugate Gaussian draw
            P = N * BtB / sigma2 + np.eye(nbasis) / qpriorvar
            R = cholesky((P + P.T) / 2)
            rhs = B.T @ qtil.sum(axis=1) / sigma2
            mu = solve_triangular(
                R, solve_triangular(R.T, rhs, lower=True), lower=False
            )
            b_curr = mu + solve_triangular(R, randn(nbasis), lower=False)
            qstar = B @ b_curr

            # noise variance: conjugate inverse gamma draw
            SSE = np.sum((qtil - qstar.reshape((M, 1))) ** 2, axis=0)
            sigma2 = 1 / np.random.gamma(
                N * M / 2 + alpha0, 1 / (SSE.sum() / 2 + beta0)
            )

            # centering constraint (Karcher mean of warps = identity):
            # project onto it every ncenter iterations
            if (
                ncenter > 0
                and ((m + 1) < burnin or center_all)
                and np.mod(m + 1, ncenter) == 0
            ):
                gam_c, psi_c = bf.bayes_coef_to_gam(c_curr, Phi, rep)
                gam_inv = _karcher_mean_inverse(gam_c, t01)
                for ii in range(0, N):
                    gam_c[:, ii] = np.interp(gam_inv, t01, gam_c[:, ii])
                c_curr = bf.bayes_gam_to_coef(gam_c, Phi, rep)
                qstar = uf.warp_q_gamma(t01, qstar, gam_inv)
                b_curr = Bpinv @ qstar
                qstar = B @ b_curr

            sigma_tr[m] = np.sqrt(sigma2)
            logl_tr[m] = -SSE.sum() / (2 * sigma2) - N * M / 2 * np.log(
                2 * np.pi * sigma2
            )
            eps_tr[:, m] = epsilon
            if ks < nsamp and m == keep[ks]:
                v_coef[:, :, ks] = c_curr
                qstar_mat[:, ks] = qstar
                ks += 1

        # posterior summaries of the warping functions
        gamma_mat = np.zeros((M, nsamp, N))
        gam_mean = np.zeros((M, N))
        psi_mean = np.zeros((M, N))
        for ii in range(0, N):
            gam_s, psi_s = bf.bayes_coef_to_gam(v_coef[:, ii, :], Phi, rep)
            for k in range(0, nsamp):
                gamma_mat[:, k, ii] = np.interp(tn, t01, gam_s[:, k])
            pm = psi_s.mean(axis=1)
            pm = pm / np.sqrt(trapezoid(pm**2, t01))
            g = cumulative_trapezoid(pm**2, t01, initial=0)
            g = (g - g[0]) / (g[-1] - g[0])
            gam_mean[:, ii] = np.interp(tn, t01, g)
            psi_mean[:, ii] = np.interp(tn, t01, pm)

        self.gam = gam_mean
        self.psi = psi_mean
        self.fn = np.zeros((M, N))
        for ii in range(0, N):
            self.fn[:, ii] = uf.warp_f_gamma(time0, self.f[:, ii], self.gam[:, ii])
        self.qn = uf.f_to_srsf(self.fn, time0)
        self.q0 = uf.f_to_srsf(self.f, time0)
        # template on the original time scale (SRVF scales by 1/sqrt(T))
        mq01 = qstar_mat.mean(axis=1)
        self.mqn = np.interp(tn, t01, mq01) / np.sqrt(time0[-1] - time0[0])
        self.fmean = self.f[0, :].mean() + cumulative_trapezoid(
            self.mqn * np.abs(self.mqn), time0, initial=0
        )

        std_f0 = self.f.std(axis=1, ddof=1)
        std_fn = self.fn.std(axis=1, ddof=1)
        fgam = np.zeros((M, N))
        for ii in range(0, N):
            fgam[:, ii] = uf.warp_f_gamma(time0, self.fmean, self.gam[:, ii])
        self.orig_var = trapezoid(std_f0**2, time0)
        self.amp_var = trapezoid(std_fn**2, time0)
        self.phase_var = trapezoid(fgam.var(axis=1, ddof=1), time0)

        self.qun = logl_tr[1:]
        self.method = "bayesian_infHMC"
        self.gamI = uf.SqrtMeanInverse(self.gam)
        self.rsamps = False
        self.type = "mean"

        self.mcmc = {
            "v_coef": v_coef,
            "sigma": sigma_tr,
            "logl": logl_tr,
            "accept": accept[:, 1:],
            "epsilon": eps_tr,
            "warp_rep": warp_rep,
            "burnin": burnin,
            "iter": iter,
        }
        if extrainfo:
            self.mcmc["gamma_mat"] = gamma_mat
            gamma_stats = np.zeros((M, 2, N))
            for ii in range(0, N):
                gamma_stats[:, :, ii] = uf.statsFun(gamma_mat[:, :, ii])
            self.mcmc["gamma_stats"] = gamma_stats
            q_star_mat = np.zeros((M, nsamp))
            for k in range(0, nsamp):
                q_star_mat[:, k] = np.interp(tn, t01, qstar_mat[:, k]) / np.sqrt(
                    time0[-1] - time0[0]
                )
            self.mcmc["q_star_mat"] = q_star_mat
            self.mcmc["q_star_stats"] = uf.statsFun(q_star_mat)
            fmean_mat = np.zeros((M, nsamp))
            f0_mean = self.f[0, :].mean()
            for k in range(0, nsamp):
                fmean_mat[:, k] = f0_mean + cumulative_trapezoid(
                    q_star_mat[:, k] * np.abs(q_star_mat[:, k]), time0, initial=0
                )
            self.mcmc["fmean_mat"] = fmean_mat
            self.mcmc["fmean_stats"] = uf.statsFun(fmean_mat)

        return

    def ppd(self, 
            max_lam = 2, 
            num_lam=10, 
            pt=0.15, 
            srvf=True,
            mu=None,
            method="mean",
            omethod="DP2",
            center=True,
            smoothdata=False,
            MaxItr=20,
            parallel=True,
            pen="l2psi",
            cores=-1,
            grid_dim=10,
            verbose=True):
        """
        This computes the peak persistance diagram over a range of
        lambda. This can help determine the proper elasticity (penalty).
        This can be slow and recommended to run in parallel.

        
        :param max_lam: max regularization parameter (default 2)
        :param num_lam: number of steps (default 10)
        :param pt: the percentile of negative curvature of raw data (default .15)
        :param srvf: perform in SRVF space (default=True)
        :param mu: vector of function to align to (default None)
        :param method: (string) warp calculate Karcher Mean or Median
                       (options = "mean" or "median") (default="mean")
        :param omethod: optimization method (DP, DP2, RBFGS, cRBFGS)
                        (default = DP2)
        :param center: center warping functions (default = T)
        :param smoothdata: Smooth the data using a box filter (default = F)
        :param MaxItr: Maximum number of iterations (default = 20)
        :param parallel: run in parallel (default = F)
        :param penalty: penalty type (default="l2psi") options are "roughness",
                "l2gam", "l2psi", "geodesic" and "none". The penalty is
                weighted by lam, so it has no effect when lam is 0.
                "roughness" and "geodesic" have no dynamic programming
                counterpart, so with a nonzero lam they need omethod
                "RBFGS" or "cRBFGS"
        :param cores: number of cores for parallel (default = -1 (all))
        :param grid_dim: size of the grid, for the DP2 method only
                         (default = 10)
        :param verbose: print status output (default = T)
        :param thresh: cost function threshold (default = .01)
        :type lam: double
        :type smoothdata: bool

        """

        # Obtain a lambda candidate set
        lam_vec = np.linspace(0, max_lam, num_lam)
        fns = []
        for i in range(num_lam):
            if mu is None:
                self.srsf_align(method,
                                omethod,
                                center,
                                smoothdata,
                                MaxItr,
                                parallel,
                                lam_vec[i],
                                pen,
                                cores,
                                grid_dim,
                                verbose)
            else:
                self.multiple_align_functions(mu,
                                              omethod,
                                              smoothdata,
                                              parallel,
                                              lam_vec[i],
                                              pen,
                                              cores,
                                              grid_dim=grid_dim)
            
            if srvf:
                fns.append(self.qn)
            else:
                fns.append(self.fn)
        
        if srvf:
            f0 = self.q0.copy()
        else:
            f0 = self.f.copy()

        # Peak Persistent Diagrams
        # Get the threshold for significant peak
        diff_t = np.mean(np.diff(self.time))
        taus = []
        for i in range(self.f.shape[1]):
            idx, _ = find_peaks(f0[:,i])
            df2 = np.gradient(np.gradient(f0[:,i], diff_t), diff_t)
            tau = -df2 / np.max(-df2)
            tau[tau < 0] = 0
            taus.append(tau[idx])

        th = np.percentile(np.concatenate(taus), pt*100)

        IndicatorMatrix, Heights, Locs, Labels, FNm = getPPDinfo(self.time, fns, lam_vec, th)
        persistent_peak_labels = getPersistentPeaks(IndicatorMatrix.T)

        # choose optimal lambda
        ref_row = np.zeros(IndicatorMatrix.shape[1])
        ref_row[persistent_peak_labels] = 1

        n_lams = IndicatorMatrix.shape[0]
        hamming_distances = np.zeros(n_lams)
        exact_match_indices = []

        for i in range(n_lams):
            comp = np.logical_not(np.isnan(IndicatorMatrix[i, :]))

            # check for exact match
            if np.all(comp == ref_row):
                exact_match_indices.append(i)
            
            # calculate hamming distance
            hamming_distances[i] = np.sum(comp != ref_row)
        
        if len(exact_match_indices) != 0:
            idx_opt = min(exact_match_indices)
        else:
            min_distance_indices = hamming_distances.argmin()
            idx_opt = min_distance_indices.min()
        
        # draw PPD bar chart
        drawPPDBarChart(IndicatorMatrix,Heights,lam_vec,idx_opt)
        #draw PPD surface
        drawPPDSurface(self.time,lam_vec,FNm,Heights,Locs,IndicatorMatrix,Labels,idx_opt)
        self.lam_opt = lam_vec[idx_opt]

    def align_new(self, f_new, smoothdata=None):
        """
        Aligns new functions to the fitted template srsf (mqn) with the same
        lam, omethod, penalty and grid_dim used to fit the object

        Usage:  fn, qn, gam = obj.align_new(f_new)

        :param f_new: numpy ndarray of shape (M,K) of K new functions sampled
                      at obj.time (a vector is a single function)
        :param smoothdata: smooth the data when computing the srsfs
                           (default = None, the value used for the fit)

        :rtype: tuple of numpy ndarray
        :return fn: (M,K) aligned functions
        :return qn: (M,K) aligned srsfs
        :return gam: (M,K) warping functions
        """
        if not hasattr(self, "mqn"):
            raise Exception(
                "Please align fdawarp class using srsf_align or "
                "multiple_align_functions first!"
            )
        if smoothdata is None:
            smoothdata = getattr(self, "smoothdata", False)
        grid_dim = getattr(self, "grid_dim", 10)
        penalty = getattr(self, "penalty", "l2psi")

        f_new = np.asarray(f_new, dtype=float)
        if f_new.ndim == 1:
            f_new = f_new.reshape(-1, 1)
        M, K = f_new.shape
        if M != self.time.shape[0]:
            raise Exception("f_new must have len(obj.time) rows")

        eps = np.finfo(np.double).eps
        f, g, g2 = uf.gradient_spline(self.time, f_new, smoothdata)
        q_new = g / np.sqrt(abs(g) + eps)

        fn = np.zeros((M, K))
        gam = np.zeros((M, K))
        for k in range(0, K):
            gam[:, k] = uf.optimum_reparam(
                self.mqn, self.time, q_new[:, k], self.method, self.lam,
                penalty, grid_dim
            )
            fn[:, k] = uf.warp_f_gamma(self.time, f_new[:, k], gam[:, k])
        qn = uf.f_to_srsf(fn, self.time)

        return fn, qn, gam

    def plot(self):
        """
        plot functional alignment results

        Usage: obj.plot()
        """

        M = self.f.shape[0]
        plot.f_plot(self.time, self.f, title="f Original Data")

        fig, ax = plot.f_plot(
            np.arange(0, M) / float(M - 1), self.gam, title="Warping Functions"
        )
        ax.set_aspect("equal")

        plot.f_plot(self.time, self.fn, title="Warped Data")

        mean_f0 = self.f.mean(axis=1)
        std_f0 = self.f.std(axis=1)
        mean_fn = self.fn.mean(axis=1)
        std_fn = self.fn.std(axis=1)
        tmp = np.array([mean_f0, mean_f0 + std_f0, mean_f0 - std_f0])
        tmp = tmp.transpose()
        fig, ax = plt.subplots()
        ax.plot(self.time, mean_f0, 'k')
        ax.plot(self.time, mean_f0 + std_f0, color="#66C2A5")
        ax.plot(self.time, mean_f0 - std_f0, color="#66C2A5")
        plt.title(r"Original Data: Mean $\pm$ STD")
        plt.style.use("seaborn-v0_8-colorblind")

        fig, ax = plt.subplots()
        ax.plot(self.time, mean_fn, 'k')
        ax.plot(self.time, mean_fn + std_fn, color="#66C2A5")
        ax.plot(self.time, mean_fn - std_fn, color="#66C2A5")
        plt.title(r"Warped Data: Mean $\pm$ STD")
        plt.style.use("seaborn-v0_8-colorblind")

        plot.f_plot(self.time, self.fmean, title="$f_{mean}$")

        # MCMC diagnostics when the Bayesian alignment was used
        mc = getattr(self, "mcmc", None)
        if isinstance(mc, dict) and ("sigma1" in mc or "sigma" in mc):
            self.plot_mcmc(show=False)

        plt.show()

        return

    def plot_mcmc(self, show=True):
        """
        plot diagnostics of the Bayesian alignment MCMC chain

        Plots trace, posterior and mixing diagnostics of the chain stored in
        self.mcmc by time_warping_bayes (pCN) or time_warping_bayes_infHMC.
        Plots that need the extra information (extrainfo=True) are only
        drawn when it is available.

        Figures:
          chain diagnostics: sigma trace, log-likelihood trace, posterior of
            sigma, autocorrelation of sigma (with effective sample size),
            running acceptance rate and pCN step usage (pCN) or HMC step
            sizes per function (infHMC)
          warping coefficient norm trace for each function
          posterior template with 95% credible band
          posterior warping functions with 95% credible bands

        :param show: call plt.show() at the end (default = True)

        Usage: obj.plot_mcmc()
        """
        mc = getattr(self, "mcmc", None)
        if not isinstance(mc, dict) or not ("sigma1" in mc or "sigma" in mc):
            raise Exception(
                "No MCMC output available, run time_warping_bayes or "
                "time_warping_bayes_infHMC first"
            )
        is_hmc = "epsilon" in mc

        if "sigma1" in mc:
            sigma1 = np.ravel(mc["sigma1"])
        else:
            sigma1 = np.ravel(mc["sigma"])
        niter = sigma1.shape[0]
        if "burnin" in mc:
            burnin = int(mc["burnin"])
        elif "q_star_mat" in mc:
            burnin = niter - mc["q_star_mat"].shape[1] + 1
        else:
            burnin = niter // 2
        burnin = min(max(burnin, 1), niter)
        if "logl" in mc:
            logl = np.ravel(mc["logl"])
        else:
            logl = np.ravel(getattr(self, "qun", np.zeros(0)))
        has_extra = "accept" in mc and np.size(mc["accept"]) > 0
        # iterations are 1-based in the labels, post burn-in includes burnin
        it_all = np.arange(1, niter + 1)
        post = sigma1[burnin - 1 :]
        col = "#66C2A5"

        def burnline(ax):
            ax.axvline(burnin, color="k", linestyle="--", linewidth=1)
            ax.text(
                burnin,
                1.0,
                " burn-in",
                transform=ax.get_xaxis_transform(),
                va="top",
                ha="left",
                fontsize=8,
            )

        def not_stored(ax, what):
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "%s not stored\n(set extrainfo=True)" % what,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # chain diagnostics
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        fig.suptitle("MCMC diagnostics", fontsize=16)

        ax = axs[0, 0]
        ax.plot(it_all, sigma1, color=col, linewidth=1)
        burnline(ax)
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\sigma$")
        ax.set_title(r"Trace of $\sigma$")

        ax = axs[0, 1]
        if logl.shape[0] == niter:
            ax.plot(it_all, logl, color=col, linewidth=1)
            burnline(ax)
            ax.set_xlabel("iteration")
            ax.set_ylabel("log-likelihood")
        else:
            not_stored(ax, "log-likelihood")
        ax.set_title("Trace of log-likelihood")

        ax = axs[0, 2]
        ax.hist(post, bins="auto", density=True, color=col)
        ax.axvline(post.mean(), color="k", linewidth=1.5)
        for qv in np.quantile(post, [0.025, 0.975]):
            ax.axvline(qv, color="k", linestyle="--")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel("density")
        ax.set_title(r"Posterior of $\sigma$ (95% CI)")

        ax = axs[1, 0]
        maxlag = min(50, post.shape[0] - 1)
        rho = _acf_vec(post, maxlag)
        lags = np.arange(0, maxlag + 1)
        ax.vlines(lags, 0, rho, color=col)
        ax.plot(lags, rho, "o", color=col, markersize=3)
        ci = 1.96 / np.sqrt(post.shape[0])
        ax.axhline(ci, color="k", linestyle="--")
        ax.axhline(-ci, color="k", linestyle="--")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("lag")
        ax.set_ylabel("ACF")
        ess = post.shape[0] / max(1 + 2 * rho[1:].sum(), 1)
        ax.set_title(r"ACF of $\sigma$ (ESS $\approx$ %.0f)" % ess)

        ax = axs[1, 1]
        if has_extra:
            acc = np.asarray(mc["accept"], dtype=float)
            if is_hmc:
                # HMC stores (N,iter-1) indicators: average over functions
                acc = acc.mean(axis=0)
                lab = "warps"
            else:
                lab = "$q^*$"
            acc = np.ravel(acc)
            it = np.arange(1, acc.shape[0] + 1) + 1
            ax.plot(
                it,
                np.cumsum(acc) / np.arange(1, acc.shape[0] + 1),
                color=col,
                linewidth=1,
            )
            burnline(ax)
            ax.set_ylim(0, 1)
            ax.set_xlabel("iteration")
            ax.set_ylabel("rate")
            ax.set_title("Running acceptance of %s (%.2f)" % (lab, acc.mean()))
        else:
            not_stored(ax, "acceptance")
            ax.set_title("Running acceptance of $q^*$")

        ax = axs[1, 2]
        if is_hmc:
            eps_tr = np.atleast_2d(mc["epsilon"])
            ax.semilogy(np.arange(1, eps_tr.shape[1] + 1), eps_tr.T, linewidth=1)
            burnline(ax)
            ax.set_xlabel("iteration")
            ax.set_ylabel(r"$\epsilon$")
            ax.set_title("HMC step size per function")
        elif has_extra and "betas_ind" in mc:
            bind = np.ravel(mc["betas_ind"]).astype(int)
            acc = np.ravel(mc["accept"]).astype(bool)
            if "zpcn" in mc:
                nb = len(mc["zpcn"]["betas"])
            else:
                nb = bind.max() + 1
            nprop = np.bincount(bind, minlength=nb)[:nb]
            nacc = np.bincount(bind[acc], minlength=nb)[:nb]
            x = np.arange(nb)
            ax.bar(x, nprop - nacc, label="rejected")
            ax.bar(x, nacc, bottom=nprop - nacc, label="accepted")
            ax.set_xticks(x)
            if "zpcn" in mc:
                ax.set_xticklabels(["%g" % b for b in mc["zpcn"]["betas"]])
                ax.set_xlabel(r"pCN $\beta$")
            else:
                ax.set_xlabel("pCN step index")
            ax.set_ylabel("proposals")
            ax.legend(loc="best")
            rate = nacc / np.maximum(nprop, 1)
            for xi, npi, ri in zip(x, nprop, rate):
                ax.text(xi, npi, "%.2f" % ri, ha="center", va="bottom")
            ax.set_ylim(0, 1.15 * max(nprop.max(), 1))
            ax.set_title("$q^*$ proposals by pCN step (acc. rate)")
        else:
            not_stored(ax, "pCN step usage")
            ax.set_title("pCN step usage")

        # warping coefficient traces
        if "g_coef" in mc and np.size(mc["g_coef"]) > 0:
            # pCN stores (iter,D,N) draws
            gnorm = np.sqrt((np.asarray(mc["g_coef"]) ** 2).sum(axis=1))
            fig, ax = plt.subplots()
            ax.plot(np.arange(1, gnorm.shape[0] + 1), gnorm, linewidth=1)
            burnline(ax)
            ax.set_xlabel("iteration")
            ax.set_ylabel("norm of warping coefficients")
            ax.set_title("Trace of warping coefficient norm per function")
        elif "v_coef" in mc and np.size(mc["v_coef"]) > 0:
            # infHMC stores (D,N,nsamp) post burn-in draws only
            vnorm = np.sqrt((np.asarray(mc["v_coef"]) ** 2).sum(axis=0)).T
            fig, ax = plt.subplots()
            ax.plot(np.arange(1, vnorm.shape[0] + 1), vnorm, linewidth=1)
            ax.set_xlabel("posterior draw (after burn-in)")
            ax.set_ylabel("norm of warping coefficients")
            ax.set_title("Trace of warping coefficient norm per function")

        if "fmean_mat" in mc and "gamma_mat" in mc:
            # posterior template
            fmean_mat = mc["fmean_mat"]
            ts = np.linspace(self.time[0], self.time[-1], fmean_mat.shape[0])
            ndraw = fmean_mat.shape[1]
            idx = np.unique(
                np.round(np.linspace(0, ndraw - 1, min(ndraw, 100))).astype(int)
            )
            fig, ax = plt.subplots()
            ax.plot(ts, fmean_mat[:, idx], color="0.7", alpha=0.3, linewidth=0.5)
            ax.fill_between(
                ts,
                mc["fmean_stats"][:, 0],
                mc["fmean_stats"][:, 1],
                color=col,
                alpha=0.35,
                linewidth=0,
            )
            ax.plot(self.time, self.fmean, "k-", linewidth=2, label="posterior mean")
            ax.legend(loc="best")
            ax.set_title("Posterior template with 95% credible band")

            # posterior warping functions
            gamma_mat = mc["gamma_mat"]
            N = gamma_mat.shape[2]
            tg = np.linspace(0, 1, gamma_mat.shape[0])
            cmap = plt.get_cmap("tab10")
            fig, ax = plt.subplots()
            for jj in range(N):
                band = np.quantile(gamma_mat[:, :, jj], [0.025, 0.975], axis=1)
                ax.fill_between(
                    tg, band[0], band[1], color=cmap(jj % 10), alpha=0.2, linewidth=0
                )
            for jj in range(N):
                ax.plot(tg, self.gam[:, jj], color=cmap(jj % 10), linewidth=1)
            ax.plot([0, 1], [0, 1], "k:")
            ax.set_aspect("equal")
            ax.set_title("Posterior warping functions with 95% credible bands")

        if show:
            plt.show()

        return

    def gauss_model(self, n=1, sort_samples=False):
        """
        This function models the functional data using a Gaussian model
        extracted from the principal components of the srvfs

        :param n: number of random samples
        :param sort_samples: sort samples (default = T)
        :type n: integer
        :type sort_samples: bool
        """
        fn = self.fn
        time = self.time
        qn = self.qn
        gam = self.gam

        # Parameters
        eps = np.finfo(np.double).eps
        binsize = np.diff(time)
        binsize = binsize.mean()
        M = time.size

        # compute mean and covariance in q-domain
        mq_new = qn.mean(axis=1)
        mididx = int(np.round(time.shape[0] / 2))
        m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
        mqn = np.append(mq_new, m_new.mean())
        qn2 = np.vstack((qn, m_new))
        C = np.cov(qn2)

        q_s = np.random.multivariate_normal(mqn, C, n)
        q_s = q_s.transpose()

        # compute the correspondence to the original function domain
        fs = np.zeros((M, n))
        for k in range(0, n):
            fs[:, k] = uf.cumtrapzmid(
                time,
                q_s[0:M, k] * np.abs(q_s[0:M, k]),
                np.sign(q_s[M, k]) * (q_s[M, k] ** 2),
                mididx,
            )
        fbar = fn.mean(axis=1)

        fsbar = fs.mean(axis=1)
        err = np.transpose(np.tile(fbar - fsbar, (n, 1)))
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
                len[i] = np.arccos(tmp.dot(psi[:, i] / M))

            seq2 = len.argsort()

            # combine x-variability and y-variability
            ft = np.zeros((M, n))
            for k in range(0, n):
                ft[:, k] = np.interp(
                    gams[:, seq2[k]], np.arange(0, M) / np.double(M - 1), fs[:, seq1[k]]
                )
                tmp = np.isnan(ft[:, k])
                while tmp.any():
                    rgam2 = uf.randomGamma(gam, 1)
                    ft[:, k] = np.interp(
                        gams[:, seq2[k]],
                        np.arange(0, M) / np.double(M - 1),
                        uf.invertGamma(rgam2),
                    )
        else:
            # combine x-variability and y-variability
            ft = np.zeros((M, n))
            for k in range(0, n):
                ft[:, k] = np.interp(
                    gams[:, k], np.arange(0, M) / np.double(M - 1), fs[:, k]
                )
                tmp = np.isnan(ft[:, k])
                while tmp.any():
                    rgam2 = uf.randomGamma(gam, 1)
                    ft[:, k] = np.interp(
                        gams[:, k],
                        np.arange(0, M) / np.double(M - 1),
                        uf.invertGamma(rgam2),
                    )

        self.rsamps = True
        self.fs = fs
        self.gams = rgam
        self.ft = ft
        self.qs = q_s[0:M, :]

        return

    def joint_gauss_model(self, n=1, no=3):
        """
        This function models the functional data using a joint Gaussian model
        extracted from the principal components of the srsfs

        :param n: number of random samples
        :param no: number of principal components (default = 3)
        :type n: integer
        :type no: integer
        """

        # Parameters
        fn = self.fn
        time = self.time
        qn = self.qn

        M = time.size

        # Perform PCA
        jfpca = fpca.fdajpca(self)
        jfpca.calc_fpca(no=no)
        s = jfpca.latent
        U = jfpca.U
        C = jfpca.C
        mu_psi = jfpca.mu_psi

        # compute mean and covariance
        mq_new = qn.mean(axis=1)
        mididx = jfpca.id
        m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
        mqn = np.append(mq_new, m_new.mean())

        # generate random samples
        vals = np.random.multivariate_normal(np.zeros(s.shape), np.diag(s), n)

        tmp = np.matmul(U, np.transpose(vals))
        qhat = np.tile(mqn.T, (n, 1)).T + tmp[0 : M + 1, :]
        tmp = np.matmul(U, np.transpose(vals) / C)
        vechat = tmp[(M + 1) :, :]
        psihat = np.zeros((M, n))
        gamhat = np.zeros((M, n))
        for ii in range(n):
            psihat[:, ii] = geo.exp_map(mu_psi, vechat[:, ii])
            gam_tmp = cumulative_trapezoid(psihat[:, ii] ** 2, np.linspace(0, 1, M), initial=0.0)
            gamhat[:, ii] = (gam_tmp - gam_tmp.min()) / (gam_tmp.max() - gam_tmp.min())

        ft = np.zeros((M, n))
        fhat = np.zeros((M, n))
        for ii in range(n):
            fhat[:, ii] = uf.cumtrapzmid(
                time,
                qhat[0:M, ii] * np.fabs(qhat[0:M, ii]),
                np.sign(qhat[M, ii]) * (qhat[M, ii] * qhat[M, ii]),
                mididx,
            )
            ft[:, ii] = uf.warp_f_gamma(
                np.linspace(0, 1, M), fhat[:, ii], gamhat[:, ii]
            )

        self.rsamps = True
        self.fs = fhat
        self.gams = gamhat
        self.ft = ft
        self.qs = qhat[0:M, :]

        return

    def multiple_align_functions(
        self,
        mu,
        omethod="DP2",
        smoothdata=False,
        parallel=False,
        lam=0.0,
        pen="l2psi",
        cores=-1,
        grid_dim=10,
    ):
        """
        This function aligns a collection of functions using the elastic
        square-root slope (srsf) framework.

        Usage:  obj.multiple_align_functions(mu)

        :param mu: vector of function to align to
        :param omethod: optimization method (DP, DP2, RBFGS, cRBFGS)
                        (default = DP2)
        :param smoothdata: Smooth the data using a box filter (default = F)
        :param parallel: run in parallel (default = F)
        :param lam: controls the elasticity (default = 0)
        :param penalty: penalty type (default="l2psi") options are "roughness",
                "l2gam", "l2psi", "geodesic" and "none". The penalty is
                weighted by lam, so it has no effect when lam is 0.
                "roughness" and "geodesic" have no dynamic programming
                counterpart, so with a nonzero lam they need omethod
                "RBFGS" or "cRBFGS"
        :param cores: number of cores for parallel (default = -1 (all))
        :param grid_dim: size of the grid, for the DP2 method only
                         (default = 10)
        :type lam: double
        :type smoothdata: bool

        """

        M = self.f.shape[0]
        N = self.f.shape[1]
        self.lam = lam
        self.penalty = pen
        self.grid_dim = grid_dim
        self.smoothdata = smoothdata

        if M > 500:
            parallel = True
        elif N > 100:
            parallel = True

        eps = np.finfo(np.double).eps
        self.method = omethod
        self.type = "multiple"
        self.mcmc = None

        # Compute SRSF function from data
        f, g, g2 = uf.gradient_spline(self.time, self.f, smoothdata)
        q = g / np.sqrt(abs(g) + eps)

        mq = uf.f_to_srsf(mu, self.time)

        if parallel:
            out = Parallel(n_jobs=cores)(
                delayed(uf.optimum_reparam)(
                    mq, self.time, q[:, n], omethod, lam, pen, grid_dim
                )
                for n in range(N)
            )
            gam = np.array(out)
            gam = gam.transpose()
        else:
            gam = np.zeros((M, N))
            for k in range(0, N):
                gam[:, k] = uf.optimum_reparam(
                    mq, self.time, q[:, k], omethod, lam, pen, grid_dim
                )

        self.gamI = uf.SqrtMeanInverse(gam)

        fn = np.zeros((M, N))
        qn = np.zeros((M, N))
        for k in range(0, N):
            fn[:, k] = np.interp(
                (self.time[-1] - self.time[0]) * gam[:, k] + self.time[0],
                self.time,
                f[:, k],
            )
            qn[:, k] = uf.f_to_srsf(f[:, k], self.time)

        # Aligned data & stats
        self.fn = fn
        self.qn = qn
        self.q0 = q
        std_f0 = f.std(axis=1, ddof=1)
        std_fn = self.fn.std(axis=1, ddof=1)
        self.gam = gam
        self.mqn = mq
        self.fmean = mu

        fgam = np.zeros((M, N))
        for k in range(0, N):
            time0 = (self.time[-1] - self.time[0]) * gam[:, k] + self.time[0]
            fgam[:, k] = np.interp(time0, self.time, self.fmean)

        var_fgam = fgam.var(axis=1, ddof=1)
        self.orig_var = trapezoid(std_f0**2, self.time)
        self.amp_var = trapezoid(std_fn**2, self.time)
        self.phase_var = trapezoid(var_fgam, self.time)

        return


def _acf_vec(x, maxlag):
    """sample autocorrelation of x for lags 0..maxlag"""
    x = np.ravel(x) - np.mean(x)
    n = x.shape[0]
    c0 = (x**2).sum()
    rho = np.zeros(maxlag + 1)
    if c0 == 0:
        rho[0] = 1
        return rho
    for k in range(maxlag + 1):
        rho[k] = (x[: n - k] * x[k:]).sum() / c0
    return rho


def pairwise_align_functions(
    f1, f2, time, omethod="DP2", lam=0, pen="l2psi", grid_dim=10
):
    """
    This function aligns f2 to f1 using the elastic square-root
        slope (srsf) framework.

    Usage:  out = pairwise_align_functions(f1, f2, time)
            out = pairwise_align_functions(f1, f2, time, omethod, lam,
                                           grid_dim)

    :param f1: vector defining M samples of function 1
    :param f2: vector defining M samples of function 2
    :param time: time vector of length M
    :param omethod: optimization method (DP, DP2, RBFGS, cRBFGS) (default = DP)
    :param lam: controls the elasticity (default = 0)
    :param penalty: penalty type (default="l2psi") options are "roughness",
                "l2gam", "l2psi", "geodesic" and "none". The penalty is
                weighted by lam, so it has no effect when lam is 0.
                "roughness" and "geodesic" have no dynamic programming
                counterpart, so with a nonzero lam they need omethod
                "RBFGS" or "cRBFGS"
    :param grid_dim: size of the grid, for the DP2 method only (default = 10)

    :rtype list containing
    :return f2n: aligned f2
    :return gam: warping function
    :return q2n: aligned q2 (srsf)

    """

    q1 = uf.f_to_srsf(f1, time)
    q2 = uf.f_to_srsf(f2, time)

    gam = uf.optimum_reparam(q1, time, q2, omethod, lam, pen, grid_dim)

    f2n = uf.warp_f_gamma(time, f2, gam)
    q2n = uf.f_to_srsf(f2n, time)

    return (f2n, gam, q2n)


def pairwise_align_bayes(f1i, f2i, time, mcmcopts=None):
    """
    This function aligns two functions using Bayesian framework. It will align
    f2 to f1. It is based on mapping warping functions to a hypersphere, and a
    subsequent exponential mapping to a tangent space. In the tangent space,
    the Z-mixture pCN algorithm is used to explore both local and global
    structure in the posterior distribution.

    The Z-mixture pCN algorithm uses a mixture distribution for the proposal
    distribution, controlled by input parameter zpcn. The zpcn$betas must be
    between 0 and 1, and are the coefficients of the mixture components, with
    larger coefficients corresponding to larger shifts in parameter space. The
    zpcn["probs"] give the probability of each shift size.

    Usage:  out = pairwise_align_bayes(f1i, f2i, time)

    :param f1i: vector defining M samples of function 1
    :param f2i: vector defining M samples of function 2
    :param time: time vector of length M
    :param mcmopts: dict of mcmc parameters
    :type mcmcopts: dict

    default mcmc options:
    tmp = {"betas":np.array([0.5,0.5,0.005,0.0001]),"probs":np.array([0.1,0.1,0.7,0.1])}
    mcmcopts = {"iter":2*(10**4) ,"burnin":np.minimum(5*(10**3),2*(10**4)//2), "alpha0":0.1, "beta0":0.1,"zpcn":tmp,"propvar":1, "initcoef":np.repeat(0,20), "npoints":200, "extrainfo":True}

    :rtype collection containing
    :return f2_warped: aligned f2
    :return gamma: warping function
    :return g_coef: final g_coef
    :return psi: final psi
    :return sigma1: final sigma

    if extrainfo
    :return accept: accept of psi samples
    :return betas_ind
    :return logl: log likelihood
    :return gamma_mat: posterior gammas
    :return gamma_stats: posterior gamma stats
    :return xdist: phase distance posterior
    :return ydist: amplitude distance posterior)
    """

    if mcmcopts is None:
        tmp = {
            "betas": np.array([0.5, 0.5, 0.005, 0.0001]),
            "probs": np.array([0.1, 0.1, 0.7, 0.1]),
        }
        mcmcopts = {
            "iter": 2 * (10**4),
            "burnin": np.minimum(5 * (10**3), 2 * (10**4) // 2),
            "alpha0": 0.1,
            "beta0": 0.1,
            "zpcn": tmp,
            "propvar": 1,
            "initcoef": np.repeat(0, 20),
            "npoints": 200,
            "extrainfo": True,
        }

    if f1i.shape[0] != f2i.shape[0]:
        raise Exception("Length of f1 and f2 must be equal")

    if f1i.shape[0] != time.shape[0]:
        raise Exception("Length of f1 and time must be equal")

    if mcmcopts["zpcn"]["betas"].shape[0] != mcmcopts["zpcn"]["probs"].shape[0]:
        raise Exception("In zpcn, betas must equal length of probs")

    if np.mod(mcmcopts["initcoef"].shape[0], 2) != 0:
        raise Exception("Length of mcmcopts.initcoef must be even")

    # Number of sig figs to report in gamma_mat
    iter = mcmcopts["iter"]

    # parameter settings
    pw_sim_global_burnin = mcmcopts["burnin"]
    valid_index = np.arange(pw_sim_global_burnin - 1, iter)
    pw_sim_global_Mg = mcmcopts["initcoef"].shape[0] // 2
    g_coef_ini = mcmcopts["initcoef"]
    numSimPoints = mcmcopts["npoints"]
    pw_sim_global_domain_par = np.linspace(0, 1, numSimPoints)
    g_basis = uf.basis_fourier(pw_sim_global_domain_par, pw_sim_global_Mg, 1)
    sigma1_ini = 1
    zpcn = mcmcopts["zpcn"]
    pw_sim_global_sigma_g = mcmcopts["propvar"]

    def propose_g_coef(g_coef_curr):
        pCN_beta = zpcn["betas"]
        pCN_prob = zpcn["probs"]
        probm = np.insert(np.cumsum(pCN_prob), 0, 0)
        z = np.random.rand()
        result = {"prop": g_coef_curr, "ind": 1}
        for i in range(0, pCN_beta.shape[0]):
            if z <= probm[i + 1] and z > probm[i]:
                g_coef_new = normal(
                    0,
                    pw_sim_global_sigma_g
                    / np.repeat(np.arange(1, pw_sim_global_Mg + 1), 2),
                )
                result["prop"] = (
                    np.sqrt(1 - pCN_beta[i] ** 2) * g_coef_curr
                    + pCN_beta[i] * g_coef_new
                )
                result["ind"] = i

        return result

    # normalize time to [0,1]
    time = (time - time.min()) / (time.max() - time.min())
    timet = np.linspace(0, 1, numSimPoints)
    f1 = uf.f_predictfunction(f1i, timet, 0)
    f2 = uf.f_predictfunction(f2i, timet, 0)

    # srsf transformation
    q1 = uf.f_to_srsf(f1, timet)
    q1i = uf.f_to_srsf(f1i, time)
    q2 = uf.f_to_srsf(f2, timet)

    tmp = uf.f_exp1(uf.f_basistofunction(g_basis["x"], 0, g_coef_ini, g_basis))

    if tmp.min() < 0:
        raise Exception("Invalid initial value of g")

    # result vectors
    g_coef = np.zeros((iter, g_coef_ini.shape[0]))
    sigma1 = np.zeros(iter)
    logl = np.zeros(iter)
    SSE = np.zeros(iter)
    accept = np.zeros(iter, dtype=bool)
    accept_betas = np.zeros(iter)

    # init
    g_coef_curr = g_coef_ini
    sigma1_curr = sigma1_ini
    SSE_curr = bf.f_SSEg_pw(
        uf.f_basistofunction(g_basis["x"], 0, g_coef_ini, g_basis), q1, q2
    )
    logl_curr = bf.f_logl_pw(
        uf.f_basistofunction(g_basis["x"], 0, g_coef_ini, g_basis),
        q1,
        q2,
        sigma1_ini**2,
        SSE_curr,
    )

    g_coef[0, :] = g_coef_ini
    sigma1[0] = sigma1_ini
    SSE[0] = SSE_curr
    logl[0] = logl_curr

    # update the chain for iter-1 times
    for m in tqdm(range(1, iter)):
        # update g
        g_coef_curr, tmp, SSE_curr, accepti, zpcnInd = bf.f_updateg_pw(
            g_coef_curr, g_basis, sigma1_curr**2, q1, q2, SSE_curr, propose_g_coef
        )

        # update sigma1
        newshape = q1.shape[0] / 2 + mcmcopts["alpha0"]
        newscale = 1 / 2 * SSE_curr + mcmcopts["beta0"]
        sigma1_curr = np.sqrt(1 / np.random.gamma(newshape, 1 / newscale))
        logl_curr = bf.f_logl_pw(
            uf.f_basistofunction(g_basis["x"], 0, g_coef_curr, g_basis),
            q1,
            q2,
            sigma1_curr**2,
            SSE_curr,
        )

        # save updates to results
        g_coef[m, :] = g_coef_curr
        sigma1[m] = sigma1_curr
        SSE[m] = SSE_curr
        if mcmcopts["extrainfo"]:
            logl[m] = logl_curr
            accept[m] = accepti
            accept_betas[m] = zpcnInd

    # calculate posterior mean of psi
    pw_sim_est_psi_matrix = np.zeros((numSimPoints, valid_index.shape[0]))
    for k in range(0, valid_index.shape[0]):
        g_temp = uf.f_basistofunction(
            g_basis["x"], 0, g_coef[valid_index[k], :], g_basis
        )
        psi_temp = uf.f_exp1(g_temp)
        pw_sim_est_psi_matrix[:, k] = psi_temp

    result_posterior_psi_simDomain = uf.f_psimean(
        pw_sim_global_domain_par, pw_sim_est_psi_matrix
    )

    # resample to same number of points as the input f1 and f2
    interp = interp1d(
        np.linspace(0, 1, result_posterior_psi_simDomain.shape[0]),
        result_posterior_psi_simDomain,
        fill_value="extrapolate",
    )
    result_posterior_psi = interp(np.linspace(0, 1, f1i.shape[0]))

    # transform posterior mean of psi to gamma
    result_posterior_gamma = uf.f_phiinv(result_posterior_psi)
    result_posterior_gamma = uf.norm_gam(result_posterior_gamma)

    # warped f2
    f2_warped = uf.warp_f_gamma(time, f2i, result_posterior_gamma)

    if mcmcopts["extrainfo"]:
        M, N = pw_sim_est_psi_matrix.shape
        gamma_mat = np.zeros((time.shape[0], N))
        one_v = np.ones(M)
        Dx = np.zeros(N)
        Dy = Dx
        for ii in range(0, N):
            interp = interp1d(
                np.linspace(0, 1, result_posterior_psi_simDomain.shape[0]),
                pw_sim_est_psi_matrix[:, ii],
                fill_value="extrapolate",
            )
            result_i = interp(time)
            tmp = uf.f_phiinv(result_i)
            gamma_mat[:, ii] = uf.norm_gam(tmp)
            v, theta = geo.inv_exp_map(one_v, pw_sim_est_psi_matrix[:, ii])
            Dx[ii] = np.sqrt(trapezoid(v**2, pw_sim_global_domain_par))
            q2warp = uf.warp_q_gamma(pw_sim_global_domain_par, q2, gamma_mat[:, ii])
            Dy[ii] = np.sqrt(trapezoid((q1i - q2warp) ** 2, time))

        gamma_stats = uf.statsFun(gamma_mat)

    results_o = collections.namedtuple(
        "align_bayes",
        [
            "f2_warped",
            "gamma",
            "g_coef",
            "psi",
            "sigma1",
            "accept",
            "betas_ind",
            "logl",
            "gamma_mat",
            "gamma_stats",
            "xdist",
            "ydist",
        ],
    )

    out = results_o(
        f2_warped,
        result_posterior_gamma,
        g_coef,
        result_posterior_psi,
        sigma1,
        accept[1:],
        accept_betas[1:],
        logl,
        gamma_mat,
        gamma_stats,
        Dx,
        Dy,
    )

    return out


def pairwise_align_bayes_infHMC(y1i, y2i, time, mcmcopts=None):
    r"""
    This function aligns two functions using Bayesian framework. It uses a
    hierarchical Bayesian framework assuming measurement error. It will
    align f2 to f1. It is based on mapping warping functions to a hypersphere,
    and a subsequent exponential mapping to a tangent space. In the tangent
    space, the :math:`\infty`-HMC algorithm is used to explore both local and
    global structure in the posterior distribution.

    Usage:  out = pairwise_align_bayes_infHMC(f1i, f2i, time)

    :param y1i: vector defining M samples of function 1
    :param y2i: vector defining M samples of function 2
    :param time: time vector of length M
    :param mcmcopts: dict of mcmc parameters, missing entries take the
                     default below
    :type mcmcopts: dict

    default mcmc options:
    mcmcopts = {"iter":1*(10**4), "nchains":1, "burnin":None (min(5e3,iter/2)),
    "thin":50, "alpha0":0.1, "beta0":0.1, "alpha":1, "beta":1, "h":0.01,
    "L":4, "adapt":True, "target_accept":0.65, "f1propvar":0.0001,
    "f2propvar":0.0001, "L1propvar":0.3, "L2propvar":0.3, "vpriorvar":1,
    "vpriordecay":1, "initcoef":np.repeat(0,20), "basis":'fourier',
    "warp_rep":'shooting', "init":'dp', "sampfreq":5, "extrainfo":True,
    "parallel":True, "seed":None}

    init = 'dp' starts chain 1 at the DP warp of the GP-smoothed functions
    ('coef' starts it at initcoef); further chains start at random
    coefficients.  Chains started at the identity can settle in a poor local
    mode.

    Basis can be 'fourier' or 'legendre'

    Warping functions: with warp_rep = 'shooting', :math:`v = \Phi c` is a
    shooting vector at the identity of the Hilbert sphere of SRSFs of warps
    and must satisfy :math:`|v| < \pi` (draws outside are rejected); with
    warp_rep = 'clr', :math:`h = \Phi c` is the centered log-derivative,
    :math:`\gamma' = \exp(h)/\int \exp(h)`, which is unconstrained. The
    likelihood gradient with respect to c is exact (see
    :func:`fdasrsf.bayesian_functions.bayes_warp_geom`) and v is updated with
    the preconditioned HMC of Beskos et al. (2011) (see
    :func:`fdasrsf.bayesian_functions.infhmc_step`). time is assumed to be
    uniformly sampled.

    :rtype collection containing
    :return f2_warped: aligned f2
    :return gamma: warping function
    :return v_coef: final v_coef
    :return psi: final psi
    :return sigma1: final sigma
    :return epsilon: HMC step size (rotation angle) at each iteration

    if extrainfo
    :return theta_accept: accept of psi samples
    :return f2_accept: accept of f2 samples
    :return SSE: SSE
    :return gamma_mat: posterior gammas
    :return gamma_stats: posterior gamma stats
    :return phasedist: phase distance posterior
    :return ampdist: amplitude distance posterior

    J. D. Tucker, L. Shand, and K. Chowdhary. “Multimodal Bayesian Registration
    of Noisy Functions using Hamiltonian Monte Carlo”, Computational Statistics
    and Data Analysis, accepted, 2021.
    """

    defaults = {
        "iter": 1 * (10**4),
        "nchains": 1,
        "burnin": None,
        "thin": 50,
        "alpha0": 0.1,
        "beta0": 0.1,
        "alpha": 1,
        "beta": 1,
        "h": 0.01,
        "L": 4,
        "adapt": True,
        "target_accept": 0.65,
        "f1propvar": 0.0001,
        "f2propvar": 0.0001,
        "L1propvar": 0.3,
        "L2propvar": 0.3,
        "vpriorvar": 1,
        "vpriordecay": 1,
        "initcoef": np.repeat(0, 20),
        "basis": "fourier",
        "warp_rep": "shooting",
        "init": "dp",
        "sampfreq": 5,
        "extrainfo": True,
        "parallel": True,
        "seed": None,
    }

    if mcmcopts is None:
        mcmcopts = {}
    opts = defaults.copy()
    opts.update(mcmcopts)
    mcmcopts = opts

    y1i = np.asarray(y1i, dtype=float).ravel()
    y2i = np.asarray(y2i, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()

    if y1i.shape[0] != y2i.shape[0]:
        raise Exception("Length of f1 and f2 must be equal")

    if y1i.shape[0] != time.shape[0]:
        raise Exception("Length of f1 and time must be equal")

    initcoef = np.asarray(mcmcopts["initcoef"], dtype=float).ravel()
    mcmcopts["initcoef"] = initcoef
    if np.mod(initcoef.shape[0], 2) != 0:
        raise Exception("Length of mcmcopts.initcoef must be even")

    if str(mcmcopts["warp_rep"]).lower() not in ("shooting", "clr"):
        raise Exception("mcmcopts.warp_rep must be 'shooting' or 'clr'")

    if str(mcmcopts["init"]).lower() not in ("dp", "coef"):
        raise Exception("mcmcopts.init must be 'dp' or 'coef'")

    iter = int(mcmcopts["iter"])
    if mcmcopts["burnin"] is None:
        mcmcopts["burnin"] = int(np.minimum(5 * (10**3), iter // 2))
    burnin = int(mcmcopts["burnin"])
    mcmcopts["burnin"] = burnin
    if burnin < 1 or burnin >= iter:
        raise Exception("mcmcopts.burnin must be between 1 and mcmcopts.iter-1")

    nchains = int(mcmcopts["nchains"])

    # set up random start points for more than 1 chain
    random_starts = np.tile(initcoef.reshape((-1, 1)), (1, nchains))
    if nchains > 1:
        rng = np.random.default_rng(mcmcopts["seed"])
        for i in range(0, nchains):
            randcoef = -1 + 2 * rng.random(initcoef.shape[0])
            if (
                str(mcmcopts["warp_rep"]).lower() == "shooting"
                and norm(randcoef) >= 0.9 * np.pi
            ):
                # keep the shooting vector inside the injectivity radius
                randcoef = randcoef * (0.9 * np.pi / norm(randcoef)) * rng.random()
            random_starts[:, i] = randcoef

    # chain 1 starts at initcoef, or at the DP warp when init = 'dp': chains
    # started at the identity can settle in a poor local mode
    random_starts[:, 0] = initcoef

    isparallel = bool(mcmcopts["parallel"]) and nchains > 1

    mcmcopts_p = []
    for i in range(0, nchains):
        opts_i = dict(mcmcopts)
        opts_i["initcoef"] = random_starts[:, i]
        opts_i["dp_start"] = (i == 0) and str(mcmcopts["init"]).lower() == "dp"
        if mcmcopts["seed"] is not None:
            opts_i["seed"] = int(mcmcopts["seed"]) + i + 1
        mcmcopts_p.append(opts_i)

    # run chains
    if isparallel:
        chains = Parallel(n_jobs=-1)(
            delayed(run_mcmc)(y1i, y2i, time, mcmcopts_p[n]) for n in range(nchains)
        )
    else:
        chains = [run_mcmc(y1i, y2i, time, mcmcopts_p[n]) for n in range(nchains)]

    # combine outputs
    Nsamples = chains[0]["f1"].shape[0]
    M = chains[0]["f1"].shape[1]
    Nsamplesa = iter - 1
    f1 = np.zeros((Nsamples * nchains, M))
    f2 = np.zeros((Nsamples * nchains, M))
    gamma = np.zeros((M, nchains))
    v_coef = np.zeros((Nsamples * nchains, chains[0]["v_coef"].shape[1]))
    psi = np.zeros((M, nchains))
    sigma = np.zeros(Nsamples * nchains)
    sigma1 = np.zeros(Nsamples * nchains)
    sigma2 = np.zeros(Nsamples * nchains)
    s1 = np.zeros(Nsamples * nchains)
    s2 = np.zeros(Nsamples * nchains)
    L1 = np.zeros(Nsamples * nchains)
    L2 = np.zeros(Nsamples * nchains)
    f2_warped_mu = np.zeros((M, nchains))
    epsilon = np.zeros((nchains, chains[0]["epsilon"].shape[0]))

    if mcmcopts["extrainfo"]:
        theta_accept = np.zeros(Nsamplesa * nchains)
        f1_accept = np.zeros(Nsamplesa * nchains)
        f2_accept = np.zeros(Nsamplesa * nchains)
        L1_accept = np.zeros(Nsamplesa * nchains)
        L2_accept = np.zeros(Nsamplesa * nchains)
        gamma_mat = np.zeros((M, Nsamples * nchains))
        gamma_stats = np.zeros((M, 2, nchains))
        SSE = np.zeros((Nsamplesa + 1) * nchains)
        logl = np.zeros((Nsamplesa + 1) * nchains)
        f2_warped = np.zeros((Nsamples * nchains, M))
        phasedist = np.zeros(Nsamples * nchains)
        ampdist = np.zeros(Nsamples * nchains)

    for i in range(0, nchains):
        a = (i) * Nsamples
        b = (i + 1) * Nsamples
        f1[a:b, :] = chains[i]["f1"]
        f2[a:b, :] = chains[i]["f2"]
        gamma[:, i] = chains[i]["gamma"]
        v_coef[a:b, :] = chains[i]["v_coef"]
        psi[:, i] = chains[i]["psi"]
        sigma[a:b] = chains[i]["sigma"]
        sigma1[a:b] = chains[i]["sigma1"]
        sigma2[a:b] = chains[i]["sigma2"]
        s1[a:b] = chains[i]["s1"]
        s2[a:b] = chains[i]["s2"]
        L1[a:b] = chains[i]["L1"]
        L2[a:b] = chains[i]["L2"]
        f2_warped_mu[:, i] = chains[i]["f2_warped_mu"]
        epsilon[i, :] = chains[i]["epsilon"]

        if mcmcopts["extrainfo"]:
            a1 = (i) * Nsamplesa
            b1 = (i + 1) * Nsamplesa
            theta_accept[a1:b1] = chains[i]["theta_accept"]
            f1_accept[a1:b1] = chains[i]["f1_accept"]
            f2_accept[a1:b1] = chains[i]["f2_accept"]
            L1_accept[a1:b1] = chains[i]["L1_accept"]
            L2_accept[a1:b1] = chains[i]["L2_accept"]
            gamma_mat[:, a:b] = chains[i]["gamma_mat"]
            gamma_stats[:, :, i] = chains[i]["gamma_stats"]
            a1 = (i) * (Nsamplesa + 1)
            b1 = (i + 1) * (Nsamplesa + 1)
            SSE[a1:b1] = chains[i]["SSE"]
            logl[a1:b1] = chains[i]["logl"]
            f2_warped[a:b, :] = chains[i]["f2_warped"]
            phasedist[a:b] = chains[i]["phasedist"]
            ampdist[a:b] = chains[i]["ampdist"]

    # finding modes
    posterior_gamma_modes = None
    posterior_gamma_modes_cr = None
    if nchains > 1:
        Dx = np.zeros((nchains, nchains))
        time1 = np.linspace(0, 1, gamma.shape[0])
        binsize = np.diff(time1).mean()
        for i in range(0, nchains):
            for j in range(i + 1, nchains):
                psi1 = np.sqrt(np.maximum(np.gradient(gamma[:, i], binsize), 0))
                psi2 = np.sqrt(np.maximum(np.gradient(gamma[:, j], binsize), 0))
                q1dotq2 = np.clip(trapezoid(psi1 * psi2, time1), -1, 1)
                Dx[i, j] = np.real(np.arccos(q1dotq2))

        Dx = Dx + Dx.T

        # cluster modes
        y = squareform(Dx)
        Z = linkage(y, method="complete")
        cutoff = np.median(Dx)
        T_c = fcluster(Z, cutoff, criterion="distance")
        N = np.unique(T_c)

        # find mean and confidence region of cluster
        posterior_gamma_modes = np.zeros((M, N.shape[0]))
        posterior_gamma_modes_cr = np.zeros((M, 2, N.shape[0]))
        for i in range(1, N.shape[0] + 1):
            idx = np.where(T_c == i)[0]
            tmp = np.zeros((M, Nsamples * idx.shape[0]))
            for j in range(0, idx.shape[0]):
                a = (j) * Nsamples
                b = (j + 1) * Nsamples
                tmp[:, a:b] = chains[idx[j]]["gamma_mat"]
            mu, gam_mu, psi_tmp, vec = uf.SqrtMean(tmp, smooth=False)
            posterior_gamma_modes[:, i - 1] = gam_mu
            posterior_gamma_modes_cr[:, :, i - 1] = uf.statsFun(tmp)

    # thining
    thin = int(mcmcopts["thin"])
    f1 = f1[0::thin, :]
    f2 = f2[0::thin, :]
    v_coef = v_coef[0::thin, :]
    sigma = sigma[0::thin]
    sigma1 = sigma1[0::thin]
    sigma2 = sigma2[0::thin]
    s1 = s1[0::thin]
    s2 = s2[0::thin]
    L1 = L1[0::thin]
    L2 = L2[0::thin]

    if mcmcopts["extrainfo"]:
        theta_accept = theta_accept[0::thin]
        f1_accept = f1_accept[0::thin]
        f2_accept = f2_accept[0::thin]
        L1_accept = L1_accept[0::thin]
        L2_accept = L2_accept[0::thin]
        gamma_mat = gamma_mat[:, 0::thin]
        SSE = SSE[0::thin]
        logl = logl[0::thin]
        f2_warped = f2_warped[0::thin, :]
        phasedist = phasedist[0::thin]
        ampdist = ampdist[0::thin]

    fields = [
        "f1",
        "f2",
        "gamma",
        "v_coef",
        "psi",
        "sigma",
        "sigma1",
        "sigma2",
        "s1",
        "s2",
        "L1",
        "L2",
        "f2_warped_mu",
        "epsilon",
    ]
    values = [
        f1,
        f2,
        gamma,
        v_coef,
        psi,
        sigma,
        sigma1,
        sigma2,
        s1,
        s2,
        L1,
        L2,
        f2_warped_mu,
        epsilon,
    ]

    if mcmcopts["extrainfo"]:
        fields += [
            "theta_accept",
            "f1_accept",
            "f2_accept",
            "L1_accept",
            "L2_accept",
            "gamma_mat",
            "gamma_stats",
            "SSE",
            "logl",
            "f2_warped",
            "phasedist",
            "ampdist",
        ]
        values += [
            theta_accept,
            f1_accept,
            f2_accept,
            L1_accept,
            L2_accept,
            gamma_mat,
            gamma_stats,
            SSE,
            logl,
            f2_warped,
            phasedist,
            ampdist,
        ]

    if nchains > 1:
        fields += ["posterior_gamma_modes", "posterior_gamma_modes_cr"]
        values += [posterior_gamma_modes, posterior_gamma_modes_cr]

    results_o = collections.namedtuple("align_bayes_HMC", fields)
    out = results_o(*values)

    return out


def _gp_prior(s, L, Dsq, Rcorr=None):
    """
    GP prior N(0, s * (corr(L) + 0.1 I)) stored through its Cholesky factor

    :param s: prior variance scale
    :param L: length scale
    :param Dsq: squared pairwise distances of the time points
    :param Rcorr: Cholesky factor of the correlation matrix, when it is known
    """
    n = Dsq.shape[0]
    if Rcorr is None:
        K = np.exp(-Dsq / (2 * L**2))
        K = K + 0.1 * np.eye(n)
        Rcorr = cholesky(K)

    logdet = 2 * np.sum(np.log(np.diag(Rcorr))) + n * np.log(s)

    return {"Rcorr": Rcorr, "s": s, "logdet": logdet}


def _quad_form(R, f):
    """f' K^{-1} f from the upper Cholesky factor R of K"""
    z = solve_triangular(R.T, np.asarray(f, dtype=float).ravel(), lower=True)
    return z @ z


def _chol_jitter(K):
    """upper Cholesky factor of K, adding jitter until it is positive definite"""
    jit = 0.0
    n = K.shape[0]
    while True:
        try:
            return cholesky(K + jit * np.eye(n))
        except np.linalg.LinAlgError:
            jit = np.maximum(2 * jit, 1e-10 * np.mean(np.diag(K)))
        except Exception:
            jit = np.maximum(2 * jit, 1e-10 * np.mean(np.diag(K)))


def _update_f(
    f_curr, q_curr, SSE_curr, y, sse_fun, gp, Rprop, to_srvf, sigma, sigmaf
):
    """
    Metropolis update of a latent function f (f1 or f2) given the data y, its
    GP prior and the SRVF alignment likelihood. f | y, prior is Gaussian with
    precision K^{-1} + I/sigmaf, so the target is that density times the
    alignment likelihood exp(-SSE/(2 sigma)).
    """
    f_prop = f_curr + randn(f_curr.shape[0]) @ Rprop
    q_prop = to_srvf(f_prop)
    SSE_prop = sse_fun(q_prop)
    lp_curr = -SSE_curr / (2 * sigma) - (
        _quad_form(gp["Rcorr"], f_curr) / gp["s"] + np.sum((y - f_curr) ** 2) / sigmaf
    ) / 2
    lp_prop = -SSE_prop / (2 * sigma) - (
        _quad_form(gp["Rcorr"], f_prop) / gp["s"] + np.sum((y - f_prop) ** 2) / sigmaf
    ) / 2
    accept = np.log(rand()) < (lp_prop - lp_curr)
    if accept:
        f_curr = f_prop
        q_curr = q_prop
        SSE_curr = SSE_prop

    return f_curr, q_curr, SSE_curr, accept


def _update_length(f, gp, s, L_curr, propsd, Dsq):
    """
    Metropolis-Hastings update of a GP length scale with a truncated normal
    random-walk proposal (flat prior on L > 0)
    """
    # truncated normal N(L_curr, propsd^2) on (0, inf) by rejection (exact)
    L_prop = L_curr + propsd * randn()
    while L_prop <= 0:
        L_prop = L_curr + propsd * randn()

    gp_prop = _gp_prior(s, L_prop, Dsq)
    lp_curr = -gp["logdet"] / 2 - _quad_form(gp["Rcorr"], f) / (2 * s)
    lp_prop = -gp_prop["logdet"] / 2 - _quad_form(gp_prop["Rcorr"], f) / (2 * s)
    # Hastings correction for the truncation: q(L'|L) = phi(.)/Phi(L/propsd)
    log_hastings = np.log(erfc(-L_curr / (propsd * np.sqrt(2)))) - np.log(
        erfc(-L_prop / (propsd * np.sqrt(2)))
    )
    accept = np.log(rand()) < (lp_prop - lp_curr + log_hastings)
    if accept:
        L_curr = L_prop
        gp = gp_prop

    return L_curr, gp, accept


def _group_geom(c, Phi, rep, qstar, Q, sigma2):
    """
    geometry of the group alignment likelihood for a batch of curves (one per
    column): aux = [SSE; q_i warped]
    """
    nll, g, SSE, gam, q2w = bf.bayes_warp_geom(c, Phi, rep, qstar, Q, sigma2)
    aux = np.vstack((SSE.reshape((1, -1)), q2w))
    return nll, g, aux


def _group_hmc_update(c, Phi, rep, qstar, Q, sigma2, C, epsilon, L):
    """one batched infinite-dimensional HMC update of the warps of curves Q"""

    def geom(x):
        return _group_geom(x, Phi, rep, qstar, Q, sigma2)

    nll, g, aux = geom(c)
    c, nll, g, aux, accept, alpha = bf.infhmc_step_batch(
        c, nll, g, aux, geom, C, epsilon, L
    )
    return c, aux, accept, alpha


def _karcher_mean_inverse(gam, t):
    """
    Inverse of the Karcher mean of warping functions gam (T,N) on the uniform
    grid t; same estimate as :func:`fdasrsf.utility_functions.SqrtMeanInverse`,
    vectorized over the curves and started from the extrinsic mean (the warps
    are near identity after the first centering step, so a few full gradient
    steps converge)
    """
    T, N = gam.shape
    dt = t[1] - t[0]
    w = np.full((T, 1), dt)
    w[0] = dt / 2
    w[-1] = dt / 2
    gd = np.gradient(gam, dt, axis=0)
    psi = np.sqrt(np.maximum(gd, 0))
    psi = psi / np.sqrt(np.sum(w * psi**2, axis=0))
    mu = psi.mean(axis=1).reshape((T, 1))
    mu = mu / np.sqrt(np.sum(w * mu**2))
    for it in range(0, 100):
        ip = np.clip(np.sum(w * (psi * mu), axis=0), -1, 1)
        th = np.arccos(ip)
        fac = np.ones(N)
        nz = th > 1e-10
        fac[nz] = th[nz] / np.sin(th[nz])
        vbar = ((psi - mu * ip) * fac).mean(axis=1).reshape((T, 1))
        nv = np.sqrt(np.sum(w * vbar**2))
        if nv < 1e-8:
            break
        mu = np.cos(nv) * mu + np.sin(nv) / nv * vbar
        mu = mu / np.sqrt(np.sum(w * mu**2))

    gam_mu = cumulative_trapezoid(mu.ravel() ** 2, t, initial=0)
    gam_mu = (gam_mu - gam_mu[0]) / (gam_mu[-1] - gam_mu[0])

    return uf.invertGamma(gam_mu)


def run_mcmc(y1i, y2i, time, mcmcopts):
    """
    single chain of the pairwise Bayesian infinite-dimensional HMC alignment,
    see :func:`pairwise_align_bayes_infHMC`
    """
    if mcmcopts.get("seed", None) is not None:
        np.random.seed(int(mcmcopts["seed"]) % (2**32))

    # Number of sig figs to report in gamma_mat
    SIG_GAM = 13
    iter = int(mcmcopts["iter"])
    burnin = int(mcmcopts["burnin"])

    y1 = np.asarray(y1i, dtype=float).ravel()
    y2 = np.asarray(y2i, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    T = time.shape[0]

    # normalize time to [0,1] (uniform sampling assumed)
    t01 = (time - time.min()) / (time.max() - time.min())

    # warping function parameter settings
    valid_index = np.arange(burnin - 1, iter)
    initcoef = np.asarray(mcmcopts["initcoef"], dtype=float).ravel()
    ncoef = initcoef.shape[0]
    rep = 1 if str(mcmcopts["warp_rep"]).lower() == "clr" else 0
    Phi = bf.bayes_warp_basis(t01, ncoef, mcmcopts["basis"])
    C = mcmcopts["vpriorvar"] / np.ceil(np.arange(1, ncoef + 1) / 2) ** mcmcopts[
        "vpriordecay"
    ]
    epsilon = np.sqrt(mcmcopts["h"])
    da = bf.dual_avg_step(epsilon)
    L = int(mcmcopts["L"])

    # f1, f2 GP priors, initialized from a GP fit
    Dmat = squareform(pdist(time.reshape((T, 1))))
    Dsq = Dmat**2

    # SRVF operator: the derivative of the interpolating cubic spline used by
    # f_to_srsf is linear in f, so build its (T,T) matrix once; to_srvf(f) is
    # identical to f_to_srsf(f, t01) without fitting a spline every iteration
    Dspl = bf.spline_deriv_matrix(t01)

    def to_srvf(f):
        return bf.srvf_from_deriv(Dspl @ np.asarray(f, dtype=float).ravel())

    sampfreq = int(mcmcopts["sampfreq"])
    L1_curr, sf1, sn1, f1_curr = gp_fit_se(time[0::sampfreq], y1[0::sampfreq], time)
    L2_curr, sf2, sn2, f2_curr = gp_fit_se(time[0::sampfreq], y2[0::sampfreq], time)
    s1_curr = sf1**2
    s2_curr = sf2**2
    gp1 = _gp_prior(s1_curr, L1_curr, Dsq)
    gp2 = _gp_prior(s2_curr, L2_curr, Dsq)

    # random-walk proposals for f1, f2 (fixed covariance, factor once)
    R_f1prop = _chol_jitter(uf.exp2corr(mcmcopts["f1propvar"], L1_curr, Dmat))
    R_f2prop = _chol_jitter(uf.exp2corr(mcmcopts["f2propvar"], L2_curr, Dmat))

    # result vectors
    v_coef = np.zeros((iter, ncoef))
    sigma = np.zeros(iter)
    sigma1 = np.zeros(iter)
    sigma2 = np.zeros(iter)
    f1 = np.zeros((iter, T))
    f2 = np.zeros((iter, T))
    s1 = np.zeros(iter)
    s2 = np.zeros(iter)
    L1 = np.zeros(iter)
    L2 = np.zeros(iter)
    logl = np.zeros(iter)
    SSE = np.zeros(iter)
    eps_tr = np.zeros(iter)
    theta_accept = np.zeros(iter, dtype=bool)
    f1_accept = np.zeros(iter, dtype=bool)
    f2_accept = np.zeros(iter, dtype=bool)
    L1_accept = np.zeros(iter, dtype=bool)
    L2_accept = np.zeros(iter, dtype=bool)

    # init
    v_coef_curr = initcoef.copy()
    sigma_curr = 1.0
    sigma1_curr = 0.01
    sigma2_curr = 0.01
    q1_curr = to_srvf(f1_curr)
    q2_curr = to_srvf(f2_curr)
    if mcmcopts.get("dp_start", False):
        # start at the elastic (DP) alignment of the GP-smoothed functions
        f2n_dp, gam_dp, q2n_dp = pairwise_align_functions(f1_curr, f2_curr, t01)
        v_coef_curr = bf.bayes_gam_to_coef(gam_dp, Phi, rep)

    nll, g, SSE_curr, gam_tmp, q2w_tmp = bf.bayes_warp_geom(
        v_coef_curr, Phi, rep, q1_curr, q2_curr, sigma_curr
    )
    if not np.isfinite(nll):
        raise Exception(
            "Initial warping coefficients are outside the injectivity radius "
            "(|v| >= pi)"
        )

    v_coef[0, :] = v_coef_curr
    f1[0, :] = f1_curr
    f2[0, :] = f2_curr
    sigma[0] = sigma_curr
    sigma1[0] = sigma1_curr
    sigma2[0] = sigma2_curr
    s1[0] = s1_curr
    s2[0] = s2_curr
    L1[0] = L1_curr
    L2[0] = L2_curr
    SSE[0] = SSE_curr
    logl[0] = -nll
    eps_tr[0] = epsilon

    for m in range(1, iter):
        # update f1 | v, f2
        f1_curr, q1_curr, SSE_curr, f1_accept1 = _update_f(
            f1_curr,
            q1_curr,
            SSE_curr,
            y1,
            lambda q, c=v_coef_curr, q2=q2_curr, sg=sigma_curr: bf.bayes_warp_geom(
                c, Phi, rep, q, q2, sg, grad=False
            )[2],
            gp1,
            R_f1prop,
            to_srvf,
            sigma_curr,
            sigma1_curr,
        )

        # update f2 | v, f1
        f2_curr, q2_curr, SSE_tmp, f2_accept1 = _update_f(
            f2_curr,
            q2_curr,
            SSE_curr,
            y2,
            lambda q, c=v_coef_curr, q1=q1_curr, sg=sigma_curr: bf.bayes_warp_geom(
                c, Phi, rep, q1, q, sg, grad=False
            )[2],
            gp2,
            R_f2prop,
            to_srvf,
            sigma_curr,
            sigma2_curr,
        )

        # update v by infinite-dimensional HMC; geometry evaluated at the
        # current f1, f2 and sigma
        # the geometry is evaluated at the current f1, f2 and sigma, bound
        # here as MATLAB's anonymous function captures them by value
        def geom(c, q1=q1_curr, q2=q2_curr, sg=sigma_curr):
            nll_c, g_c, SSE_c, gam_c, q2w_c = bf.bayes_warp_geom(
                c, Phi, rep, q1, q2, sg
            )
            return nll_c, g_c, None

        nll, g, aux = geom(v_coef_curr)
        v_coef_curr, nll, g, aux, theta_accept1, nsteps, alpha = bf.infhmc_step(
            v_coef_curr, nll, g, aux, geom, C, epsilon, L
        )
        SSE_curr = 2 * sigma_curr * (nll - T / 2 * np.log(sigma_curr))

        # adapt step size during burn-in (dual averaging); keep the averaged
        # step size afterwards
        if mcmcopts["adapt"] and (m + 1) <= burnin:
            epsilon, da = bf.dual_avg_step(da, alpha, mcmcopts["target_accept"])
            if (m + 1) == burnin:
                epsilon = np.exp(da["log_eps_bar"])

        # update sigma^2 (SRVF model variance)
        newshape = T / 2 + mcmcopts["alpha"]
        newscale = 1 / 2 * SSE_curr + mcmcopts["beta"]
        sigma_curr = 1 / np.random.gamma(newshape, 1 / newscale)

        # update sigma1^2, sigma2^2 (measurement error variances)
        newshape = T / 2 + mcmcopts["alpha0"]
        sigma1_curr = 1 / np.random.gamma(
            newshape, 1 / (np.sum((y1 - f1_curr) ** 2) / 2 + mcmcopts["beta0"])
        )
        sigma2_curr = 1 / np.random.gamma(
            newshape, 1 / (np.sum((y2 - f2_curr) ** 2) / 2 + mcmcopts["beta0"])
        )

        # update GP variances s1^2, s2^2
        s1_curr = 1 / np.random.gamma(
            newshape,
            1 / (_quad_form(gp1["Rcorr"], f1_curr) / 2 + mcmcopts["beta0"]),
        )
        s2_curr = 1 / np.random.gamma(
            newshape,
            1 / (_quad_form(gp2["Rcorr"], f2_curr) / 2 + mcmcopts["beta0"]),
        )
        gp1 = _gp_prior(s1_curr, L1_curr, Dsq, gp1["Rcorr"])
        gp2 = _gp_prior(s2_curr, L2_curr, Dsq, gp2["Rcorr"])

        # update GP length scales L1, L2
        L1_curr, gp1, L1_accept1 = _update_length(
            f1_curr, gp1, s1_curr, L1_curr, mcmcopts["L1propvar"], Dsq
        )
        L2_curr, gp2, L2_accept1 = _update_length(
            f2_curr, gp2, s2_curr, L2_curr, mcmcopts["L2propvar"], Dsq
        )

        # save update to results
        v_coef[m, :] = v_coef_curr
        f1[m, :] = f1_curr
        f2[m, :] = f2_curr
        sigma[m] = sigma_curr
        sigma1[m] = sigma1_curr
        sigma2[m] = sigma2_curr
        s1[m] = s1_curr
        s2[m] = s2_curr
        L1[m] = L1_curr
        L2[m] = L2_curr
        SSE[m] = SSE_curr
        logl[m] = -(SSE_curr / (2 * sigma_curr) + T / 2 * np.log(sigma_curr))
        eps_tr[m] = epsilon
        theta_accept[m] = theta_accept1
        f1_accept[m] = f1_accept1
        f2_accept[m] = f2_accept1
        L1_accept[m] = L1_accept1
        L2_accept[m] = L2_accept1

    # posterior warping functions
    gamma_mat, psi_mat = bf.bayes_coef_to_gam(v_coef[valid_index, :].T, Phi, rep)
    gamma_mat = np.round(gamma_mat, SIG_GAM)

    # posterior mean of psi (extrinsic mean, projected back to the sphere)
    psi_mean = psi_mat.mean(axis=1)
    psi_mean = psi_mean / np.sqrt(trapezoid(psi_mean**2, t01))
    gam_mean = np.concatenate(
        (
            [0],
            np.cumsum(
                np.diff(t01) * (psi_mean[0:-1] ** 2 + psi_mean[1:] ** 2) / 2
            ),
        )
    )
    gam_mean = uf.norm_gam(gam_mean)

    # return object
    f1 = f1[valid_index, :]
    f2 = f2[valid_index, :]
    v_coef = v_coef[valid_index, :]
    sigma = sigma[valid_index]
    sigma1 = sigma1[valid_index]
    sigma2 = sigma2[valid_index]
    s1 = s1[valid_index]
    s2 = s2[valid_index]
    L1 = L1[valid_index]
    L2 = L2[valid_index]
    f2_warped_mu = uf.warp_f_gamma(t01, f2.mean(axis=0), gam_mean)

    out_dict = {
        "f1": f1,
        "f2": f2,
        "gamma": gam_mean,
        "v_coef": v_coef,
        "psi": psi_mean,
        "sigma": sigma,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "s1": s1,
        "s2": s2,
        "L1": L1,
        "L2": L2,
        "epsilon": eps_tr,
        "f2_warped_mu": f2_warped_mu,
    }

    if mcmcopts["extrainfo"]:
        nv = valid_index.shape[0]
        Dx = np.real(np.arccos(np.clip(trapezoid(psi_mat, t01, axis=0), -1, 1)))
        Dy = np.zeros(nv)
        f2_warped = np.zeros((nv, T))
        for ii in range(0, nv):
            q1s = to_srvf(f1[ii, :])
            q2s = to_srvf(f2[ii, :])
            q2warp = uf.warp_q_gamma(t01, q2s, gamma_mat[:, ii])
            Dy[ii] = np.sqrt(trapezoid((q1s - q2warp) ** 2, t01))
            f2_warped[ii, :] = uf.warp_f_gamma(t01, f2[ii, :], gamma_mat[:, ii])

        out_dict["theta_accept"] = theta_accept[1:]
        out_dict["f1_accept"] = f1_accept[1:]
        out_dict["f2_accept"] = f2_accept[1:]
        out_dict["L1_accept"] = L1_accept[1:]
        out_dict["L2_accept"] = L2_accept[1:]
        out_dict["gamma_mat"] = gamma_mat
        out_dict["gamma_stats"] = uf.statsFun(gamma_mat)
        out_dict["SSE"] = SSE
        out_dict["logl"] = logl
        out_dict["phasedist"] = Dx
        out_dict["ampdist"] = Dy
        out_dict["f2_warped"] = f2_warped

    return out_dict


def align_fPCA(f, time, num_comp=3, showplot=True, smoothdata=False, cores=-1):
    """
    aligns a collection of functions while extracting principal components.
    The functions are aligned to the principal components

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param num_comp: number of fPCA components
    :param showplot: Shows plots of results using matplotlib (default = T)
    :param smooth_data: Smooth the data using a box filter (default = F)
    :param cores: number of cores for parallel (default = -1 (all))
    :type sparam: double
    :type smooth_data: bool
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of numpy array
    :return fn: aligned functions - numpy ndarray of shape (M,N) of N
                functions with M samples
    :return qn: aligned srvfs - similar structure to fn
    :return q0: original srvf - similar structure to fn
    :return mqn: srvf mean or median - vector of length M
    :return gam: warping functions - similar structure to fn
    :return q_pca: srsf principal directions
    :return f_pca: functional principal directions
    :return latent: latent values
    :return coef: coefficients
    :return U: eigenvectors
    :return orig_var: Original Variance of Functions
    :return amp_var: Amplitude Variance
    :return phase_var: Phase Variance

    """
    lam = 0.0
    MaxItr = 50
    coef = np.arange(-2.0, 3.0)
    Nstd = coef.shape[0]
    M = f.shape[0]
    N = f.shape[1]
    if M > 500:
        parallel = True
    elif N > 100:
        parallel = True
    else:
        parallel = False

    eps = np.finfo(np.double).eps
    f0 = f

    if showplot:
        plot.f_plot(time, f, title="Original Data")

    # Compute SRSF function from data
    f, g, g2 = uf.gradient_spline(time, f, smoothdata)
    q = g / np.sqrt(abs(g) + eps)

    print("Initializing...")
    mnq = q.mean(axis=1)
    a = mnq.repeat(N)
    d1 = a.reshape(M, N)
    d = (q - d1) ** 2
    dqq = np.sqrt(d.sum(axis=0))
    min_ind = dqq.argmin()

    print(
        "Aligning %d functions in SRVF space to %d fPCA components..." % (N, num_comp)
    )
    itr = 0
    mq = np.zeros((M, MaxItr + 1))
    mq[:, itr] = q[:, min_ind]
    fi = np.zeros((M, N, MaxItr + 1))
    fi[:, :, 0] = f
    qi = np.zeros((M, N, MaxItr + 1))
    qi[:, :, 0] = q
    gam = np.zeros((M, N, MaxItr + 1))
    cost = np.zeros(MaxItr + 1)

    while itr < MaxItr:
        print("updating step: r=%d" % (itr + 1))
        if itr == MaxItr:
            print("maximal number of iterations is reached")

        # PCA Step
        a = mq[:, itr].repeat(N)
        d1 = a.reshape(M, N)
        qhat_cent = qi[:, :, itr] - d1
        K = np.cov(qi[:, :, itr])
        U, s, V = svd(K)

        alpha_i = np.zeros((num_comp, N))
        for ii in range(0, num_comp):
            for jj in range(0, N):
                alpha_i[ii, jj] = trapezoid(qhat_cent[:, jj] * U[:, ii], time)

        U1 = U[:, 0:num_comp]
        tmp = U1.dot(alpha_i)
        qhat = d1 + tmp

        # Matching Step
        if parallel:
            out = Parallel(n_jobs=cores)(
                delayed(uf.optimum_reparam)(qhat[:, n], time, qi[:, n, itr], "DP2", lam)
                for n in range(N)
            )
            gam_t = np.array(out)
            gam[:, :, itr] = gam_t.transpose()
        else:
            gam[:, :, itr] = uf.optimum_reparam(qhat, time, qi[:, :, itr], "DP2", lam)

        for k in range(0, N):
            time0 = (time[-1] - time[0]) * gam[:, k, itr] + time[0]
            fi[:, k, itr + 1] = np.interp(time0, time, fi[:, k, itr])
            qi[:, k, itr + 1] = uf.f_to_srsf(fi[:, k, itr + 1], time)

        qtemp = qi[:, :, itr + 1]
        mq[:, itr + 1] = qtemp.mean(axis=1)

        cost_temp = np.zeros(N)

        for ii in range(0, N):
            cost_temp[ii] = norm(qtemp[:, ii] - qhat[:, ii]) ** 2

        cost[itr + 1] = cost_temp.mean()

        if abs(cost[itr + 1] - cost[itr]) < 1e-06:
            break

        itr += 1

    if itr >= MaxItr:
        itrf = MaxItr
    else:
        itrf = itr + 1
    cost = cost[1 : (itrf + 1)]

    # Aligned data & stats
    fn = fi[:, :, itrf]
    qn = qi[:, :, itrf]
    q0 = qi[:, :, 0]
    mean_f0 = f0.mean(axis=1)
    std_f0 = f0.std(axis=1)
    mqn = mq[:, itrf]
    # fi[:, :, itrf] has warps gam[:, :, 0..itrf-1] applied, so compose all of
    # them (copy so the per-iteration warps in gam are not overwritten)
    gamf = gam[:, :, 0].copy()
    for k in range(1, itrf):
        gam_k = gam[:, :, k]
        for l in range(0, N):
            time0 = (time[-1] - time[0]) * gam_k[:, l] + time[0]
            gamf[:, l] = np.interp(time0, time, gamf[:, l])

    # Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    time0 = (time[-1] - time[0]) * gamI + time[0]
    mqn = np.interp(time0, time, mqn) * np.sqrt(gamI_dev)
    for k in range(0, N):
        qn[:, k] = np.interp(time0, time, qn[:, k]) * np.sqrt(gamI_dev)
        fn[:, k] = np.interp(time0, time, fn[:, k])
        gamf[:, k] = np.interp(time0, time, gamf[:, k])

    mean_fn = fn.mean(axis=1)
    std_fn = fn.std(axis=1)

    # Get Final PCA
    mididx = int(np.round(time.shape[0] / 2))
    m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
    mqn2 = np.append(mqn, m_new.mean())
    qn2 = np.vstack((qn, m_new))
    K = np.cov(qn2)

    U, s, V = svd(K)
    stdS = np.sqrt(s)

    # compute the PCA in the q domain
    q_pca = np.ndarray(shape=(M + 1, Nstd, num_comp), dtype=float)
    for k in range(0, num_comp):
        for l in range(0, Nstd):
            q_pca[:, l, k] = mqn2 + coef[l] * stdS[k] * U[:, k]

    # compute the correspondence in the f domain
    f_pca = np.ndarray(shape=(M, Nstd, num_comp), dtype=float)
    for k in range(0, num_comp):
        for l in range(0, Nstd):
            q_pca_tmp = q_pca[0:M, l, k] * np.abs(q_pca[0:M, l, k])
            q_pca_tmp2 = np.sign(q_pca[M, l, k]) * (q_pca[M, l, k] ** 2)
            f_pca[:, l, k] = uf.cumtrapzmid(time, q_pca_tmp, q_pca_tmp2, mididx)

    N2 = qn.shape[1]
    c = np.zeros((N2, num_comp))
    for k in range(0, num_comp):
        for l in range(0, N2):
            c[l, k] = sum((np.append(qn[:, l], m_new[l]) - mqn2) * U[:, k])

    if showplot:
        CBcdict = {
            "Bl": (0, 0, 0),
            "Or": (0.9, 0.6, 0),
            "SB": (0.35, 0.7, 0.9),
            "bG": (0, 0.6, 0.5),
            "Ye": (0.95, 0.9, 0.25),
            "Bu": (0, 0.45, 0.7),
            "Ve": (0.8, 0.4, 0),
            "rP": (0.8, 0.6, 0.7),
        }
        cl = sorted(CBcdict.keys())

        # Align Plots
        fig, ax = plot.f_plot(
            np.arange(0, M) / float(M - 1), gamf, title="Warping Functions"
        )
        ax.set_aspect("equal")

        plot.f_plot(time, fn, title="Warped Data")

        tmp = np.array([mean_f0, mean_f0 + std_f0, mean_f0 - std_f0])
        tmp = tmp.transpose()
        plot.f_plot(time, tmp, title=r"Original Data: Mean $\pm$ STD")

        tmp = np.array([mean_fn, mean_fn + std_fn, mean_fn - std_fn])
        tmp = tmp.transpose()
        plot.f_plot(time, tmp, title=r"Warped Data: Mean $\pm$ STD")

        # PCA Plots
        fig, ax = plt.subplots(2, num_comp)
        for k in range(0, num_comp):
            axt = ax[0, k]
            for l in range(0, Nstd):
                axt.plot(time, q_pca[0:M, l, k], color=CBcdict[cl[l]])
                axt.hold(True)

            axt.set_title("q domain: PD %d" % (k + 1))
            plot.rstyle(axt)
            axt = ax[1, k]
            for l in range(0, Nstd):
                axt.plot(time, f_pca[:, l, k], color=CBcdict[cl[l]])
                axt.hold(True)

            axt.set_title("f domain: PD %d" % (k + 1))
            plot.rstyle(axt)
        fig.set_tight_layout(True)

        cumm_coef = 100 * np.cumsum(s) / sum(s)
        idx = np.arange(0, M + 1) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.xlabel("Percentage")
        plt.ylabel("Index")
        plt.show()

    mean_f0 = f0.mean(axis=1)
    std_f0 = f0.std(axis=1, ddof=1)
    mean_fn = fn.mean(axis=1)
    std_fn = fn.std(axis=1, ddof=1)
    fmean = uf.srsf_to_f(mqn, time, np.mean(f0[0, :]))

    fgam = np.zeros((M, N))
    for k in range(0, N):
        time0 = (time[-1] - time[0]) * gamf[:, k] + time[0]
        fgam[:, k] = np.interp(time0, time, fmean)

    var_fgam = fgam.var(axis=1, ddof=1)
    orig_var = trapezoid(std_f0**2, time)
    amp_var = trapezoid(std_fn**2, time)
    phase_var = trapezoid(var_fgam, time)

    K = np.cov(fn)

    U, s, V = svd(K)

    align_fPCAresults = collections.namedtuple(
        "align_fPCA",
        [
            "fn",
            "qn",
            "q0",
            "mqn",
            "gam",
            "q_pca",
            "f_pca",
            "latent",
            "coef",
            "U",
            "orig_var",
            "amp_var",
            "phase_var",
            "cost",
        ],
    )

    out = align_fPCAresults(
        fn, qn, q0, mqn, gamf, q_pca, f_pca, s, c, U, orig_var, amp_var, phase_var, cost
    )
    return out


def align_fPLS(
    f, g, time, comps=3, showplot=True, smoothdata=False, delta=0.01, max_itr=100
):
    """
    This function aligns a collection of functions while performing
    principal least squares

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param g: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param comps: number of fPLS components
    :param showplot: Shows plots of results using matplotlib (default = T)
    :param smooth_data: Smooth the data using a box filter (default = F)
    :param delta: gradient step size
    :param max_itr: maximum number of iterations
    :type smooth_data: bool
    :type f: np.ndarray
    :type g: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of numpy array
    :return fn: aligned functions - numpy ndarray of shape (M,N) of N functions with M samples
    :return gn: aligned functions - numpy ndarray of shape (M,N) of N functions with M samples
    :return qfn: aligned srvfs - similar structure to fn
    :return qgn: aligned srvfs - similar structure to fn
    :return qf0: original srvf - similar structure to fn
    :return qg0: original srvf - similar structure to fn
    :return gam: warping functions - similar structure to fn
    :return wqf: srsf principal weight functions
    :return wqg: srsf principal weight functions
    :return wf: srsf principal weight functions
    :return wg: srsf principal weight functions
    :return cost: cost function value

    """
    print("Initializing...")
    binsize = np.diff(time)
    binsize = binsize.mean()
    eps = np.finfo(np.double).eps
    M = f.shape[0]
    N = f.shape[1]

    if showplot:
        plot.f_plot(time, f, title="f Original Data")
        plot.f_plot(time, g, title="g Original Data")

    # Compute q-function of f and g
    f, g1, g2 = uf.gradient_spline(time, f, smoothdata)
    qf = g1 / np.sqrt(abs(g1) + eps)
    g, g1, g2 = uf.gradient_spline(time, g, smoothdata)
    qg = g1 / np.sqrt(abs(g1) + eps)

    print("Calculating fPLS weight functions for %d Warped Functions..." % N)
    itr = 0
    fi = np.zeros((M, N, max_itr + 1))
    fi[:, :, itr] = f
    gi = np.zeros((M, N, max_itr + 1))
    gi[:, :, itr] = g
    qfi = np.zeros((M, N, max_itr + 1))
    qfi[:, :, itr] = qf
    qgi = np.zeros((M, N, max_itr + 1))
    qgi[:, :, itr] = qg
    wqf1, wqg1, alpha, values, costmp = pls_svd(
        time, qfi[:, :, itr], qgi[:, :, itr], 2, 0
    )
    wqf = np.zeros((M, max_itr + 1))
    wqf[:, itr] = wqf1[:, 0]
    wqg = np.zeros((M, max_itr + 1))
    wqg[:, itr] = wqg1[:, 0]
    gam = np.zeros((M, N, max_itr + 1))
    tmp = np.tile(np.linspace(0, 1, M), (N, 1))
    gam[:, :, itr] = tmp.transpose()
    wqf_diff = np.zeros(max_itr + 1)
    cost = np.zeros(max_itr + 1)
    cost_diff = 1

    while itr <= max_itr:
        # warping
        gamtmp = np.ascontiguousarray(gam[:, :, 0])
        qftmp = np.ascontiguousarray(qfi[:, :, 0])
        qgtmp = np.ascontiguousarray(qgi[:, :, 0])
        wqftmp = np.ascontiguousarray(wqf[:, itr])
        wqgtmp = np.ascontiguousarray(wqg[:, itr])
        gam[:, :, itr + 1] = fpls.fpls_warp(
            time,
            gamtmp,
            qftmp,
            qgtmp,
            wqftmp,
            wqgtmp,
            display=0,
            delta=delta,
            tol=1e-6,
            max_iter=4000,
        )

        for k in range(0, N):
            gam_k = gam[:, k, itr + 1]
            time0 = (time[-1] - time[0]) * gam_k + time[0]
            fi[:, k, itr + 1] = np.interp(time0, time, fi[:, k, 0])
            gi[:, k, itr + 1] = np.interp(time0, time, gi[:, k, 0])
            qfi[:, k, itr + 1] = uf.warp_q_gamma(time, qfi[:, k, 0], gam_k)
            qgi[:, k, itr + 1] = uf.warp_q_gamma(time, qgi[:, k, 0], gam_k)

        # PLS
        wqfi, wqgi, alpha, values, costmp = pls_svd(
            time, qfi[:, :, itr + 1], qgi[:, :, itr + 1], 2, 0
        )
        wqf[:, itr + 1] = wqfi[:, 0]
        wqg[:, itr + 1] = wqgi[:, 0]

        wqf_diff[itr] = np.sqrt(sum(wqf[:, itr + 1] - wqf[:, itr]) ** 2)

        rfi = np.zeros(N)
        rgi = np.zeros(N)

        for l in range(0, N):
            rfi[l] = uf.innerprod_q(time, qfi[:, l, itr + 1], wqf[:, itr + 1])
            rgi[l] = uf.innerprod_q(time, qgi[:, l, itr + 1], wqg[:, itr + 1])

        cost[itr] = np.cov(rfi, rgi)[1, 0]

        if itr > 1:
            cost_diff = cost[itr] - cost[itr - 1]

        print(
            "Iteration: %d - Diff Value: %f - %f" % (itr + 1, wqf_diff[itr], cost[itr])
        )
        if wqf_diff[itr] < 1e-1 or abs(cost_diff) < 1e-3:
            break

        itr += 1

    cost = cost[0 : (itr + 1)]

    # Aligned data & stats
    fn = fi[:, :, itr + 1]
    gn = gi[:, :, itr + 1]
    qfn = qfi[:, :, itr + 1]
    qf0 = qfi[:, :, 0]
    qgn = qgi[:, :, itr + 1]
    qg0 = qgi[:, :, 0]
    wqfn, wqgn, alpha, values, costmp = pls_svd(time, qfn, qgn, comps, 0)

    wf = np.zeros((M, comps))
    wg = np.zeros((M, comps))
    for ii in range(0, comps):
        wf[:, ii] = cumulative_trapezoid(wqfn[:, ii] * np.abs(wqfn[:, ii]), time, initial=0)
        wg[:, ii] = cumulative_trapezoid(wqgn[:, ii] * np.abs(wqgn[:, ii]), time, initial=0)

    gam_f = gam[:, :, itr + 1]

    if showplot:
        # Align Plots
        fig, ax = plot.f_plot(
            np.arange(0, M) / float(M - 1), gam_f, title="Warping Functions"
        )
        ax.set_aspect("equal")

        plot.f_plot(time, fn, title="fn Warped Data")
        plot.f_plot(time, gn, title="gn Warped Data")
        plot.f_plot(time, wf, title="wf")
        plot.f_plot(time, wg, title="wg")

        plt.show()

    align_fPLSresults = collections.namedtuple(
        "align_fPLS",
        [
            "wf",
            "wg",
            "fn",
            "gn",
            "qfn",
            "qgn",
            "qf0",
            "qg0",
            "wqf",
            "wqg",
            "gam",
            "values",
            "cost",
        ],
    )

    out = align_fPLSresults(
        wf, wg, fn, gn, qfn, qgn, qf0, qg0, wqfn, wqgn, gam_f, values, cost
    )
    return out
