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
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
from scipy.linalg import svd, cholesky
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
import GPy
from numpy.linalg import norm, inv
from numpy.random import rand, normal
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
            raise Exception('Columns of f and time must be equal')

        self.f = f
        self.time = time
        self.rsamps = False
    

    def srsf_align(self, method="mean", omethod="DP2", center=True, 
                   smoothdata=False, MaxItr=20, parallel=False, lam=0.0, 
                   cores=-1, grid_dim=7, verbose=True):
        """
        This function aligns a collection of functions using the elastic
        square-root slope (srsf) framework.

        :param method: (string) warp calculate Karcher Mean or Median 
                       (options = "mean" or "median") (default="mean")
        :param omethod: optimization method (DP, DP2, RBFGS) (default = DP2)
        :param center: center warping functions (default = T)
        :param smoothdata: Smooth the data using a box filter (default = F)
        :param MaxItr: Maximum number of iterations (default = 20)
        :param parallel: run in parallel (default = F)
        :param lam: controls the elasticity (default = 0)
        :param cores: number of cores for parallel (default = -1 (all))
        :param grid_dim: size of the grid, for the DP2 method only (default = 7)
        :param verbose: print status output (default = T)
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

        if M > 500:
            parallel = True
        elif N > 100:
            parallel = True

        eps = np.finfo(np.double).eps
        f0 = self.f
        self.method = omethod

        methods = ["mean", "median"]
        self.type = method

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
            out = Parallel(n_jobs=cores)(delayed(uf.optimum_reparam)(mq, self.time,
                                    q[:, n], omethod, lam, grid_dim) for n in range(N))
            gam = np.array(out)
            gam = gam.transpose()
        else:
            gam = np.zeros((M,N))
            for k in range(0,N):
                gam[:,k] = uf.optimum_reparam(mq,self.time,q[:,k],omethod,lam,grid_dim)

        gamI = uf.SqrtMeanInverse(gam)
        mf = np.interp((self.time[-1] - self.time[0]) * gamI + self.time[0], self.time, mf)
        mq = uf.f_to_srsf(mf, self.time)

        # Compute Karcher Mean
        if verbose:
            if method == 0:
                print("Compute Karcher Mean of %d function in SRSF space..." % N)
            if method == 1:
                print("Compute Karcher Median of %d function in SRSF space..." % N)

        ds = np.repeat(0.0, MaxItr + 2)
        ds[0] = np.inf
        qun = np.repeat(0.0, MaxItr + 1)
        tmp = np.zeros((M, MaxItr + 2))
        tmp[:, 0] = mq
        mq = tmp
        tmp = np.zeros((M, MaxItr+2))
        tmp[:,0] = mf
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
                out = Parallel(n_jobs=cores)(delayed(uf.optimum_reparam)(mq[:, r],
                                        self.time, q[:, n, 0], omethod, lam, grid_dim) for n in range(N))
                gam = np.array(out)
                gam = gam.transpose()
            else:
                for k in range(0,N):
                    gam[:,k] = uf.optimum_reparam(mq[:, r], self.time, q[:, k, 0],
                            omethod, lam, grid_dim)

            gam_dev = np.zeros((M, N))
            vtil = np.zeros((M,N))
            dtil = np.zeros(N)
            for k in range(0, N):
                f[:, k, r + 1] = np.interp((self.time[-1] - self.time[0]) * gam[:, k]
                                        + self.time[0], self.time, f[:, k, 0])
                q[:, k, r + 1] = uf.f_to_srsf(f[:, k, r + 1], self.time)
                gam_dev[:, k] = np.gradient(gam[:, k], 1 / float(M - 1))
                v = q[:, k, r + 1] - mq[:,r]
                d = np.sqrt(trapz(v*v, self.time))
                vtil[:,k] = v/d
                dtil[k] = 1.0/d

            mqt = mq[:, r]
            a = mqt.repeat(N)
            d1 = a.reshape(M, N)
            d = (q[:, :, r + 1] - d1) ** 2
            if method == 0:
                d1 = sum(trapz(d, self.time, axis=0))
                d2 = sum(trapz((1 - np.sqrt(gam_dev)) ** 2, self.time, axis=0))
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
                d1 = np.sqrt(sum(trapz(d, self.time, axis=0)))
                d2 = sum(trapz((1 - np.sqrt(gam_dev)) ** 2, self.time, axis=0))
                ds_tmp = d1 + lam * d2
                ds[r + 1] = ds_tmp

                # Minimization Step
                # compute the mean of the matched function
                stp = .3
                vbar = vtil.sum(axis=1)*(1/dtil.sum())
                qtemp = q[:, :, r + 1] 
                ftemp = f[:, :, r + 1] 
                mq[:, r + 1] = mq[:,r] + stp*vbar
                tmp = np.zeros(M)
                tmp[1:] = cumtrapz(mq[:, r + 1] * np.abs(mq[:, r + 1]), self.time)
                mf[:, r + 1] = np.median(f0[1, :])+tmp

                qun[r] = norm(mq[:, r + 1] - mq[:, r]) / norm(mq[:, r])

            if qun[r] < 1e-2 or r >= MaxItr:
                break

        # Last Step with centering of gam
        r += 1
        if parallel:
            out = Parallel(n_jobs=cores)(delayed(uf.optimum_reparam)(mq[:, r], self.time,
                q[:, n, 0], omethod, lam, grid_dim) for n in range(N))
            gam = np.array(out)
            gam = gam.transpose()
        else:
            for k in range(0,N):
                gam[:,k] = uf.optimum_reparam(mq[:, r], self.time, q[:, k, 0], omethod,
                        lam, grid_dim)

        gam_dev = np.zeros((M, N))
        for k in range(0, N):
            gam_dev[:, k] = np.gradient(gam[:, k], 1 / float(M - 1))

        if center:
            gamI = uf.SqrtMeanInverse(gam)
            gamI_dev = np.gradient(gamI, 1 / float(M - 1))
            time0 = (self.time[-1] - self.time[0]) * gamI + self.time[0]
            mq[:, r + 1] = np.interp(time0, self.time, mq[:, r]) * np.sqrt(gamI_dev)

            for k in range(0, N):
                q[:, k, r + 1] = np.interp(time0, self.time, q[:, k, r]) * np.sqrt(gamI_dev)
                f[:, k, r + 1] = np.interp(time0, self.time, f[:, k, r])
                gam[:, k] = np.interp(time0, self.time, gam[:, k])
        else:
            gamI = uf.SqrtMeanInverse(gam)
            gamI_dev = np.gradient(gamI, 1 / float(M - 1))

        # Aligned data & stats
        self.center = center
        self.fn = f[:, :, r + 1]
        self.qn = q[:, :, r + 1]
        self.q0 = q[:, :, 0]
        self.gamI = gamI
        mean_f0 = f0.mean(axis=1)
        std_f0 = f0.std(axis=1)
        mean_fn = self.fn.mean(axis=1)
        std_fn = self.fn.std(axis=1)
        self.gam = gam
        self.mqn = mq[:, r + 1]
        tmp = np.zeros(M)
        tmp[1:] = cumtrapz(self.mqn * np.abs(self.mqn), self.time)
        self.fmean = np.mean(f0[1, :]) + tmp

        fgam = np.zeros((M, N))
        for k in range(0, N):
            time0 = (self.time[-1] - self.time[0]) * gam[:, k] + self.time[0]
            fgam[:, k] = np.interp(time0, self.time, self.fmean)

        var_fgam = fgam.var(axis=1)
        self.orig_var = trapz(std_f0 ** 2, self.time)
        self.amp_var = trapz(std_fn ** 2, self.time)
        self.phase_var = trapz(var_fgam, self.time)

        return


    def plot(self):
        """
        plot plot functional alignment results
        
        Usage: obj.plot()
        """

        M = self.f.shape[0]
        plot.f_plot(self.time, self.f, title="f Original Data")

        fig, ax = plot.f_plot(np.arange(0, M) / float(M - 1), self.gam,
                                title="Warping Functions")
        ax.set_aspect('equal')

        plot.f_plot(self.time, self.fn, title="Warped Data")

        mean_f0 = self.f.mean(axis=1)
        std_f0 = self.f.std(axis=1)
        mean_fn = self.fn.mean(axis=1)
        std_fn = self.fn.std(axis=1)
        tmp = np.array([mean_f0, mean_f0 + std_f0, mean_f0 - std_f0])
        tmp = tmp.transpose()
        plot.f_plot(self.time, tmp, title=r"Original Data: Mean $\pm$ STD")

        tmp = np.array([mean_fn, mean_fn + std_fn, mean_fn - std_fn])
        tmp = tmp.transpose()
        plot.f_plot(self.time, tmp, title=r"Warped Data: Mean $\pm$ STD")

        plot.f_plot(self.time, self.fmean, title="$f_{mean}$")
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
                len[i] = np.arccos(tmp.dot(psi[:, i] / M))

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

        
        self.rsamps = True
        self.fs = fs
        self.gams = rgam
        self.ft = ft
        self.qs = q_s[0:M,:]

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
        gam = self.gam

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
        qhat = np.tile(mqn.T,(n,1)).T + tmp[0:M+1,:]
        tmp = np.matmul(U, np.transpose(vals)/C)
        vechat = tmp[(M+1):,:]
        psihat = np.zeros((M,n))
        gamhat = np.zeros((M,n))
        for ii in range(n):
            psihat[:,ii] = geo.exp_map(mu_psi,vechat[:,ii])
            gam_tmp = cumtrapz(psihat[:,ii]**2,np.linspace(0,1,M),initial=0.0)
            gamhat[:,ii] = (gam_tmp - gam_tmp.min())/(gam_tmp.max()-gam_tmp.min())
        
        ft = np.zeros((M,n))
        fhat = np.zeros((M,n))
        for ii in range(n):
            fhat[:,ii] = uf.cumtrapzmid(time, qhat[0:M,ii]*np.fabs(qhat[0:M,ii]), np.sign(qhat[M,ii])*(qhat[M,ii]*qhat[M,ii]), mididx)
            ft[:,ii] = uf.warp_f_gamma(np.linspace(0,1,M),fhat[:,ii],gamhat[:,ii])


        self.rsamps = True
        self.fs = fhat
        self.gams = gamhat
        self.ft = ft
        self.qs = qhat[0:M,:]

        return

    def multiple_align_functions(self, mu, omethod="DP2", smoothdata=False,
                                 parallel=False, lam=0.0, cores=-1, grid_dim=7):
        """
        This function aligns a collection of functions using the elastic square-root
        slope (srsf) framework.

        Usage:  obj.multiple_align_functions(mu)
                obj.multiple_align_functions(lambda)
        obj.multiple_align_functions(lambda, ...)
    
        :param mu: vector of function to align to
        :param omethod: optimization method (DP, DP2, RBFGS) (default = DP)
        :param smoothdata: Smooth the data using a box filter (default = F)
        :param parallel: run in parallel (default = F)
        :param lam: controls the elasticity (default = 0)
        :param cores: number of cores for parallel (default = -1 (all))
        :param grid_dim: size of the grid, for the DP2 method only (default = 7)
        :type lam: double
        :type smoothdata: bool

        """

        M = self.f.shape[0]
        N = self.f.shape[1]
        self.lam = lam

        if M > 500:
            parallel = True
        elif N > 100:
            parallel = True

        eps = np.finfo(np.double).eps
        self.method = omethod
        self.type = "multiple"

        # Compute SRSF function from data
        f, g, g2 = uf.gradient_spline(self.time, self.f, smoothdata)
        q = g / np.sqrt(abs(g) + eps)

        mq = uf.f_to_srsf(mu, self.time)

        if parallel:
            out = Parallel(n_jobs=cores)(delayed(uf.optimum_reparam)(mq, self.time,
                                    q[:, n], omethod, lam, grid_dim) for n in range(N))
            gam = np.array(out)
            gam = gam.transpose()
        else:
            gam = np.zeros((M,N))
            for k in range(0,N):
                gam[:,k] = uf.optimum_reparam(mq,self.time,q[:,k],omethod,lam,grid_dim)

        self.gamI = uf.SqrtMeanInverse(gam)

        fn = np.zeros((M,N))
        qn = np.zeros((M,N))
        for k in range(0, N):
            fn[:, k] = np.interp((self.time[-1] - self.time[0]) * gam[:, k]
                                    + self.time[0], self.time, f[:, k])
            qn[:, k] = uf.f_to_srsf(f[:, k], self.time)


        # Aligned data & stats
        self.fn = fn
        self.qn = qn
        self.q0 = q
        mean_f0 = f.mean(axis=1)
        std_f0 = f.std(axis=1)
        mean_fn = self.fn.mean(axis=1)
        std_fn = self.fn.std(axis=1)
        self.gam = gam
        self.mqn = mq
        self.fmean = mu

        fgam = np.zeros((M, N))
        for k in range(0, N):
            time0 = (self.time[-1] - self.time[0]) * gam[:, k] + self.time[0]
            fgam[:, k] = np.interp(time0, self.time, self.fmean)

        var_fgam = fgam.var(axis=1)
        self.orig_var = trapz(std_f0 ** 2, self.time)
        self.amp_var = trapz(std_fn ** 2, self.time)
        self.phase_var = trapz(var_fgam, self.time)            

        return


def pairwise_align_functions(f1, f2, time, omethod="DP2", lam=0, grid_dim=7):
    """
    This function aligns f2 to f1 using the elastic square-root
        slope (srsf) framework.

    Usage:  out = pairwise_align_functions(f1, f2, time)
            out = pairwise_align_functions(f1, f2, time, omethod, lam, grid_dim)
    
    :param f1: vector defining M samples of function 1
    :param f2: vector defining M samples of function 2
    :param time: time vector of length M
    :param omethod: optimization method (DP, DP2, RBFGS) (default = DP)
    :param lam: controls the elasticity (default = 0)
    :param grid_dim: size of the grid, for the DP2 method only (default = 7)

    :rtype list containing
    :return f2n: aligned f2
    :return gam: warping function
    :return q2n: aligned q2 (srsf)

    """

    q1 = uf.f_to_srsf(f1, time)
    q2 = uf.f_to_srsf(f2, time)

    gam = uf.optimum_reparam(q1, time, q2, omethod, lam, grid_dim)

    f2n = uf.warp_f_gamma(time, f2 , gam)
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
            out = pairwise_align_bayes(f1i, f2i, time, mcmcopts)
    
    :param f1i: vector defining M samples of function 1
    :param f2i: vector defining M samples of function 2
    :param time: time vector of length M
    :param mcmopts: dict of mcmc parameters
    :type mcmcopts: dict
  
    default mcmc options:
    tmp = {"betas":np.array([0.5,0.5,0.005,0.0001]),"probs":np.array([0.1,0.1,0.7,0.1])}
    mcmcopts = {"iter":2*(10**4) ,"burnin":np.minimum(5*(10**3),2*(10**4)//2),
                "alpha0":0.1, "beta0":0.1,"zpcn":tmp,"propvar":1,
                "initcoef":np.repeat(0,20), "npoints":200, "extrainfo":True}
   
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
        tmp = {"betas":np.array([0.5,0.5,0.005,0.0001]),"probs":np.array([0.1,0.1,0.7,0.1])}
        mcmcopts = {"iter":2*(10**4) ,"burnin":np.minimum(5*(10**3),2*(10**4)//2),"alpha0":0.1,
                    "beta0":0.1,"zpcn":tmp,"propvar":1,
                    "initcoef":np.repeat(0,20), "npoints":200, "extrainfo":True}

    if f1i.shape[0] != f2i.shape[0]:
        raise Exception('Length of f1 and f2 must be equal')

    if f1i.shape[0] != time.shape[0]:
        raise Exception('Length of f1 and time must be equal')
    
    if mcmcopts["zpcn"]["betas"].shape[0] != mcmcopts["zpcn"]["probs"].shape[0]:
        raise Exception('In zpcn, betas must equal length of probs')

    if np.mod(mcmcopts["initcoef"].shape[0], 2) != 0:
        raise Exception('Length of mcmcopts.initcoef must be even')

    # Number of sig figs to report in gamma_mat
    SIG_GAM = 13
    iter = mcmcopts["iter"]
    
    # parameter settings
    pw_sim_global_burnin = mcmcopts["burnin"]
    valid_index = np.arange(pw_sim_global_burnin-1,iter)
    pw_sim_global_Mg = mcmcopts["initcoef"].shape[0]//2
    g_coef_ini = mcmcopts["initcoef"]
    numSimPoints = mcmcopts["npoints"]
    pw_sim_global_domain_par = np.linspace(0,1,numSimPoints)
    g_basis = uf.basis_fourier(pw_sim_global_domain_par, pw_sim_global_Mg, 1)
    sigma1_ini = 1
    zpcn = mcmcopts["zpcn"]
    pw_sim_global_sigma_g = mcmcopts["propvar"] 

    def propose_g_coef(g_coef_curr):
        pCN_beta = zpcn["betas"]
        pCN_prob = zpcn["probs"]
        probm = np.insert(np.cumsum(pCN_prob),0,0)
        z = np.random.rand()
        result = {"prop":g_coef_curr,"ind":1}
        for i in range (0,pCN_beta.shape[0]):
            if z <= probm[i+1] and z > probm[i]:
                g_coef_new = normal(0, pw_sim_global_sigma_g / np.repeat(np.arange(1,pw_sim_global_Mg+1),2))
                result["prop"] = np.sqrt(1-pCN_beta[i]**2) * g_coef_curr + pCN_beta[i] * g_coef_new
                result["ind"] = i

        return result

    # normalize time to [0,1]
    time = (time - time.min())/(time.max()-time.min())
    timet = np.linspace(0,1,numSimPoints)
    f1 = uf.f_predictfunction(f1i,timet,0)
    f2 = uf.f_predictfunction(f2i,timet,0)

    # srsf transformation
    q1 = uf.f_to_srsf(f1,timet)
    q1i = uf.f_to_srsf(f1i,time)
    q2 = uf.f_to_srsf(f2,timet)

    tmp = uf.f_exp1(uf.f_basistofunction(g_basis["x"],0,g_coef_ini,g_basis))

    if tmp.min() < 0:
        raise Exception("Invalid initial value of g")

    # result vectors
    g_coef = np.zeros((iter,g_coef_ini.shape[0]))
    sigma1 = np.zeros(iter)
    logl = np.zeros(iter)
    SSE = np.zeros(iter)
    accept = np.zeros(iter, dtype=bool)
    accept_betas = np.zeros(iter)

    # init
    g_coef_curr = g_coef_ini
    sigma1_curr = sigma1_ini
    SSE_curr = bf.f_SSEg_pw(uf.f_basistofunction(g_basis["x"],0,g_coef_ini,g_basis),q1,q2)
    logl_curr = bf.f_logl_pw(uf.f_basistofunction(g_basis["x"],0,g_coef_ini,g_basis),q1,q2,sigma1_ini**2,SSE_curr)
    
    g_coef[0,:] = g_coef_ini
    sigma1[0] = sigma1_ini
    SSE[0] = SSE_curr
    logl[0] = logl_curr

    # update the chain for iter-1 times
    for m in tqdm(range(1,iter)):
        # update g
        g_coef_curr, tmp, SSE_curr, accepti, zpcnInd = bf.f_updateg_pw(g_coef_curr, g_basis, sigma1_curr**2, q1, q2, SSE_curr, propose_g_coef)
        
        # update sigma1
        newshape = q1.shape[0]/2 + mcmcopts["alpha0"]
        newscale = 1/2 * SSE_curr + mcmcopts["beta0"]
        sigma1_curr = np.sqrt(1/np.random.gamma(newshape,1/newscale))
        logl_curr = bf.f_logl_pw(uf.f_basistofunction(g_basis["x"],0,g_coef_curr,g_basis), q1, q2, sigma1_curr**2, SSE_curr)

        # save updates to results
        g_coef[m,:] = g_coef_curr
        sigma1[m] = sigma1_curr
        SSE[m] = SSE_curr
        if mcmcopts["extrainfo"]:
            logl[m] = logl_curr
            accept[m] = accepti
            accept_betas[m] = zpcnInd

    # calculate posterior mean of psi
    pw_sim_est_psi_matrix = np.zeros((numSimPoints,valid_index.shape[0]))
    for k in range(0,valid_index.shape[0]):
        g_temp = uf.f_basistofunction(g_basis["x"],0,g_coef[valid_index[k],:],g_basis)
        psi_temp = uf.f_exp1(g_temp)
        pw_sim_est_psi_matrix[:,k] = psi_temp

    result_posterior_psi_simDomain = uf.f_psimean(pw_sim_global_domain_par, pw_sim_est_psi_matrix)

    # resample to same number of points as the input f1 and f2
    interp = interp1d(np.linspace(0,1,result_posterior_psi_simDomain.shape[0]), result_posterior_psi_simDomain, fill_value="extrapolate")
    result_posterior_psi = interp(np.linspace(0,1,f1i.shape[0]))

    # transform posterior mean of psi to gamma
    result_posterior_gamma = uf.f_phiinv(result_posterior_psi)
    result_posterior_gamma = uf.norm_gam(result_posterior_gamma)

    # warped f2
    f2_warped = uf.warp_f_gamma(time, f2i, result_posterior_gamma)

    if mcmcopts["extrainfo"]:
        M,N = pw_sim_est_psi_matrix.shape
        gamma_mat = np.zeros((time.shape[0],N))
        one_v = np.ones(M)
        Dx = np.zeros(N)
        Dy = Dx
        for ii in range(0,N):
            interp = interp1d(np.linspace(0,1,result_posterior_psi_simDomain.shape[0]), pw_sim_est_psi_matrix[:,ii], fill_value="extrapolate")
            result_i = interp(time)
            tmp = uf.f_phiinv(result_i)
            gamma_mat[:,ii] = uf.norm_gam(tmp)
            v, theta = geo.inv_exp_map(one_v,pw_sim_est_psi_matrix[:,ii])
            Dx[ii] = np.sqrt(trapz(v**2,pw_sim_global_domain_par))
            q2warp = uf.warp_q_gamma(pw_sim_global_domain_par,q2,gamma_mat[:,ii])
            Dy[ii] = np.sqrt(trapz((q1i-q2warp)**2,time))

        gamma_stats = uf.statsFun(gamma_mat)

    
    results_o = collections.namedtuple('align_bayes', ['f2_warped', 'gamma','g_coef', 'psi', 'sigma1', 'accept', 'betas_ind', 'logl', 'gamma_mat', 'gamma_stats', 'xdist', 'ydist'])

    out = results_o(f2_warped, result_posterior_gamma, g_coef, result_posterior_psi, sigma1, accept[1:], accept_betas[1:], logl, gamma_mat, gamma_stats, Dx, Dy)

    return(out)


def pairwise_align_bayes_infHMC(y1i, y2i, time, mcmcopts=None):
    """
    This function aligns two functions using Bayesian framework. It uses a 
    hierarchical Bayesian framework assuming mearsurement error error It will 
    align f2 to f1. It is based on mapping warping functions to a hypersphere, 
    and a subsequent exponential mapping to a tangent space. In the tangent space,
    the \infty-HMC algorithm is used to explore both local and global
    structure in the posterior distribution.
   
    Usage:  out = pairwise_align_bayes_infHMC(f1i, f2i, time)
            out = pairwise_align_bayes_infHMC(f1i, f2i, time, mcmcopts)
    
    :param y1i: vector defining M samples of function 1
    :param y2i: vector defining M samples of function 2
    :param time: time vector of length M
    :param mcmopts: dict of mcmc parameters
    :type mcmcopts: dict
  
    default mcmc options:
    mcmcopts = {"iter":1*(10**4), "nchains":4, "vpriorvar":1, 
                "burnin":np.minimum(5*(10**3),2*(10**4)//2),
                "alpha0":0.1, "beta0":0.1, "alpha":1, "beta":1,
                "h":0.01, "L":4, "f1propvar":0.0001, "f2propvar":0.0001,
                "L1propvar":0.3, "L2propvar":0.3, "npoints":200, "thin":1,
                "sampfreq":1, "initcoef":np.repeat(0,20), "nbasis":10, 
                "basis":'fourier', "extrainfo":True}
    
    Basis can be 'fourier' or 'legendre'
   
    :rtype collection containing
    :return f2_warped: aligned f2
    :return gamma: warping function
    :return v_coef: final v_coef
    :return psi: final psi
    :return sigma1: final sigma
    
    if extrainfo
    :return theta_accept: accept of psi samples
    :return f2_accept: accept of f2 samples
    :return SSE: SSE
    :return gamma_mat: posterior gammas
    :return gamma_stats: posterior gamma stats
    :return xdist: phase distance posterior
    :return ydist: amplitude distance posterior)

    J. D. Tucker, L. Shand, and K. Chowdhary. “Multimodal Bayesian Registration of Noisy Functions using Hamiltonian Monte Carlo”, Computational Statistics and Data Analysis, accepted, 2021.
    """

    if mcmcopts is None:
        mcmcopts = {"iter":1*(10**4), "nchains":4 , "vpriorvar":1, 
                    "burnin":np.minimum(5*(10**3),2*(10**4)//2),
                    "alpha0":0.1, "beta0":0.1, "alpha":1, "beta":1,
                    "h":0.01, "L":4, "f1propvar":0.0001, "f2propvar":0.0001,
                    "L1propvar":0.3, "L2propvar":0.3, "npoints":200, "thin":1,
                    "sampfreq":1, "initcoef":np.repeat(0,20), "nbasis":10, 
                    "basis":'fourier', "extrainfo":True}

    if y1i.shape[0] != y2i.shape[0]:
        raise Exception('Length of f1 and f2 must be equal')

    if y1i.shape[0] != time.shape[0]:
        raise Exception('Length of f1 and time must be equal')

    if np.mod(mcmcopts["initcoef"].shape[0], 2) != 0:
        raise Exception('Length of mcmcopts.initcoef must be even')
    
    if np.mod(mcmcopts["nbasis"], 2) != 0:
        raise Exception('Length of mcmcopts.nbasis must be even')

    # set up random start points for more than 1 chain
    random_starts = np.zeros((mcmcopts["initcoef"].shape[0], mcmcopts["nchains"]))
    if mcmcopts["nchains"] > 1:
        for i in range(0, mcmcopts["nchains"]):
            randcoef = -1 + (2)*rand(mcmcopts["initcoef"].shape[0])
            random_starts[:, i] = randcoef
    
    isparallel = True
    if mcmcopts["nchains"] == 1:
        isparallel = False
    
    if isparallel:
        mcmcopts_p = []
        for i in range(0, mcmcopts["nchains"]):
            mcmcopts["initcoef"] = random_starts[:, i]
            mcmcopts_p.append(mcmcopts)
    
    # run chains
    if isparallel:
        chains = Parallel(n_jobs=-1)(delayed(run_mcmc)(y1i, y2i, time, 
                               mcmcopts_p[n]) for n in range(mcmcopts["nchains"]))

    else:
        chains = []
        chains1 = run_mcmc(y1i, y2i, time, mcmcopts)
        chains.append(chains1)
    
    # combine outputs
    Nsamples = chains[0]['f1'].shape[0]
    M = chains[0]['f1'].shape[1]
    f1 = np.zeros((Nsamples*mcmcopts["nchains"], M))
    f2 = np.zeros((Nsamples*mcmcopts["nchains"], M))
    gamma = np.zeros((M, mcmcopts["nchains"]))
    v_coef = np.zeros((Nsamples*mcmcopts["nchains"], chains[0]['v_coef'].shape[1]))
    psi = np.zeros((M, Nsamples*mcmcopts["nchains"]))
    sigma = np.zeros(Nsamples*mcmcopts["nchains"])
    sigma1 = np.zeros(Nsamples*mcmcopts["nchains"])
    sigma2 = np.zeros(Nsamples*mcmcopts["nchains"])
    s1 = np.zeros(Nsamples*mcmcopts["nchains"])
    s2 = np.zeros(Nsamples*mcmcopts["nchains"])
    L1 = np.zeros(Nsamples*mcmcopts["nchains"])
    L2 = np.zeros(Nsamples*mcmcopts["nchains"])
    f2_warped_mu = np.zeros((M, mcmcopts["nchains"]))

    if mcmcopts["extrainfo"]:
        Nsamplesa = chains[0]['theta_accept'].shape[0]
        theta_accept = np.zeros(Nsamplesa*mcmcopts["nchains"])
        f1_accept = np.zeros(Nsamplesa*mcmcopts["nchains"])
        f2_accept = np.zeros(Nsamplesa*mcmcopts["nchains"])
        L1_accept = np.zeros(Nsamplesa*mcmcopts["nchains"])
        L2_accept = np.zeros(Nsamplesa*mcmcopts["nchains"])
        gamma_mat = np.zeros((M,Nsamplesa*mcmcopts["nchains"]))
        SSE = np.zeros((Nsamplesa+1)*mcmcopts["nchains"])
        logl = np.zeros((Nsamplesa+1)*mcmcopts["nchains"])
        f2_warped = np.zeros((Nsamples*mcmcopts["nchains"], M))
        phasedist = np.zeros(Nsamples*mcmcopts["nchains"])
        ampdist = np.zeros(Nsamples*mcmcopts["nchains"])

    for i in range(0, mcmcopts["nchains"]):
        a = (i)*Nsamples
        b = (i+1)*Nsamples
        f1[a:b, :] = chains[i]['f1']
        f2[a:b, :] = chains[i]['f2']
        gamma[:, i] = chains[i]['gamma']
        v_coef[a:b, :] = chains[i]['v_coef']
        psi[:, i] = chains[i]['psi']
        sigma[a:b] = chains[i]['sigma']
        sigma1[a:b] = chains[i]['sigma1']
        sigma2[a:b] = chains[i]['sigma2']
        s1[a:b] = chains[i]['s1']
        s2[a:b] = chains[i]['s2']
        L1[a:b] = chains[i]['L1']
        L2[a:b] = chains[i]['L2']
        f2_warped_mu[:, i] = chains[i]['f2_warped_mu']

        if mcmcopts["extrainfo"]:
            a1 = (i)*Nsamplesa
            b1 = (i+1)*Nsamplesa
            theta_accept[a1:b1] = chains[i]['theta_accept']
            f1_accept[a1:b1] = chains[i]['f1_accept']
            f2_accept[a1:b1] = chains[i]['f2_accept']
            L1_accept[a1:b1] = chains[i]['L1_accept']
            L2_accept[a1:b1] = chains[i]['L2_accept']
            gamma_mat[:, a:b] = chains[i]['gamma_mat']
            a1 = (i)*(Nsamplesa)
            b1 = (i+1)*Nsamplesa
            SSE[a1:b1] = chains[i]['SSE']
            logl[a1:b1] = chains[i]['logl']
            f2_warped[a:b, :] = chains[i]['f2_warped']
            phasedist[a:b] = chains[i]['phasedist']
            ampdist[a:b] = chains[i]['ampdist']
    
    # finding modes
    if mcmcopts["nchains"] > 1:
        Dx = np.zeros((mcmcopts["nchains"], mcmcopts["nchains"]))
        time1 = np.linspace(0,1,gamma.shape[0])
        binsize = np.diff(time1)
        binsize = binsize.mean()
        for i in range(0, mcmcopts["nchains"]):
            for j in range(i+1,mcmcopts["nchains"]):
                psi1 = np.sqrt(np.gradient(gamma[:, i], binsize))
                psi2 = np.sqrt(np.gradient(gamma[:, j], binsize))
                q1dotq2 = trapz(psi1*psi2, time1)
                if q1dotq2 > 1:
                    q1dotq2 = 1
                elif q1dotq2 < -1:
                    q1dotq2 = -1

                Dx[i,j] = np.real(np.arccos(q1dotq2))
        
        Dx = Dx + Dx.T

        # cluster modes
        y = squareform(Dx)
        Z = linkage(y, method='complete')
        cutoff = np.median(Dx)
        T = fcluster(Z, cutoff, criterion='distance')
        N = np.unique(T)

        # find mean and confidence region of cluster
        posterior_gamma_modes = np.zeros((M, N.shape[0]))
        posterior_gamma_modes_cr = np.zeros((M, 2, N.shape[0]))
        for i in range(1, N.shape[0]+1):
            idx = np.where(T == i)[0]
            tmp = np.zeros((M, Nsamples*idx.shape[0]))
            for j in range(0, idx.shape[0]):
                a = (j)*Nsamples
                b = (j+1)*Nsamples
                tmp[:, a:b] = chains[idx[j]]['gamma_mat']
            mu, gam_mu, psi, vec = uf.SqrtMean(tmp)
            posterior_gamma_modes[:, i-1] = gam_mu
            posterior_gamma_modes_cr[:, :, i-1] = uf.statsFun(tmp)
        
    # thining
    f1 = f1[0::mcmcopts["thin"], :]
    f2 = f2[0::mcmcopts["thin"], :]
    v_coef = v_coef[0::mcmcopts["thin"], :]
    sigma = sigma[0::mcmcopts["thin"]]
    sigma1 = sigma1[0::mcmcopts["thin"]]
    sigma2 = sigma2[0::mcmcopts["thin"]]
    s1 = s1[0::mcmcopts["thin"]]
    s2 = s2[0::mcmcopts["thin"]]
    L1 = L1[0::mcmcopts["thin"]]
    L2 = L2[0::mcmcopts["thin"]]

    if mcmcopts["extrainfo"]:
        theta_accept = theta_accept[0::mcmcopts["thin"]]
        f1_accept = f1_accept[0::mcmcopts["thin"]]
        f2_accept = f2_accept[0::mcmcopts["thin"]]
        L1_accept = L1_accept[0::mcmcopts["thin"]]
        L2_accept = L2_accept[0::mcmcopts["thin"]]
        gamma_mat = gamma_mat[:, 0::mcmcopts["thin"]]
        SSE = SSE[0::mcmcopts["thin"]]
        logl = logl[0::mcmcopts["thin"]]
        f2_warped = f2_warped[0::mcmcopts["thin"], :]
        phasedist = phasedist[0::mcmcopts["thin"]]
        ampdist = ampdist[0::mcmcopts["thin"]]


    if mcmcopts["extrainfo"]:
        results_o = collections.namedtuple('align_bayes_HMC', ['f1', 'f2', 'gamma', 'v_coef', 'psi', 'sigma', 'sigma1', 'sigma2', 's1', 's2', 'L1', 'L2', 'f2_warped_mu', 'theta_accept', 'f1_accept', 'f2_accept', 'L1_accept', 'L2_accept', 'gamma_mat', 'SSE', 'logl', 'f2_warped', 'phasedist', 'ampdist'])

        out = results_o(f1, f2, gamma, v_coef, psi, sigma, sigma1, sigma2, s1, s2, L1, L2, f2_warped_mu,
                        theta_accept, f1_accept, f2_accept, L1_accept, L2_accept, gamma_mat, SSE, logl,
                        f2_warped, phasedist, ampdist)

    else:
        results_o = collections.namedtuple('align_bayes_HMC', ['f1', 'f2', 'gamma', 'v_coef', 'psi', 'sigma', 'sigma1', 'sigma2', 's1', 's2', 'L1', 'L2', 'f2_warped_mu'])

        out = results_o(f1, f2, gamma, v_coef, psi, sigma, sigma1, sigma2, s1, s2, L1, L2, f2_warped_mu)

    return(out)


def run_mcmc(y1i, y2i, time, mcmcopts):
    # Number of sig figs to report in gamma_mat
    SIG_GAM = 13
    iter = mcmcopts["iter"]
    T = time.shape[0]

    # normalize time to [0,1]
    time = (time - time.min())/(time.max()-time.min())

    # parameter settings
    pw_sim_global_burnin = mcmcopts["burnin"]
    valid_index = np.arange(pw_sim_global_burnin-1,iter)
    ncoef = mcmcopts["initcoef"].shape[0]
    nbasis = mcmcopts["nbasis"]
    pw_sim_global_Mv = ncoef//2
    numSimPoints = T
    pw_sim_global_domain_par = np.linspace(0,1,numSimPoints)
    d_basis = uf.basis_fourierd(pw_sim_global_domain_par, nbasis)
    if mcmcopts["basis"] == 'fourier':
        v_basis = uf.basis_fourier(pw_sim_global_domain_par, pw_sim_global_Mv, 1)
    elif mcmcopts["basis"] == 'legendre':
        v_basis = uf.basis_legendre(pw_sim_global_domain_par, pw_sim_global_Mv, 1)
    else:
        raise Exception('Incorrect Basis Specified')
    sigma_ini = 1
    v_priorvar = mcmcopts["vpriorvar"]
    v_coef_ini = mcmcopts["initcoef"]
    D = pdist(time.reshape((time.shape[0],1)))
    Dmat = squareform(D)
    C = v_priorvar / np.repeat(np.arange(1,pw_sim_global_Mv+1), 2)
    cholC = cholesky(np.diag(C))
    h = mcmcopts["h"]
    L = mcmcopts["L"]

    def propose_v_coef(v_coef_curr):
        v_coef_new = normal(v_coef_curr, C.T)
        return v_coef_new

    # f1,f2 prior, propoposal params
    sigma1_ini = 0.01
    sigma2_ini = 0.01
    f1_propvar = mcmcopts["f1propvar"]
    f2_propvar = mcmcopts["f2propvar"]
    y1itmp = y1i[0::mcmcopts["sampfreq"]]
    timetmp = time[0::mcmcopts["sampfreq"]]
    kernel1 = GPy.kern.RBF(input_dim=1, variance=y1itmp.std()/np.sqrt(2), lengthscale=np.mean(timetmp.std()))
    y2itmp = y2i[0::mcmcopts["sampfreq"]]
    kernel2 = GPy.kern.RBF(input_dim=1, variance=y2itmp.std()/np.sqrt(2), lengthscale=np.mean(timetmp.std()))
    M1 = timetmp.shape[0]
    model1 = GPy.models.GPRegression(timetmp.reshape((M1,1)),y1itmp.reshape((M1,1)),kernel1)
    model1.optimize()
    model2 = GPy.models.GPRegression(timetmp.reshape((M1,1)),y2itmp.reshape((M1,1)),kernel2)
    model2.optimize()

    s1_ini = model1.kern.param_array[0]
    s2_ini = model2.kern.param_array[0]
    L1_propvar = mcmcopts["L1propvar"]
    L2_propvar = mcmcopts["L2propvar"]
    L1_ini = model2.kern.param_array[1]
    L2_ini = model2.kern.param_array[1]

    K_f1_corr = uf.exp2corr2(L1_ini,Dmat)+0.1 * np.eye(y1i.shape[0])
    K_f1 = s1_ini * K_f1_corr
    K_f1 = inv(K_f1)
    K_f2_corr = uf.exp2corr2(L2_ini,Dmat)+0.1 * np.eye(y2i.shape[0])
    K_f2 = s2_ini * K_f2_corr
    K_f2 = inv(K_f2)
    K_f1prop= uf.exp2corr(f1_propvar,L1_ini,Dmat)
    K_f2prop= uf.exp2corr(f2_propvar,L2_ini,Dmat)

    # result vectors
    v_coef = np.zeros((iter,v_coef_ini.shape[0]))
    sigma = np.zeros(iter)
    sigma1 = np.zeros(iter)
    sigma2 = np.zeros(iter)
    f1 = np.zeros((iter,time.shape[0]))
    f2 = np.zeros((iter,time.shape[0]))
    f2_warped = np.zeros((iter,time.shape[0]))
    s1 = np.zeros(iter)
    s2 = np.zeros(iter)
    L1 = np.zeros(iter)
    L2 = np.zeros(iter)
    logl = np.zeros(iter)
    SSE = np.zeros(iter)
    SSEprop = np.zeros(iter)
    theta_accept = np.zeros(iter, dtype=bool)
    f1_accept = np.zeros(iter, dtype=bool)
    f2_accept = np.zeros(iter, dtype=bool)
    L1_accept = np.zeros(iter, dtype=bool)
    L2_accept = np.zeros(iter, dtype=bool)

    # init
    v_coef_curr = v_coef_ini
    v_curr = uf.f_basistofunction(v_basis["x"],0,v_coef_ini,v_basis)
    sigma_curr = sigma_ini
    sigma1_curr = sigma1_ini
    sigma2_curr = sigma2_ini
    L1_curr = L1_ini
    L2_curr = L2_ini

    f1_curr, predvar = model1.predict(time.reshape((T,1)))
    f1_curr = f1_curr.reshape(T)
    f2_curr, predvar = model2.predict(time.reshape((T,1)))
    f2_curr = f2_curr.reshape(T)

    # srsf transformation
    q1_curr = uf.f_to_srsf(f1_curr, time)
    q2_curr = uf.f_to_srsf(f2_curr, time)

    SSE_curr = bf.f_SSEv_pw(v_curr, q1_curr, q2_curr)
    logl_curr, SSEv = bf.f_vpostlogl_pw(v_curr, q1_curr, q2_curr, sigma_curr, SSE_curr)

    v_coef[0,:] = v_coef_ini
    f1[0,:] = f1_curr
    f2[0,:] = f2_curr
    f2_warped[0,:] = f2_curr
    sigma[0] = sigma_ini
    sigma1[0] = sigma1_ini
    sigma2[0] = sigma2_ini
    s1[0] = s1_ini
    s2[0] = s2_ini
    L1[0] = L1_ini
    L2[0] = L2_ini
    SSE[0] = SSE_curr
    SSEprop[0] = SSE_curr
    logl[0] = logl_curr

    n = f1_curr.shape[0]

    nll, g, SSE_curr = bf.f_dlogl_pw(v_coef_curr, v_basis, d_basis, sigma_curr, q1_curr, q2_curr)

    # update the chain for iter-1 times
    for m in range(1,iter):

        # update f1
        f1_curr, q1_curr, f1_accept1 = bf.f_updatef1_pw(f1_curr,q1_curr, y1i, q2_curr,v_coef_curr, v_basis,
                                                       SSE_curr,K_f1,K_f1prop,sigma_curr,np.sqrt(sigma1_curr))

        # update f2
        f2_curr, q2_curr, f2_accept1 = bf.f_updatef2_pw(f2_curr,q2_curr, y2i, q1_curr,v_coef_curr, v_basis,
                                                       SSE_curr,K_f2,K_f2prop,sigma_curr,np.sqrt(sigma2_curr))

        # update v
        v_coef_curr, nll, g, SSE_curr, theta_accept1 = bf.f_updatev_pw(v_coef_curr, v_basis, np.sqrt(sigma_curr),
                                                                      q1_curr, q2_curr,nll, g,SSE_curr,
                                                                      propose_v_coef,d_basis,cholC,h,L)
        
        # update sigma^2
        newshape = q1_curr.shape[0]/2 + mcmcopts["alpha"]
        newscale = 1/2 * SSE_curr + mcmcopts["beta"]
        sigma_curr = 1/np.random.gamma(newshape, 1/newscale)
        
        # update sigma1^2
        newshape = n/2 + mcmcopts["alpha0"]
        newscale = np.sum((y1i-f1_curr)**2)/2 + mcmcopts["beta0"]
        sigma1_curr = 1/np.random.gamma(newshape, 1/newscale)
        
        # update sigma^2
        newshape = n/2 + mcmcopts["alpha0"]
        newscale = np.sum((y2i-f2_curr)**2)/2 + mcmcopts["beta0"]
        sigma2_curr = 1/np.random.gamma(newshape, 1/newscale)

        # update hyperparameters
        # update s1^2
        newshape = n/2 + mcmcopts["alpha0"]
        newscale = (uf.mrdivide(f1_curr,K_f1_corr) @ f1_curr.T)/2 + mcmcopts["beta0"]
        s1_curr = 1/np.random.gamma(newshape, 1/newscale)

        # update s2^2
        newshape = n/2 + mcmcopts["alpha0"]
        newscale = (uf.mrdivide(f2_curr,K_f2_corr) @ f2_curr.T)/2 + mcmcopts["beta0"]
        s2_curr = 1/np.random.gamma(newshape, 1/newscale)

        # update L1
        L1_curr, L1_accept1 = bf.f_updatephi_pw(f1_curr,K_f1,s1_curr, L1_curr, L1_propvar, Dmat)

        # update L2
        L2_curr, L2_accept1 = bf.f_updatephi_pw(f2_curr,K_f2,s2_curr, L2_curr, L2_propvar, Dmat)

        K_f1_corr = uf.exp2corr2(L1_curr,Dmat)+0.1 * np.eye(y1i.shape[0])
        K_f1 = s1_curr * K_f1_corr
        K_f1 = inv(K_f1)
        K_f2_corr = uf.exp2corr2(L2_curr,Dmat)+0.1 * np.eye(y2i.shape[0])
        K_f2 = s2_curr * K_f2_corr
        K_f2 = inv(K_f2)

        v_curr = uf.f_basistofunction(v_basis["x"], 0, v_coef_curr, v_basis)
        logl_curr, SSEv1 = bf.f_vpostlogl_pw(v_curr, q1_curr, q2_curr, sigma_curr, SSE_curr)

        # save updates to results
        v_coef[m,:] = v_coef_curr
        f1[m,:] = f1_curr
        f2[m,:] = f2_curr
        sigma[m] = sigma_curr
        sigma1[m] = sigma1_curr
        sigma2[m] = sigma2_curr
        s1[m] = s1_curr
        s2[m] = s2_curr
        L1[m] = L1_curr
        L2[m] = L2_curr
        SSE[m] = SSE_curr
        logl[m] = logl_curr
        if mcmcopts["extrainfo"]:
            theta_accept[m] = theta_accept1
            f1_accept[m] = f1_accept1
            f2_accept[m] = f2_accept1
            L1_accept[m] = L1_accept1
            L2_accept[m] = L2_accept1

    # calculate posterior mean of psi
    pw_sim_est_psi_matrix = np.zeros((pw_sim_global_domain_par.shape[0],valid_index.shape[0]))
    for k in range(0,valid_index.shape[0]):
        v_temp = uf.f_basistofunction(v_basis["x"],0,v_coef[valid_index[k],:],v_basis)
        psi_temp = uf.f_exp1(v_temp)
        pw_sim_est_psi_matrix[:,k] = psi_temp

    result_posterior_psi_simDomain = uf.f_psimean(pw_sim_global_domain_par, pw_sim_est_psi_matrix)

    # resample to same number of points as the input f1 and f2
    interp = interp1d(np.linspace(0,1,result_posterior_psi_simDomain.shape[0]), result_posterior_psi_simDomain, fill_value="extrapolate")
    result_posterior_psi = interp(np.linspace(0,1,y1i.shape[0]))

    # transform posterior mean of psi to gamma
    result_posterior_gamma = uf.f_phiinv(result_posterior_psi)
    result_posterior_gamma = uf.norm_gam(result_posterior_gamma)

    if mcmcopts["extrainfo"]:
        M,N = pw_sim_est_psi_matrix.shape
        gamma_mat = np.zeros((time.shape[0],N))
        one_v = np.ones(M)
        Dx = np.zeros(N)
        Dy = Dx
        for ii in range(0,N):
            interp = interp1d(np.linspace(0,1,result_posterior_psi_simDomain.shape[0]), pw_sim_est_psi_matrix[:,ii], fill_value="extrapolate")
            result_i = interp(time)
            tmp = uf.f_phiinv(result_i)
            gamma_mat[:,ii] = uf.norm_gam(tmp)
            v, theta = geo.inv_exp_map(one_v,pw_sim_est_psi_matrix[:,ii])
            Dx[ii] = np.sqrt(trapz(v**2,pw_sim_global_domain_par))
            q2warp = uf.warp_q_gamma(pw_sim_global_domain_par,q2_curr,gamma_mat[:,ii])
            Dy[ii] = np.sqrt(trapz((q1_curr-q2warp)**2,time))

        gamma_stats = uf.statsFun(gamma_mat)


    f1 = f1[valid_index, :]
    f2 = f2[valid_index, :]
    gamma = result_posterior_gamma
    v_coef = v_coef[valid_index, :]
    psi = result_posterior_psi
    sigma = sigma[valid_index]
    sigma1 = sigma1[valid_index]
    sigma2 = sigma2[valid_index]
    s1 = s1[valid_index]
    s2 = s2[valid_index]
    L1 = L1[valid_index]
    L2 = L2[valid_index]
    SSE = SSE[valid_index]
    logl = logl[valid_index]
    f2_warped_mu = uf.warp_f_gamma(time, f2.mean(axis=0), gamma)

    if mcmcopts["extrainfo"]:
        theta_accept = theta_accept[valid_index]
        f1_accept = f1_accept[valid_index]
        f2_accept = f2_accept[valid_index]
        L1_accept = L1_accept[valid_index]
        L2_accept = L2_accept[valid_index]

        phasedist = Dx
        ampdist = Dy

        f2_warped = np.zeros((valid_index.shape[0], result_posterior_gamma.shape[0]))
        for ii in range(0, valid_index.shape[0]):
            f2_warped[ii,:] = uf.warp_f_gamma(time, f2[ii,:], gamma_mat[:,ii])

    if mcmcopts["extrainfo"]:
        out_dict = {"v_coef":v_coef, "sigma":sigma, "sigma1":sigma1, "sigma2":sigma2, "f1":f1,   
                    "f2_warped_mu":f2_warped_mu, "f2":f2, "s1":s1, "gamma":gamma, "psi":psi, "s2":s2, 
                    "L1":L1, "L2":L2, "logl":logl, "SSE":SSE, "theta_accept":theta_accept,"f1_accept":f1_accept, 
                    "f2_accept":f2_accept, "L1_accept":L1_accept, "L2_accept":L2_accept, "phasedist":phasedist, 
                    "ampdist":ampdist, "f2_warped":f2_warped, "gamma_mat":gamma_mat, "gamma_stats":gamma_stats}
    else:
        out_dict = {"v_coef":v_coef, "sigma":sigma, "sigma1":sigma1, "sigma2":sigma2, "f1":f1, 
                    "f2_warped_mu":f2_warped_mu, "f2":f2, "gamma":gamma, "psi":psi, "s1":s1, "s2":s2, 
                    "L1":L1, "L2":L2, "logl":logl, "SSE":SSE}

    return(out_dict)


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
    coef = np.arange(-2., 3.)
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

    print ("Initializing...")
    mnq = q.mean(axis=1)
    a = mnq.repeat(N)
    d1 = a.reshape(M, N)
    d = (q - d1) ** 2
    dqq = np.sqrt(d.sum(axis=0))
    min_ind = dqq.argmin()

    print("Aligning %d functions in SRVF space to %d fPCA components..."
          % (N, num_comp))
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
                alpha_i[ii, jj] = trapz(qhat_cent[:, jj] * U[:, ii], time)

        U1 = U[:, 0:num_comp]
        tmp = U1.dot(alpha_i)
        qhat = d1 + tmp

        # Matching Step
        if parallel:
            out = Parallel(n_jobs=cores)(
                delayed(uf.optimum_reparam)(qhat[:, n], time, qi[:, n, itr],
                                            "DP", lam) for n in range(N))
            gam_t = np.array(out)
            gam[:, :, itr] = gam_t.transpose()
        else:
            gam[:, :, itr] = uf.optimum_reparam(qhat, time, qi[:, :, itr], "DP",  lam)

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
        itrf = itr+1
    cost = cost[1:(itrf+1)]

    # Aligned data & stats
    fn = fi[:, :, itrf]
    qn = qi[:, :, itrf]
    q0 = qi[:, :, 0]
    mean_f0 = f0.mean(axis=1)
    std_f0 = f0.std(axis=1)
    mqn = mq[:, itrf]
    gamf = gam[:, :, 0]
    for k in range(1, itr):
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

        # Align Plots
        fig, ax = plot.f_plot(np.arange(0, M) / float(M - 1), gamf,
                              title="Warping Functions")
        ax.set_aspect('equal')

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
        idx = np.arange(0, M + 1) + 1
        plot.f_plot(idx, cumm_coef, "Coefficient Cumulative Percentage")
        plt.xlabel("Percentage")
        plt.ylabel("Index")
        plt.show()

    mean_f0 = f0.mean(axis=1)
    std_f0 = f0.std(axis=1)
    mean_fn = fn.mean(axis=1)
    std_fn = fn.std(axis=1)
    tmp = np.zeros(M)
    tmp[1:] = cumtrapz(mqn * np.abs(mqn), time)
    fmean = np.mean(f0[1, :]) + tmp

    fgam = np.zeros((M, N))
    for k in range(0, N):
        time0 = (time[-1] - time[0]) * gamf[:, k] + time[0]
        fgam[:, k] = np.interp(time0, time, fmean)

    var_fgam = fgam.var(axis=1)
    orig_var = trapz(std_f0 ** 2, time)
    amp_var = trapz(std_fn ** 2, time)
    phase_var = trapz(var_fgam, time)

    K = np.cov(fn)

    U, s, V = svd(K)

    align_fPCAresults = collections.namedtuple('align_fPCA', ['fn', 'qn',
                                               'q0', 'mqn', 'gam', 'q_pca',
                                               'f_pca', 'latent', 'coef',
                                               'U', 'orig_var', 'amp_var',
                                               'phase_var', 'cost'])

    out = align_fPCAresults(fn, qn, q0, mqn, gamf, q_pca, f_pca, s, c,
                            U, orig_var, amp_var, phase_var, cost)
    return out


def align_fPLS(f, g, time, comps=3, showplot=True, smoothdata=False,
               delta=0.01, max_itr=100):
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
    :return fn: aligned functions - numpy ndarray of shape (M,N) of N
    functions with M samples
    :return gn: aligned functions - numpy ndarray of shape (M,N) of N
    functions with M samples
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
    print ("Initializing...")
    binsize = np.diff(time)
    binsize = binsize.mean()
    eps = np.finfo(np.double).eps
    M = f.shape[0]
    N = f.shape[1]
    f0 = f
    g0 = g

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
    wqf1, wqg1, alpha, values, costmp = pls_svd(time, qfi[:, :, itr],
                                                qgi[:, :, itr], 2, 0)
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
        gam[:, :, itr + 1] = fpls.fpls_warp(time, gamtmp, qftmp, qgtmp,
                                            wqftmp, wqgtmp, display=0,
                                            delta=delta, tol=1e-6,
                                            max_iter=4000)

        for k in range(0, N):
            gam_k = gam[:, k, itr + 1]
            time0 = (time[-1] - time[0]) * gam_k + time[0]
            fi[:, k, itr + 1] = np.interp(time0, time, fi[:, k, 0])
            gi[:, k, itr + 1] = np.interp(time0, time, gi[:, k, 0])
            qfi[:, k, itr + 1] = uf.warp_q_gamma(time, qfi[:, k, 0], gam_k)
            qgi[:, k, itr + 1] = uf.warp_q_gamma(time, qgi[:, k, 0], gam_k)

        # PLS
        wqfi, wqgi, alpha, values, costmp = pls_svd(time, qfi[:, :, itr + 1],
                                                    qgi[:, :, itr + 1], 2, 0)
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

        print("Iteration: %d - Diff Value: %f - %f" % (itr + 1, wqf_diff[itr],
                                                       cost[itr]))
        if wqf_diff[itr] < 1e-1 or abs(cost_diff) < 1e-3:
            break

        itr += 1

    cost = cost[0:(itr + 1)]

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
        wf[:, ii] = cumtrapz(wqfn[:, ii] * np.abs(wqfn[:, ii]), time, initial=0)
        wg[:, ii] = cumtrapz(wqgn[:, ii] * np.abs(wqgn[:, ii]), time, initial=0)

    gam_f = gam[:, :, itr + 1]

    if showplot:
        # Align Plots
        fig, ax = plot.f_plot(np.arange(0, M) / float(M - 1), gam_f,
                              title="Warping Functions")
        ax.set_aspect('equal')

        plot.f_plot(time, fn, title="fn Warped Data")
        plot.f_plot(time, gn, title="gn Warped Data")
        plot.f_plot(time, wf, title="wf")
        plot.f_plot(time, wg, title="wg")

        plt.show()

    align_fPLSresults = collections.namedtuple('align_fPLS', ['wf', 'wg', 'fn',
                                               'gn', 'qfn', 'qgn', 'qf0',
                                               'qg0', 'wqf', 'wqg', 'gam',
                                               'values', 'cost'])

    out = align_fPLSresults(wf, wg, fn, gn, qfn, qgn, qf0, qg0, wqfn,
                            wqgn, gam_f, values, cost)
    return out
