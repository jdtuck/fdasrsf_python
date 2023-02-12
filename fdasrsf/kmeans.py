"""
Elastic Functional Clustering

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf.utility_functions as uf
from scipy.integrate import trapz
from joblib import Parallel, delayed


def kmeans_align(f, time, K, seeds=None, lam=0, showplot=True, smooth_data=False,
                 parallel=False, alignment=True, omethod="DP2", 
                 MaxItr=50, thresh=0.01):
    """
    This function clusters functions and aligns using the elastic square-root
    slope (srsf) framework.

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param K number of clusters
    :param seeds indexes of cluster center functions (default = None)
    :param lam controls the elasticity (default = 0)
    :param showplot shows plots of functions (default = T)
    :param smooth_data smooth data using box filter (default = F)
    :param parallel enable parallel mode using \code{\link{joblib}} and
     \code{doParallel} package (default=F)
    :param alignment whether to perform alignment (default = T)
    :param omethod optimization method (DP,DP2,RBFGS)
    :param MaxItr maximum number of iterations
    :param thresh cost function threshold
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: dictionary
    :return fn: aligned functions - matrix (N x M) of M functions with N samples which is a list for each cluster
    :return qn: aligned SRSFs - similar structure to fn
    :return q0: original SRSFs
    :return labels: cluster labels
    :return templates: cluster center functions
    :return templates_q: cluster center SRSFs
    :return gam: warping functions - similar structure to fn
    :return qun: Cost Function

    """

    w = 0.0
    k = 1
    cores = -1
    eps = np.finfo(np.double).eps

    M = f.shape[0]
    N = f.shape[1]

    if seeds is None:
        a = np.arange(0, N, dtype=int)
        template_ind = np.random.choice(a, K)
    else:
        template_ind = seeds
    
    templates = np.zeros((M,K))
    for i in range(K):
        templates[:,i] = f[:,template_ind[i]]

    cluster_id = np.zeros(N, dtype=int)
    qun = np.zeros(MaxItr)

    # convert to SRSF
    f, g, g2 = uf.gradient_spline(time, f, smooth_data)
    q = g / np.sqrt(abs(g) + eps)
    templates_q = np.zeros((M,K))
    for i in range(K):
        templates_q[:,i] = q[:,template_ind[i]]
    
    for itr in range(0, MaxItr):
        print("updating step: r=%d" % (itr + 1))

        # Alignment
        gam = {}
        Dy = np.zeros((K,N))
        qn = {}
        fn = {}

        for k in range(K):
            gam_tmp = np.zeros((M,N))
            if alignment:
                if parallel:
                    out = Parallel(n_jobs=cores)(delayed(uf.optimum_reparam)(templates_q[:, k],
                                            time, q[:, n], omethod, lam) for n in range(N))
                    gam_tmp = np.array(out)
                    gam_tmp = gam.transpose()
                else:
                    for n in range(0,N):
                        gam_tmp[:,k] = uf.optimum_reparam(templates_q[:, k], time, q[:, n],
                                omethod, lam)
            else:
                for n in range(0,N):
                    gam_tmp[:,k] = np.linspace(0,1,M)
            
            fw = np.zeros((M,N))
            qw = np.zeros((M,N))
            dist = np.zeros(N)
            for i in range(0, N):
                fw[:, i] = uf.warp_f_gamma(time, f[:,i], gam_tmp[:,i])
                qw[:, k] = uf.f_to_srsf(fw[:, i], time)
                dist[i] = np.sqrt(trapz((qw - templates_q[:, k]) ** 2, time))
            
            Dy[k,:] = dist
            qn[k] = qw
            fn[k] = fw
            gam[k] = gam_tmp

        # Assignment
        cluster_id = Dy.argmin(axis=0)

        # Normalization
        for k in range(K):
            idx = np.where(cluster_id == k)
            ftmp = fn[i][:,idx]
            gamtmp = gam[i][:,idx]
            gamI = uf.SqrtMeanInverse(gamtmp)
            N1 = idx.shape[0]

            gamt = np.zeros((M,N1))
            f_temp = np.zeros((M,N1))
            q_temp = np.zeros((M,N1))
            if parallel:
                out = Parallel(n_jobs=cores)(delayed(norm_sub)(ftmp[:, i],
                                            time, gamtmp[:,i], gamI) for i in range(N1))
                for i in range(0, fns):
                    f_temp[:,i] = out[i][0]
                    q_temp[:, i] = out[i][1]
                    gamt[:, i] = out[i][2]
            else:
                for i in range(N1):
                    f_temp[:,i], q_temp[:, i], gamt[:, i] = norm_sub(ftmp[:, i], time, gamtmp[:,i], gamI)

            qn[k][:,idx] = q_temp
            fn[k][:,idx] = f_temp
            gam[k][:,idx] = gamt
        
        # Template Identification




    

    return


def norm_sub(f,time,gam,gamI):
    fw = uf.warp_f_gamma(time, f, gamI)
    qw = uf.f_to_srsf(fw, time)
    time0 = (time[-1] - time[0]) * gamI + time[0]
    gamw = np.interp(time0, time, gam)

    return(fw, qw, gamw)