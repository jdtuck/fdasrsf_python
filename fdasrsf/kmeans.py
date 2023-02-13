"""
Elastic Functional Clustering

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf.utility_functions as uf
from scipy.integrate import trapz
from numpy.linalg import norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
                    gam_tmp = gam_tmp.transpose()
                else:
                    for n in range(0,N):
                        gam_tmp[:,n] = uf.optimum_reparam(templates_q[:, k], time, q[:, n],
                                omethod, lam)
            else:
                for n in range(0,N):
                    gam_tmp[:,k] = np.linspace(0,1,M)
            
            fw = np.zeros((M,N))
            qw = np.zeros((M,N))
            dist = np.zeros(N)
            for i in range(0, N):
                fw[:, i] = uf.warp_f_gamma(time, f[:,i], gam_tmp[:,i])
                qw[:, i] = uf.f_to_srsf(fw[:, i], time)
                dist[i] = np.sqrt(trapz((qw[:, i] - templates_q[:, k]) ** 2, time))
            
            Dy[k,:] = dist
            qn[k] = qw
            fn[k] = fw
            gam[k] = gam_tmp

        # Assignment
        cluster_id = Dy.argmin(axis=0)

        # Normalization
        for k in range(K):
            idx = np.where(cluster_id == k)[0]
            ftmp = fn[k][:,idx]
            gamtmp = gam[k][:,idx]
            gamI = uf.SqrtMeanInverse(gamtmp)
            N1 = idx.shape[0]

            gamt = np.zeros((M,N1))
            f_temp = np.zeros((M,N1))
            q_temp = np.zeros((M,N1))
            if parallel:
                out = Parallel(n_jobs=cores)(delayed(norm_sub)(ftmp[:, i],
                                            time, gamtmp[:,i], gamI) for i in range(N1))
                for i in range(0, N1):
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
        qun_t = np.zeros(K)
        old_templates_q = templates_q.copy()
        for k in range(K):
            idx = np.where(cluster_id == k)[0]
            templates_q[:,k] = qn[k][:,idx].mean(axis=1)
            templates[:,k] = fn[k][:,idx].mean(axis=1)

            qun_t[k] = norm(templates_q[:,k] - old_templates_q[:,k])/norm(old_templates_q[:,k])
        
        qun[itr] = qun_t.mean()

        if qun[itr] < thresh:
            break

    # Output
    ftmp = {}
    qtmp = {}
    gamtmp = {}
    for k in range(K):
        idx = np.where(cluster_id == k)[0]
        ftmp[k] = fn[k][:,idx]
        qtmp[k] = qn[k][:,idx]
        gamtmp[k] = gam[k][:,idx]
    
    out = {}
    out['f0'] = f
    out['q0'] = q
    out['time'] = time
    out['fn'] = ftmp
    out['qn'] = qtmp
    out['gam'] = gamtmp
    out['labels'] = cluster_id
    out['templates'] = templates
    out['templates_q'] = templates_q
    out['lambda'] = lam
    out['omethod'] = omethod
    out['qun'] = qun[0:itr]

    if showplot:
        num_plot = int(np.ceil(K/6))
        a = mcolors.TABLEAU_COLORS
        colors = list(a.keys())
        plt.figure()
        plt.plot(time, f)
        plt.title('Original Data')

        plt.figure()
        plt.plot(time, templates)
        plt.title('Cluster Mean Functions')

        for k in range(num_plot):
            cnt = 1
            plt.figure()
            for n in np.arange(k*6,min(K,(k+1)*6),dtype=int):
                ax = plt.subplot(2, 3, cnt)
                ax.plot(time, ftmp[n], color='lightgrey')
                ax.plot(time, templates[:, n], color=colors[cnt-1])
                ax.set_title('Cluster f: %d' % n)
                cnt += 1
            
        for k in range(num_plot):
            cnt = 1
            plt.figure()
            for n in np.arange(k*6,min(K,(k+1)*6),dtype=int):
                ax = plt.subplot(2, 3, cnt)
                ax.plot(time, qtmp[n], color='lightgrey')
                ax.plot(time, templates_q[:, n], color=colors[cnt-1])
                ax.set_title('Cluster q: %d' % n)
                cnt += 1
            
        plt.show()

    return out


def norm_sub(f,time,gam,gamI):
    fw = uf.warp_f_gamma(time, f, gamI)
    qw = uf.f_to_srsf(fw, time)
    time0 = (time[-1] - time[0]) * gamI + time[0]
    gamw = np.interp(time0, time, gam)

    return(fw, qw, gamw)
