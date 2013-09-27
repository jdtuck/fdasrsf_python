"""
Group-wise function alignment using SRSF framework and Dynamic Programming

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""
import numpy as np
import matplotlib.pyplot as plt
import utility_functions as uf
from scipy.integrate import simps, cumtrapz, trapz
from numpy.linalg import norm
from joblib import Parallel, delayed
import plot_style as plot
import collections


def srsf_align(f, time, method="mean", showplot=True, smoothdata=False, lam=0.0):
    """
    This function aligns a collection of functions using the elastic square-root slope (srsf) framework.

    :param f: numpy ndarray of shape (M,N) of M functions with N samples
    :param time: vector of size N describing the sample points
    :param method: (string) warp calculate Karcher Mean or Median (options = "mean" or "median") (default="mean")
    :param showplot: Shows plots of results using matplotlib (default = T)
    :param smoothdata: Smooth the data using a box filter (default = F)
    :param sparam: Number of times to run box filter (default = 25)
    :param lam: controls the elasticity (default = 0)
    :type lam: double
    :type sparam: double
    :type smoothdata: bool
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of numpy array
    :return fn: aligned functions - numpy ndarray of shape (M,N) of M functions with N samples
    :return qn: aligned srvfs - similar structure to fn
    :return q0: original srvf - similar structure to fn
    :return fmean: function mean or median - vector of length N
    :return mqn: srvf mean or median - vector of length N
    :return gam: warping functions - similar structure to fn
    :return orig_var: Original Variance of Functions
    :return amp_var: Amplitude Variance
    :return phase_var: Phase Variance

    Examples
    >>> import tables
    >>> fun=tables.open_file("../Data/simu_data.h5")
    >>> f = fun.root.f[:]
    >>> f = f.transpose()
    >>> time = fun.root.time[:]
    >>> out = srsf_align(f,time)

    """
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

    methods = ["mean", "median"]
    method = [i for i, x in enumerate(methods) if x == method]  # 0 mean, 1-median

    if method != 0 or method != 1:
        method = 0

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
    mq = q[:, min_ind]
    mf = f[:, min_ind]

    if parallel:
        out = Parallel(n_jobs=-1)(delayed(uf.optimum_reparam)(mq, time, q[:, n], lam) for n in range(N))
        gam = np.array(out)
        gam = gam.transpose()
    else:
        gam = uf.optimum_reparam(mq, time, q, lam)

    gamI = uf.SqrtMeanInverse(gam)
    mf = np.interp((time[-1] - time[0]) * gamI + time[0], time, mf)
    mq = uf.f_to_srsf(mf, time)

    # Compute Karcher Mean
    if method == 0:
        print "Compute Karcher Mean of %d function in SRSF space..." % N
    if method == 1:
        print "Compute Karcher Median of %d function in SRSF space..." % N

    MaxItr = 20
    ds = np.repeat(0.0, MaxItr + 2)
    ds[0] = np.inf
    qun = np.repeat(0.0, MaxItr + 1)
    tmp = np.zeros((M, MaxItr + 2))
    tmp[:, 0] = mq
    mq = tmp
    tmp = np.zeros((M, N, MaxItr + 2))
    tmp[:, :, 0] = f
    f = tmp
    tmp = np.zeros((M, N, MaxItr + 2))
    tmp[:, :, 0] = q
    q = tmp

    for r in xrange(0, MaxItr):
        print "updating step: r=%d" % (r + 1)
        if r == (MaxItr - 1):
            print "maximal number of iterations is reached"

        # Matching Step
        if parallel:
            out = Parallel(n_jobs=-1)(delayed(uf.optimum_reparam)(mq[:, r], time, q[:, n, 0], lam) for n in range(N))
            gam = np.array(out)
            gam = gam.transpose()
        else:
            gam = uf.optimum_reparam(mq[:, r], time, q[:, :, 0], lam)

        gam_dev = np.zeros((M, N))
        for k in xrange(0, N):
            f[:, k, r + 1] = np.interp((time[-1] - time[0]) * gam[:, k] + time[0], time, f[:, k, 0])
            q[:, k, r + 1] = uf.f_to_srsf(f[:, k, r + 1], time)
            gam_dev[:, k] = np.gradient(gam[:, k], 1 / float(M - 1))

        mqt = mq[:, r]
        a = mqt.repeat(N)
        d1 = a.reshape(M, N)
        d = (q[:, :, r + 1] - d1) ** 2
        if method == 0:
            ds_tmp = sum(simps(d, time, axis=0)) + lam * sum(simps((1 - np.sqrt(gam_dev)) ** 2, time, axis=0))
            ds[r + 1] = ds_tmp

            # Minimization Step
            # compute the mean of the matched function
            qtemp = q[:, :, r + 1]
            mq[:, r + 1] = qtemp.mean(axis=1)

            qun[r] = norm(mq[:, r + 1] - mq[:, r]) / norm(mq[:, r])

        if method == 1:
            ds_tmp = np.sqrt(sum(simps(d, time, axis=0))) + lam * sum(simps((1 - np.sqrt(gam_dev)) ** 2, time, axis=0))
            ds[r + 1] = ds_tmp

            # Minimization Step
            # compute the mean of the matched function
            dist_iinv = ds[r + 1] ** (-1)
            qtemp = q[:, :, r + 1] / ds[r + 1]
            mq[:, r + 1] = qtemp.sum(axis=1) * dist_iinv

            qun[r] = norm(mq[:, r + 1] - mq[:, r]) / norm(mq[:, r])

        if qun[r] < 1e-2 or r >= MaxItr:
            break

    # Last Step with centering of gam
    r += 1
    if parallel:
        out = Parallel(n_jobs=-1)(delayed(uf.optimum_reparam)(mq[:, r], time, q[:, n, 0], lam) for n in range(N))
        gam = np.array(out)
        gam = gam.transpose()
    else:
        gam = uf.optimum_reparam(mq[:, r], time, q[:, :, 0], lam)

    gam_dev = np.zeros((M, N))
    for k in xrange(0, N):
        gam_dev[:, k] = np.gradient(gam[:, k], 1 / float(M - 1))

    gamI = uf.SqrtMeanInverse(gam)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    mq[:, r + 1] = np.interp((time[-1] - time[0]) * gamI + time[0], time, mq[:, r]) * np.sqrt(gamI_dev)

    for k in xrange(0, N):
        q[:, k, r + 1] = np.interp((time[-1] - time[0]) * gamI + time[0], time, q[:, k, r]) * np.sqrt(gamI_dev)
        f[:, k, r + 1] = np.interp((time[-1] - time[0]) * gamI + time[0], time, f[:, k, r])
        gam[:, k] = np.interp((time[-1] - time[0]) * gamI + time[0], time, gam[:, k])

    # Aligned data & stats
    fn = f[:, :, r + 1]
    qn = q[:, :, r + 1]
    q0 = q[:, :, 0]
    mean_f0 = f0.mean(axis=1)
    std_f0 = f0.std(axis=1)
    mean_fn = fn.mean(axis=1)
    std_fn = fn.std(axis=1)
    mqn = mq[:, r + 1]
    tmp = np.zeros((1, M))
    tmp = tmp.flatten()
    tmp[1:] = cumtrapz(mqn * np.abs(mqn), time)
    fmean = np.mean(f0[1, :]) + tmp

    fgam = np.zeros((M, N))
    for k in xrange(0, N):
        fgam[:, k] = np.interp((time[-1] - time[0]) * gam[:, k] + time[0], time, fmean)

    var_fgam = fgam.var(axis=1)
    orig_var = trapz(std_f0 ** 2, time)
    amp_var = trapz(std_fn ** 2, time)
    phase_var = trapz(var_fgam, time)

    if showplot:
        fig, ax = plot.f_plot(np.arange(0, M) / float(M - 1), gam, title="Warping Functions")
        ax.set_aspect('equal')

        plot.f_plot(time, fn, title="Warped Data")

        tmp = np.array([mean_f0, mean_f0 + std_f0, mean_f0 - std_f0])
        tmp = tmp.transpose()
        plot.f_plot(time, tmp, title="Original Data: Mean $\pm$ STD")

        tmp = np.array([mean_fn, mean_fn + std_fn, mean_fn - std_fn])
        tmp = tmp.transpose()
        plot.f_plot(time, tmp, title="Warped Data: Mean $\pm$ STD")

        plot.f_plot(time, fmean, title="$f_{mean}$")
        plt.show()

    align_results = collections.namedtuple('align', ['fn', 'qn', 'q0', 'fmean', 'mqn', 'gam', 'orig_var', 'amp_var',
                                                     'phase_var'])
    out = align_results(fn, qn, q0, fmean, mqn, gam, orig_var, amp_var, phase_var)
    return out


def align_fPCA(f, time, num_comp=3, showplot=True, smooth_data=False, sparam=25):
    """
    aligns a collection of functions while extracting principal components. The functions are aligned to the principal
    components

    :param f: numpy ndarray of shape (M,N) of M functions with N samples
    :param time: vector of size N describing the sample points
    :param num_comp: number of fPCA components
    :param showplot: Shows plots of results using matplotlib (default = T)
    :param smooth_data: Smooth the data using a box filter (default = F)
    :param sparam: Number of times to run box filter (default = 25)
    :type sparam: double
    :type smooth_data: bool
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of numpy array
    :return fn: aligned functions - numpy ndarray of shape (M,N) of M functions with N samples
    :return qn: aligned srvfs - similar structure to fn
    :return q0: original srvf - similar structure to fn
    :return mqn: srvf mean or median - vector of length N
    :return gam: warping functions - similar structure to fn
    :return q_pca: srsf principal directions
    :return f_pca: functional principal directions
    :return latent: latent values
    :return coef: coefficients
    :return U: eigenvectors

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

    if smooth_data:
        f = uf.smooth_data(f, sparam)

    if showplot:
        plot.f_plot(time, f, title="Original Data")

    # Compute SRSF function from data
    f, g, g2 = uf.gradient_spline(time, f)
    q = g / np.sqrt(abs(g) + eps)

    print ("Initializing...")
    mnq = q.mean(axis=1)
    a = mnq.repeat(N)
    d1 = a.reshape(M, N)
    d = (q - d1) ** 2
    dqq = np.sqrt(d.sum(axis=0))
    min_ind = dqq.argmin()

    print "Aligning %d functions in SRVF space to %d fPCA components..." % (N, num_comp)
    itr = 0
    mq = np.zeros((M, MaxItr + 1))
    mq[:, itr] = q[:, min_ind]
    fi = np.zeros((M, N, MaxItr + 1))
    fi[:, :, 0] = f
    qi = np.zeros((M, N, MaxItr + 1))
    qi[:, :, 0] = q
    gam = np.zeros((M, N, MaxItr))
    cost = np.zeros(MaxItr + 1)

    while itr <= MaxItr:
        print "updating step: r=%d" % (itr + 1)
        if itr == MaxItr:
            print "maximal number of iterations is reached"

        # PCA Step
        a = mq[:, itr].repeat(N)
        d1 = a.reshape(M, N)
        qhat_cent = qi[:, :, itr] - d1
        K = np.cov(qi[:, :, itr])
        U, s, V = np.linalg.svd(K)

        alpha_i = np.zeros((num_comp, N))
        for ii in xrange(0, num_comp):
            for jj in xrange(0, N):
                alpha_i[ii, jj] = simps(qhat_cent[:, jj] * U[:, ii], time)

        U1 = U[:, 0:num_comp]
        tmp = U1.dot(alpha_i)
        qhat = d1 + tmp

        # Matching Step
        if parallel:
            out = Parallel(n_jobs=-1)(delayed(uf.optimum_reparam)(qhat[:, n], time, qi[:, n, itr], lam) for n in range(N))
            gam_t = np.array(out)
            gam[:, :, itr] = gam_t.transpose()
        else:
            gam[:, :, itr] = uf.optimum_reparam(qhat, time, qi[:, :, itr], lam)

        for k in xrange(0, N):
            fi[:, k, itr + 1] = np.interp((time[-1] - time[0]) * gam[:, k, itr] + time[0], time, fi[:, k, itr])
            qi[:, k, itr + 1] = uf.f_to_srsf(fi[:, k, itr + 1], time)

        qtemp = qi[:, :, itr + 1]
        mq[:, itr + 1] = qtemp.mean(axis=1)

        cost_temp = np.zeros(N)

        for ii in xrange(0, N):
            cost_temp[ii] = norm(qtemp[:, ii] - qhat[:, ii]) ** 2

        cost[itr + 1] = cost_temp.mean()

        if abs(cost[itr + 1] - cost[itr]) < 1e-05:
            break

        itr += 1

    cost = cost[1:itr + 1]

    # Aligned data & stats
    fn = fi[:, :, itr + 1]
    qn = qi[:, :, itr + 1]
    q0 = qi[:, :, 0]
    mean_f0 = f0.mean(axis=1)
    std_f0 = f0.std(axis=1)
    mqn = mq[:, itr + 1]
    gamf = gam[:, :, 0]
    for k in xrange(1, itr):
        gam_k = gam[:, :, k]
        for l in xrange(0, N):
            gamf[:, l] = np.interp((time[-1] - time[0]) * gam_k[:, l] + time[0], time, gamf[:, l])

    # Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    mqn = np.interp((time[-1] - time[0]) * gamI + time[0], time, mqn) * np.sqrt(gamI_dev)
    for k in xrange(0, N):
        qn[:, k] = np.interp((time[-1] - time[0]) * gamI + time[0], time, qn[:, k]) * np.sqrt(gamI_dev)
        fn[:, k] = np.interp((time[-1] - time[0]) * gamI + time[0], time, fn[:, k])
        gamf[:, k] = np.interp((time[-1] - time[0]) * gamI + time[0], time, gamf[:, k])

    mean_fn = fn.mean(axis=1)
    std_fn = fn.std(axis=1)

    # Get Final PCA
    mididx = np.round(time.shape[0] / 2)
    m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
    mqn2 = np.append(mqn, m_new.mean())
    qn2 = np.vstack((qn, m_new))
    K = np.cov(qn2)

    U, s, V = np.linalg.svd(K)
    stdS = np.sqrt(s)

    # compute the PCA in the q domain
    q_pca = np.ndarray(shape=(M + 1, Nstd, num_comp), dtype=float)
    for k in xrange(0, num_comp):
        for l in xrange(0, Nstd):
            q_pca[:, l, k] = mqn2 + coef[l] * stdS[k] * U[:, k]

    # compute the correspondence in the f domain
    f_pca = np.ndarray(shape=(M, Nstd, num_comp), dtype=float)
    for k in xrange(0, num_comp):
        for l in xrange(0, Nstd):
            f_pca[:, l, k] = uf.cumtrapzmid(time, q_pca[0:M, l, k] * np.abs(q_pca[0:M, l, k]),
                                            np.sign(q_pca[M, l, k]) * (q_pca[M, l, k] ** 2))

    N2 = qn.shape[1]
    c = np.zeros((N2, num_comp))
    for k in xrange(0, num_comp):
        for l in xrange(0, N2):
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
        fig, ax = plot.f_plot(np.arange(0, M) / float(M - 1), gamf, title="Warping Functions")
        ax.set_aspect('equal')

        plot.f_plot(time, fn, title="Warped Data")

        tmp = np.array([mean_f0, mean_f0 + std_f0, mean_f0 - std_f0])
        tmp = tmp.transpose()
        plot.f_plot(time, tmp, title="Original Data: Mean $\pm$ STD")

        tmp = np.array([mean_fn, mean_fn + std_fn, mean_fn - std_fn])
        tmp = tmp.transpose()
        plot.f_plot(time, tmp, title="Warped Data: Mean $\pm$ STD")

        # PCA Plots
        fig, ax = plt.subplots(2, num_comp)
        for k in xrange(0, num_comp):
            axt = ax[0, k]
            for l in xrange(0, Nstd):
                axt.plot(time, q_pca[0:M, l, k], color=CBcdict[cl[l]])
                axt.hold(True)

            axt.set_title('q domain: PD %d' % (k + 1))
            plot.rstyle(axt)
            axt = ax[1, k]
            for l in xrange(0, Nstd):
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

    align_fPCAresults = collections.namedtuple('align_fPCA', ['fn', 'qn', 'q0', 'mqn', 'gam', 'q_pca', 'f_pca',
                                                              'latent', 'coef', 'U'])
    out = align_fPCAresults(fn, qn, q0, mqn, gam, q_pca, f_pca, s, c, U)
    return out
