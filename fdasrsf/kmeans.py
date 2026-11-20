"""
Elastic Functional Clustering

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf.utility_functions as uf
from scipy.integrate import trapezoid
from numpy.linalg import norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def kmeans_align(
    f,
    time,
    K,
    seeds=None,
    lam=0,
    showplot=True,
    smooth_data=False,
    parallel=False,
    alignment=True,
    omethod="DP2",
    MaxItr=50,
    thresh=0.01,
):
    r"""
    This function clusters functions and aligns using the elastic square-root
    slope (srsf) framework.

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param K: number of clusters
    :param seeds: indexes of cluster center functions (default = None)
    :param lam: controls the elasticity (default = 0)
    :param showplot: shows plots of functions (default = T)
    :param smooth_data: smooth data using a smoothing spline (default = F)
    :param parallel: enable parallel mode using joblib (default = F)
    :param alignment: whether to perform alignment (default = T)
    :param omethod: optimization method (DP,DP2,RBFGS)
    :param MaxItr: maximum number of iterations
    :param thresh: cost function threshold
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
    :return gamI: inverse Karcher mean warp of each cluster
    :return qun: Cost Function

    """

    cores = -1
    M = f.shape[0]
    N = f.shape[1]

    if seeds is None:
        template_ind = np.random.choice(N, K, replace=False)
    else:
        template_ind = np.asarray(seeds, dtype=int)

    # convert to SRSF (smoothing first, so the templates are smoothed too)
    f, q = _to_srsf(f, time, smooth_data)

    templates = f[:, template_ind].copy()
    templates_q = q[:, template_ind].copy()

    cluster_id = np.zeros(N, dtype=int)
    qun = np.zeros(MaxItr)
    gamI = [np.linspace(0, 1, M) for k in range(K)]

    for itr in range(0, MaxItr):
        print("updating step: r=%d" % (itr + 1))
        if itr == MaxItr - 1:
            print("maximal number of iterations is reached.")

        # Alignment
        gam = {}
        Dy = np.zeros((K, N))
        qn = {}
        fn = {}
        for k in range(K):
            gam[k], fn[k], qn[k], Dy[k, :] = _align_to_template(
                f, q, time, templates_q[:, k], omethod, lam, alignment, parallel, cores
            )

        # Assignment
        cluster_id = Dy.argmin(axis=0)

        # Normalization
        for k in range(K):
            idx = np.where(cluster_id == k)[0]
            if idx.shape[0] == 0:
                # empty cluster: nothing to normalize
                gamI[k] = np.linspace(0, 1, M)
                continue
            gamI[k], fn[k][:, idx], qn[k][:, idx], gam[k][:, idx] = _normalize(
                fn[k][:, idx], gam[k][:, idx], time, parallel, cores
            )

        # Template Identification
        qun_t = np.zeros(K)
        for k in range(K):
            idx = np.where(cluster_id == k)[0]
            if idx.shape[0] == 0:
                # empty cluster: keep the previous template
                continue
            old_template_q = templates_q[:, k].copy()
            templates_q[:, k] = qn[k][:, idx].mean(axis=1)
            # keep the f-template consistent with the SRSF template
            templates[:, k] = uf.srsf_to_f(
                templates_q[:, k], time, fn[k][0, idx].mean()
            )
            qun_t[k] = norm(templates_q[:, k] - old_template_q) / norm(old_template_q)

        qun[itr] = qun_t.mean()

        if qun[itr] < thresh:
            break

    # Output
    ftmp = {}
    qtmp = {}
    gamtmp = {}
    for k in range(K):
        idx = np.where(cluster_id == k)[0]
        ftmp[k] = fn[k][:, idx]
        qtmp[k] = qn[k][:, idx]
        gamtmp[k] = gam[k][:, idx]

    out = {}
    out["f0"] = f
    out["q0"] = q
    out["time"] = time
    out["fn"] = ftmp
    out["qn"] = qtmp
    out["gam"] = gamtmp
    out["gamI"] = gamI
    out["labels"] = cluster_id
    out["templates"] = templates
    out["templates_q"] = templates_q
    out["lambda"] = lam
    out["omethod"] = omethod
    out["qun"] = qun[0 : itr + 1]

    if showplot:
        _plot_clusters(time, f, ftmp, qtmp, templates, templates_q)

    return out


def dpmeans_align(
    f,
    time,
    lam=None,
    lam_align=0,
    showplot=True,
    smooth_data=False,
    parallel=False,
    alignment=True,
    omethod="DP2",
    MaxItr=20,
    thresh=0.01,
):
    r"""
    This function clusters functions and aligns using the elastic square-root
    slope (srsf) framework with the DP-means algorithm (Kulis and Jordan,
    2012), the small-variance limit of a Dirichlet process mixture. A new
    cluster is opened whenever a function's elastic distance to every
    existing template exceeds lam, so the number of clusters is learned from
    the data. Each pass assigns and aligns the functions, normalizes the
    warps in each cluster, and updates the templates as in kmeans_align.

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param lam: cluster penalty, as an elastic distance (default = None:
                chosen by a k-means++ style heuristic). The objective
                minimized is sum of squared distances + lam^2 * K
    :param lam_align: alignment penalty passed to optimum_reparam (default = 0)
    :param showplot: shows plots of functions (default = T)
    :param smooth_data: smooth data using a smoothing spline (default = F)
    :param parallel: enable parallel mode using joblib (default = F)
    :param alignment: whether to perform alignment (default = T)
    :param omethod: optimization method (DP,DP2,RBFGS)
    :param MaxItr: maximum number of iterations
    :param thresh: relative change in the objective for convergence
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: dictionary
    :return fn: aligned functions in a dict of clusters
    :return qn: aligned SRSFs - similar structure to fn
    :return q0: original SRSFs
    :return labels: cluster labels
    :return templates: cluster center functions
    :return templates_q: cluster center SRSFs
    :return gam: warping functions - similar structure to fn
    :return gamI: inverse Karcher mean warp of each cluster
    :return K: number of clusters
    :return lambda: cluster penalty used
    :return qun: DP-means objective at each iteration

    """
    cores = -1
    M = f.shape[0]
    N = f.shape[1]

    f, q = _to_srsf(f, time, smooth_data)

    # start with one cluster whose template is the function closest to the
    # cross-sectional mean srsf
    mnq = q.mean(axis=1)
    dqq = np.sqrt(trapezoid((q - mnq[:, np.newaxis]) ** 2, time, axis=0))
    min_ind = dqq.argmin()
    K = 1
    templates = f[:, [min_ind]].copy()
    templates_q = q[:, [min_ind]].copy()

    def align(tq):
        return _align_to_template(
            f, q, time, tq, omethod, lam_align, alignment, parallel, cores
        )

    if lam is None or np.isnan(lam):
        print("Initializing lambda")
        lam = _kpp_init(f, q, align, 4)
    print("lambda = %5.3f" % lam)

    cluster_id = np.zeros(N, dtype=int)
    dmin = np.zeros(N)
    fa = np.zeros((M, N))
    qa = np.zeros((M, N))
    ga = np.zeros((M, N))
    gamI = []
    qun = np.zeros(MaxItr)

    for itr in range(0, MaxItr):
        print("updating step: r=%d" % (itr + 1))
        if itr == MaxItr - 1:
            print("maximal number of iterations is reached.")

        # Assignment and alignment: align every function to every template
        gam1 = []
        fn1 = []
        qn1 = []
        Dy = []
        for k in range(K):
            g, fw, qw, d = align(templates_q[:, k])
            gam1.append(g)
            fn1.append(fw)
            qn1.append(qw)
            Dy.append(d)

        # sequential DP-means pass: a function farther than lam from every
        # template opens a new cluster, which later functions in the pass
        # can join
        for n in range(N):
            dn = np.array([Dy[k][n] for k in range(K)])
            ik = dn.argmin()
            dk = dn[ik]
            if dk > lam:
                templates = np.column_stack((templates, f[:, n]))
                templates_q = np.column_stack((templates_q, q[:, n]))
                g, fw, qw, d = align(q[:, n])
                gam1.append(g)
                fn1.append(fw)
                qn1.append(qw)
                Dy.append(d)
                ik = K
                dk = d[n]
                K += 1
            cluster_id[n] = ik
            dmin[n] = dk

        # drop clusters that lost all their functions
        used, cluster_id = np.unique(cluster_id, return_inverse=True)
        cluster_id = cluster_id.ravel()
        K = used.shape[0]
        templates = templates[:, used]
        templates_q = templates_q[:, used]
        gam1 = [gam1[i] for i in used]
        fn1 = [fn1[i] for i in used]
        qn1 = [qn1[i] for i in used]

        # DP-means objective at this assignment
        qun[itr] = (dmin**2).sum() + lam**2 * K

        # keep each function aligned to its own template
        for n in range(N):
            ga[:, n] = gam1[cluster_id[n]][:, n]
            fa[:, n] = fn1[cluster_id[n]][:, n]
            qa[:, n] = qn1[cluster_id[n]][:, n]

        # Normalization
        gamI = []
        for k in range(K):
            idx = np.where(cluster_id == k)[0]
            gIk, fa[:, idx], qa[:, idx], ga[:, idx] = _normalize(
                fa[:, idx], ga[:, idx], time, parallel, cores
            )
            gamI.append(gIk)

        # Template Identification
        for k in range(K):
            idx = np.where(cluster_id == k)[0]
            templates_q[:, k] = qa[:, idx].mean(axis=1)
            templates[:, k] = uf.srsf_to_f(templates_q[:, k], time, fa[0, idx].mean())

        if itr > 0 and abs(qun[itr] - qun[itr - 1]) < thresh * qun[itr - 1]:
            break

    # Output
    ftmp = {}
    qtmp = {}
    gamtmp = {}
    for k in range(K):
        idx = np.where(cluster_id == k)[0]
        ftmp[k] = fa[:, idx]
        qtmp[k] = qa[:, idx]
        gamtmp[k] = ga[:, idx]

    out = {}
    out["f0"] = f
    out["q0"] = q
    out["time"] = time
    out["fn"] = ftmp
    out["qn"] = qtmp
    out["gam"] = gamtmp
    out["gamI"] = gamI
    out["labels"] = cluster_id
    out["templates"] = templates
    out["templates_q"] = templates_q
    out["K"] = K
    out["lambda"] = lam
    out["lambda_align"] = lam_align
    out["omethod"] = omethod
    out["qun"] = qun[0 : itr + 1]

    if showplot:
        _plot_clusters(time, f, ftmp, qtmp, templates, templates_q)

    return out


def _to_srsf(f, time, smooth_data):
    eps = np.finfo(np.double).eps
    f, g, g2 = uf.gradient_spline(time, f, smooth_data)
    q = g / np.sqrt(abs(g) + eps)
    return f, q


def _align_to_template(f, q, time, tq, omethod, lam, alignment, parallel, cores):
    """
    align every function to one template srsf tq

    returns gam, fn, qn of shape (M,N) and the elastic distances (N,)
    """
    M, N = f.shape
    if not alignment:
        gam = np.tile(np.linspace(0, 1, M)[:, np.newaxis], (1, N))
    elif parallel:
        out = Parallel(n_jobs=cores)(
            delayed(uf.optimum_reparam)(tq, time, q[:, n], omethod, lam)
            for n in range(N)
        )
        gam = np.array(out).transpose()
    else:
        gam = np.zeros((M, N))
        for n in range(N):
            gam[:, n] = uf.optimum_reparam(tq, time, q[:, n], omethod, lam)

    fw = np.zeros((M, N))
    qw = np.zeros((M, N))
    dist = np.zeros(N)
    for n in range(N):
        fw[:, n] = uf.warp_f_gamma(time, f[:, n], gam[:, n])
        qw[:, n] = uf.f_to_srsf(fw[:, n], time)
        dist[n] = np.sqrt(trapezoid((qw[:, n] - tq) ** 2, time))

    return gam, fw, qw, dist


def _normalize(fn, gam, time, parallel, cores):
    """
    center the warps of one cluster: warp the aligned functions and warps
    by the inverse Karcher mean warp gamI (warps as columns)
    """
    M, N1 = fn.shape
    gamI = uf.SqrtMeanInverse(gam)
    f_temp = np.zeros((M, N1))
    q_temp = np.zeros((M, N1))
    gamt = np.zeros((M, N1))
    if parallel:
        out = Parallel(n_jobs=cores)(
            delayed(norm_sub)(fn[:, i], time, gam[:, i], gamI) for i in range(N1)
        )
    else:
        out = [norm_sub(fn[:, i], time, gam[:, i], gamI) for i in range(N1)]
    for i in range(N1):
        f_temp[:, i], q_temp[:, i], gamt[:, i] = out[i]

    return gamI, f_temp, q_temp, gamt


def _kpp_init(f, q, align, K):
    """
    choose the DP-means penalty from a k-means++ seeding

    Picks K centers by k-means++ (sampling proportional to the squared
    elastic distance to the nearest chosen center) and returns the largest
    distance from any function to its nearest center (Kulis and Jordan, 2012).
    """
    N = q.shape[1]
    D = []
    idx = np.random.randint(N)
    for i in range(K):
        D.append(align(q[:, idx])[3])
        if i < K - 1:
            w = np.min(np.array(D), axis=0) ** 2
            if w.sum() == 0:
                break
            idx = np.searchsorted(np.cumsum(w), np.random.rand() * w.sum(), "right")
            idx = min(idx, N - 1)
    return np.max(np.min(np.array(D), axis=0))


def _plot_clusters(time, f, ftmp, qtmp, templates, templates_q):
    K = templates.shape[1]
    num_plot = int(np.ceil(K / 6))
    a = mcolors.TABLEAU_COLORS
    colors = list(a.keys())
    plt.figure()
    plt.plot(time, f)
    plt.title("Original Data")

    plt.figure()
    plt.plot(time, templates)
    plt.title("Cluster Mean Functions")

    for k in range(num_plot):
        cnt = 1
        plt.figure()
        for n in np.arange(k * 6, min(K, (k + 1) * 6), dtype=int):
            ax = plt.subplot(2, 3, cnt)
            ax.plot(time, ftmp[n], color="lightgrey")
            ax.plot(time, templates[:, n], color=colors[cnt - 1])
            ax.set_title("Cluster f: %d" % n)
            cnt += 1

    for k in range(num_plot):
        cnt = 1
        plt.figure()
        for n in np.arange(k * 6, min(K, (k + 1) * 6), dtype=int):
            ax = plt.subplot(2, 3, cnt)
            ax.plot(time, qtmp[n], color="lightgrey")
            ax.plot(time, templates_q[:, n], color=colors[cnt - 1])
            ax.set_title("Cluster q: %d" % n)
            cnt += 1

    plt.show()


def norm_sub(f, time, gam, gamI):
    fw = uf.warp_f_gamma(time, f, gamI)
    qw = uf.f_to_srsf(fw, time)
    time0 = (time[-1] - time[0]) * gamI + time[0]
    gamw = np.interp(time0, time, gam)

    return (fw, qw, gamw)
