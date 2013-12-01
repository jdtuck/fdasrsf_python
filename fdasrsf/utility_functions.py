"""
Utility functions for SRSF Manipulations

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""

from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapz, cumtrapz, trapz
from scipy.linalg import norm, svd, cholesky, inv
from scipy.stats.mstats import mquantiles
from numpy import zeros, interp, finfo, double, sqrt, diff, linspace, arccos, sin, cos, arange, ascontiguousarray, round
from numpy import ones, real, pi, cumsum, fabs, cov, diagflat, inner, gradient, column_stack, row_stack, append
from numpy import insert, vectorize
import numpy.random as rn
import optimum_reparamN as orN
import sys


def smooth_data(f, sparam):
    """
    This function smooths a collection of functions using a box filter

    :param f: numpy ndarray of shape (M,N) of M functions with N samples
    :param sparam: Number of times to run box filter (default = 25)

    :rtype: numpy ndarray
    :return f: smoothed functions functions

    """
    M = f.shape[0]
    N = f.shape[1]

    for k in range(1, sparam):
        for r in range(0, N):
            f[1:(M - 2), r] = (f[0:(M - 3), r] + 2 * f[1:(M - 2), r] +
                               f[2:(M - 1), r]) / 4
    return f


def gradient_spline(time, f, smooth=False):
    """
    This function takes the gradient of f using b-spline smoothing

    :param time: vector of size N describing the sample points
    :param f: numpy ndarray of shape (M,N) of M functions with N samples
    :param smooth: smooth data (default = F)

    :rtype: tuple of numpy ndarray
    :return f0: smoothed functions functions
    :return g: first derivative of each function
    :return g2: second derivative of each function

    """
    M = f.shape[0]

    if f.ndim > 1:
        N = f.shape[1]
        f0 = zeros((M, N))
        g = zeros((M, N))
        g2 = zeros((M, N))
        for k in range(0, N):
            if smooth:
                spar = time.shape[0] * (.025 * fabs(f[:, k]).max()) ** 2
            else:
                spar = 0
            tmp_spline = UnivariateSpline(time, f[:, k], s=spar)
            f0[:, k] = tmp_spline(time)
            g[:, k] = tmp_spline(time, 1)
            g2[:, k] = tmp_spline(time, 2)
    else:
        if smooth:
            spar = time.shape[0] * (.025 * fabs(f).max()) ** 2
        else:
            spar = 0
        tmp_spline = UnivariateSpline(time, f, s=spar)
        f0 = tmp_spline(time)
        g = tmp_spline(time, 1)
        g2 = tmp_spline(time, 2)

    return f0, g, g2


def f_to_srsf(f, time):
    """
    converts f to a square-root slope function (SRSF)

    :param f: vector of size N samples
    :param time: vector of size N describing the sample points

    :rtype: vector
    :return q: srsf of f

    """
    eps = finfo(double).eps
    f0, g, g2 = gradient_spline(time, f)
    q = g / sqrt(abs(g) + eps)
    return q


def optimum_reparam(q1, time, q2, lam=0.0):
    """
    calculates the warping to align srsf q2 to q1

    :param q1: vector of size N or array of NxM samples of first SRSF
    :param time: vector of size N describing the sample points
    :param q2: vector of size N or array of NxM samples samples of second SRSF
    :param lam: controls the amount of elasticity (default = 0.0)

    :rtype: vector
    :return gam: describing the warping function used to align q2 with q1

    """
    if q1.ndim == 1 and q2.ndim == 1:
        gam = orN.coptimum_reparam(ascontiguousarray(q1), time,
                                   ascontiguousarray(q2), lam)

    if q1.ndim == 1 and q2.ndim == 2:
        gam = orN.coptimum_reparamN(ascontiguousarray(q1), time,
                                    ascontiguousarray(q2), lam)

    if q1.ndim == 2 and q2.ndim == 2:
        gam = orN.coptimum_reparamN2(ascontiguousarray(q1), time,
                                     ascontiguousarray(q2), lam)

    return gam


def optimum_reparam_pair(q, time, q1, q2, lam=0.0):
    """
    calculates the warping to align srsf pair q1 and q2 to q

    :param q: vector of size N or array of NxM samples of first SRSF
    :param time: vector of size N describing the sample points
    :param q1: vector of size N or array of NxM samples samples of second SRSF
    :param q2: vector of size N or array of NxM samples samples of second SRSF
    :param lam: controls the amount of elasticity (default = 0.0)

    :rtype: vector
    :return gam: describing the warping function used to align q2 with q1

    """
    if q1.ndim == 1 and q2.ndim == 1:
        q_c = column_stack((q1, q2))
        gam = orN.coptimum_reparam_pair(ascontiguousarray(q), time,
                                        ascontiguousarray(q_c), lam)

    if q1.ndim == 2 and q2.ndim == 2:
        gam = orN.coptimum_reparamN2_pair(ascontiguousarray(q), time,
                                          ascontiguousarray(q1),
                                          ascontiguousarray(q2), lam)

    return gam


def elastic_distance(f1, f2, time, lam=0.0):
    """"
    calculates the distances between function, where f1 is aligned to
    f2. In other words
    calculates the elastic distances

    :param f1: vector of size N
    :param f2: vector of size N
    :param time: vector of size N describing the sample points
    :param lam: controls the elasticity (default = 0.0)

    :rtype: scalar
    :return Dy: amplitude distance
    :return Dx: phase distance

    """
    q1 = f_to_srsf(f1, time)
    q2 = f_to_srsf(f2, time)

    gam = optimum_reparam(q1, time, q2, lam)
    fw = interp((time[-1] - time[0]) * gam + time[0], time, f2)
    qw = f_to_srsf(fw, time)

    Dy = sqrt(trapz((qw - q1) ** 2, time))
    M = time.shape[0]
    psi = sqrt(diff(gam) * (M - 1))
    mu = ones(M - 1)
    Dx = real(arccos(sum(mu * psi) / double(M - 1)))

    return Dy, Dx


def invertGamma(gam):
    """
    finds the inverse of the diffeomorphism gamma

    :param gam: vector describing the warping function

    :rtype: vector
    :return gamI: inverse of gam

    """
    N = gam.size
    x = arange(0, N) / float(N - 1)
    gamI = interp(x, gam, x)
    gamI[-1] = 1.0
    gamI = gamI / gamI[-1]
    return gamI


def SqrtMeanInverse(gam):
    """
    finds the inverse of the mean of the set of the diffeomorphisms gamma

    :param gam: numpy ndarray of shape (M,N) of M warping functions
    with N samples

    :rtype: vector
    :return gamI: inverse of gam

    """
    eps = finfo(double).eps
    n = gam.shape[1]
    T1 = gam.shape[0]
    dt = 1 / float(T1 - 1)
    psi = zeros((T1 - 1, n))
    for k in range(0, n):
        psi[:, k] = sqrt(diff(gam[:, k]) / dt + eps)

    # Find Direction
    mnpsi = psi.mean(axis=1)
    a = mnpsi.repeat(n)
    d1 = a.reshape(T1 - 1, n)
    d = (psi - d1) ** 2
    dqq = sqrt(d.sum(axis=0))
    min_ind = dqq.argmin()
    mu = psi[:, min_ind]
    maxiter = 20
    tt = 1
    lvm = zeros(maxiter)
    vec = zeros((T1 - 1, n))
    for itr in range(0, maxiter):
        for k in range(0, n):
            dot = trapz(mu * psi[:, k], linspace(0, 1, T1 - 1))
            if dot > 1:
                dot = 1
            elif dot < (-1):
                dot = -1
            leng = arccos(dot)
            if leng > 0.0001:
                vec[:, k] = (leng / sin(leng)) * (psi[:, k] - cos(leng) * mu)
            else:
                vec[:, k] = zeros(T1 - 1)
        vm = vec.mean(axis=1)
        vm1 = vm * vm
        lvm[itr] = sqrt(vm1.sum() * dt)
        if lvm[itr] == 0:
            mu = mu
            break

        mu = cos(tt * lvm[itr]) * mu + (sin(tt * lvm[itr]) / lvm[itr]) * vm
        if lvm[itr] < 1e-6 or itr >= maxiter:
            break

    tmp = mu * mu
    gam_mu = zeros(T1)
    gam_mu[1:] = tmp.cumsum() / T1
    gam_mu = (gam_mu - gam_mu.min()) / (gam_mu.max() - gam_mu.min())
    gamI = invertGamma(gam_mu)
    return gamI


def SqrtMean(gam):
    """
    calculates the srsf of warping functions with corresponding shooting vectors

    :param gam: numpy ndarray of shape (M,N) of M warping functions
    with N samples

    :rtype: 2 numpy ndarray and vector
    :return mu: Karcher mean psi function
    :return gam_mu: vector of dim N which is the Karcher mean warping function
    :return psi: numpy ndarray of shape (M,N) of M SRSF of the warping functions
    :return vec: numpy ndarray of shape (M,N) of M shooting vectors

    """
    n = gam.shape[1]
    TT = double(gam.shape[0])
    psi = zeros((TT - 1, n))
    for k in range(0, n):
        psi[:, k] = sqrt(diff(gam[:, k]) * TT)

    # Find Direction
    mnpsi = psi.mean(axis=1)
    a = mnpsi.repeat(n)
    d1 = a.reshape(TT - 1, n)
    d = (psi - d1) ** 2
    dqq = sqrt(d.sum(axis=0))
    min_ind = dqq.argmin()
    mu = psi[:, min_ind]
    maxiter = 20
    tt = 1
    lvm = zeros(maxiter)
    vec = zeros((TT - 1, n))
    for itr in range(0, maxiter):
        for k in range(0, n):
            dot = trapz(mu * psi[:, k], linspace(0, 1, TT - 1))
            if dot > 1:
                dot = 1
            elif dot < (-1):
                dot = -1
            leng = arccos(dot)
            if leng > 0.0001:
                vec[:, k] = (leng / sin(leng)) * (psi[:, k] - cos(leng) * mu)
            else:
                vec[:, k] = zeros(TT - 1)
        vm = vec.mean(axis=1)
        vm1 = vm * vm
        lvm[itr] = sqrt(vm1.sum() / TT)
        if lvm[itr] == 0:
            mu = mu
            break

        mu = cos(tt * lvm[itr]) * mu + (sin(tt * lvm[itr]) / lvm[itr]) * vm
        if lvm[itr] < 1e-6 or itr >= maxiter:
            break

    tmp = mu * mu
    gam_mu = zeros(TT)
    gam_mu[1:] = tmp.cumsum() / TT
    gam_mu = (gam_mu - gam_mu.min()) / (gam_mu.max() - gam_mu.min())

    return mu, gam_mu, psi, vec


def cumtrapzmid(x, y, c):
    """
    cumulative trapezoidal numerical integration taken from midpoint

    :param x: vector of size N describing the time samples
    :param y: vector of size N describing the function
    :param c: midpoint

    :rtype: vector
    :return fa: cumulative integration

    """
    a = x.shape[0]
    mid = int(round(a / 2.))

    # case < mid
    fa = zeros(a)
    tmpx = x[0:mid]
    tmpy = y[0:mid]
    tmp = c + cumtrapz(tmpy[::-1], tmpx[::-1], initial=0)
    fa[0:mid] = tmp[::-1]

    # case >= mid
    fa[mid:a] = c + cumtrapz(y[mid - 1:a - 1], x[mid - 1:a - 1], initial=0)

    return fa


def rgam(N, sigma, num):
    """
    Generates random warping functions

    :param N: length of warping function
    :param sigma: variance of warping functions
    :param num: number of warping functions
    :return: gam: numpy ndarray of warping functions

    """
    gam = zeros((N, num))

    TT = N - 1
    time = linspace(0, 1, TT)
    mu = sqrt(ones(N - 1) * TT / double(N - 1))
    omega = (2 * pi) / double(TT)
    for k in range(0, num):
        alpha_i = rn.normal(scale=sqrt(sigma))
        v = alpha_i * ones(TT)
        cnt = 1
        for l in range(2, 11):
            alpha_i = rn.normal(scale=sqrt(sigma))
            #odd
            if l % 2 != 0:
                v = v + alpha_i * sqrt(2) * cos(cnt * omega * time)
                cnt += 1

            #even
            if l % 2 == 0:
                v = v + alpha_i * sqrt(2) * cos(cnt * omega * time)
        v = v.reshape((TT, 1))
        mu = mu.reshape((TT, 1))
        tmp = mu.dot(v.transpose())
        v = v - tmp.dot(mu) / double(TT)
        vn = norm(v) / sqrt(TT)
        psi = cos(vn) * mu + sin(vn) * v / vn
        gam[1:, k] = cumsum(psi * psi) / double(TT)

    return gam


def outlier_detection(q, time, mq, k=1.5):
    """
    calculates outlier's using geodesic distances of the SRSFs from the median

    :param q: numpy ndarray of N x M of M SRS functions with N samples
    :param time: vector of size N describing the sample points
    :param mq: median calculated using :func:`time_warping.srsf_align`
    :param k: cutoff threshold (default = 1.5)

    :return: q_outlier: outlier functions

    """
    N = q.shape[1]
    ds = zeros(N)
    for kk in range(0, N):
        ds[kk] = sqrt(trapz((mq - q[:, kk]) ** 2, time))

    quartile_range = mquantiles(ds)
    IQR = quartile_range[2] - quartile_range[0]

    thresh = quartile_range[2] + k * IQR

    ind = (ds > thresh).nonzero()

    q_outlier = q[:, ind]

    return q_outlier


def randomGamma(gam, num):
    """
    generates random warping functions

    :param gam: numpy ndarray of N x M of M of warping functions
    :param num: number of random functions

    :return: rgam: random warping functions

    """
    mu, gam_mu, psi, vec = SqrtMean(gam)
    K = cov(vec)

    U, s, V = svd(K)
    n = 5
    TT = vec.shape[0] + 1
    vm = vec.mean(axis=1)

    rgam = zeros((TT, num))
    for k in range(0, num):
        a = rn.standard_normal(n)
        v = zeros(vm.size)
        for i in range(0, n):
            v = v + a[i] * sqrt(s[i]) * U[:, i]

        vn = norm(v) / sqrt(TT)
        psi = cos(vn) * mu + sin(vn) * v / vn
        tmp = zeros(TT)
        tmp[1:TT] = cumsum(psi * psi) / TT
        rgam[:, k] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])

    return rgam


def update_progress(progress):
    """
    This function creates a progress bar

    :param progress: fraction of progress

    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#" * block + "-" *
                                              (barLength - block),
                                              progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def diffop(n, binsize=1):
    """
    Creates a second order differential operator

    :param n: dimension
    :param binsize: dx (default = 1)

    :rtype: numpy ndarray
    :return m: matrix describing differential operator

    """
    m = diagflat(ones(n - 1), k=1) + diagflat(ones(n - 1), k=-1) + diagflat(2 * ones(n))
    m = inner(m.transpose(), m)
    m[0, 0] = 6
    m[-1, -1] = 6
    m /= (binsize ** 4)

    return m


def geigen(Amat, Bmat, Cmat):
    """
    generalized eigenvalue problem of the form

    max tr L'AM / sqrt(tr L'BL tr M'CM) w.r.t. L and M

    :param Amat numpy ndarray of shape (M,N)
    :param Bmat numpy ndarray of shape (M,N)
    :param Bmat numpy ndarray of shape (M,N)

    :rtype: numpy ndarray
    :return values: eigenvalues
    :return Lmat: left eigenvectors
    :return Mmat: right eigenvectors

    """
    if Bmat.shape[0] != Bmat.shape[1]:
        print("BMAT is not square.\n")
        sys.exit(1)

    if Cmat.shape[0] != Cmat.shape[1]:
        print("CMAT is not square.\n")
        sys.exit(1)

    p = Bmat.shape[0]
    q = Cmat.shape[0]

    s = min(p, q)
    tmp = fabs(Bmat - Bmat.transpose())
    tmp1 = fabs(Bmat)
    if tmp.max() / tmp1.max() > 1e-10:
        print("BMAT not symmetric..\n")
        sys.exit(1)

    tmp = fabs(Cmat - Cmat.transpose())
    tmp1 = fabs(Cmat)
    if tmp.max() / tmp1.max() > 1e-10:
        print("CMAT not symmetric..\n")
        sys.exit(1)

    Bmat = (Bmat + Bmat.transpose()) / 2
    Cmat = (Cmat + Cmat.transpose()) / 2
    Bfac = cholesky(Bmat)
    Cfac = cholesky(Cmat)
    Bfacinv = inv(Bfac)
    Bfacinvt = Bfacinv.transpose()
    Cfacinv = inv(Cfac)
    Dmat = Bfacinvt.dot(Amat).dot(Cfacinv)
    if p >= q:
        u, d, v = svd(Dmat)
        values = d
        Lmat = Bfacinv.dot(u)
        Mmat = Cfacinv.dot(v.transpose())
    else:
        u, d, v = svd(Dmat.transpose())
        values = d
        Lmat = Bfacinv.dot(u)
        Mmat = Cfacinv.dot(v.transpose())

    return values, Lmat, Mmat


def innerprod_q(time, q1, q2):
    """
    calculates the innerproduct between two srsfs

    :param time vector descrbing time samples
    :param q1 vector of srsf 1
    :param q2 vector of srsf 2

    :rtype: scalar
    :return val: inner product value

    """
    val = trapz(q1 * q2, time)
    return val


def warp_q_gamma(time, q, gam):
    """
    warps a srsf q by gam

    :param time vector describing time samples
    :param q vector describing srsf
    :param gam vector describing warping function

    :rtype: numpy ndarray
    :return q_temp: warped srsf

    """
    M = gam.size
    gam_dev = gradient(gam, 1 / double(M - 1))
    tmp = interp((time[-1] - time[0]) * gam + time[0], time, q)

    q_temp = tmp * sqrt(gam_dev)

    return q_temp


def f_K_fold(Nobs, K=5):
    """
    generates sample indices for K-fold cross validation

    :param Nobs number of observations
    :param K number of folds

    :rtype: numpy ndarray
    :return train: train indexes (Nobs*(K-1)/K X K)
    :return test: test indexes (Nobs*(1/K) X K)

    """
    rs = rn.uniform(size=Nobs)
    ids = rs.ravel().argsort()
    k = Nobs * arange(1, K) / K
    tmp = append(k.repeat(2), Nobs)
    tmp = insert(tmp, 0, 0)
    k = tmp.reshape((K, 2))
    k[:, 0] = k[:, 0] + 1
    k = k - 1
    train = zeros((Nobs * (K - 1) / K, K))
    test = zeros((Nobs * 1 / K, K))
    for ii in range(0, K):
        tf = vectorize(lambda x: x in arange(k[ii, 0],
                       k[ii, 1] + 1))(arange(0, Nobs))
        train[:, ii] = ids[~tf]
        test[:, ii] = ids[tf]

    return train, test


def zero_crossing(Y, q, bt, time, y_max, y_min, gmax, gmin):
    """
    finds zero-crossing of optimal gamma, gam = s*gmax + (1-s)*gmin
    from elastic regression model

    :param Y: response
    :param q: predicitve function
    :param bt: basis function
    :param time: time samples
    :param y_max: maximum repsonse for warping function gmax
    :param y_min: minimum response for warping function gmin
    :param gmax: max warping function
    :param gmin: min warping fucntion

    :rtype: numpy array
    :return gamma: optimal warping function

    """
    # simple iterative method based on intermediate theorem
    a = zeros(100)
    a[0] = 1
    f = zeros(100)
    f[0] = y_max - Y
    f[1] = y_min - Y
    mrp = f[0]
    mrn = f[1]
    mrp_ind = 0  # most recent positive index
    mrn_ind = 1  # most recent negative index

    for ii in range(2, 100):
        x1 = a[mrp_ind]
        x2 = a[mrn_ind]
        y1 = mrp
        y2 = mrn
        a[ii] = (x1 * y2 - x2 * y1) / (y2 - y1)

        gam_m = a[ii] * gmax + (1 - a[ii]) * gmin
        qtmp = warp_q_gamma(time, q, gam_m)
        f[ii] = trapz(qtmp * bt, time) - Y

        if fabs(f[ii]) < 1e-5:
            break
        elif f[ii] > 0:
            mrp = f[ii]
            mrp_ind = ii
        else:
            mrn = f[ii]
            mrn_ind = ii

    gamma = a[ii] * gmax + (1 - a[ii]) * gmin

    return gamma
