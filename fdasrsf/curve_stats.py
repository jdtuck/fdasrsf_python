"""
statistic calculation for SRVF (curves) open and closed using Karcher
Mean and Variance

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""
from numpy import zeros, sqrt, fabs, cos, sin, tile, vstack, empty
from numpy.linalg import svd
from numpy.random import randn
import fdasrsf.curve_functions as cf
import fdasrsf.utility_functions as uf
from joblib import Parallel, delayed
import collections


def curve_karcher_mean(beta, mode='O'):
    """
    This claculates the mean of a set of curves
    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param mode: Open ('O') or closed curve ('C') (default 'O')

    :rtype: tuple of numpy array
    :return mu: mean srvf
    :return betamean: mean curve
    :return v: shooting vectors
    :return q: srvfs

    """
    n, T, N = beta.shape
    q = zeros((n, T, N))
    for ii in range(0, N):
        q[:, :, ii] = cf.curve_to_q(beta[:, :, ii])

    modes = ['O', 'C']
    mode = [i for i, x in enumerate(modes) if x == mode]
    if len(mode) == 0:
        mode = 0
    else:
        mode = mode[0]

    # Initialize mu as one of the shapes
    mu = q[:, :, 0]
    betamean = beta[:, :, 0]

    delta = 0.5
    tolv = 1e-4
    told = 5*1e-3
    maxit = 20
    itr = 0
    sumd = zeros(maxit+1)
    v = zeros((n, T, N))
    normvbar = zeros(maxit+1)

    while itr < maxit:
        print("Iteration: %d" % itr)

        mu = mu / sqrt(cf.innerprod_q2(mu, mu))

        sumv = zeros((2, T))
        sumd[itr+1] = 0
        out = Parallel(n_jobs=-1)(delayed(karcher_calc)(beta[:, :, n],
                                  q[:, :, n], betamean, mu, mode) for n in range(N))
        v = zeros((n, T, N))
        for i in range(0, N):
            v[:, :, i] = out[i][0]
            sumd[itr+1] = sumd[itr+1] + out[i][1]**2

        sumv = v.sum(axis=2)

        # Compute average direction of tangent vectors v_i
        vbar = sumv/float(N)

        normvbar[itr] = sqrt(cf.innerprod_q2(vbar, vbar))
        normv = normvbar[itr]

        if normv > tolv and fabs(sumd[itr+1]-sumd[itr]) > told:
            # Update mu in direction of vbar
            mu = cos(delta*normvbar[itr])*mu + sin(delta*normvbar[itr]) * vbar/normvbar[itr]

            if mode == 1:
                mu = cf.project_curve(mu)

            x = cf.q_to_curve(mu)
            a = -1*cf.calculatecentroid(x)
            betamean = x + tile(a, [T, 1]).T
        else:
            break

        itr += 1

    return(mu, betamean, v, q)


def oc_srvf_align(beta, mode='O'):
    """
    This claculates the mean of a set of curves and aligns them
    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param mode: Open ('O') or closed curve ('C') (default 'O')

    :rtype: tuple of numpy array
    :return betan: aligned curves
    :return qn: aligned srvf
    :return betamean: mean curve
    :return mu: mean srvf
    """
    n, T, N = beta.shape
    # find mean
    mu, betamean, v, q = curve_karcher_mean(beta, mode=mode)

    qn = zeros((n, T, N))
    betan = zeros((n, T, N))
    centroid2 = cf.calculatecentroid(betamean)
    betamean = betamean - tile(centroid2, [T, 1]).T
    q_mu = cf.curve_to_q(betamean)
    # align to mean
    for ii in range(0, N):
        beta1 = beta[:, :, ii]
        centroid1 = cf.calculatecentroid(beta1)
        beta1 = beta1 - tile(centroid1, [T, 1]).T

        # Iteratively optimize over SO(n) x Gamma
        for i in range(0, 1):
            # Optimize over SO(n)
            beta1, O_hat, tau = cf.find_rotation_and_seed_coord(betamean,
                                                                beta1)
            q1 = cf.curve_to_q(beta1)

            # Optimize over Gamma
            gam = cf.optimum_reparam_curve(q1, q_mu, 0.0)
            gamI = uf.invertGamma(gam)
            # Applying optimal re-parameterization to the second curve
            beta1 = cf.group_action_by_gamma_coord(beta1, gamI)

        # Optimize over SO(n)
        beta1, O_hat, tau = cf.find_rotation_and_seed_coord(betamean, beta1)
        qn[:, :, ii] = cf.curve_to_q(beta1)
        betan[:, :, ii] = beta1

    align_results = collections.namedtuple('align', ['betan', 'qn', 'betamean', 'mu'])
    out = align_results(betan, qn, betamean, q_mu)
    return out


def curve_karcher_cov(betamean, beta, mode='O'):
    """
    This claculates the mean of a set of curves
    :param betamean: numpy ndarray of shape (n, M) describing the mean curve
    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param mode: Open ('O') or closed curve ('C') (default 'O')

    :rtype: tuple of numpy array
    :return K: Covariance Matrix

    """
    n, T, N = beta.shape
    modes = ['O', 'C']
    mode = [i for i, x in enumerate(modes) if x == mode]
    if len(mode) == 0:
        mode = 0
    else:
        mode = mode[0]

    # Compute Karcher covariance of uniformly sampled mean
    betamean = cf.resamplecurve(betamean, T)
    mu = cf.curve_to_q(betamean)
    if mode == 1:
        mu = cf.project_curve(mu)
        basis = cf.find_basis_normal(mu)

    v = zeros((n, T, N))
    for i in range(0, N):
        beta1 = beta[:, :, i]

        w, dist = cf.inverse_exp_coord(betamean, beta1)
        # Project to the tangent sapce of manifold to obtain v_i
        if mode == 0:
            v[:, :, i] = w
        else:
            v[:, :, i] = cf.project_tangent(w, mu, basis)

    K = zeros((2*T, 2*T))

    for i in range(0, N):
        w = v[:, :, i]
        wtmp = w.reshape((T*n, 1), order='C')
        K = K + wtmp.dot(wtmp.T)

    K = K/(N-1)

    return(K)


def curve_principal_directions(betamean, mu, K, mode='O', no=3, N=5):
    """
    Computes principal direction of variation specified by no. N is
    Number of shapes away from mean. Creates 2*N+1 shape sequence

    :param betamean: numpy ndarray of shape (n, M) describing the mean curve
    :param mu: numpy ndarray of shape (n, M) describing the mean srvf
    :param K: numpy ndarray of shape (M, M) describing the covariance
    :param mode: Open ('O') or closed curve ('C') (default 'O')
    :param no: number of direction (default 3)
    :param N: number of shapes (2*N+1) (default 5)

    :rtype: tuple of numpy array
    :return pd: principal directions

    """
    n, T = betamean.shape
    modes = ['O', 'C']
    mode = [i for i, x in enumerate(modes) if x == mode]
    if len(mode) == 0:
        mode = 0
    else:
        mode = mode[0]

    U, s, V = svd(K)

    qarray = empty((no, 2*N+1), dtype=object)
    qarray1 = empty(N, dtype=object)
    qarray2 = empty(N, dtype=object)
    pd = empty((no, 2*N+1), dtype=object)
    pd1 = empty(N, dtype=object)
    pd2 = empty(N, dtype=object)
    for m in range(0, no):
        princDir = vstack((U[0:T, m], U[T:2*T, m]))
        v = sqrt(s[m]) * princDir
        q1 = mu
        epsilon = 2./N

        # Forward direction from mean
        for i in range(0, N):
            normv = sqrt(cf.innerprod_q2(v, v))

            if normv < 1e-4:
                q2 = mu
            else:
                q2 = cos(epsilon*normv)*q1 + sin(epsilon*normv)*v/normv
                if mode == 1:
                    q2 = cf.project_curve(q2)

            qarray1[i] = q2
            p = cf.q_to_curve(q2)
            centroid1 = -1*cf.calculatecentroid(p)
            beta_scaled, scale = cf.scale_curve(p + tile(centroid1, [T, 1]).T)
            pd1[i] = beta_scaled

            # Parallel translate tangent vector
            basis2 = cf.find_basis_normal(q2)
            v = cf.parallel_translate(v, q1, q2, basis2, mode)

            q1 = q2

        # Backward direction from mean
        v = -sqrt(s[m])*princDir
        q1 = mu
        for i in range(0, N):
            normv = sqrt(cf.innerprod_q2(v, v))

            if normv < 1e-4:
                q2 = mu
            else:
                q2 = cos(epsilon*normv)*q1+sin(epsilon*normv)*v/normv
                if mode == 1:
                    q2 = cf.project_curve(q2)

            qarray2[i] = q2
            p = cf.q_to_curve(q2)
            centroid1 = -1*cf.calculatecentroid(p)
            beta_scaled, scale = cf.scale_curve(p + tile(centroid1, [T, 1]).T)
            pd2[i] = beta_scaled

            # Parallel translate tangent vector
            basis2 = cf.find_basis_normal(q2)
            v = cf.parallel_translate(v, q1, q2, basis2, mode)

            q1 = q2

        for i in range(0, N):
            qarray[m, i] = qarray2[(N-1)-i]
            pd[m, i] = pd2[(N-1)-i]

        qarray[m, N] = mu
        centroid1 = -1*cf.calculatecentroid(betamean)
        beta_scaled, scale = cf.scale_curve(betamean +
                                            tile(centroid1, [T, 1]).T)
        pd[m, N] = beta_scaled

        for i in range(N+1, 2*N+1):
            qarray[m, i] = qarray1[i-(N+1)]
            pd[m, i] = pd1[i-(N+1)]

    return(pd)


def sample_shapes(mu, K, mode='O', no=3, numSamp=10):
    """
    Computes sample shapes from mean and covariance

    :param betamean: numpy ndarray of shape (n, M) describing the mean curve
    :param mu: numpy ndarray of shape (n, M) describing the mean srvf
    :param K: numpy ndarray of shape (M, M) describing the covariance
    :param mode: Open ('O') or closed curve ('C') (default 'O')
    :param no: number of direction (default 3)
    :param numSamp: number of samples (default 10)

    :rtype: tuple of numpy array
    :return samples: sample shapes

    """
    n, T = mu.shape
    modes = ['O', 'C']
    mode = [i for i, x in enumerate(modes) if x == mode]
    if len(mode) == 0:
        mode = 0
    else:
        mode = mode[0]

    U, s, V = svd(K)

    if mode == 0:
        N = 2
    else:
        N = 10

    epsilon = 1./(N-1)

    samples = empty(numSamp, dtype=object)
    for i in range(0, numSamp):
        v = zeros((2, T))
        for m in range(0, no):
            v = v + randn()*sqrt(s[m])*vstack((U[0:T, m], U[T:2*T, m]))

        q1 = mu
        for j in range(0, N-1):
            normv = sqrt(cf.innerprod_q2(v, v))

            if normv < 1e-4:
                q2 = mu
            else:
                q2 = cos(epsilon*normv)*q1+sin(epsilon*normv)*v/normv
                if mode == 1:
                    q2 = cf.project_curve(q2)

            # Parallel translate tangent vector
            basis2 = cf.find_basis_normal(q2)
            v = cf.parallel_translate(v, q1, q2, basis2, mode)

            q1 = q2

        samples[i] = cf.q_to_curve(q2)

    return(samples)


def karcher_calc(beta, q, betamean, mu, mode=0):
    if mode == 1:
        basis = cf.find_basis_normal(mu)
    # Compute shooting vector from mu to q_i
    w, d = cf.inverse_exp_coord(betamean, beta)

    # Project to tangent space of manifold to obtain v_i
    if mode == 0:
        v = w
    else:
        v = cf.project_tangent(w, q, basis)

    return(v, d)
