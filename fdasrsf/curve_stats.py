"""
statistic calculation for SRVF (curves) open and closed using Karcher
Mean and Variance

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""
from numpy import zeros, sqrt, fabs, cos, sin, tile, vstack, empty
from numpy.linalg import svd
from numpy.random import randn
import fdasrsf.curve_functions as cf


def curve_karcher_mean(q, beta, mode='O'):
    """
    This claculates the mean of a set of curves
    :param q: numpy ndarray of shape (n, M, N) describing N curves
    in srvf space
    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param mode: Open ('O') or closed curve ('C') (default 'O')

    :rtype: tuple of numpy array
    :return mu: mean srvf
    :return betamean: mean curve
    :return v: shooting vectors

    """
    n, T, N = q.shape
    modes = ['O', 'C']
    mode = [i for i, x in enumerate(modes) if x == mode]
    mode = mode[0]
    if mode != 0 and mode != 1:
        mode = 0

    # Initialize mu as one of the shapes
    mu = q[:, :, 0]
    betamean = beta[:, :, 0]

    delta = 0.5
    tolv = 1e-4
    told = 5*1e-3
    maxit = 20
    itr = 0
    sumd = zeros(maxit)
    v = zeros((n, T, N))
    normvbar = zeros(maxit)

    while itr < maxit:
        print("Iteration: %d" % itr)

        mu = mu / sqrt(cf.innerprod_q(mu, mu))
        if mode == 1:
            basis = cf.find_basis_normal(mu)

        sumv = zeros((2, T))
        sumd[itr+1] = 0
        for i in range(0, N):
            q1 = q[:, :, i]
            beta1 = beta[:, :, i]

            # Compute shooting vector from mu to q_i
            w, d = cf.inverse_exp_coord(betamean, beta1)

            # Project to tangent space of manifold to obtain v_i
            if mode == 0:
                v[:, :, i] = w
            else:
                v[:, :, i] = cf.project_tangent(w, q1, basis)

            sumv = sumv + v[:, :, i]
            sumd[itr+1] = sumd[itr+1] + d**2

        # Compute average direction of tangent vectors v_i
        vbar = sumv/N

        normvbar[itr] = sqrt(cf.innerprod_q(vbar, vbar))
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

    return(mu, betamean, v)


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
    mode = mode[0]
    if mode != 0 and mode != 1:
        mode = 0

    # Compute Karcher covariance of uniformly sampled mean
    betamean = cf.resamplecurve(betamean, T)
    mu_q = cf.curve_to_q(betamean)
    mu = cf.project_curve(mu_q)
    basis = cf.find_basis_normal(mu)

    v = zeros((n, T, N))
    for i in range(0, N):
        beta1 = beta[:, :, i]

        w = cf.inverse_exp_coord(betamean, beta1)
        # Project to the tangent sapce of manifold to obtain v_i
        if mode == 0:
            v[:, :, i] = w
        else:
            v[:, :, i] = cf.project_tangent(w, mu, basis)

    K = zeros((2*T, 2*T))

    for i in range(0, N):
        w = v[:, :, i]
        wtmp = w.reshape((20, 1), order='C')
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
    mode = mode[0]
    if mode != 0 and mode != 1:
        mode = 0

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
            normv = sqrt(cf.innerprod_q(v, v))

            if normv < 1e-4:
                q2 = mu
            else:
                q2 = cos(epsilon*normv)*q1 + sin(epsilon*normv)*v/normv
                if mode == 1:
                    q2 = cf.project_curve(q2)

            qarray1[i] = q2
            p = cf.q_to_curve(q2)
            centroid1 = -1*cf.calculatecentroid(p)
            pd1[i] = cf.scale_curve(p + tile(centroid1, [T, 1]).T)

            # Parallel translate tangent vector
            basis2 = cf.find_basis_normal(q2)
            v = cf.parallel_translate(v, q1, q2, basis2)

            q1 = q2

        # Backward direction from mean
        v = -sqrt(s[m])*princDir
        q1 = mu
        for i in range(0, N):
            normv = sqrt(cf.innerprod_q(v, v))

            if normv < 1e-4:
                q2 = mu
            else:
                q2 = cos(epsilon*normv)*q1+sin(epsilon*normv)*v/normv
                if mode == 1:
                    q2 = cf.project_curve(q2)

            qarray2[i] = q2
            p = cf.q_to_curve(q2)
            centroid1 = -1*cf.calculatecentroid(p)
            pd2[i] = cf.scale_curve(p + tile(centroid1, [T, 1]).T)

            # Parallel translate tangent vector
            basis2 = cf.find_basis_normal(q2)
            v = cf.parallel_translate(v, q1, q2, basis2)

            q1 = q2

        for i in range(0, N):
            qarray[m, i] = qarray2[(N-1)-i]
            pd[m, i] = pd2[(N-1)-i]

        qarray[m, N] = mu
        centroid1 = -1*cf.calculatecentroid(betamean)
        pd[m, N] = cf.scale_curve(betamean + tile(centroid1, [T, 1]).T)

        for i in range(N+1, 2*N+1):
            qarray1[m, i] = qarray1[i-(N+1)]
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
    mode = mode[0]
    if mode != 0 and mode != 1:
        mode = 0

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
            normv = sqrt(cf.innerprod_q(v, v))

            if normv < 1e-4:
                q2 = mu
            else:
                q2 = cos(epsilon*normv)*q1+sin(epsilon*normv)*v/normv
                if mode == 1:
                    q2 = cf.project_curve(q2)

            # Parallel translate tangent vector
            basis2 = cf.find_basis_normal(q2)
            v = cf.parallel_translate(v, q1, q2, basis2)

            q1 = q2

        samples[i] = cf.q_to_curve(q2)

    return(samples)
