"""
geodesic calculation for SRVF (curves) open and closed

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

from numpy import tile, eye, arccos, zeros, sin, arange, linspace, empty, isnan
from numpy import sqrt
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.linalg import norm
import fdasrsf.utility_functions as uf
import fdasrsf.curve_functions as cf


def plot_geod(path):
    r"""
    Plots the geodesic path as a sequence of curves

    :param path: numpy ndarray of shape (2,M,K) of M sample points of K samples along path

    """
    fig, ax = plt.subplots()
    mv = 0.2
    for i in range(0,path.shape[2]):
        ax.plot(mv*i + path[0,:,i],path[1,:,i], linewidth=2)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()

    return


def geod_sphere(beta1, beta2, k=5, scale=False, rotation=True, center=True):
    """
    This function calculates the geodesics between open curves beta1 and
    beta2 with k steps along path

    :param beta1: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples
    :param k: number of samples along path (Default = 5)
    :param scale: include length (Default = False)
    :param rotation: include rotation (Default = True)
    :param center: center curves at origin (Default = True)

    :rtype: numpy ndarray
    :return dist: geodesic distance
    :return path: geodesic path
    :return PsiQ: geodesic path in SRVF

    """
    lam = 0.0
    returnpath = 1
    n, T = beta1.shape

    if center:
        centroid1 = cf.calculatecentroid(beta1)
        beta1 = beta1 - tile(centroid1, [T, 1]).T
        centroid2 = cf.calculatecentroid(beta2)
        beta2 = beta2 - tile(centroid2, [T, 1]).T

    q1, len1, lenq1 = cf.curve_to_q(beta1)
    if scale:
        q2, len2, lenq2 = cf.curve_to_q(beta2)
    beta2, q2n, O1, gamI = cf.find_rotation_and_seed_coord(beta1, beta2, rotation=rotation)
    
    # Forming geodesic between the registered curves
    val = cf.innerprod_q2(q1, q2n)
    if val > 1:
        if val < 1.0001: # assume numerical error
            import warnings
            warnings.warn(f"Corrected a numerical error in geod_sphere: rounded {val} to 1")
            val = 1
        else:
            raise Exception(f"innerpod_q2 computed an inner product of {val} which is much greater than 1")
    elif val < -1:
        if val > -1.0001: # assume numerical error
            import warnings
            warnings.warn(f"Corrected a numerical error in geod_sphere: rounded {val} to -1")
            val = -1
        else:
            raise Exception(f"innerpod_q2 computed an inner product of {val} which is much less than -1")

    dist = arccos(val)
    if isnan(dist):
        raise Exception("geod_sphere computed a dist value which is NaN")

    if returnpath:
        PsiQ = zeros((n, T, k))
        PsiX = zeros((n, T, k))
        for tau in range(0, k):
            if tau == 0:
                tau1 = 0
            else:
                tau1 = tau / (k - 1.)
                
            s = dist * tau1
            if dist > 0:
                PsiQ[:, :, tau] = (sin(dist-s)*q1+sin(s)*q2n)/sin(dist)
            elif dist == 0:
                PsiQ[:, :, tau] = (1 - tau1)*q1 + (tau1)*q2n
            else:
                raise Exception("geod_sphere computed a negative distance")
                
            if scale:
                scl = len1**(1-tau1)*len2**(tau1)
            else:
                scl = 1
            beta = scl*cf.q_to_curve(PsiQ[:, :, tau])
            if center:
                centroid = cf.calculatecentroid(beta)
                beta = beta - tile(centroid, [T, 1]).T
            PsiX[:, :, tau] = beta

        path = PsiX
    else:
        path = 0

    return(dist, path, PsiQ)


def path_straightening(beta1, beta2, betamid=None, init="rand", T=100, k=5):
    """
    Perform path straightening to find geodesic between two shapes in either
    the space of closed curves or the space of affine standardized curves.
    This algorithm follows the steps outlined in section 4.6 of the
    manuscript.

    :param beta1: numpy ndarray of shape (2,M) of M samples (first curve)
    :param beta2: numpy ndarray of shape (2,M) of M samples (end curve)
    :param betamid: numpy ndarray of shape (2,M) of M samples (mid curve
     Default = None, only needed for init "geod")
    :param init: initialize path geodesic or random (Default = "rand")
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return dist: geodesic distance
    :return path: geodesic path
    :return pathsqnc: geodesic path sequence
    :return E: energy

    """
    inits = ["rand", "geod"]
    init = [i for i, x in enumerate(inits) if x == init]
    if len(init) == 0:
        init = 0
    else:
        init = init[0]

    betanew1, qnew1, A1 = cf.pre_proc_curve(beta1, T)
    betanew2, qnew2, A2 = cf.pre_proc_curve(beta2, T)

    if init == 0:
        betanewmid, qnewmid, Amid = cf.pre_proc_curve(beta2, T)

    if init == 0:
        alpha, beta, O = init_path_rand(betanew1, betanewmid,
                                        betanew2, T, k)
    elif init == 1:
        alpha, beta, O = init_path_geod(betanew1, betanew2, T, k)

    # path straightening
    tol = 1e-2
    n = beta.shape[0]
    T = beta.shape[1]
    maxit = 20
    i = 0
    g = 1
    delta = 0.5
    E = zeros(maxit+1)
    gradEnorm = zeros(maxit+1)
    pathsqnc = zeros((n, T, k, maxit+1))

    pathsqnc[:, :, :, 0] = beta

    while i < maxit:
        # algorithm 8:
        # compute dalpha/dt along alpha using finite difference approx
        # First calculate basis for normal sapce at each point in alpha
        basis = find_basis_normal_path(alpha, k)
        alphadot = calc_alphadot(alpha, basis, T, k)
        E[i] = calculate_energy(alphadot, T, k)

        # algorithm 9:
        # compute covariant integral of alphadot along alpha. This is
        # the gradient
        # of E in \cal{H}. Later we will project it to the space \cal{H}_{O}
        u1 = cov_integral(alpha, alphadot, basis, T, k)

        # algorithm 10:
        # backward parallel transport of u(1)
        utilde = back_parallel_transport(u1[:, :, -1], alpha, basis, T, k)

        # algorithm 11:
        # compute gradient vector field of E in \cal{H}_{O}
        gradE, normgradE = calculate_gradE(u1, utilde, T, k)
        gradEnorm[i] = norm(normgradE)
        g = gradEnorm[i]

        # algorithm 12:
        # update the path along the direction -gradE
        alpha, beta = update_path(alpha, beta, gradE, delta, T, k)

        # path evolution
        pathsqnc[:, :, :, i+1] = beta

        if g < tol:
            break

        i += 1

    if i > 0:
        E = E[0:i]
        gradEnorm = gradEnorm[0:i]
        pathsqnc = pathsqnc[:, :, :, 0:(i+2)]
    else:
        E = E[0]
        gradEnorm = gradEnorm[0]
        pathsqnc = pathsqnc[:, :, :, 0:(i+2)]

    path = beta
    dist = geod_dist_path_strt(beta, k)

    return(dist, path, pathsqnc, E)


# path straightening helper functions
def init_path_rand(beta1, beta_mid, beta2, T=100, k=5):
    r"""
    Initializes a path in :math:`\cal{C}`. beta1, beta_mid beta2 are already
    standardized curves. Creates a path from beta1 to beta_mid to beta2 in
    shape space, then projects to the closed shape manifold.

    :param beta1: numpy ndarray of shape (2,M) of M samples (first curve)
    :param betamid: numpy ndarray of shape (2,M) of M samples (mid curve)
    :param beta2: numpy ndarray of shape (2,M) of M samples (end curve)
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return alpha: a path between two q-functions
    :return beta:  a path between two curves
    :return O: rotation matrix

    """
    alpha = zeros((2, T, k))
    beta = zeros((2, T, k))

    q1 = cf.curve_to_q(beta1)[0]
    q_mid = cf.curve_to_q(beta_mid)[0]

    # find optimal rotation of q2
    beta2, q2best, O1, gamIbest = cf.find_rotation_and_seed_coord(beta1, beta2)
    q2 = cf.curve_to_q(beta2)[0]

    # find the optimal coorespondence
    gam = cf.optimum_reparam_curve(q2, q1)
    gamI = uf.invertGamma(gam)

    # apply optimal reparametrization
    beta2n = cf.group_action_by_gamma_coord(beta2, gamI)

    # find optimal rotation of q2
    beta2n, q2n1, O2, gamIbest1 = cf.find_rotation_and_seed_coord(beta1, beta2n)
    centroid2 = cf.calculatecentroid(beta2n)
    beta2n = beta2n - tile(centroid2, [T, 1]).T
    q2n = cf.curve_to_q(beta2n)[0]
    O = O1 @ O2

    # Initialize a path as a geodesic through q1 --- q_mid --- q2
    theta1 = arccos(cf.innerprod_q2(q1, q_mid))
    theta2 = arccos(cf.innerprod_q2(q_mid, q2n))
    tmp = arange(2, int((k-1)/2)+1)
    t = zeros(tmp.size)
    alpha[:, :, 0] = q1
    beta[:, :, 0] = beta1

    i = 0
    for tau in range(2, int((k-1)/2)+1):
        t[i] = (tau-1.)/((k-1)/2.)
        qnew = (1/sin(theta1))*(sin((1-t[i])*theta1)*q1+sin(t[i]*theta1)*q_mid)
        alpha[:, :, tau-1] = cf.project_curve(qnew)
        x = cf.q_to_curve(alpha[:, :, tau-1])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau-1] = x + tile(a, [T, 1]).T
        i += 1

    alpha[:, :, int((k-1)/2)] = q_mid
    beta[:, :, int((k-1)/2)] = beta_mid

    i = 0
    for tau in range(int((k-1)/2)+1, k-1):
        qnew = (1/sin(theta2))*(sin((1-t[i])*theta2)*q_mid
                                + sin(t[i]*theta2)*q2n)
        alpha[:, :, tau] = cf.project_curve(qnew)
        x = cf.q_to_curve(alpha[:, :, tau])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau] = x + tile(a, [T, 1]).T
        i += 1

    alpha[:, :, k-1] = q2n
    beta[:, :, k-1] = beta2n

    return(alpha, beta, O)


def init_path_geod(beta1, beta2, T=100, k=5):
    r"""
    Initializes a path in :math:`\cal{C}`. beta1, beta2 are already
    standardized curves. Creates a path from beta1 to beta2 in
    shape space, then projects to the closed shape manifold.

    :param beta1: numpy ndarray of shape (2,M) of M samples (first curve)
    :param beta2: numpy ndarray of shape (2,M) of M samples (end curve)
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return alpha: a path between two q-functions
    :return beta:  a path between two curves
    :return O: rotation matrix

    """
    alpha = zeros((2, T, k))
    beta = zeros((2, T, k))

    dist, pathq, O = geod_sphere(beta1, beta2, k)

    for tau in range(0, k):
        alpha[:, :, tau] = cf.project_curve(pathq[:, :, tau])
        x = cf.q_to_curve(alpha[:, :, tau])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau] = x + tile(a, [T, 1]).T

    return(alpha, beta, O)


def find_basis_normal_path(alpha, k=5):
    """
    computes orthonormalized basis vectors to the normal space at each of the
    k points (q-functions) of the path alpha

    :param alpha: numpy ndarray of shape (2,M) of M samples (path)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return basis: basis vectors along the path

    """
    basis = empty(k, dtype=object)
    for tau in range(0, k):
        q = alpha[:, :, tau]
        b = cf.find_basis_normal(q)
        basis_tmp = cf.gram_schmidt(b)
        basis[tau] = basis_tmp

    return(basis)


def calc_alphadot(alpha, basis, T=100, k=5):
    """
    calculates derivative along the path alpha

    :param alpha: numpy ndarray of shape (2,M) of M samples
    :param basis: list of numpy ndarray of shape (2,M) of M samples
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return alphadot: derivative of alpha

    """
    alphadot = zeros((2, T, k))

    for tau in range(0, k):
        if tau == 0:
            v = (k-1)*(alpha[:, :, tau+1] - alpha[:, :, tau])
        elif tau == (k-1):
            v = (k-1)*(alpha[:, :, tau] - alpha[:, :, (tau-1)])
        else:
            v = ((k-1)/2.0)*(alpha[:, :, tau+1] - alpha[:, :, (tau-1)])

        alphadot[:, :, tau] = cf.project_tangent(v, alpha[:, :, tau],
                                                 basis[tau])

    return(alphadot)


def calculate_energy(alphadot, T=100, k=5):
    """
    calculates energy along path

    :param alphadot: numpy ndarray of shape (2,M) of M samples
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy scalar
    :return E: energy

    """
    integrand1 = zeros((k, T))
    integrand2 = zeros(k)

    for i in range(0, k):
        for j in range(1, T):
            tmp = alphadot[:, j, i].T
            integrand1[i, j] = tmp.dot(alphadot[:, j, i])

        integrand2[i] = trapz(integrand1[i, :], linspace(0, 1, T))

    E = 0.5*trapz(integrand2, linspace(0, 1, k))

    return(E)


def cov_integral(alpha, alphadot, basis, T=100, k=5):
    """
    Calculates covariance along path alpha

    :param alpha: numpy ndarray of shape (2,M) of M samples (first curve)
    :param alphadot: numpy ndarray of shape (2,M) of M samples
    :param basis: list numpy ndarray of shape (2,M) of M samples
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return u: covariance

    """
    u = zeros((2, T, k))

    for tau in range(1, k):
        w = u[:, :, tau-1]
        q1 = alpha[:, :, tau-1]
        q2 = alpha[:, :, tau]
        b = basis[tau]
        wbar = cf.parallel_translate(w, q1, q2, b)
        u[:, :, tau] = (1./(k-1))*alphadot[:, :, tau]+wbar

    return(u)


def back_parallel_transport(u1, alpha, basis, T=100, k=5):
    """
    backwards parallel translates q1 and q2 along manifold

    :param u1: numpy ndarray of shape (2,M) of M samples
    :param alpha: numpy ndarray of shape (2,M) of M samples
    :param basis: list numpy ndarray of shape (2,M) of M samples
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return utilde: translated vector

    """
    utilde = zeros((2, T, k))

    utilde[:, :, k-1] = u1
    for tau in arange(k-2, -1, -1):
        w = utilde[:, :, tau+1]
        q1 = alpha[:, :, tau+1]
        q2 = alpha[:, :, tau]
        b = basis[tau]
        utilde[:, :, tau] = cf.parallel_translate(w, q1, q2, b)

    return(utilde)


def calculate_gradE(u, utilde, T=100, k=5):
    """
    calculates gradient of energy along path

    :param u: numpy ndarray of shape (2,M) of M samples
    :param utilde: numpy ndarray of shape (2,M) of M samples
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy scalar
    :return gradE: gradient of energy
    :return normgradE: norm of gradient of energy

    """
    gradE = zeros((2, T, k))
    normgradE = zeros(k)

    for tau in range(2, k+1):
        gradE[:, :, tau-1] = u[:, :, tau-1] - ((tau-1.)/(k-1.)) * utilde[:, :, tau-1]
        normgradE[tau-1] = sqrt(cf.innerprod_q2(gradE[:, :, tau-1], gradE[:, :, tau-1]))

    return(gradE, normgradE)


def update_path(alpha, beta, gradE, delta, T=100, k=5):
    """
    Update the path along the direction -gradE

    :param alpha: numpy ndarray of shape (2,M) of M samples
    :param beta: numpy ndarray of shape (2,M) of M samples
    :param gradE: numpy ndarray of shape (2,M) of M samples
    :param delta: gradient paramenter
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy scalar
    :return alpha: updated path of srvfs
    :return beta: updated path of curves

    """
    for tau in range(1, k-1):
        alpha_new = alpha[:, :, tau] - delta*gradE[:, :, tau]
        alpha[:, :, tau] = cf.project_curve(alpha_new)
        x = cf.q_to_curve(alpha[:, :, tau])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau] = x + tile(a, [T, 1]).T

    return(alpha, beta)


def geod_dist_path_strt(beta, k=5):
    """
    calculate geodisc distance for path straightening

    :param beta: numpy ndarray of shape (2,M) of M samples
    :param k: number of samples along path (Default = 5)

    :rtype: numpy scalar
    :return dist: geodesic distance

    """
    dist = 0

    for i in range(1, k):
        beta1 = beta[:, :, i-1]
        beta2 = beta[:, :, i]
        q1 = cf.curve_to_q(beta1)[0]
        q2 = cf.curve_to_q(beta2)[0]
        d = arccos(cf.innerprod_q2(q1, q2))
        dist += d

    return(dist)
