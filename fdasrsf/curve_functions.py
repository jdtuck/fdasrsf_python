"""
functions for SRVF curve manipulations

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import trapz, cumtrapz
from numpy import zeros, cumsum, linspace, gradient, sqrt, ascontiguousarray
from numpy import finfo, double, eye, roll, tile, vstack, array, cos, sin
from scipy.linalg import norm, svd, det, solve
import optimum_reparamN as orN


def resamplecurve(x, N=100):
    """
    This function resamples a curve to have N samples

    :param x: numpy ndarray of shape (2,M) of M samples
    :param N: Number of samples for new curve (default = 100)

    :rtype: numpy ndarray
    :return xn: resampled curve

    """
    n, T = x.shape
    xn = zeros((n, N))

    delta = zeros(T)
    for r in range(1, T):
        delta[r] = norm(x[:, r] - x[:, r-1])

    cumdel = cumsum(delta)/delta.sum()
    newdel = linspace(0, 1, N)

    for r in range(0, n):
        s = InterpolatedUnivariateSpline(cumdel, x[r, :], k=3)
        xn[r, :] = s(newdel)

    return(xn)


def calculatecentroid(beta):
    """
    This function calculates centroid of a parameterized curve

    :param beta: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return centroid: center coordinates

    """
    n, T = beta.shape
    betadot = gradient(beta, 1./(T - 1))
    betadot = betadot[1]
    normbetadot = zeros(T)
    integrand = zeros((n, T))
    for i in range(0, T):
        normbetadot[i] = norm(betadot[:, i])
        integrand[:, i] = beta[:, i] * normbetadot[i]

    scale = trapz(normbetadot, linspace(0, 1, T))
    centroid = trapz(integrand, linspace(0, 1, T), axis=1)/scale

    return(centroid)


def curve_to_q(beta):
    """
    This function converts curve beta to srvf q

    :param beta: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return q: srvf of curve

    """
    n, T = beta.shape
    v = gradient(beta, 1./(T - 1))
    v = v[1]

    length = sum(sqrt(sum(v*v)))/T
    v = v/length
    q = zeros((n, T))
    for i in range(0, T):
        L = sqrt(norm(v[:, i]))
        if L > 0.0001:
            q[:, i] = v[:, i]/L
        else:
            q[:, i] = v[:, i]*0.0001

    return(q)


def q_to_curve(q):
    """
    This function converts srvf to beta

    :param q: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return beta: parameterized curve

    """
    T = q.shape[1]
    qnorm = zeros(T)
    for i in range(0, T):
        qnorm[i] = norm(q[:, i])

    integrand = zeros((2, T))
    integrand[0, :] = q[0, :] * qnorm
    integrand[1, :] = q[1, :] * qnorm
    beta = cumtrapz(integrand, axis=1, initial=0)/T

    return(beta)


def optimum_reparam_curve(q1, q2, lam=0.0):
    """
    calculates the warping to align srsf q2 to q1

    :param q1: matrix of size nxN or array of NxM samples of first SRVF
    :param time: vector of size N describing the sample points
    :param q2: matrix of size nxN or array of NxM samples samples of second SRVF
    :param lam: controls the amount of elasticity (default = 0.0)

    :rtype: vector
    :return gam: describing the warping function used to align q2 with q1

    """
    time = linspace(0, 1, q1.shape[1])
    gam = orN.coptimum_reparam_curve(ascontiguousarray(q1), time,
                                     ascontiguousarray(q2), lam)

    return gam


def innerprod_q(q1, q2):
    """
    This function calculates the innerproduct in srvf space

    :param q1: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return val: inner product

    """
    T = q1.shape[1]
    val = sum(sum(q1*q2))/T

    return(val)


def find_best_rotation(q1, q2):
    """
    This function calculates the best rotation between two srvfs using
    procustes rigid alignment

    :param q1: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return q2new: optimal rotated q2 to q1
    :return R: rotation matrix

    """
    eps = finfo(double).eps
    n, T = q1.shape
    A = q1.dot(q2.T)
    U, s, V = svd(A)

    if (abs(det(U)*det(V)-1) < 10*eps):
        S = eye(n)
    else:
        S = eye(n)
        S[:, -1] = -S[:, -1]

    R = U.dot(S).dot(V.T)
    q2new = R.dot(q2)

    return(q2new, R)


def calculate_variance(beta):
    """
    This function calculates variance of curve beta

    :param beta: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return variance: variance

    """
    n, T = beta.shape
    betadot = gradient(beta, 1./(T - 1))
    betadot = betadot[1]
    normbetadot = zeros(T)
    centroid = calculatecentroid(beta)
    integrand = zeros((n, n, T))
    t = linspace(0, 1, T)
    for i in range(0, T):
        normbetadot[i] = norm(betadot[:, i])
        a1 = (beta[:, i]-centroid)
        a1 = a1.reshape((n, 1))
        integrand[:, :, i] = a1.dot(a1.T) * normbetadot[i]

    l = trapz(normbetadot, t)
    variance = trapz(integrand, t, axis=2)
    variance /= l

    return(variance)


def psi(x, a, q):
    """
    This function formats variance output

    :param x: numpy ndarray of shape (2,M) of M samples curve
    :param a: numpy ndarray of shape (2,1) mean
    :param q: numpy ndarray of shape (2,M) of M samples srvf

    :rtype: numpy ndarray
    :return psi1: variance
    :return psi2: cross variance
    :return psi3: curve end
    :return psi4: curve end

    """
    T = q.shape[1]
    covmat = calculate_variance(x + tile(a, [T, 1]).T)
    psi1 = covmat[0, 0] - covmat[1, 1]
    psi2 = covmat[0, 1]
    psi3 = x[0, -1]
    psi4 = x[1, -1]

    return(psi1, psi2, psi3, psi4)


def find_basis_normal(q):
    """
    Finds the basis normal to the srvf

    :param q1: numpy ndarray of shape (2,M) of M samples

    :rtype: list of numpy ndarray
    :return basis: list containing basis vectors

    """
    n, T = q.shape

    f1 = zeros((n, T))
    f2 = zeros((n, T))
    for i in range(0, T):
        f1[:, i] = q[0, i] * q[:, i]/norm(q[:, i]) + array([norm(q[:, i]),
                                                           0])
        f2[:, i] = q[1, i] * q[:, i]/norm(q[:, i]) + array([0,
                                                           norm(q[:, i])])

    h3 = f1
    h4 = f2
    integrandb3 = zeros(T)
    integrandb4 = zeros(T)
    for i in range(0, T):
        a = q[:, i].T
        integrandb3[i] = a.dot(h3[:, i])
        integrandb4[i] = a.dot(h4[:, i])

    b3 = h3 - q * trapz(integrandb3, linspace(0, 1, T))
    b4 = h4 - q * trapz(integrandb4, linspace(0, 1, T))

    basis = [b3, b4]

    return(basis)


def calc_j(basis):
    """
    Calculates Jacobian matrix from normal basis

    :param basis: list of numpy ndarray of shape (2,M) of M samples basis

    :rtype: numpy ndarray
    :return j: Jacobian

    """
    b1 = basis[0]
    b2 = basis[1]
    T = b1.shape[1]
    integrand11 = zeros(T)
    integrand12 = zeros(T)
    integrand22 = zeros(T)

    for i in range(0, T):
        a = b1[:, i].T
        b = b2[:, i].T
        integrand11[i] = a.dot(b1[:, i])
        integrand12[i] = a.dot(b2[:, i])
        integrand22[i] = b.dot(b2[:, i])

    j = zeros((2, 2))
    j[0, 0] = trapz(integrand11, linspace(0, 1, T))
    j[0, 1] = trapz(integrand12, linspace(0, 1, T))
    j[1, 1] = trapz(integrand22, linspace(0, 1, T))
    j[1, 0] = j[0, 1]

    return(j)


def shift_f(f, tau):
    """
    shifts a curve f by tau

    :param f: numpy ndarray of shape (2,M) of M samples
    :param tau: scalar

    :rtype: numpy ndarray
    :return fn: shifted curve

    """
    n, T = f.shape
    fn = zeros((n, T))
    fn[:, 0:(T-1)] = roll(f[:, 0:(T-1)], tau, axis=1)
    fn[:, T-1] = fn[:, 0]

    return(fn)


def find_rotation_and_seed_coord(beta1, beta2):
    """
    This function returns a candidate list of optimally oriented and
    registered (seed) shapes w.r.t. beta1

    :param beta2: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return beta2new: optimal rotated beta2 to beta1
    :return O: rotation matrix
    :return tau: seed

    """
    n, T = beta1.shape
    q1 = curve_to_q(beta1)
    Ltwo = zeros(T)
    Rlist = zeros((n, n, T))
    for ctr in range(0, T):
        beta2n = shift_f(beta2, ctr)
        beta2new, R = find_best_rotation(beta1, beta2n)
        q2new = curve_to_q(beta2new)
        Ltwo[ctr] = innerprod_q(q1 - q2new, q1 - q2new)
        Rlist[:, :, ctr] = R

    tau = Ltwo.argmin()
    O_hat = Rlist[:, :, tau]
    beta2new = shift_f(beta2, tau)
    beta2new = O_hat.dot(beta2new)

    return(beta2new, O_hat, tau)


def group_action_by_gamma_coord(f, gamma):
    """
    This function reparamerized curve f by gamma

    :param f: numpy ndarray of shape (2,M) of M samples
    :param gamma: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return fn: reparatermized curve

    """
    n, T = f.shape
    fn = zeros((n, T))

    for j in range(0, n):
        s = InterpolatedUnivariateSpline(linspace(0, 1, T), f[j, :], k=3)
        fn[j, :] = s(gamma)

    return(fn)


def group_action_by_gamma(q, gamma):
    """
    This function reparamerized srvf q by gamma

    :param f: numpy ndarray of shape (2,M) of M samples
    :param gamma: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return qn: reparatermized srvf

    """
    n, T = q.shape
    gammadot = gradient(gamma, 1./T)
    qn = zeros((n, T))

    for j in range(0, n):
        s = InterpolatedUnivariateSpline(linspace(0, 1, T), q[j, :], k=3)
        qn[j, :] = s(gamma)*sqrt(gammadot)

    qn = qn/sqrt(innerprod_q(qn, qn))

    return(qn)


def project_curve(q):
    """
    This function projects srvf q to set of close curves

    :param q: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return qproj: project srvf

    """
    T = q.shape[1]
    tol = 1e-5
    maxit = 200
    itr = 1
    delta = 0.5
    x = q_to_curve(q)
    a = -1*calculatecentroid(x)

    psi1, psi2, psi3, psi4 = psi(x, a, q)

    r = vstack((psi3, psi4))
    rnorm = zeros(maxit+1)
    rnorm[0] = norm(r)

    while itr <= maxit:
        basis = find_basis_normal(q)

        # calculate Jacobian
        j = calc_j(basis)

        # Newton-Raphson step to update q
        y = solve(j, -r)
        dq = delta * (y[0]*basis[0] + y[1]*basis[1])
        normdq = sqrt(innerprod_q(dq, dq))
        q = cos(normdq)*q + sin(normdq)*dq/normdq
        q /= sqrt(innerprod_q(q, q))

        # update x and a from the new q
        beta_new = q_to_curve(q)
        x = beta_new
        a = -1*calculatecentroid(x)
        beta_new = x + tile(a, [T, 1]).T

        # calculate the new value of psi
        psi1, psi2, psi3, psi4 = psi(x, a, q)
        r = vstack((psi3, psi4))
        rnorm[itr] = norm(r)

        if norm(r) < tol:
            break

        itr += 1

    rnorm = rnorm[0:itr]
    qproj = q

    return(qproj)


def pre_proc_curve(beta, T=100):
    """
    This function prepcoessed a curve beta to set of closed curves

    :param beta: numpy ndarray of shape (2,M) of M samples
    :param T: number of samples (default = 100)

    :rtype: numpy ndarray
    :return betanew: projected beta
    :return qnew: projected srvf
    :return A: alignment matrix (not used currently)

    """
    beta = resamplecurve(beta, T)
    q = curve_to_q(beta)
    qnew = project_curve(q)
    x = q_to_curve(qnew)
    a = -1*calculatecentroid(x)
    betanew = x + tile(a, [T, 1]).T
    A = eye(2)

    return(betanew, qnew, A)
