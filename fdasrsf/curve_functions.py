"""
functions for SRVF curve manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.integrate import trapz, cumtrapz
from numpy import zeros, ones, cumsum, linspace, gradient, sqrt, ascontiguousarray
from numpy import finfo, double, eye, roll, tile, vstack, array, cos, sin
from numpy import arccos, fabs, floor
from scipy.linalg import norm, svd, det, solve
import optimum_reparam_N as orN
import fdasrsf.utility_functions as uf


def resamplecurve(x, N=100, mode='O'):
    """
    This function resamples a curve to have N samples

    :param x: numpy ndarray of shape (2,M) of M samples
    :param N: Number of samples for new curve (default = 100)
    :param mode: Open ('O') or closed curve ('C') (default 'O')

    :rtype: numpy ndarray
    :return xn: resampled curve

    """
    n, T = x.shape
    xn = zeros((n, N))

    delta = zeros(T)
    for r in range(1, T):
        delta[r] = norm(x[:, r] - x[:, r - 1])

    cumdel = cumsum(delta) / delta.sum()
    newdel = linspace(0, 1, N)

    for r in range(0, n):
        s = InterpolatedUnivariateSpline(cumdel, x[r, :], k=3)
        xn[r, :] = s(newdel)

    if mode == 'C':
        q = curve_to_q(xn)
        qn = project_curve(q)
        xn = q_to_curve(qn)

    return (xn)


def calculatecentroid(beta):
    """
    This function calculates centroid of a parameterized curve

    :param beta: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return centroid: center coordinates

    """
    n, T = beta.shape
    betadot = gradient(beta, 1. / (T - 1))
    betadot = betadot[1]
    normbetadot = zeros(T)
    integrand = zeros((n, T))
    for i in range(0, T):
        normbetadot[i] = norm(betadot[:, i])
        integrand[:, i] = beta[:, i] * normbetadot[i]

    scale = trapz(normbetadot, linspace(0, 1, T))
    centroid = trapz(integrand, linspace(0, 1, T), axis=1) / scale

    return (centroid)


def curve_to_q(beta,scale=True,mode='O'):
    """
    This function converts curve beta to srvf q

    :param beta: numpy ndarray of shape (2,M) of M samples
    :param scale: scale curve to length 1
    :param mode: Open ('O') or closed curve ('C') (default 'O')

    :rtype: numpy ndarray
    :return q: srvf of curve
    :return len: length of curve

    """
    n, T = beta.shape
    v = gradient(beta, 1. / (T - 1))
    v = v[1]

    q = zeros((n, T))
    for i in range(0, T):
        L = sqrt(norm(v[:, i]))
        if L > 0.0001:
            q[:, i] = v[:, i] / L
        else:
            q[:, i] = v[:, i] * 0.0001

    len1 = sqrt(innerprod_q2(q,q))
    if scale:
        q = q / sqrt(innerprod_q2(q, q))
    
    if mode == 'C':
        q = project_curve(q)

    return q


def q_to_curve(q,scale=1):
    """
    This function converts srvf to beta

    :param q: numpy ndarray of shape (n,M) of M samples
    :param scale: scale of curve

    :rtype: numpy ndarray
    :return beta: parameterized curve

    """
    n,T = q.shape
    qnorm = zeros(T)
    for i in range(0, T):
        qnorm[i] = norm(q[:, i])

    beta = zeros((n,T))
    for i in range(0,n):
        beta[i,:] = cumtrapz(q[i, :] * qnorm * scale, initial=0)/T

    return (beta)


def Basis_Normal_A(q):
    """
    Find Normal Basis

    :param q: numpy ndarray (n,T) defining T points on n dimensional SRVF

    :rtype list
    :return delg: basis
    """
    n,T = q.shape
    e = eye(n)
    Ev = zeros((n,T,n))
    for i in range(0,n):
        Ev[:,:,i] = tile(e[:,i], (T, 1)).T
    
    qnorm = zeros(T)
    for t in range(0,T):
        qnorm[t] = norm(q[:,t])
    
    delG = list()
    for i in range(0,n):
        tmp1 = tile(q[i,:]/qnorm, (n,1))
        tmp2 = tile(qnorm, (n,1))
        delG.append(tmp1*q + tmp2*Ev[:,:,i])
    
    return delG


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


def innerprod_q2(q1, q2):
    """
    This function calculates the inner product in srvf space

    :param q1: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return val: inner product

    """
    T = q1.shape[1]
    val = sum(sum(q1 * q2)) / T

    return (val)


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
    n = q1.shape[0]
    A = q1.dot(q2.T)
    U, s, V = svd(A)

    if (det(A) > 0):
        S = eye(n)
    else:
        S = eye(n)
        S[:, -1] = -S[:, -1]

    R = U.dot(S).dot(V.T)
    q2new = R.dot(q2)

    return (q2new, R)


def calculate_variance(beta):
    """
    This function calculates variance of curve beta

    :param beta: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return variance: variance

    """
    n, T = beta.shape
    betadot = gradient(beta, 1. / (T - 1))
    betadot = betadot[1]
    normbetadot = zeros(T)
    centroid = calculatecentroid(beta)
    integrand = zeros((n, n, T))
    t = linspace(0, 1, T)
    for i in range(0, T):
        normbetadot[i] = norm(betadot[:, i])
        a1 = (beta[:, i] - centroid)
        a1 = a1.reshape((n, 1))
        integrand[:, :, i] = a1.dot(a1.T) * normbetadot[i]

    l = trapz(normbetadot, t)
    variance = trapz(integrand, t, axis=2)
    variance /= l

    return (variance)


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

    return (psi1, psi2, psi3, psi4)


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
        f1[:, i] = q[0, i] * q[:, i] / norm(q[:, i]) + array([norm(q[:, i]),
                                                              0])
        f2[:, i] = q[1, i] * q[:, i] / norm(q[:, i]) + array([0,
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

    return (basis)


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

    return (j)


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
    fn[:, 0:(T - 1)] = roll(f[:, 0:(T - 1)], tau, axis=1)
    fn[:, T - 1] = fn[:, 0]

    return (fn)


def find_rotation_and_seed_coord(beta1, beta2, mode=0):
    """
    This function returns a candidate list of optimally oriented and
    registered (seed) shapes w.r.t. beta1

    :param beta1: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples
    :param mode: Open (0) or Closed (1)

    :rtype: numpy ndarray
    :return beta2new: optimal rotated beta2 to beta1
    :return O: rotation matrix
    :return tau: seed

    """
    n, T = beta1.shape
    q1 = curve_to_q(beta1)
    scl = 4.
    minE = 1000
    if mode == 1:
        end_idx = int(floor(T/scl))
        scl = 4
    else:
        end_idx = 0
    
    for ctr in range(0, end_idx+1):
        if mode == 1:
            beta2n = shift_f(beta2, scl*ctr)
        else:
            beta2n = beta2
        
        beta2new, R = find_best_rotation(beta1, beta2n)
        q2new = curve_to_q(beta2new)

        # Reparam
        if norm(q1-q2new,'fro') > 0.0001:
            gam = optimum_reparam_curve(q2new, q1, 0.0)
            gamI = uf.invertGamma(gam)
            beta2new = group_action_by_gamma_coord(beta2new,gamI)
            q2new = curve_to_q(beta2new)
            if mode == 1:
                q2new = project_curve(q2new)
        else:
            gamI = linspace(0,1,T)
        
        tmp = innerprod_q2(q1,q2new)
        if tmp > 1:
            tmp = 1
        Ec = arccos(tmp)
        if Ec < minE:
            Rbest = R
            beta2best = beta2new
            q2best = q2new
            gamIbest = gamI
            minE = Ec

    return (beta2best, Rbest, gamIbest)


def find_rotation_and_seed_q(q1, q2):
    """
    This function returns a candidate list of optimally oriented and
    registered (seed) shapes w.r.t. beta1

    :param q1: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return beta2new: optimal rotated beta2 to beta1
    :return O: rotation matrix
    :return tau: seed

    """
    n, T = q1.shape
    Ltwo = zeros(T)
    Rlist = zeros((n, n, T))
    for ctr in range(0, T):
        q2n = shift_f(q2, ctr)
        q2new, R = find_best_rotation(q1, q2n)
        Ltwo[ctr] = innerprod_q2(q1 - q2new, q1 - q2new)
        Rlist[:, :, ctr] = R

    tau = Ltwo.argmin()
    O_hat = Rlist[:, :, tau]
    q2new = shift_f(q2, tau)
    q2new = O_hat.dot(q2new)

    return (q2new, O_hat, tau)


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
        s = interp1d(linspace(0, 1, T), f[j, :])
        fn[j, :] = s(gamma)

    return (fn)


def group_action_by_gamma(q, gamma):
    """
    This function reparamerized srvf q by gamma

    :param f: numpy ndarray of shape (2,M) of M samples
    :param gamma: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return qn: reparatermized srvf

    """
    n, T = q.shape
    gammadot = gradient(gamma, 1. / T)
    qn = zeros((n, T))

    for j in range(0, n):
        s = InterpolatedUnivariateSpline(linspace(0, 1, T), q[j, :], k=3)
        qn[j, :] = s(gamma) * sqrt(gammadot)

    qn = qn / sqrt(innerprod_q2(qn, qn))

    return (qn)


def project_curve(q):
    """
    This function projects srvf q to set of close curves

    :param q: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return qproj: project srvf
    """
    n,T = q.shape
    if n==2:
        dt = 0.35
    if n==3:
        dt = 0.2
    epsilon = 1e-6

    iter = 1
    res = ones(n)
    J = zeros((n,n))

    s = linspace(0,1,T)

    qnew = q.copy()
    qnew = qnew / sqrt(innerprod_q2(qnew,qnew))

    qnorm = zeros(T)
    G = zeros(n)
    C = zeros(300)
    while (norm(res) > epsilon):
        if iter > 300:
            break

        # Jacobian
        for i in range(0,n):
            for j in range(0,n):
                J[i,j] = 3 * trapz(qnew[i,:]*qnew[j,:],s)
        
        J += eye(n)

        for i in range(0,T):
            qnorm[i] = norm(qnew[:,i])
        
        # Compute the residue
        for i in range(0,n):
            G[i] = trapz(qnew[i,:]*qnorm,s)
        
        res = -G

        if (norm(res) < epsilon):
            break

        x = solve(J,res)
        C[iter] = norm(res)

        delG = Basis_Normal_A(qnew)
        temp = 0
        for i in range(0,n):
            temp += x[i]*delG[i]*dt
        
        qnew += temp
        iter += 1
    
    qnew = qnew/sqrt(innerprod_q2(qnew,qnew))

    return qnew


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
    a = -1 * calculatecentroid(x)
    betanew = x + tile(a, [T, 1]).T
    A = eye(2)

    return (betanew, qnew, A)


def elastic_distance_curve(beta1, beta2):
    """
    Calculates the two elastic distances between two curves
    :param beta1: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples

    :rtype: scalar
    :return dist: distance
    """

    if (beta1 == beta2).all():
        d = 0.0
    else:
        v,d = inverse_exp_coord(beta1, beta2)

    return d
    

def inverse_exp_coord(beta1, beta2, mode=0):
    """
    Calculate the inverse exponential to obtain a shooting vector from
    beta1 to beta2 in shape space of open curves

    :param beta1: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples
    :param mode: open (0) or closed (1) curve

    :rtype: numpy ndarray
    :return v: shooting vectors
    :return dist: distance

    """
    T = beta1.shape[1]
    centroid1 = calculatecentroid(beta1)
    beta1 = beta1 - tile(centroid1, [T, 1]).T
    centroid2 = calculatecentroid(beta2)
    beta2 = beta2 - tile(centroid2, [T, 1]).T

    q1 = curve_to_q(beta1)

    # Iteratively optimize over SO(n) x Gamma
    beta2n, O_hat, gamI = find_rotation_and_seed_coord(beta1, beta2, mode)
    q2n = curve_to_q(beta2n)

    # Compute geodesic distance
    q1dotq2 = innerprod_q2(q1, q2n)
    if q1dotq2 > 1:
        q1dotq2 = 1
    elif q1dotq2 < -1:
        q1dotq2 = -1
    dist = arccos(q1dotq2)

    # Compute shooting vector
    if q1dotq2 > 1:
        q1dotq2 = 1

    u = q2n - q1dotq2 * q1
    normu = sqrt(innerprod_q2(u, u))

    if normu > 1e-4:
        v = u * arccos(q1dotq2) / normu
    else:
        v = zeros((2, T))

    return (v, dist)


def inverse_exp(q1, q2, beta2):
    """
    Calculate the inverse exponential to obtain a shooting vector from
    q1 to q2 in shape space of open curves

    :param q1: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return v: shooting vectors

    """
    T = q1.shape[1]
    centroid1 = calculatecentroid(beta2)
    beta2 = beta2 - tile(centroid1, [T, 1]).T

    # Optimize over SO(n)
    q2, O_hat, tau = find_rotation_and_seed_q(q1, q2)

    # Optimize over Gamma
    gam = optimum_reparam_curve(q2, q1, 0.0)
    gamI = uf.invertGamma(gam)

    # Applying optimal re-parameterization to the second curve
    beta2 = group_action_by_gamma_coord(beta2, gamI)
    q2 = curve_to_q(beta2)

    # Optimize over SO(n)
    q2, O2, tau = find_rotation_and_seed_q(q1, q2)

    # Compute geodesic distance
    q1dotq2 = innerprod_q2(q1, q2)
    dist = arccos(q1dotq2)

    # Compute shooting vector
    if q1dotq2 > 1:
        q1dotq2 = 1

    u = q2 - q1dotq2 * q1
    normu = sqrt(innerprod_q2(u, u))

    if normu > 1e-4:
        v = u * arccos(q1dotq2) / normu
    else:
        v = zeros((2, T))

    return v


def gram_schmidt(basis):
    """
   Performs Gram Schmidt Orthogonlization of a basis_o

    :param basis: list of numpy ndarray of shape (2,M) of M samples

    :rtype: list of numpy ndarray
    :return basis_o: orthogonlized basis

    """
    b1 = basis[0]
    b2 = basis[1]

    basis1 = b1 / sqrt(innerprod_q2(b1, b1))
    b2 = b2 - innerprod_q2(basis1, b2) * basis1
    basis2 = b2 / sqrt(innerprod_q2(b2, b2))

    basis_o = [basis1, basis2]

    return (basis_o)


def project_tangent(w, q, basis):
    """
    projects srvf to tangent space w using basis

    :param w: numpy ndarray of shape (2,M) of M samples
    :param q: numpy ndarray of shape (2,M) of M samples
    :param basis: list of numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return wproj: projected q

    """
    w = w - innerprod_q2(w, q) * q
    bo = gram_schmidt(basis)

    wproj = w - innerprod_q2(w, bo[0]) * bo[0] - innerprod_q2(w, bo[1]) * bo[1]

    return (wproj)


def scale_curve(beta):
    """
    scales curve to length 1

    :param beta: numpy ndarray of shape (2,M) of M samples

    :rtype: numpy ndarray
    :return beta_scaled: scaled curve
    :return scale: scale factor used

    """
    n, T = beta.shape
    normbetadot = zeros(T)
    betadot = gradient(beta, 1. / T)
    betadot = betadot[1]
    for i in range(0, T):
        normbetadot[i] = norm(betadot[:, i])

    scale = trapz(normbetadot, linspace(0, 1, T))
    beta_scaled = beta / scale

    return (beta_scaled, scale)


def parallel_translate(w, q1, q2, basis, mode=0):
    """
    parallel translates q1 and q2 along manifold

    :param w: numpy ndarray of shape (2,M) of M samples
    :param q1: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples
    :param basis: list of numpy ndarray of shape (2,M) of M samples
    :param mode: open 0 or closed curves 1 (default 0)

    :rtype: numpy ndarray
    :return wbar: translated vector

    """
    modes = [0, 1]
    mode = [i for i, x in enumerate(modes) if x == mode]
    if len(mode) == 0:
        mode = 0
    else:
        mode = mode[0]

    wtilde = w - 2 * innerprod_q2(w, q2) / innerprod_q2(q1 + q2, q1 + q2) * (q1 + q2)
    l = sqrt(innerprod_q2(wtilde, wtilde))

    if mode == 1:
        wbar = project_tangent(wtilde, q2, basis)
        normwbar = sqrt(innerprod_q2(wbar, wbar))
        if normwbar > 10 ** (-4):
            wbar = wbar * l / normwbar
    else:
        wbar = wtilde

    return (wbar)


def curve_zero_crossing(Y, beta, bt, y_max, y_min, gmax, gmin):
    """
    finds zero-crossing of optimal gamma, gam = s*gmax + (1-s)*gmin
    from elastic curve regression model

    :param Y: response
    :param beta: predicitve function
    :param bt: basis function
    :param y_max: maximum repsonse for warping function gmax
    :param y_min: minimum response for warping function gmin
    :param gmax: max warping function
    :param gmin: min warping fucntion

    :rtype: numpy array
    :return gamma: optimal warping function
    :return O_hat: rotation matrix

    """
    # simple iterative method based on intermediate theorem
    T = beta.shape[1]
    betanu = q_to_curve(bt)
    max_itr = 100
    a = zeros(max_itr)
    a[0] = 1
    f = zeros(max_itr)
    f[0] = y_max - Y
    f[1] = y_min - Y
    mrp = f[0]
    mrn = f[1]
    mrp_ind = 0  # most recent positive index
    mrn_ind = 1  # most recent negative index

    for ii in range(2, max_itr):
        x1 = a[mrp_ind]
        x2 = a[mrn_ind]
        y1 = mrp
        y2 = mrn
        a[ii] = (x1 * y2 - x2 * y1) / float(y2 - y1)

        beta1, O_hat, tau = find_rotation_and_seed_coord(betanu, beta)

        gamma = a[ii] * gmax + (1 - a[ii]) * gmin

        beta1 = group_action_by_gamma_coord(beta1, gamma)
        beta1, O_hat1, tau = find_rotation_and_seed_coord(betanu, beta1)
        q1 = curve_to_q(beta1)
        f[ii] = innerprod_q2(q1, bt) - Y

        if fabs(f[ii]) < 1e-5:
            break
        elif f[ii] > 0:
            mrp = f[ii]
            mrp_ind = ii
        else:
            mrn = f[ii]
            mrn_ind = ii

    gamma = a[ii] * gmax + (1 - a[ii]) * gmin

    beta1, O_hat, tau = find_rotation_and_seed_coord(betanu, beta)
    beta1 = group_action_by_gamma_coord(beta1, gamma)
    beta1, O_hat1, tau = find_rotation_and_seed_coord(betanu, beta1)
    O_hat = O_hat.dot(O_hat1)

    return (gamma, O_hat, tau)


def elastic_shooting(q1,v):
    """
    Calculates shooting vector from v to q1

    :param q1: vector of srvf
    :param v: shooting vector

    :rtype numpy ndarray
    :return q2n: vector of srvf
    """
    d = sqrt(innerprod_q2(v,v))
    if d < 0.00001:
        q2n = q1
    else:
        q2n = cos(d)*q1 + (sin(d)/d)*v
        q2n = project_curve(q2n)
    
    return (q2n)


def rot_mat(theta):
    O = array([(cos(theta), -1*sin(theta)), (sin(theta), cos(theta))])

    return (O)
