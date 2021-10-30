"""
functions for SRVF curve manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.integrate import trapz, cumtrapz
from numpy import zeros, ones, cumsum, linspace, gradient, sqrt, ascontiguousarray
from numpy import finfo, double, eye, roll, tile, vstack, array, cos, sin
from numpy import arccos, fabs, floor, fliplr, log, real, diff, mean
from scipy.linalg import norm, svd, det, solve
import optimum_reparam_N as orN
import fdasrsf.utility_functions as uf
from fdasrsf.rbfgs import rlbfgs


def resamplecurve(x, N=100, time=None, mode='O'):
    """
    This function resamples a curve to have N samples

    :param x: numpy ndarray of shape (2,M) of M samples
    :param N: Number of samples for new curve (default = 100)
    :param time: timing vector (Default=None)
    :param mode: Open ('O') or closed curve ('C') (default 'O')

    :rtype: numpy ndarray
    :return xn: resampled curve

    """
    n, T = x.shape
    xn = zeros((n, N))

    tst = x[:,1]-x[:,0]
    if tst[0] < 0:
        x = fliplr(x)

    delta = zeros(T)
    for r in range(1, T):
        delta[r] = norm(x[:, r] - x[:, r - 1])

    if time is None: 
        cumdel = cumsum(delta) / delta.sum()
    else:
        time -= time[0]
        time /= time[-1]
        cumdel = time
    newdel = linspace(0, 1, N)

    for r in range(0, n):
        s = InterpolatedUnivariateSpline(cumdel, x[r, :], k=3)
        xn[r, :] = s(newdel)

    if mode == 'C':
        q = curve_to_q(xn)[0]
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


def curve_to_q(beta,mode='O'):
    """
    This function converts curve beta to srvf q

    :param beta: numpy ndarray of shape (2,M) of M samples
    :param mode: Open ('O') or closed curve ('C') (default 'O')

    :rtype: numpy ndarray
    :return q: srvf of curve
    :return lenb: length of curve
    :return lenq: length of srvf

    """
    n, T = beta.shape
    v = gradient(beta, 1. / (T - 1))
    v = v[1]

    q = zeros((n, T))
    lenb = sqrt(innerprod_q2(v,v))
    for i in range(0, T):
        L = sqrt(norm(v[:, i]))
        if L > 0.0001:
            q[:, i] = v[:, i] / L
        else:
            q[:, i] = v[:, i] * 0.0001

    lenq = sqrt(innerprod_q2(q,q))
    q = q / lenq
    
    if mode == 'C':
        q = project_curve(q)

    return (q, lenb, lenq)


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
        beta[i,:] = cumtrapz(q[i, :] * qnorm, initial=0)/T

    beta = scale*beta

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


def optimum_reparam_curve(q1, q2, lam=0.0, method="DP"):
    """
    calculates the warping to align srsf q2 to q1

    :param q1: matrix of size nxN or array of NxM samples of first SRVF
    :param time: vector of size N describing the sample points
    :param q2: matrix of size nxN or array of NxM samples samples of second SRVF
    :param lam: controls the amount of elasticity (default = 0.0)
    :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"

    :rtype: vector
    :return gam: describing the warping function used to align q2 with q1

    """
    time = linspace(0, 1, q1.shape[1])
    if method == "DP":
        gam = orN.coptimum_reparam_curve(ascontiguousarray(q1), time,
                                         ascontiguousarray(q2), lam)
    elif method == "RBFGS":
        obj = rlbfgs(q1,q2,time)
        obj.solve()
        gam = obj.gammaOpt
    else:
        raise Exception('Invalid Optimization Method')

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


def find_best_rotation(q1, q2, allow_reflection = False, only_xy = False):
    """
    This function calculates the best rotation between two srvfs using
    procustes rigid alignment

    :param q1: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples
    :param allow_reflection: bool indicating if reflection is allowed 
                             (i.e. if the determinant of the optimal 
                             rotation can be -1)
    :param only_xy: bool indicating if rotation should only be allowed 
                    in the first two dimensions of the space

    :rtype: numpy ndarray
    :return q2new: optimal rotated q2 to q1
    :return R: rotation matrix

    """
    if q1.ndim != 2 or q2.ndim != 2:
        raise Exception("This only supports curves of shape (N,M) for N dimensions and M samples")

    n = q1.shape[0]

    # if only_xy, strip everything but the x and y coordinates of q1 and q2
    if only_xy:
        _q1 = q1[0:2, :]
        _q2 = q2[0:2, :]
    else:
        _q1 = q1
        _q2 = q2

    _n = _q1.shape[0]
    A = _q1@_q2.T
    U, s, Vh = svd(A)
    S = eye(_n)

    # if reflections are not allowed and the determinant of A is negative,
    # then the entry corresponding to the smallest singular value is negated
    # as in the Kabsch algorithm
    if det(A) < 0 and not allow_reflection:
        S[-1, -1] = -1 # the last entry of the matrix becomes -1

    _R = U@S@Vh # optimal
    
    # if only_xy, the top left block of the matrix is _R and the rest is identity matrix
    if only_xy:
        R = eye(n)
        R[0:2, 0:2] = _R
    else:
        R = _R
        
    q2new = R@q2

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
        integrand[:, :, i] = a1 @ a1.T * normbetadot[i]

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


def find_rotation_and_seed_unique(q1, q2, closed=0, rotation=True, method="DP"):
    """
    This function returns a candidate list of optimally oriented and
    registered (seed) shapes w.r.t. beta1

    :param beta1: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples
    :param closed: Open (0) or Closed (1)
    :param rotation: find rotation (default=True)
    :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"

    :rtype: numpy ndarray
    :return beta2new: optimal rotated beta2 to beta1
    :return O: rotation matrix
    :return tau: seed
    """

    n, T = q1.shape

    scl = 4.
    minE = 1000
    if closed == 1:
        end_idx = int(floor(T/scl))
        scl = 4
    else:
        end_idx = 0
    
    for ctr in range(0, end_idx+1):
        if closed == 1:
            q2n = shift_f(q2, scl*ctr)
        else:
            q2n = q2.copy()
        
        if rotation:
            q2new, R = find_best_rotation(q1, q2n)
        else:
            q2new = q2n
            R = eye(n)

        # Reparam
        if norm(q1-q2new,'fro') > 0.0001:
            gam = optimum_reparam_curve(q2new, q1, 0.0, method)
            gamI = uf.invertGamma(gam)
            p2n = q_to_curve(q2n)
            p2n = group_action_by_gamma_coord(p2n,gamI)
            q2new = curve_to_q(p2n)[0]
            if closed == 1:
                q2new = project_curve(q2new)
        else:
            gamI = linspace(0,1,T)
        
        tmp = innerprod_q2(q1,q2new)
        if tmp > 1:
            tmp = 1
        if tmp < -1:
            tmp = -1
        Ec = arccos(tmp)
        if Ec < minE:
            Rbest = R
            q2best = q2new
            gamIbest = gamI
            minE = Ec

    return (q2best, Rbest, gamIbest)


def find_rotation_and_seed_coord(beta1, beta2, closed=0, rotation=True, method="DP"):
    """
    This function returns a candidate list of optimally oriented and
    registered (seed) shapes w.r.t. beta1

    :param beta1: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples
    :param closed: Open (0) or Closed (1)
    :param rotation: find rotation (default=True)
    :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"

    :rtype: numpy ndarray
    :return beta2new: optimal aligned beta2 to beta1
    :return q2best: optimal aligned q2 to q1 
    :return Rbest: rotation matrix
    :return gamIbest: warping function
    """

    n, T = beta1.shape
    q1 = curve_to_q(beta1)[0]
    scl = 4.
    minE = 1000
    if closed == 1:
        end_idx = int(floor(T/scl))
        scl = 4
    else:
        end_idx = 0
    
    for ctr in range(0, end_idx+1):
        if closed == 1:
            beta2n = shift_f(beta2, scl*ctr)
        else:
            beta2n = beta2
        
        if rotation:
            beta2new, R = find_best_rotation(beta1, beta2n)
        else:
            beta2new = beta2n
            R = eye(n)
        q2new = curve_to_q(beta2new)[0]

        # Reparam
        if norm(q1-q2new,'fro') > 0.0001:
            gam = optimum_reparam_curve(q2new, q1, 0.0, method)
            gamI = uf.invertGamma(gam)
            beta2new = group_action_by_gamma_coord(beta2new,gamI)
            q2new = curve_to_q(beta2new)[0]
            if closed == 1:
                q2new = project_curve(q2new)
        else:
            gamI = linspace(0,1,T)
        
        tmp = innerprod_q2(q1,q2new)
        if tmp > 1:
            tmp = 1
        if tmp < -1:
            tmp = -1
        Ec = arccos(tmp)
        if Ec < minE:
            Rbest = R
            beta2best = beta2new
            q2best = q2new
            gamIbest = gamI
            minE = Ec

    return (beta2best, q2best, Rbest, gamIbest)


def find_rotation_and_seed_q(q1, q2, closed=0, rotation=True, method="DP"):
    """
    This function returns a candidate list of optimally oriented and
    registered (seed) srvs w.r.t. q1

    :param q1: numpy ndarray of shape (2,M) of M samples
    :param q2: numpy ndarray of shape (2,M) of M samples
    :param closed: Open (0) or Closed (1)
    :param rotation: find rotation (default=True)
    :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"

    :rtype: numpy ndarray
    :return q2best: optimal aligned q2 to q1 
    :return Rbest: rotation matrix
    :return gamIbest: warping function
    """

    n, T = q1.shape
    scl = 4.
    minE = 4000
    if closed == 1:
        end_idx = int(floor(T/scl))
        scl = 4
    else:
        end_idx = 0
    
    for ctr in range(0, end_idx+1):
        if closed == 1:
            q2n = shift_f(q2, scl*ctr)
        else:
            q2n = q2
        
        if rotation:
            q2new, R = find_best_rotation(q1, q2n)
        else:
            q2new = q2n.copy()
            R = eye(n)

        # Reparam
        if norm(q1-q2new,'fro') > 0.0001:
            gam = optimum_reparam_curve(q2new, q1, 0.0, method)
            gamI = uf.invertGamma(gam)
            q2new = group_action_by_gamma(q2new,gamI)
            if closed == 1:
                q2new = project_curve(q2new)
        else:
            gamI = linspace(0,1,T)
        
        tmp = innerprod_q2(q1,q2new)
        if tmp > 1:
            tmp = 1
        if tmp < -1:
            tmp = -1
        Ec = arccos(tmp)
        if Ec < minE:
            Rbest = R
            q2best = q2new
            gamIbest = gamI
            minE = Ec

    return (q2best, Rbest, gamIbest)


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
        temp = zeros((n,T))
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
    q = curve_to_q(beta)[0]
    qnew = project_curve(q)
    x = q_to_curve(qnew)
    a = -1 * calculatecentroid(x)
    betanew = x + tile(a, [T, 1]).T
    A = eye(2)

    return (betanew, qnew, A)


def elastic_distance_curve(beta1, beta2, closed=0, rotation=True, scale=False, method="DP"):
    """
    Calculates the two elastic distances between two curves
    :param beta1: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples
    :param closed: open (0) or closed (1) curve (default=0)
    :param rotation: compute optimal rotation (default=True)
    :param scale: include scale (default=False)
    :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"

    :rtype: tuple
    :return dist: shape distance
    :return dx: phase distance
    """

    if (beta1 == beta2).all():
        d = 0.0
        dx = 0.0
    else:
        N = beta1.shape[1]
        a = -calculatecentroid(beta1)
        beta1 += tile(a, (N,1)).T
        a = -calculatecentroid(beta2)
        beta2 += tile(a, (N,1)).T

        q1, len1, lenq1 = curve_to_q(beta1)
        q2, len2, lenq2 = curve_to_q(beta2)

        # compute shooting vector from q1 to q2
        beta2best, qn_t, Rbest, gam = find_rotation_and_seed_coord(beta1, beta2, closed, rotation, method)

        q1dotq2 = innerprod_q2(q1, qn_t)
        if q1dotq2 > 1:
            q1dotq2 = 1
        elif q1dotq2 < -1:
            q1dotq2 = -1
        
        if scale:
            d = sqrt(arccos(q1dotq2)**2+log(lenq1/lenq2)**2)
        else:
            d = arccos(q1dotq2)
        
        time1 = linspace(0,1,N)
        binsize = mean(diff(time1))
        psi = sqrt(gradient(gam,binsize))
        q1dotq2 = trapz(psi, time1)
        if q1dotq2 > 1:
            q1dotq2 = 1
        elif q1dotq2 < -1:
            q1dotq2 = -1

        dx = real(arccos(q1dotq2))

    return d, dx
    

def inverse_exp_coord(beta1, beta2, closed=0, method="DP"):
    """
    Calculate the inverse exponential to obtain a shooting vector from
    beta1 to beta2 in shape space of open curves

    :param beta1: numpy ndarray of shape (2,M) of M samples
    :param beta2: numpy ndarray of shape (2,M) of M samples
    :param closed: open (0) or closed (1) curve
    :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"

    :rtype: numpy ndarray
    :return v: shooting vectors
    :return dist: distance

    """
    T = beta1.shape[1]
    centroid1 = calculatecentroid(beta1)
    beta1 = beta1 - tile(centroid1, [T, 1]).T
    centroid2 = calculatecentroid(beta2)
    beta2 = beta2 - tile(centroid2, [T, 1]).T

    q1 = curve_to_q(beta1)[0]

    # Iteratively optimize over SO(n) x Gamma
    beta2n, q2n, O_hat, gamI = find_rotation_and_seed_coord(beta1, beta2, closed, method)

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
    q2, O_hat, gamI = find_rotation_and_seed_q(q1, q2)

    # Applying optimal re-parameterization to the second curve
    beta2 = group_action_by_gamma_coord(beta2, gamI)
    q2 = curve_to_q(beta2)

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


def curve_zero_crossing(Y, q, bt, y_max, y_min, gmax, gmin):
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
    n = q.shape[0]
    T = q.shape[1]
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

        gamma = a[ii] * gmax + (1 - a[ii]) * gmin

        q1 = group_action_by_gamma(q, gamma)
        #q1, O_hat1 = find_best_rotation(bt, q1)
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

    #q1 = group_action_by_gamma(q, gamma)
    #q1, O_hat = find_best_rotation(bt, q1)
    O_hat = eye(n)

    return (gamma, O_hat)


def elastic_shooting(q1,v, mode=0):
    """
    Calculates shooting vector from v to q1

    :param q1: vector of srvf
    :param v: shooting vector
    :param mode: closed or open (1/0)

    :rtype numpy ndarray
    :return q2n: vector of srvf
    """
    d = sqrt(innerprod_q2(v,v))
    if d < 0.00001:
        q2n = q1
    else:
        q2n = cos(d)*q1 + (sin(d)/d)*v
        if mode == 1:
            q2n = project_curve(q2n)
    
    return (q2n)


def elastic_shooting_vector(q1,q2, mode=0):
    """
    Calculates shooting between two srvfs

    :param q1: vector of srvf
    :param q2: vector of srvf
    :param mode: closed or open (1/0)

    :rtype numpy ndarray
    :return v: shooting vector
    :return d: distance
    :return q2n: aligned srvf
    """
    
    (q2n, Rbest, gamIbest) = find_rotation_and_seed_unique(q1, q2, closed=mode)   
    lenq = sqrt(innerprod_q2(q2n,q2n))
    q2n = q2n / lenq
    
    d = sqrt(innerprod_q2(q1,q2n))
    if d < 0.00001:
        v = np.zeros(q1.shape)
    else:
        v = (d/sin(d))*(q2n-cos(d)*q1)
        q2n = cos(d)*q1 + (sin(d)/d)*v
        if mode == 1:
            q2n = project_curve(q2n)
    
    return (v,d,q2n)


def rot_mat(theta):
    O = array([(cos(theta), -1*sin(theta)), (sin(theta), cos(theta))])

    return (O)
