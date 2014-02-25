"""
Warping Invariant Regression using SRVF

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""

import numpy as np
import fdasrsf.utility_functions as uf
import fdasrsf.curve_functions as cf
from scipy import dot
from scipy.optimize import fmin_l_bfgs_b
from scipy.integrate import trapz, cumtrapz
from scipy.linalg import inv, norm, expm
from patsy import bs
from joblib import Parallel, delayed
# import ocmlogit_warp as mw
import collections


def oc_elastic_regression(beta, y, B=None, df=20, T=100, max_itr=20, cores=-1):
    """
    This function identifies a regression model for open curves
    using elastic methods

    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param y: numpy array of N responses
    :param B: optional matrix describing Basis elements
    :param df: number of degrees of freedom B-spline (default 20)
    :param T: number of desired samples along curve (default 100)
    :param max_itr: maximum number of iterations (default 20)
    :param cores: number of cores for parallel processing (default all)
    :type beta: np.ndarray

    :rtype: tuple of numpy array
    :return alpha: alpha parameter of model
    :return beta: beta(t) of model
    :return fn: aligned functions - numpy ndarray of shape (M,N) of M
    functions with N samples
    :return qn: aligned srvfs - similar structure to fn
    :return gamma: calculated warping functions
    :return q: original training SRSFs
    :return B: basis matrix
    :return b: basis coefficients
    :return SSE: sum of squared error

    """
    n = beta.shape[0]
    N = beta.shape[2]
    time = np.linspace(0, 1, T)

    if n > 500:
        parallel = True
    elif T > 100:
        parallel = True
    else:
        parallel = False

    # Create B-Spline Basis if none provided
    if B is None:
        B = bs(time, df=df, degree=4, include_intercept=True)
    Nb = B.shape[1]

    q, beta = preproc_open_curve(beta, T)
    beta0 = beta.copy()
    qn = q.copy()

    gamma = np.tile(np.linspace(0, 1, T), (N, 1))
    gamma = gamma.transpose()
    O_hat = np.zeros((n, n, N))

    itr = 1
    SSE = np.zeros(max_itr)
    while itr <= max_itr:
        print("Iteration: %d" % itr)
        # align data

        # OLS using basis
        Phi = np.ones((N, n * Nb + 1))
        for ii in range(0, N):
            for jj in range(0, n):
                for kk in range(1, Nb + 1):
                    Phi[ii, jj * Nb + kk] = trapz(qn[jj, :, ii] * B[:, kk - 1], time)

        xx = dot(Phi.T, Phi)
        inv_xx = inv(xx)
        xy = dot(Phi.T, y)
        b = dot(inv_xx, xy)

        alpha = b[0]
        nu = np.zeros((n, T))
        for ii in range(0, n):
            nu[ii, :] = B.dot(b[(ii * Nb + 1):((ii + 1) * Nb + 1)])

        # compute the SSE
        int_X = np.zeros(N)
        for ii in range(0, N):
            int_X[ii] = cf.innerprod_q(qn[:, :, ii], nu)

        SSE[itr - 1] = sum((y.reshape(N) - alpha - int_X) ** 2)

        # find gamma
        gamma_new = np.zeros((T, N))
        if parallel:
            out = Parallel(n_jobs=cores)(delayed(regression_warp)(nu, beta[:, :, n], y[n], alpha) for n in range(N))
            for ii in range(0, N):
                gamma_new[:, ii] = out[ii][0]
                beta[:, :, ii] = out[ii][2]
                qn[:, :, ii] = cf.curve_to_q(beta[:, :, ii])
        else:
            for ii in range(0, N):
                beta1 = beta[:, :, ii]
                gammatmp, Otmp, beta1, tau = regression_warp(nu, beta1,
                                                             y[ii], alpha)
                gamma_new[:, ii] = gammatmp
                beta[:, :, ii] = beta1
                qn[:, :, ii] = cf.curve_to_q(beta1)


        if np.abs(SSE[itr - 1] - SSE[itr - 2]) < 1e-15:
            break
        else:
            gamma = gamma_new

        itr += 1

    tau = np.zeros(N)
    if parallel:
        out = Parallel(n_jobs=cores)(delayed(regression_warp)(nu, beta0[:, :, ii], y[ii], alpha) for ii in range(N))
        for ii in range(0, N):
            gamma[:, ii] = out[ii][0]
            O_hat[:, :, ii] = out[ii][1]
            tau[ii] = out[ii][3]
    else:
        for ii in range(0, N):
            beta1 = beta0[:, :, ii]
            gammatmp, Otmp, beta1, tautmp = regression_warp(nu, beta1,
                                                            y[ii], alpha)
            gamma_new[:, ii] = gammatmp
            O_hat[:, :, ii] = Otmp
            tau[ii] = tautmp

    model = collections.namedtuple('model', ['alpha', 'nu', 'betan' 'q', 'gamma',
                                             'O', 'tau', 'B', 'b', 'SSE', 'type'])
    out = model(alpha, nu, beta, q, gamma, O_hat, tau, B, b[1:-1], SSE[0:itr], 'oclinear')
    return out


def oc_elastic_logistic(beta, y, B=None, df=20, T=100, max_itr=20, cores=-1):
    """
    This function identifies a logistic regression model with
    phase-variablity using elastic methods for open curves

    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param y: numpy array of N responses
    :param B: optional matrix describing Basis elements
    :param df: number of degrees of freedom B-spline (default 20)
    :param T: number of desired samples along curve (default 100)
    :param max_itr: maximum number of iterations (default 20)
    :param cores: number of cores for parallel processing (default all)
    :type beta: np.ndarray

    :rtype: tuple of numpy array
    :return alpha: alpha parameter of model
    :return nu: nu(t) of model
    :return betan: aligned curves - numpy ndarray of shape (n,T,N)
    :return O: calulated rotation matrices
    :return gamma: calculated warping functions
    :return B: basis matrix
    :return b: basis coefficients
    :return Loss: logistic loss

    """
    n = beta.shape[0]
    N = beta.shape[2]
    time = np.linspace(0, 1, T)

    if n > 500:
        parallel = True
    elif T > 100:
        parallel = True
    else:
        parallel = False

    # Create B-Spline Basis if none provided
    if B is None:
        B = bs(time, df=df, degree=4, include_intercept=True)
    Nb = B.shape[1]

    q, beta = preproc_open_curve(beta, T)
    beta0 = beta.copy()
    qn = q.copy()

    gamma = np.tile(np.linspace(0, 1, T), (N, 1))
    gamma = gamma.transpose()
    O_hat = np.zeros((n, n, N))

    itr = 1
    LL = np.zeros(max_itr + 1)
    while itr <= max_itr:
        print("Iteration: %d" % itr)

        Phi = np.ones((N, n * Nb + 1))
        for ii in range(0, N):
            for jj in range(0, n):
                for kk in range(1, Nb + 1):
                    Phi[ii, jj * Nb + kk] = trapz(qn[jj, :, ii] * B[:, kk - 1], time)

        # Find alpha and beta using l_bfgs
        b0 = np.zeros(n * Nb + 1)
        out = fmin_l_bfgs_b(logit_loss, b0, fprime=logit_gradient,
                            args=(Phi, y), pgtol=1e-10, maxiter=200,
                            maxfun=250, factr=1e-30)
        b = out[0]
        alpha = b[0]
        nu = np.zeros((n, T))
        for ii in range(0, n):
            nu[ii, :] = B.dot(b[(ii * Nb + 1):((ii + 1) * Nb + 1)])

        # compute the logistic loss
        LL[itr] = logit_loss(b, Phi, y)

        # find gamma
        gamma_new = np.zeros((T, N))
        if parallel:
            out = Parallel(n_jobs=cores)(delayed(logistic_warp)(nu, beta[:, :, ii], y[ii]) for ii in range(N))
            for ii in range(0, N):
                gamma_new[:, ii] = out[ii][0]
                beta[:, :, ii] = out[ii][2]
                qn[:, :, ii] = cf.curve_to_q(beta[:, :, ii])
        else:
            for ii in range(0, N):
                beta1 = beta[:, :, ii]
                gammatmp, Otmp, beta1, tautmp = logistic_warp(nu, beta1, y[ii])
                gamma_new[:, ii] = gammatmp
                beta[:, :, ii] = beta1
                qn[:, :, ii] = cf.curve_to_q(beta1)

        if np.abs(LL[itr] - LL[itr - 1]) < 1e-18:
            break
        else:
            gamma = gamma_new

        itr += 1

    tau = np.zeros(N)
    if parallel:
        out = Parallel(n_jobs=cores)(delayed(logistic_warp)(nu,
                                                            beta0[:, :, ii], y[ii]) for ii in range(N))
        for ii in range(0, N):
            gamma_new[:, ii] = out[ii][0]
            O_hat[:, :, ii] = out[ii][1]
            tau[ii] = out[ii][3]
    else:
        for ii in range(0, N):
            beta1 = beta0[:, :, ii]
            gammatmp, Otmp, beta1, tautmp = logistic_warp(nu, beta1, y[ii])
            gamma_new[:, ii] = gammatmp
            O_hat[:, :, ii] = Otmp
            tau[ii] = tautmp

    model = collections.namedtuple('model', ['alpha', 'nu', 'betan', 'q',
                                             'gamma', 'O', 'tau', 'B', 'b', 'Loss',
                                             'type'])
    out = model(alpha, nu, beta, q, gamma_new, O_hat, tau, B, b[1:-1],
                LL[1:itr], 'oclogistic')
    return out


def oc_elastic_mlogistic(beta, y, B=None, df=20, T=100, max_itr=20, cores=-1,
                         deltaO=.003, deltag=.003):
    """
    This function identifies a multinomial logistic regression model with
    phase-variablity using elastic methods for open curves

    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param y: numpy array of labels {1,2,...,m} for m classes
    :param B: optional matrix describing Basis elements
    :param df: number of degrees of freedom B-spline (default 20)
    :param T: number of desired samples along curve (default 100)
    :param max_itr: maximum number of iterations (default 20)
    :param cores: number of cores for parallel processing (default all)
    :type beta: np.ndarray

    :rtype: tuple of numpy array
    :return alpha: alpha parameter of model
    :return nu: nu(t) of model
    :return betan: aligned curves - numpy ndarray of shape (n,T,N)
    :return O: calulated rotation matrices
    :return gamma: calculated warping functions
    :return B: basis matrix
    :return b: basis coefficients
    :return Loss: logistic loss

    """
    n = beta.shape[0]
    N = beta.shape[2]
    time = np.linspace(0, 1, T)

    if n > 500:
        parallel = True
    elif T > 100:
        parallel = True
    else:
        parallel = False

    # Code labels
    m = y.max()
    Y = np.zeros((N, m), dtype=int)
    for ii in range(0, N):
        Y[ii, y[ii] - 1] = 1

    # Create B-Spline Basis if none provided
    if B is None:
        B = bs(time, df=df, degree=4, include_intercept=True)
    Nb = B.shape[1]

    q, beta = preproc_open_curve(beta, T)
    qn = q.copy()

    gamma = np.tile(np.linspace(0, 1, T), (N, 1))
    gamma = gamma.transpose()
    O_hat = np.zeros((n, n, N))

    itr = 1
    LL = np.zeros(max_itr+1)
    while itr <= max_itr:
        print("Iteration: %d" % itr)

        Phi = np.ones((N, n * Nb + 1))
        for ii in range(0, N):
            for jj in range(0, n):
                for kk in range(1, Nb + 1):
                    Phi[ii, jj * Nb + kk] = trapz(qn[jj, :, ii] * B[:, kk - 1], time)

        # Find alpha and beta using l_bfgs
        b0 = np.zeros(m * (n * Nb + 1))
        out = fmin_l_bfgs_b(mlogit_loss, b0, fprime=mlogit_gradient,
                            args=(Phi, Y), pgtol=1e-10, maxiter=200,
                            maxfun=250, factr=1e-30)
        b = out[0]
        B0 = b.reshape(n * Nb + 1, m)
        alpha = B0[0, :]
        nu = np.zeros((n, T, m))
        for i in range(0, m):
            for j in range(0, n):
                nu[j, :, i] = B.dot(B0[(j * Nb + 1):((j + 1) * Nb + 1), i])

        # compute the logistic loss
        LL[itr] = mlogit_loss(b, Phi, Y)

        # find gamma
        gamma_new = np.zeros((T, N))
        if parallel:
            out = Parallel(n_jobs=cores)(delayed(mlogit_warp_grad)(alpha, nu, q[:, :, n], Y[n, :],
                                                                   deltaO=deltaO, deltag=deltag) for n in range(N))
            for ii in range(0, N):
                gamma_new[:, ii] = out[ii][0]
                O_hat[:, :, ii] = out[ii][1]
                beta_tmp = O_hat[:, :, ii].dot(beta[:, :, ii])
                beta[:, :, ii][:, :, ii] = cf.group_action_by_gamma_coord(beta_tmp, gamma_new[:, ii])
                qn[:, :, ii] = cf.curve_to_q(beta[:, :, ii])
        else:
            for ii in range(0, N):
                gammatmp, Otmp = mlogit_warp_grad(alpha, nu, q[:, :, ii], Y[ii, :],
                                                  deltaO=deltaO, deltag=deltag)
                gamma_new[:, ii] = gammatmp
                O_hat[:, :, ii] = Otmp
                beta_tmp = O_hat[:, :, ii].dot(beta[:, :, ii])
                beta[:, :, ii] = cf.group_action_by_gamma_coord(beta_tmp, gamma_new[:, ii])
                qn[:, :, ii] = cf.curve_to_q(beta[:, :, ii])

        if norm(gamma - gamma_new) < 1e-5:
            break
        else:
            gamma = gamma_new

        itr += 1

    if parallel:
        out = Parallel(n_jobs=cores)(delayed(mlogit_warp_grad)(alpha, nu, q[:, :, n], Y[n, :],
                                                               deltaO=deltaO, deltag=deltag) for n in range(N))
        for ii in range(0, N):
            gamma_new[:, ii] = out[ii][0]
            O_hat[:, :, ii] = out[ii][1]
    else:
        for ii in range(0, N):
            gammatmp, Otmp = mlogit_warp_grad(alpha, nu, q[:, :, ii], Y[ii, :],
                                              deltaO=deltaO, deltag=deltag)
            gamma_new[:, ii] = gammatmp
            O_hat[:, :, ii] = Otmp

    model = collections.namedtuple('model', ['alpha', 'nu', 'betan', 'q',
                                             'gamma', 'O', 'B', 'b',
                                             'Loss', 'n_classes', 'type'])
    out = model(alpha, nu, beta, q, gamma_new, O_hat, B, b[1:-1], LL[1:itr],
                m, 'ocmlogistic')
    return out


def oc_elastic_prediction(beta, model, y=None):
    """
    This function identifies a regression model with phase-variablity
    using elastic methods

    :param f: numpy ndarray of shape (M,N) of M functions with N samples
    :param model: identified model from elastic_regression
    :param y: truth, optional used to calculate SSE

    :rtype: tuple of numpy array
    :return alpha: alpha parameter of model
    :return beta: beta(t) of model
    :return fn: aligned functions - numpy ndarray of shape (M,N) of M
    functions with N samples
    :return qn: aligned srvfs - similar structure to fn
    :return gamma: calculated warping functions
    :return q: original training SRSFs
    :return B: basis matrix
    :return b: basis coefficients
    :return SSE: sum of squared error

    """
    T = model.q.shape[1]
    n = beta.shape[2]

    q, beta = preproc_open_curve(beta, T)

    if model.type == 'oclinear' or model.type == 'oclogistic':
        y_pred = np.zeros(n)
    elif model.type == 'ocmlogistic':
        m = model.n_classes
        y_pred = np.zeros((n, m))

    for ii in range(0, n):
        diff = model.q - q[:, :, ii][:, :, np.newaxis]
        dist = np.linalg.norm(np.abs(diff) ** 2, axis=(0, 1))
        if model.type == 'oclinear' or model.type == 'oclogistic':
            beta1 = cf.shift_f(beta[:, :, ii], int(model.tau[dist.argmin()]))
        else:
            beta1 = beta[:, :, ii]
        beta1 = model.O[:, :, dist.argmin()].dot(beta1)
        beta1 = cf.group_action_by_gamma_coord(beta1,
                                               model.gamma[:, dist.argmin()])
        q_tmp = cf.curve_to_q(beta1)

        if model.type == 'oclinear':
            y_pred[ii] = model.alpha + cf.innerprod_q(q_tmp, model.nu)
        elif model.type == 'oclogistic':
            y_pred[ii] = model.alpha + cf.innerprod_q(q_tmp, model.nu)
        elif model.type == 'ocmlogistic':
            for jj in range(0, m):
                y_pred[ii, jj] = model.alpha[jj] + cf.innerprod_q(q_tmp, model.nu[:, :, jj])

    if y is None:
        if model.type == 'oclinear':
            SSE = None
        elif model.type == 'oclogistic':
            y_pred = phi(y_pred)
            y_labels = np.ones(n)
            y_labels[y_pred < 0.5] = -1
            PC = None
        elif model.type == 'ocmlogistic':
            y_pred = phi(y_pred.ravel())
            y_pred = y_pred.reshape(n, m)
            y_labels = y_pred.argmax(axis=1) + 1
            PC = None
    else:
        if model.type == 'oclinear':
            SSE = sum((y - y_pred) ** 2)
        elif model.type == 'oclogistic':
            y_pred = phi(y_pred)
            y_labels = np.ones(n)
            y_labels[y_pred < 0.5] = -1
            TP = sum(y[y_labels == 1] == 1)
            FP = sum(y[y_labels == -1] == 1)
            TN = sum(y[y_labels == -1] == -1)
            FN = sum(y[y_labels == 1] == -1)
            PC = (TP + TN) / float(TP + FP + FN + TN)
        elif model.type == 'ocmlogistic':
            y_pred = phi(y_pred.ravel())
            y_pred = y_pred.reshape(n, m)
            y_labels = y_pred.argmax(axis=1) + 1
            PC = np.zeros(m)
            cls_set = np.arange(1, m + 1)
            for ii in range(0, m):
                cls_sub = np.delete(cls_set, ii)
                TP = sum(y[y_labels == (ii + 1)] == (ii + 1))
                FP = sum(y[np.in1d(y_labels, cls_sub)] == (ii + 1))
                TN = sum(y[np.in1d(y_labels, cls_sub)] ==
                         y_labels[np.in1d(y_labels, cls_sub)])
                FN = sum(np.in1d(y[y_labels == (ii + 1)], cls_sub))
                PC[ii] = (TP + TN) / float(TP + FP + FN + TN)

            PC = sum(y == y_labels) / float(y_labels.size)

    if model.type == 'oclinear':
        prediction = collections.namedtuple('prediction', ['y_pred', 'SSE'])
        out = prediction(y_pred, SSE)
    elif model.type == 'oclogistic':
        prediction = collections.namedtuple('prediction', ['y_prob',
                                                           'y_labels', 'PC'])
        out = prediction(y_pred, y_labels, PC)
    elif model.type == 'ocmlogistic':
        prediction = collections.namedtuple('prediction', ['y_prob',
                                                           'y_labels', 'PC'])
        out = prediction(y_pred, y_labels, PC)

    return out


# helper function for curve manipulation
def preproc_open_curve(beta, T=100):
    n, M, k = beta.shape

    q = np.zeros((n, T, k))
    beta2 = np.zeros((n, T, k))
    for i in range(0, k):
        beta1 = cf.resamplecurve(beta[:, :, i], T)
        centroid1 = cf.calculatecentroid(beta1)
        beta1 = beta1 - np.tile(centroid1, [T, 1]).T
        beta2[:, :, i] = beta1
        q[:, :, i] = cf.curve_to_q(beta1)

    return (q, beta2)


# helper functions for linear regression
def regression_warp(nu, beta, y, alpha):
    """
    calculates optimal warping for function linear regression

    :param nu: numpy ndarray of shape (M,N) of M functions with N samples
    :param beta: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses
    :param alpha: numpy scalar

    :rtype: numpy array
    :return gamma_new: warping function

    """
    T = beta.shape[1]
    betanu = cf.q_to_curve(nu)

    betaM, O_M, tauM = cf.find_rotation_and_seed_coord(betanu, beta)
    q = cf.curve_to_q(betaM)
    gamma = cf.optimum_reparam_curve(q, nu)
    gam_M = uf.invertGamma(gamma)
    betaM = cf.group_action_by_gamma_coord(betaM, gam_M)
    betaM, O_hat1, tauM = cf.find_rotation_and_seed_coord(betanu, betaM)
    centroid2 = cf.calculatecentroid(betaM)
    betaM = betaM - np.tile(centroid2, [T, 1]).T
    O_M = O_M.dot(O_hat1)
    qM = cf.curve_to_q(betaM)
    y_M = cf.innerprod_q(qM, nu)

    betam, O_m, taum = cf.find_rotation_and_seed_coord(-1 * betanu, beta)
    q = cf.curve_to_q(betam)
    gamma = cf.optimum_reparam_curve(q, -1 * nu)
    gam_m = uf.invertGamma(gamma)
    betam = cf.group_action_by_gamma_coord(betam, gam_m)
    betam, O_hat1, taum = cf.find_rotation_and_seed_coord(betanu, betaM)
    centroid2 = cf.calculatecentroid(betam)
    betam = betam - np.tile(centroid2, [T, 1]).T
    O_m = O_m.dot(O_hat1)
    qm = cf.curve_to_q(betam)
    y_m = cf.innerprod_q(qm, nu)

    if y > alpha + y_M:
        O_hat = O_M
        gamma_new = gam_M
        tau = tauM
        beta1n = betaM
    elif y < alpha + y_m:
        O_hat = O_m
        gamma_new = gam_m
        tau = taum
        beta1n = betam
    else:
        gamma_new, O_hat, beta1n, tau = cf.curve_zero_crossing(y - alpha, beta, nu, y_M, y_m, gam_M,
                                                               gam_m)

    return(gamma_new, O_hat, beta1n, tau)


# helper functions for logistic regression
def logistic_warp(nu, beta, y):
    """
    calculates optimal warping for function logistic regression

    :param nu: numpy ndarray of shape (M,N) of M functions with N samples
    :param beta: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses

    :rtype: numpy array
    :return gamma: warping function

    """
    betanu = cf.q_to_curve(nu)
    T = beta.shape[1]
    if y == 1:
        beta1, O_hat, tau = cf.find_rotation_and_seed_coord(betanu, beta)
        q = cf.curve_to_q(beta1)
        gamma = cf.optimum_reparam_curve(q, nu)
        gamI = uf.invertGamma(gamma)
        beta1n = cf.group_action_by_gamma_coord(beta1, gamI)
        beta1n, O_hat1, tau = cf.find_rotation_and_seed_coord(betanu, beta1n)
        centroid2 = cf.calculatecentroid(beta1n)
        beta1n = beta1n - np.tile(centroid2, [T, 1]).T
        O = O_hat.dot(O_hat1)
    elif y == -1:
        beta1, O_hat, tau = cf.find_rotation_and_seed_coord(-1 * betanu, beta)
        q = cf.curve_to_q(beta1)
        gamma = cf.optimum_reparam_curve(q, -1 * nu)
        gamI = uf.invertGamma(gamma)
        beta1n = cf.group_action_by_gamma_coord(beta1, gamI)
        beta1n, O_hat1, tau = cf.find_rotation_and_seed_coord(-1 * betanu, beta1n)
        centroid2 = cf.calculatecentroid(beta1n)
        beta1n = beta1n - np.tile(centroid2, [T, 1]).T
        O = O_hat.dot(O_hat1)
    return (gamI, O, beta1n, tau)


def phi(t):
    """
    calculates logistic function, returns 1 / (1 + exp(-t))

    :param t: scalar

    :rtype: numpy array
    :return out: return value

    """
    # logistic function, returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=np.float)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def logit_loss(b, X, y):
    """
    logistic loss function, returns Sum{-log(phi(t))}

    :param b: numpy ndarray of shape (M,N) of M functions with N samples
    :param X: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses

    :rtype: numpy array
    :return out: loss value

    """
    z = X.dot(b)
    yz = y * z
    idx = yz > 0
    out = np.zeros_like(yz)
    out[idx] = np.log(1 + np.exp(-yz[idx]))
    out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
    out = out.sum()
    return out


def logit_gradient(b, X, y):
    """
    calculates gradient of the logistic loss

    :param b: numpy ndarray of shape (M,N) of M functions with N samples
    :param X: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses

    :rtype: numpy array
    :return grad: gradient of logisitc loss

    """
    z = X.dot(b)
    z = phi(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0)
    return grad


def logit_hessian(s, b, X, y):
    """
    calculates hessian of the logistic loss

    :param s: numpy ndarray of shape (M,N) of M functions with N samples
    :param b: numpy ndarray of shape (M,N) of M functions with N samples
    :param X: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses

    :rtype: numpy array
    :return out: hessian of logistic loss

    """
    z = X.dot(b)
    z = phi(y * z)
    d = z * (1 - z)
    wa = d * X.dot(s)
    Hs = X.T.dot(wa)
    out = Hs
    return out


# helper functions for multinomial logistic regression
def mlogit_warp_grad(alpha, nu, q, y, max_itr=8000, tol=1e-6,
                     deltaO=0.008, deltag=0.008, display=0):
    """
    calculates optimal warping for functional multinomial logistic regression

    :param alpha: scalar
    :param nu: numpy ndarray of shape (M,N) of M functions with N samples
    :param q: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses
    :param max_itr: maximum number of iterations (Default=8000)
    :param tol: stopping tolerance (Default=1e-10)
    :param deltaO: gradient step size for rotation (Default=0.008)
    :param deltag: gradient step size for warping (Default=0.008
    :param display: display iterations (Default=0)

    :rtype: tuple of numpy array
    :return gam_old: warping function

    """
    n = q.shape[0]
    TT = q.shape[1]
    m = nu.shape[2]
    time = np.linspace(0, 1, TT)
    binsize = 1. / (TT - 1)

    alpha = alpha/norm(alpha)
    for i in range(0, m):
        nu[:, :, i] = nu[:, :, i]/np.sqrt(cf.innerprod_q(nu[:, :, i],
                                          nu[:, :, i]))

    gam = np.linspace(0, 1, TT)
    O = np.eye(n)
    O_old = O.copy()
    gam_old = gam.copy()
    qtilde = q.copy()
    # rotation basis (Skew Symmetric)
    E = np.array([[0, -1.], [1., 0]])
    # warping basis (Fourier)
    p = 20
    f_basis = np.zeros((TT, p))
    for i in range(0, int(p/2)):
        f_basis[:, 2*i] = 1/np.sqrt(np.pi) * np.sin(2*np.pi*(i+1)*time)
        f_basis[:, 2*i + 1] = 1/np.sqrt(np.pi) * np.cos(2*np.pi*(i+1)*time)

    itr = 0
    max_val = np.zeros(max_itr+1)
    while itr <= max_itr:
        # inner product value
        A = np.zeros(m)
        for i in range(0, m):
            A[i] = cf.innerprod_q(qtilde, nu[:, :, i])

        # form gradient for rotation
        B = np.zeros((n, n, m))
        for i in range(0, m):
            B[:, :, i] = cf.innerprod_q(E.dot(qtilde), nu[:, :, i]) * E

        tmp1 = np.sum(np.exp(alpha + A))
        tmp2 = np.sum(np.exp(alpha + A) * B, axis=2)
        hO = np.sum(y * B, axis=2) - (tmp2 / tmp1)
        O_new = O_old.dot(expm(deltaO * hO))

        # form gradient for warping
        qtilde_diff = np.gradient(qtilde, binsize)
        qtilde_diff = qtilde_diff[1]
        c = np.zeros((TT, m))
        for i in range(0, m):
            tmp3 = np.zeros((TT, p))
            for j in range(0, p):
                cbar = cumtrapz(f_basis[:, j] * f_basis[:, j], time, initial=0)
                ctmp = 2*qtilde_diff*cbar + qtilde*f_basis[:, j]
                tmp3[:, j] = cf.innerprod_q(ctmp, nu[:, :, i]) * f_basis[:, j]

            c[:, i] = np.sum(tmp3, axis=1)

        tmp2 = np.sum(np.exp(alpha + A) * c, axis=1)
        hpsi = np.sum(y * c, axis=1) - (tmp2 / tmp1)

        vecnorm = norm(hpsi)
        costmp = np.cos(deltag * vecnorm) * np.ones(TT)
        sintmp = np.sin(deltag * vecnorm) * (hpsi / vecnorm)
        psi_new = costmp + sintmp
        gam_tmp = cumtrapz(psi_new * psi_new, time, initial=0)
        gam_tmp = (gam_tmp - gam_tmp[0]) / (gam_tmp[-1] - gam_tmp[0])
        gam_new = np.interp(gam_tmp, time, gam_old)

        max_val[itr] = np.sum(y * (alpha + A)) - np.log(tmp1)

        if display == 1:
            print("Iteration %d : Cost %f" % (itr+1, max_val[itr]))

        gam_old = gam_new.copy()
        O_old = O_new.copy()
        qtilde = cf.group_action_by_gamma(O_old.dot(q), gam_old)

        if itr >= 2:
            max_val_change = max_val[itr] - max_val[itr-1]
            if np.abs(max_val_change) < tol:
                break

        itr += 1

    # gam_old = mw.mlogit_warp(np.ascontiguousarray(alpha),
    #                          np.ascontiguousarray(beta),
    #                          time, np.ascontiguousarray(q),
    #                          np.ascontiguousarray(y, dtype=np.int32), max_itr,
    #                          tol, delta, display)

    return (gam_old, O_old)


def mlogit_loss(b, X, Y):
    """
    calculates multinomial logistic loss (negative log-likelihood)

    :param b: numpy ndarray of shape (M,N) of M functions with N samples
    :param X: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses

    :rtype: numpy array
    :return nll: negative log-likelihood

    """
    N, m = Y.shape  # n_samples, n_classes
    M = X.shape[1]  # n_features
    B = b.reshape(M, m)
    Yhat = np.dot(X, B)
    Yhat -= Yhat.min(axis=1)[:, np.newaxis]
    Yhat = np.exp(-Yhat)
    # l1-normalize
    Yhat /= Yhat.sum(axis=1)[:, np.newaxis]

    Yhat = Yhat * Y
    nll = np.sum(np.log(Yhat.sum(axis=1)))
    nll /= -float(N)

    return nll


def mlogit_gradient(b, X, Y):
    """
    calculates gradient of the multinomial logistic loss

    :param b: numpy ndarray of shape (M,N) of M functions with N samples
    :param X: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses

    :rtype: numpy array
    :return grad: gradient

    """
    N, m = Y.shape  # n_samples, n_classes
    M = X.shape[1]  # n_features
    B = b.reshape(M, m)
    Yhat = np.dot(X, B)
    Yhat -= Yhat.min(axis=1)[:, np.newaxis]
    Yhat = np.exp(-Yhat)
    # l1-normalize
    Yhat /= Yhat.sum(axis=1)[:, np.newaxis]

    _Yhat = Yhat * Y
    _Yhat /= _Yhat.sum(axis=1)[:, np.newaxis]
    Yhat -= _Yhat
    grad = np.dot(X.T, Yhat)
    grad /= -float(N)
    grad = grad.ravel()

    return grad
