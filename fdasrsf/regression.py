"""
Warping Invariant Regression using SRSF

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf.utility_functions as uf
from scipy import dot
from scipy.optimize import fmin_l_bfgs_b
from scipy.integrate import trapz
from scipy.linalg import inv, norm
from patsy import bs
from joblib import Parallel, delayed
import mlogit_warp as mw


class elastic_regression:
    """
    This class provides elastic regression for functional data using the
    SRVF framework accounting for warping
    
    Usage:  obj = elastic_regression(f,y,time)

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy array of N responses
    :param time: vector of size M describing the sample points
    :param B: optional matrix describing Basis elements
    :param lam: regularization parameter (default 0)
    :param df: number of degrees of freedom B-spline (default 20)
    :param max_itr: maximum number of iterations (default 20)
    :param cores: number of cores for parallel processing (default all)
    :type f: np.ndarray
    :type time: np.ndarray

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  29-Oct-2021
    """

    def __init__(self, f, y, time):
        """
        Construct an instance of the elastic_regression class
        :param f: numpy ndarray of shape (M,N) of N functions with M samples
        :param y: response vector
        :param time: vector of size M describing the sample points
        """
        a = time.shape[0]

        if f.shape[0] != a:
            raise Exception('Columns of f and time must be equal')

        self.f = f
        self.y = y
        self.time = time

    def calc_model(self, B=None, lam=0, df=20, max_itr=20, cores=-1, smooth=False):
        """
        This function identifies a regression model with phase-variability
        using elastic pca

        :param B: optional matrix describing Basis elements
        :param lam: regularization parameter (default 0)
        :param df: number of degrees of freedom B-spline (default 20)
        :param max_itr: maximum number of iterations (default 20)
        :param cores: number of cores for parallel processing (default all)
        """
    
        M = self.f.shape[0]
        N = self.f.shape[1]

        if M > 500:
            parallel = True
        elif N > 100:
            parallel = True
        else:
            parallel = False

        binsize = np.diff(self.time)
        binsize = binsize.mean()

        # Create B-Spline Basis if none provided
        if B is None:
            B = bs(self.time, df=df, degree=4, include_intercept=True)
        Nb = B.shape[1]

        self.B = B

        # second derivative for regularization
        Bdiff = np.zeros((M, Nb))
        for ii in range(0, Nb):
            Bdiff[:, ii] = np.gradient(np.gradient(B[:, ii], binsize), binsize)
        
        self.Bdiff = Bdiff

        self.q = uf.f_to_srsf(self.f, self.time, smooth)
        
        gamma = np.tile(np.linspace(0, 1, M), (N, 1))
        gamma = gamma.transpose()

        itr = 1
        self.SSE = np.zeros(max_itr)
        while itr <= max_itr:
            print("Iteration: %d" % itr)
            # align data
            fn = np.zeros((M, N))
            qn = np.zeros((M, N))
            for ii in range(0, N):
                fn[:, ii] = np.interp((self.time[-1] - self.time[0]) * gamma[:, ii] +
                                    self.time[0], self.time, self.f[:, ii])
                qn[:, ii] = uf.warp_q_gamma(self.time, self.q[:, ii], gamma[:, ii])

            # OLS using basis
            Phi = np.ones((N, Nb+1))
            for ii in range(0, N):
                for jj in range(1, Nb+1):
                    Phi[ii, jj] = trapz(qn[:, ii] * B[:, jj-1], self.time)

            R = np.zeros((Nb+1, Nb+1))
            for ii in range(1, Nb+1):
                for jj in range(1, Nb+1):
                    R[ii, jj] = trapz(Bdiff[:, ii-1] * Bdiff[:, jj-1], self.time)

            xx = dot(Phi.T, Phi)
            inv_xx = inv(xx + lam * R)
            xy = dot(Phi.T, self.y)
            b = dot(inv_xx, xy)

            alpha = b[0]
            beta = B.dot(b[1:Nb+1])
            beta = beta.reshape(M)

            # compute the SSE
            int_X = np.zeros(N)
            for ii in range(0, N):
                int_X[ii] = trapz(qn[:, ii] * beta, self.time)

            SSE[itr - 1] = sum((self.y.reshape(N) - alpha - int_X) ** 2)

            # find gamma
            gamma_new = np.zeros((M, N))
            if parallel:
                out = Parallel(n_jobs=cores)(delayed(regression_warp)(beta,
                                            self.time, self.q[:, n], self.y[n], alpha) for n in range(N))
                gamma_new = np.array(out)
                gamma_new = gamma_new.transpose()
            else:
                for ii in range(0, N):
                    gamma_new[:, ii] = regression_warp(beta, self.time, self.q[:, ii],
                                                       self.y[ii], alpha)

            if norm(gamma - gamma_new) < 1e-5:
                break
            else:
                gamma = gamma_new

            itr += 1

        # Last Step with centering of gam
        gamI = uf.SqrtMeanInverse(gamma_new)
        gamI_dev = np.gradient(gamI, 1 / float(M - 1))
        beta = np.interp((self.time[-1] - self.time[0]) * gamI + time[0], self.time,
                        beta) * np.sqrt(gamI_dev)

        for ii in range(0, N):
            qn[:, ii] = np.interp((self.time[-1] - self.time[0]) * gamI + self.time[0],
                                self.time, qn[:, ii]) * np.sqrt(gamI_dev)
            fn[:, ii] = np.interp((self.time[-1] - self.time[0]) * gamI + self.time[0],
                                self.time, fn[:, ii])
            gamma[:, ii] = np.interp((self.time[-1] - self.time[0]) * gamI + self.time[0],
                                    self.time, gamma_new[:, ii])

        self.qn = qn
        self.fn = fn
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.b = b[1:-1]
        self.SSE = self.SSE[0:itr]

        return

        def predict(self, newdata=None):
        """
        This function performs prediction on regression model on new data if available or current stored data in object
        Usage:  obj.predict()
                obj.predict(newdata)

        :param newdata: dict containing new data for prediction (needs the keys below, if None predicts on training data)
        :type newdata: dict
        :param f: (M,N) matrix of functions
        :param time: vector of time points
        :param y: truth if available
        :param smooth: smooth data if needed
        :param sparam: number of times to run filter
        """

        if newdata != None:
            f = newdata['f']
            time = newdata['time']
            y = newdata['y']
            
            q = uf.f_to_srsf(f, time, newdata['smooth'])

            n = f.shape[1]
            yhat = np.zeros(n)
            for ii in range(0, n):
                diff = self.q - q[:, ii][:, np.newaxis]
                dist = np.sum(np.abs(diff) ** 2, axis=0) ** (1. / 2)
                q_tmp = uf.warp_q_gamma(time, q[:, ii],
                                        self.gamma[:, dist.argmin()])
                yhat[ii] = self.alpha + trapz(q_tmp * self.beta, time)

            if y == None:
                self.SSE = np.nan
            else:
                self.SSE = np.sum((y-yhat)**2)
            
            self.y_pred = yhat

        else:
            n = self.f.shape[1]
            yhat = np.zeros(n)
            for ii in range(0, n):
                diff = self.q - self.q[:, ii][:, np.newaxis]
                dist = np.sum(np.abs(diff) ** 2, axis=0) ** (1. / 2)
                q_tmp = uf.warp_q_gamma(self.time, self.q[:, ii],
                                        self.gamma[:, dist.argmin()])
                yhat[ii] = self.alpha + trapz(q_tmp * self.beta, self.time)

            self.SSE = np.sum((self.y-yhat)**2)
            self.y_pred = yhat

        return


def elastic_logistic(f, y, time, B=None, df=20, max_itr=20, cores=-1,
                     smooth=False):
    """
    This function identifies a logistic regression model with
    phase-variability using elastic methods

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy array of labels (1/-1)
    :param time: vector of size M describing the sample points
    :param B: optional matrix describing Basis elements
    :param df: number of degrees of freedom B-spline (default 20)
    :param max_itr: maximum number of iterations (default 20)
    :param cores: number of cores for parallel processing (default all)
    :type f: np.ndarray
    :type time: np.ndarray

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
    :return Loss: logistic loss

    """
    M = f.shape[0]
    N = f.shape[1]

    if M > 500:
        parallel = True
    elif N > 100:
        parallel = True
    else:
        parallel = False

    binsize = np.diff(time)
    binsize = binsize.mean()

    # Create B-Spline Basis if none provided
    if B is None:
        B = bs(time, df=df, degree=4, include_intercept=True)
    Nb = B.shape[1]

    q = uf.f_to_srsf(f, time, smooth)

    gamma = np.tile(np.linspace(0, 1, M), (N, 1))
    gamma = gamma.transpose()

    itr = 1
    LL = np.zeros(max_itr)
    while itr <= max_itr:
        print("Iteration: %d" % itr)
        # align data
        fn = np.zeros((M, N))
        qn = np.zeros((M, N))
        for ii in range(0, N):
            fn[:, ii] = np.interp((time[-1] - time[0]) * gamma[:, ii] +
                                  time[0], time, f[:, ii])
            qn[:, ii] = uf.warp_q_gamma(time, q[:, ii], gamma[:, ii])

        Phi = np.ones((N, Nb+1))
        for ii in range(0, N):
            for jj in range(1, Nb+1):
                Phi[ii, jj] = trapz(qn[:, ii] * B[:, jj-1], time)

        # Find alpha and beta using l_bfgs
        b0 = np.zeros(Nb+1)
        out = fmin_l_bfgs_b(logit_loss, b0, fprime=logit_gradient,
                            args=(Phi, y), pgtol=1e-10, maxiter=200,
                            maxfun=250, factr=1e-30)
        b = out[0]
        alpha = b[0]
        beta = B.dot(b[1:Nb+1])
        beta = beta.reshape(M)

        # compute the logistic loss
        LL[itr - 1] = logit_loss(b, Phi, y)

        # find gamma
        gamma_new = np.zeros((M, N))
        if parallel:
            out = Parallel(n_jobs=cores)(delayed(logistic_warp)(beta, time,
                                         q[:, n], y[n]) for n in range(N))
            gamma_new = np.array(out)
            gamma_new = gamma_new.transpose()
        else:
            for ii in range(0, N):
                gamma_new[:, ii] = logistic_warp(beta, time, q[:, ii], y[ii])

        if norm(gamma - gamma_new) < 1e-5:
            break
        else:
            gamma = gamma_new

        itr += 1

    gamma = gamma_new

    model = collections.namedtuple('model', ['alpha', 'beta', 'fn',
                                   'qn', 'gamma', 'q', 'B', 'b',
                                   'Loss', 'type'])
    out = model(alpha, beta, fn, qn, gamma, q, B, b[1:-1], LL[0:itr],
                'logistic')
    return out


def elastic_mlogistic(f, y, time, B=None, df=20, max_itr=20, cores=-1,
                      delta=.01, parallel=True, smooth=False):
    """
    This function identifies a multinomial logistic regression model with
    phase-variability using elastic methods

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy array of labels {1,2,...,m} for m classes
    :param time: vector of size M describing the sample points
    :param B: optional matrix describing Basis elements
    :param df: number of degrees of freedom B-spline (default 20)
    :param max_itr: maximum number of iterations (default 20)
    :param cores: number of cores for parallel processing (default all)
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of numpy array
    :return alpha: alpha parameter of model
    :return beta: beta(t) of model
    :return fn: aligned functions - numpy ndarray of shape (M,N) of N
    functions with M samples
    :return qn: aligned srvfs - similar structure to fn
    :return gamma: calculated warping functions
    :return q: original training SRSFs
    :return B: basis matrix
    :return b: basis coefficients
    :return Loss: logistic loss

    """
    M = f.shape[0]
    N = f.shape[1]
    # Code labels
    m = y.max()
    Y = np.zeros((N, m), dtype=int)
    for ii in range(0, N):
        Y[ii, y[ii]-1] = 1

    binsize = np.diff(time)
    binsize = binsize.mean()

    # Create B-Spline Basis if none provided
    if B is None:
        B = bs(time, df=df, degree=4, include_intercept=True)
    Nb = B.shape[1]

    q = uf.f_to_srsf(f, time, smooth)

    gamma = np.tile(np.linspace(0, 1, M), (N, 1))
    gamma = gamma.transpose()

    itr = 1
    LL = np.zeros(max_itr)
    while itr <= max_itr:
        print("Iteration: %d" % itr)
        # align data
        fn = np.zeros((M, N))
        qn = np.zeros((M, N))
        for ii in range(0, N):
            fn[:, ii] = np.interp((time[-1] - time[0]) * gamma[:, ii] +
                                  time[0], time, f[:, ii])
            qn[:, ii] = uf.warp_q_gamma(time, q[:, ii], gamma[:, ii])

        Phi = np.ones((N, Nb+1))
        for ii in range(0, N):
            for jj in range(1, Nb+1):
                Phi[ii, jj] = trapz(qn[:, ii] * B[:, jj-1], time)

        # Find alpha and beta using l_bfgs
        b0 = np.zeros(m * (Nb+1))
        out = fmin_l_bfgs_b(mlogit_loss, b0, fprime=mlogit_gradient,
                            args=(Phi, Y), pgtol=1e-10, maxiter=200,
                            maxfun=250, factr=1e-30)
        b = out[0]
        B0 = b.reshape(Nb+1, m)
        alpha = B0[0, :]
        beta = np.zeros((M, m))
        for i in range(0, m):
            beta[:, i] = B.dot(B0[1:Nb+1, i])

        # compute the logistic loss
        LL[itr - 1] = mlogit_loss(b, Phi, Y)

        # find gamma
        gamma_new = np.zeros((M, N))
        if parallel:
            out = Parallel(n_jobs=cores)(delayed(mlogit_warp_grad)(alpha, beta,
                                         time, q[:, n], Y[n, :], delta=delta) for n in range(N))
            gamma_new = np.array(out)
            gamma_new = gamma_new.transpose()
        else:
            for ii in range(0, N):
                gamma_new[:, ii] = mlogit_warp_grad(alpha, beta, time,
                                                    q[:, ii], Y[ii, :], delta=delta)

        if norm(gamma - gamma_new) < 1e-5:
            break
        else:
            gamma = gamma_new

        itr += 1

    gamma = gamma_new

    model = collections.namedtuple('model', ['alpha', 'beta', 'fn',
                                   'qn', 'gamma', 'q', 'B', 'b',
                                   'Loss', 'n_classes', 'type'])
    out = model(alpha, beta, fn, qn, gamma, q, B, b[1:-1], LL[0:itr],
                m, 'mlogistic')
    return out


def elastic_prediction(f, time, model, y=None, smooth=False):
    """
    This function performs prediction from an elastic regression model
    with phase-variability

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param model: identified model from elastic_regression
    :param y: truth, optional used to calculate SSE

    :rtype: tuple of numpy array
    :return alpha: alpha parameter of model
    :return beta: beta(t) of model
    :return fn: aligned functions - numpy ndarray of shape (M,N) of N
    functions with M samples
    :return qn: aligned srvfs - similar structure to fn
    :return gamma: calculated warping functions
    :return q: original training SRSFs
    :return B: basis matrix
    :return b: basis coefficients
    :return SSE: sum of squared error

    """
    q = uf.f_to_srsf(f, time, smooth)
    n = q.shape[1]

    if model.type == 'linear' or model.type == 'logistic':
        y_pred = np.zeros(n)
    elif model.type == 'mlogistic':
        m = model.n_classes
        y_pred = np.zeros((n, m))

    for ii in range(0, n):
        diff = model.q - q[:, ii][:, np.newaxis]
        dist = np.sum(np.abs(diff) ** 2, axis=0) ** (1. / 2)
        q_tmp = uf.warp_q_gamma(time, q[:, ii],
                                model.gamma[:, dist.argmin()])
        if model.type == 'linear':
            y_pred[ii] = model.alpha + trapz(q_tmp * model.beta, time)
        elif model.type == 'logistic':
            y_pred[ii] = model.alpha + trapz(q_tmp * model.beta, time)
        elif model.type == 'mlogistic':
            for jj in range(0, m):
                y_pred[ii, jj] = model.alpha[jj] + trapz(q_tmp * model.beta[:, jj], time)

    if y is None:
        if model.type == 'linear':
            SSE = None
        elif model.type == 'logistic':
            y_pred = phi(y_pred)
            y_labels = np.ones(n)
            y_labels[y_pred < 0.5] = -1
            PC = None
        elif model.type == 'mlogistic':
            y_pred = phi(y_pred.ravel())
            y_pred = y_pred.reshape(n, m)
            y_labels = y_pred.argmax(axis=1)+1
            PC = None
    else:
        if model.type == 'linear':
            SSE = sum((y - y_pred) ** 2)
        elif model.type == 'logistic':
            y_pred = phi(y_pred)
            y_labels = np.ones(n)
            y_labels[y_pred < 0.5] = -1
            TP = sum(y[y_labels == 1] == 1)
            FP = sum(y[y_labels == -1] == 1)
            TN = sum(y[y_labels == -1] == -1)
            FN = sum(y[y_labels == 1] == -1)
            PC = (TP+TN)/float(TP+FP+FN+TN)
        elif model.type == 'mlogistic':
            y_pred = phi(y_pred.ravel())
            y_pred = y_pred.reshape(n, m)
            y_labels = y_pred.argmax(axis=1)+1
            PC = np.zeros(m)
            cls_set = np.arange(1, m+1)
            for ii in range(0, m):
                cls_sub = np.delete(cls_set, ii)
                TP = sum(y[y_labels == (ii+1)] == (ii+1))
                FP = sum(y[np.in1d(y_labels, cls_sub)] == (ii+1))
                TN = sum(y[np.in1d(y_labels, cls_sub)] ==
                         y_labels[np.in1d(y_labels, cls_sub)])
                FN = sum(np.in1d(y[y_labels == (ii+1)], cls_sub))
                PC[ii] = (TP+TN)/float(TP+FP+FN+TN)

            PC = sum(y == y_labels) / float(y_labels.size)

    if model.type == 'linear':
        prediction = collections.namedtuple('prediction', ['y_pred', 'SSE'])
        out = prediction(y_pred, SSE)
    elif model.type == 'logistic':
        prediction = collections.namedtuple('prediction', ['y_prob',
                                            'y_labels', 'PC'])
        out = prediction(y_pred, y_labels, PC)
    elif model.type == 'mlogistic':
        prediction = collections.namedtuple('prediction', ['y_prob',
                                            'y_labels', 'PC'])
        out = prediction(y_pred, y_labels, PC)

    return out


# helper functions for linear regression
def regression_warp(beta, time, q, y, alpha):
    """
    calculates optimal warping for function linear regression

    :param beta: numpy ndarray of shape (M,N) of M functions with N samples
    :param time: vector of size N describing the sample points
    :param q: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses
    :param alpha: numpy scalar

    :rtype: numpy array
    :return gamma_new: warping function

    """
    gam_M = uf.optimum_reparam(beta, time, q)
    qM = uf.warp_q_gamma(time, q, gam_M)
    y_M = trapz(qM * beta, time)

    gam_m = uf.optimum_reparam(-1 * beta, time, q)
    qm = uf.warp_q_gamma(time, q, gam_m)
    y_m = trapz(qm * beta, time)

    if y > alpha + y_M:
        gamma_new = gam_M
    elif y < alpha + y_m:
        gamma_new = gam_m
    else:
        gamma_new = uf.zero_crossing(y - alpha, q, beta, time, y_M, y_m,
                                     gam_M, gam_m)

    return gamma_new


# helper functions for logistic regression
def logistic_warp(beta, time, q, y):
    """
    calculates optimal warping for function logistic regression

    :param beta: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size N describing the sample points
    :param q: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy ndarray of shape (1,N) responses

    :rtype: numpy array
    :return gamma: warping function

    """
    if y == 1:
        gamma = uf.optimum_reparam(beta, time, q)
    elif y == -1:
        gamma = uf.optimum_reparam(-1*beta, time, q)
    return gamma


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

    :param b: numpy ndarray of shape (M,N) of N functions with M samples
    :param X: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy ndarray of shape (1,N) of N responses

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

    :param b: numpy ndarray of shape (M,N) of N functions with M samples
    :param X: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy ndarray of shape (1,N) responses

    :rtype: numpy array
    :return grad: gradient of logistic loss

    """
    z = X.dot(b)
    z = phi(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0)
    return grad


def logit_hessian(s, b, X, y):
    """
    calculates hessian of the logistic loss

    :param s: numpy ndarray of shape (M,N) of N functions with M samples
    :param b: numpy ndarray of shape (M,N) of N functions with M samples
    :param X: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy ndarray of shape (1,N) responses

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
def mlogit_warp_grad(alpha, beta, time, q, y, max_itr=8000, tol=1e-10,
                     delta=0.008, display=0):
    """
    calculates optimal warping for functional multinomial logistic regression

    :param alpha: scalar
    :param beta: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param q: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy ndarray of shape (1,N) responses
    :param max_itr: maximum number of iterations (Default=8000)
    :param tol: stopping tolerance (Default=1e-10)
    :param delta: gradient step size (Default=0.008)
    :param display: display iterations (Default=0)

    :rtype: tuple of numpy array
    :return gam_old: warping function

    """

    gam_old = mw.mlogit_warp(np.ascontiguousarray(alpha),
                             np.ascontiguousarray(beta),
                             time, np.ascontiguousarray(q),
                             np.ascontiguousarray(y, dtype=np.int32), max_itr,
                             tol, delta, display)

    return gam_old


def mlogit_loss(b, X, Y):
    """
    calculates multinomial logistic loss (negative log-likelihood)

    :param b: numpy ndarray of shape (M,N) of N functions with M samples
    :param X: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy ndarray of shape (1,N) responses

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

    :param b: numpy ndarray of shape (M,N) of N functions with M samples
    :param X: numpy ndarray of shape (M,N) of N functions with M samples
    :param y: numpy ndarray of shape (1,N) responses

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
