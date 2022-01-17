"""
Warping Invariant Regression using SRVF

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf.utility_functions as uf
import fdasrsf.curve_functions as cf
from scipy.interpolate import interp1d
from scipy.optimize import fmin_l_bfgs_b
from scipy.integrate import trapz, cumtrapz
from scipy.linalg import inv, norm, expm
from patsy import bs
from joblib import Parallel, delayed
import ocmlogit_warp as mw
import oclogit_warp as lw

class oc_elastic_regression:
    """
    This class provides elastic regression for for open curves
    using elastic methods
    
    Usage:  obj = oc_elastic_regression(beta,y)

    :param beta: numpy ndarray of shape (n, M, N) describing N curves in R^M
    :param B: optional matrix describing Basis elements
    :param y: numpy array of N responses
    :param alpha: alpha parameter of model
    :param beta: beta(t) of model
    :param betan: aligned curves - numpy ndarray of shape (n,M,N) describing N curves
    in R^M
    :param qn: aligned srvfs - similar structure to betan
    :param gamma: calculated warping functions
    :param q: original training SRSFs
    :param b: basis coefficients
    :param SSE: sum of squared error
    :param nu: regressor curve

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  29-Oct-2021
    """

    def __init__(self, beta, y, ):
        """
        Construct an instance of the oc_elastic_regression class
        :param beta: numpy ndarray of shape (n, M, N) describing N curves
        :param y: response vector
        :param time: vector of size M describing the sample points
        """

        self.beta = beta
        self.y = y
    
    def calc_model(self, B=None, lam=0, df=40, T=200, max_itr=20, cores=-1):
        """
        This function identifies a regression model for open curves
        using elastic methods

        :param B: optional matrix describing Basis elements
        :param lam: regularization parameter (default 0)
        :param df: number of degrees of freedom B-spline (default 20)
        :param T: number of desired samples along curve (default 100)
        :param max_itr: maximum number of iterations (default 20)
        :param cores: number of cores for parallel processing (default all)
        """
        n = self.beta.shape[0]
        N = self.beta.shape[2]
        time = np.linspace(0, 1, T)

        if n > 500:
            parallel = True
        elif T > 100:
            parallel = True
        else:
            parallel = False

        binsize = np.diff(time)
        binsize = binsize.mean()

        # Create B-Spline Basis if none provided
        if B is None:
            B = bs(time, df=df, degree=4, include_intercept=True)
        Nb = B.shape[1]

        # second derivative for regularization
        Bdiff = np.zeros((T, Nb))
        for ii in range(0, Nb):
            Bdiff[:, ii] = np.gradient(np.gradient(B[:, ii], binsize), binsize)

        q, beta = preproc_open_curve(self.beta, T)
        self.q = q
        beta0 = beta.copy()
        qn = q.copy()

        gamma = np.tile(np.linspace(0, 1, T), (N, 1))
        gamma = gamma.transpose()
        O_hat = np.tile(np.eye(n), (N, 1, 1)).T

        itr = 1
        self.SSE = np.zeros(max_itr)
        while itr <= max_itr:
            print("Iteration: %d" % itr)
            # align data

            # OLS using basis
            Phi = np.ones((N, n * Nb + 1))
            for ii in range(0, N):
                for jj in range(0, n):
                    for kk in range(1, Nb + 1):
                        Phi[ii, jj * Nb + kk] = trapz(qn[jj, :, ii] * B[:, kk - 1], time)
            
            R = np.zeros((n * Nb+1, n * Nb+1))
            for kk in range(0, n):
                for ii in range(1, Nb+1):
                    for jj in range(1, Nb+1):
                        R[kk * Nb + ii, kk * Nb + jj] = trapz(Bdiff[:, ii-1] * Bdiff[:, jj-1], time)

            xx = np.dot(Phi.T, Phi)
            inv_xx = inv(xx + lam * R)
            xy = np.dot(Phi.T, self.y)
            b = np.dot(inv_xx, xy)

            alpha = b[0]
            nu = np.zeros((n, T))
            for ii in range(0, n):
                nu[ii, :] = B.dot(b[(ii * Nb + 1):((ii + 1) * Nb + 1)])

            # compute the SSE
            int_X = np.zeros(N)
            for ii in range(0, N):
                int_X[ii] = cf.innerprod_q2(qn[:, :, ii], nu)

            self.SSE[itr - 1] = sum((self.y.reshape(N) - alpha - int_X) ** 2)

            # find gamma
            gamma_new = np.zeros((T, N))
            if parallel:
                out = Parallel(n_jobs=cores)(delayed(regression_warp)(nu, q[:, :, n], self.y[n], alpha) for n in range(N))
                for ii in range(0, N):
                    gamma_new[:, ii] = out[ii][0]
                    beta1n = cf.group_action_by_gamma_coord(out[ii][1].dot(beta0[:, :, ii]), out[ii][0])
                    beta[:, :, ii] = beta1n
                    O_hat[:, :, ii] = out[ii][1]
                    qn[:, :, ii] = cf.curve_to_q(beta1n)[0]
            else:
                for ii in range(0, N):
                    q1 = q[:, :, ii]
                    gammatmp, Otmp = regression_warp(nu, q1, self.y[ii], alpha)
                    gamma_new[:, ii] = gammatmp
                    beta1n = cf.group_action_by_gamma_coord(Otmp.dot(beta0[:, :, ii]), gammatmp)
                    beta[:, :, ii] = beta1n
                    O_hat[:, :, ii] = Otmp
                    qn[:, :, ii] = cf.curve_to_q(beta1n)[0]


            if np.abs(self.SSE[itr - 1] - self.SSE[itr - 2]) < 1e-15:
                break
            else:
                gamma = gamma_new

            itr += 1

        tau = np.zeros(N)
        self.alpha = alpha
        self.nu = nu
        self.beta0 = beta0
        self.betan = beta
        self.gamma = gamma
        self.qn = qn
        self.B = B
        self.O = O_hat
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
        :param beta: numpy ndarray of shape (M,N) of M functions with N samples
        :param y: truth if available
        """ 

        if newdata != None:
            beta = newdata['beta']
            y = newdata['y']

            T = self.q.shape[1]
            n = beta.shape[2]
            N = self.q.shape[2]

            q, beta = preproc_open_curve(beta, T)

            y_pred = np.zeros(n)
            for ii in range(0, n):
                diff = self.q - q[:, :, ii][:, :, np.newaxis]
                # dist = np.linalg.norm(np.abs(diff), axis=(0, 1)) ** 2
                dist = np.zeros(N)
                for jj in range(0, N):
                    dist[jj] = np.linalg.norm(np.abs(diff[:, :, jj])) ** 2
                beta1 = beta[:, :, ii]
                beta1 = self.O[:, :, dist.argmin()].dot(beta1)
                beta1 = cf.group_action_by_gamma_coord(beta1,
                                                       self.gamma[:, dist.argmin()])
                q_tmp = cf.curve_to_q(beta1)[0]

                y_pred[ii] = self.alpha + cf.innerprod_q2(q_tmp, self.nu)
            
            if y is None:
                self.SSE = None
            else:
                self.SSE = sum((y - y_pred) ** 2)
            
            self.y_pred = y_pred

        else:
            n = self.q.shape[2]
            N = self.q.shape[2]
            y_pred = np.zeros(n)
            for ii in range(0, n):
                diff = self.q - self.q[:, :, ii][:, :, np.newaxis]
                # dist = np.linalg.norm(np.abs(diff), axis=(0, 1)) ** 2
                dist = np.zeros(N)
                for jj in range(0, N):
                    dist[jj] = np.linalg.norm(np.abs(diff[:, :, jj])) ** 2
                beta1 = self.beta0[:, :, ii]
                beta1 = self.O[:, :, dist.argmin()].dot(beta1)
                beta1 = cf.group_action_by_gamma_coord(beta1,
                                                       self.gamma[:, dist.argmin()])
                q_tmp = cf.curve_to_q(beta1)[0]

                y_pred[ii] = self.alpha + cf.innerprod_q2(q_tmp, self.nu)
            
            self.y_pred = y_pred

        return


class oc_elastic_logistic:
    """
    This class identifies a logistic regression model with
    phase-variability using elastic methods for open curves

    Usage:  obj = oc_elastic_logistic(beta,y)

    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param y: numpy array of N responses
    :param B: optional matrix describing Basis elements
    :param alpha: alpha parameter of model
    :param nu: nu(t) of model
    :param betan: aligned curves - numpy ndarray of shape (n,T,N)
    :param O: calculated rotation matrices
    :param gamma: calculated warping functions
    :param B: basis matrix
    :param b: basis coefficients
    :param Loss: logistic loss

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  29-Oct-2021

    """

    def __init__(self, beta, y, ):
        """
        Construct an instance of the elastic_regression class
        :param beta: numpy ndarray of shape (n, M, N) describing N curves
        :param y: response vector
        :param time: vector of size M describing the sample points
        """

        self.beta = beta
        self.y = y
    
    def calc_model(self, B=None, df=60, T=100, max_itr=40, cores=-1,
                    deltaO=.1, deltag=.05, method=1):
        """
        This function identifies a logistic regression model with
        phase-variability using elastic methods for open curves

        :param B: optional matrix describing Basis elements
        :param df: number of degrees of freedom B-spline (default 20)
        :param T: number of desired samples along curve (default 100)
        :param max_itr: maximum number of iterations (default 20)
        :param cores: number of cores for parallel processing (default all)
        
        """
        n = self.beta.shape[0]
        N = self.beta.shape[2]
        time = np.linspace(0, 1, T)

        if n > 500:
            parallel = True
        elif T > 100:
            parallel = True
        else:
            parallel = True

        # Create B-Spline Basis if none provided
        if B is None:
            B = bs(time, df=df, degree=4, include_intercept=True)
        Nb = B.shape[1]

        q, beta = preproc_open_curve(self.beta, T)
        beta0 = beta.copy()
        qn = q.copy()

        gamma = np.tile(np.linspace(0, 1, T), (N, 1))
        gamma = gamma.transpose()
        O_hat = np.tile(np.eye(n), (N, 1, 1)).T

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
                                args=(Phi, self.y), pgtol=1e-10, maxiter=200,
                                maxfun=250, factr=1e-30)
            b = out[0]
            b = b/norm(b)
            # alpha_norm = b1[0]
            alpha = b[0]
            nu = np.zeros((n, T))
            for ii in range(0, n):
                nu[ii, :] = B.dot(b[(ii * Nb + 1):((ii + 1) * Nb + 1)])

            # compute the logistic loss
            LL[itr] = logit_loss(b, Phi, self.y)

            # find gamma
            gamma_new = np.zeros((T, N))
            if parallel:
                out = Parallel(n_jobs=cores)(delayed(logistic_warp)(alpha, nu, q[:, :, ii], self.y[ii], deltaO=deltaO, deltag=deltag, method=method) for ii in range(N))
                for ii in range(0, N):
                    gamma_new[:, ii] = out[ii][0]
                    beta1n = cf.group_action_by_gamma_coord(out[ii][1].dot(beta0[:, :, ii]), out[ii][0])
                    beta[:, :, ii] = beta1n
                    O_hat[:, :, ii] = out[ii][1]
                    qn[:, :, ii] = cf.curve_to_q(beta[:, :, ii])[0]
            else:
                for ii in range(0, N):
                    q1 = q[:, :, ii]
                    gammatmp, Otmp, tautmp = logistic_warp(alpha, nu, q1, self.y[ii], deltaO=deltaO, 
                                                           deltag=deltag, method=method)
                    gamma_new[:, ii] = gammatmp
                    beta1n = cf.group_action_by_gamma_coord(Otmp.dot(beta0[:, :, ii]), gammatmp)
                    beta[:, :, ii] = beta1n
                    O_hat[:, :, ii] = Otmp
                    qn[:, :, ii] = cf.curve_to_q(beta[:, :, ii])[0]

            if norm(gamma - gamma_new) < 1e-5:
                break
            else:
                gamma = gamma_new.copy()

            itr += 1

        tau = np.zeros(N)
        self.alpha = alpha
        self.nu = nu
        self.betan = beta
        self.beta0 = beta0
        self.gamma = gamma_new
        self.q = q
        self.O = O_hat
        self.tau = tau
        self.B = B
        self.b = b[1:-1]
        self.Loss = LL[1:itr]
        self.qn = qn

        return

    def predict(self, newdata=None):
        """
        This function performs prediction on regression model on new data if available or current stored data in object
        Usage:  obj.predict()
                obj.predict(newdata)

        :param newdata: dict containing new data for prediction (needs the keys below, if None predicts on training data)
        :type newdata: dict
        :param beta: numpy ndarray of shape (M,N) of M functions with N samples
        :param y: truth if available
        """ 

        if newdata != None:
            beta = newdata['beta']
            y = newdata['y']

            T = self.q.shape[1]
            n = beta.shape[2]
            N = self.q.shape[2]

            q, beta = preproc_open_curve(beta, T)

            y_pred = np.zeros(n)
            for ii in range(0, n):
                diff = self.q - q[:, :, ii][:, :, np.newaxis]
                dist = np.zeros(N)
                for jj in range(0, N):
                    dist[jj] = np.linalg.norm(np.abs(diff[:, :, jj])) ** 2
                beta1 = beta[:, :, ii]
                beta1 = self.O[:, :, dist.argmin()].dot(beta1)
                beta1 = cf.group_action_by_gamma_coord(beta1,
                                                       self.gamma[:, dist.argmin()])
                q_tmp = cf.curve_to_q(beta1)[0]

                y_pred[ii] = self.alpha + cf.innerprod_q2(q_tmp, self.nu)
            
            if y is None:
                y_pred = phi(y_pred)
                y_labels = np.ones(n)
                y_labels[y_pred < 0.5] = -1
                self.PC = None
            else:
                y_pred = phi(y_pred)
                y_labels = np.ones(n)
                y_labels[y_pred < 0.5] = -1
                TP = sum(y[y_labels == 1] == 1)
                FP = sum(y[y_labels == -1] == 1)
                TN = sum(y[y_labels == -1] == -1)
                FN = sum(y[y_labels == 1] == -1)
                self.PC = (TP + TN) / float(TP + FP + FN + TN)
            
            self.y_pred = y_pred
            self.y_labels = y_labels

        else:
            n = self.q.shape[2]
            N = self.q.shape[2]
            y_pred = np.zeros(n)
            for ii in range(0, n):
                diff = self.q - self.q[:, :, ii][:, :, np.newaxis]
                dist = np.zeros(N)
                for jj in range(0, N):
                    dist[jj] = np.linalg.norm(np.abs(diff[:, :, jj])) ** 2
                beta1 = self.beta0[:, :, ii]
                beta1 = self.O[:, :, dist.argmin()].dot(beta1)
                beta1 = cf.group_action_by_gamma_coord(beta1,
                                                       self.gamma[:, dist.argmin()])
                q_tmp = cf.curve_to_q(beta1)[0]

                y_pred[ii] = self.alpha + cf.innerprod_q2(q_tmp, self.nu)

            y_pred = phi(y_pred)
            y_labels = np.ones(n)
            y_labels[y_pred < 0.5] = -1
            TP = sum(self.y[y_labels == 1] == 1)
            FP = sum(self.y[y_labels == -1] == 1)
            TN = sum(self.y[y_labels == -1] == -1)
            FN = sum(self.y[y_labels == 1] == -1)
            self.PC = (TP + TN) / float(TP + FP + FN + TN)
            
            self.y_pred = y_pred
            self.y_labels = y_labels

        return



class oc_elastic_mlogistic:
    """
    This class identifies a multinomial logistic regression model with
    phase-variability using elastic methods for open curves

    Usage:  obj = oc_elastic_logistic(beta,y)

    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param y: numpy array of N responses
    :param B: optional matrix describing Basis elements
    :param alpha: alpha parameter of model
    :param nu: nu(t) of model
    :param betan: aligned curves - numpy ndarray of shape (n,T,N)
    :param O: calculated rotation matrices
    :param gamma: calculated warping functions
    :param B: basis matrix
    :param b: basis coefficients
    :param Loss: logistic loss

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  29-Oct-2021

    """
    def __init__(self, beta, y, ):
        """
        Construct an instance of the elastic_regression class
        :param beta: numpy ndarray of shape (n, M, N) describing N curves
        :param y: response vector
        :param time: vector of size M describing the sample points
        """

        self.beta = beta
        self.y = y

        # Code labels
        m = y.max()
        N = beta.shape[2]
        Y = np.zeros((N, m), dtype=int)
        for ii in range(0, N):
            Y[ii, y[ii]-1] = 1
        self.Y = Y
    
    
    def calc_model(self, B=None, df=20, T=100, max_itr=30, cores=-1,
                   deltaO=.003, deltag=.003):
        """
        This function identifies a multinomial logistic regression model with
        phase-variability using elastic methods for open curves

        :param B: optional matrix describing Basis elements
        :param df: number of degrees of freedom B-spline (default 20)
        :param T: number of desired samples along curve (default 100)
        :param max_itr: maximum number of iterations (default 20)
        :param cores: number of cores for parallel processing (default all)

        """
        n = self.beta.shape[0]
        N = self.beta.shape[2]
        time = np.linspace(0, 1, T)
        m = self.y.max()

        if n > 500:
            parallel = True
        elif T > 100:
            parallel = True
        else:
            parallel = True


        # Create B-Spline Basis if none provided
        if B is None:
            B = bs(time, df=df, degree=4, include_intercept=True)
        Nb = B.shape[1]

        q, beta = preproc_open_curve(self.beta, T)
        qn = q.copy()
        beta0 = beta.copy()

        gamma = np.tile(np.linspace(0, 1, T), (N, 1))
        gamma = gamma.transpose()
        O_hat = np.tile(np.eye(n), (N, 1, 1)).T

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
                                args=(Phi, self.Y), pgtol=1e-10, maxiter=200,
                                maxfun=250, factr=1e-30)
            b = out[0]
            B0 = b.reshape(n * Nb + 1, m)
            alpha = B0[0, :]
            nu = np.zeros((n, T, m))
            for i in range(0, m):
                for j in range(0, n):
                    nu[j, :, i] = B.dot(B0[(j * Nb + 1):((j + 1) * Nb + 1), i])

            # compute the logistic loss
            LL[itr] = mlogit_loss(b, Phi, self.Y)

            # find gamma
            gamma_new = np.zeros((T, N))
            if parallel:
                out = Parallel(n_jobs=cores)(delayed(mlogit_warp_grad)(alpha, nu, q[:, :, n], self.Y[n, :], deltaO=deltaO, deltag=deltag) for n in range(N))
                for ii in range(0, N):
                    gamma_new[:, ii] = out[ii][0]
                    beta1n = cf.group_action_by_gamma_coord(out[ii][1].dot(beta0[:, :, ii]), out[ii][0])
                    beta[:, :, ii] = beta1n
                    O_hat[:, :, ii] = out[ii][1]
                    qn[:, :, ii] = cf.curve_to_q(beta[:, :, ii])[0]
            else:
                for ii in range(0, N):
                    gammatmp, Otmp = mlogit_warp_grad(alpha, nu, q[:, :, ii], self.Y[ii, :], deltaO=deltaO, deltag=deltag)
                    gamma_new[:, ii] = gammatmp
                    beta1n = cf.group_action_by_gamma_coord(Otmp.dot(beta0[:, :, ii]), gammatmp)
                    beta[:, :, ii] = beta1n
                    O_hat[:, :, ii] = Otmp
                    qn[:, :, ii] = cf.curve_to_q(beta[:, :, ii])[0]

            if norm(gamma - gamma_new) < 1e-5:
                break
            else:
                gamma = gamma_new.copy()

            itr += 1
        
        self.alpha = alpha
        self.nu = nu
        self.beta0 = beta0
        self.betan = beta
        self.q = q
        self.qn = qn
        self.gamma = gamma_new
        self.O = O_hat
        self.B = B
        self.b = b[1:-1]
        self.Loss = LL[1:itr]
        self.n_classes = m

        return 


    def predict(self, newdata=None):
        """
        This function performs prediction on regression model on new data if available or current stored data in object
        Usage:  obj.predict()
                obj.predict(newdata)

        :param newdata: dict containing new data for prediction (needs the keys below, if None predicts on training data)
        :type newdata: dict
        :param beta: numpy ndarray of shape (M,N) of M functions with N samples
        :param y: truth if available
        """ 

        if newdata != None:
            beta = newdata['beta']
            y = newdata['y']

            T = self.q.shape[1]
            n = beta.shape[2]
            N = self.q.shape[2]

            q, beta = preproc_open_curve(beta, T)

            m = self.n_classes
            y_pred = np.zeros((n, m))
            for ii in range(0, n):
                diff = self.q - q[:, :, ii][:, :, np.newaxis]
                dist = np.zeros(N)
                for jj in range(0, N):
                    dist[jj] = np.linalg.norm(np.abs(diff[:, :, jj])) ** 2
                beta1 = beta[:, :, ii]
                beta1 = self.O[:, :, dist.argmin()].dot(beta1)
                beta1 = cf.group_action_by_gamma_coord(beta1,
                                                       self.gamma[:, dist.argmin()])
                q_tmp = cf.curve_to_q(beta1)[0]

                for jj in range(0, m):
                    y_pred[ii, jj] = self.alpha[jj] + cf.innerprod_q2(q_tmp, self.nu[:, :, jj])
            
            if y is None:
                y_pred = phi(y_pred.ravel())
                y_pred = y_pred.reshape(n, m)
                y_labels = y_pred.argmax(axis=1) + 1
                self.PC = None
            else:
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

                self.PC = sum(y == y_labels) / float(y_labels.size)
            
            self.y_pred = y_pred
            self.y_labels = y_labels

        else:
            n = self.q.shape[2]
            N = self.q.shape[2]
            m = self.n_classes
            y_pred = np.zeros((n, m))
            for ii in range(0, n):
                diff = self.q - self.q[:, :, ii][:, :, np.newaxis]
                dist = np.zeros(N)
                for jj in range(0, N):
                    dist[jj] = np.linalg.norm(np.abs(diff[:, :, jj])) ** 2
                beta1 = self.beta0[:, :, ii]
                beta1 = self.O[:, :, dist.argmin()].dot(beta1)
                beta1 = cf.group_action_by_gamma_coord(beta1,
                                                       self.gamma[:, dist.argmin()])
                q_tmp = cf.curve_to_q(beta1)[0]

                for jj in range(0, m):
                    y_pred[ii, jj] = self.alpha[jj] + cf.innerprod_q2(q_tmp, self.nu[:, :, jj])

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

            self.PC = sum(self.y == y_labels) / float(y_labels.size)
            
            self.y_pred = y_pred
            self.y_labels = y_labels

        return


# helper function for curve manipulation
def preproc_open_curve(beta, T=100):
    n, M, k = beta.shape

    q = np.zeros((n, T, k))
    beta2 = np.zeros((n, T, k))
    for i in range(0, k):
        beta1 = beta[:, :, i]
        beta1, scale = cf.scale_curve(beta1)
        if T != M:
            beta1 = cf.resamplecurve(beta1, T)
        centroid1 = cf.calculatecentroid(beta1)
        beta1 = beta1 - np.tile(centroid1, [T, 1]).T
        beta2[:, :, i] = beta1
        q[:, :, i] = cf.curve_to_q(beta1)[0]

    return (q, beta2)


# helper functions for linear regression
def regression_warp(nu, q, y, alpha):
    """
    calculates optimal warping for function linear regression

    :param nu: numpy ndarray of srvf (M,N) of M functions with N samples
    :param q: numpy ndarray of srvf (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses
    :param alpha: numpy scalar

    :rtype: numpy array
    :return gamma_new: warping function

    """
    T = q.shape[1]

    qM, O_M, gam_M = cf.find_rotation_and_seed_q(nu, q, rotation=False)
    y_M = cf.innerprod_q2(qM, nu)

    qm, O_m, gam_m = cf.find_rotation_and_seed_q(-1 * nu, q, rotation=False)
    y_m = cf.innerprod_q2(qm, nu)

    if y > alpha + y_M:
        O_hat = O_M
        gamma_new = gam_M
    elif y < alpha + y_m:
        O_hat = O_m
        gamma_new = gam_m
    else:
        gamma_new, O_hat = cf.curve_zero_crossing(y - alpha, q, nu, y_M, y_m, gam_M,
                                                  gam_m)

    return(gamma_new, O_hat)


# helper functions for logistic regression
def logistic_warp(alpha, nu, q, y, deltaO=.1, deltag=.05, max_itr=8000,
                  tol=1e-4, display=0, method=1):
    """
    calculates optimal warping for function logistic regression

    :param alpha: scalar
    :param nu: numpy ndarray of shape (M,N) of M functions with N samples
    :param q: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses

    :rtype: numpy array
    :return gamma: warping function

    """
    if method == 1:
        tau = 0
        # q, scale = cf.scale_curve(q)
        q = q/norm(q)
        # nu, scale = cf.scale_curve(nu)
        # alpha = alpha/scale

        gam_old, O_old = lw.oclogit_warp(np.ascontiguousarray(alpha),
                                         np.ascontiguousarray(nu),
                                         np.ascontiguousarray(q),
                                         np.ascontiguousarray(y, dtype=np.int32),
                                         max_itr, tol, deltaO, deltag, display)
    elif method == 2:
        betanu = cf.q_to_curve(nu)
        beta = cf.q_to_curve(q)
        T = beta.shape[1]
        if y == 1:
            beta1, O_old, tau = cf.find_rotation_and_seed_coord(betanu, beta)
            q = cf.curve_to_q(beta1)[0]
            gam_old = cf.optimum_reparam_curve(nu, q)
        elif y == -1:
            beta1, O_old, tau = cf.find_rotation_and_seed_coord(-1 * betanu, beta)
            q = cf.curve_to_q(beta1)[0]
            gam_old = cf.optimum_reparam_curve(-1 * nu, q)


    return (gam_old, O_old, tau)


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


def logit_loss(b, X, y, lam=0.0):
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
    out = out.sum() + .5 * lam * b.dot(b)
    return out


def logit_gradient(b, X, y, lam=0.0):
    """
    calculates gradient of the logistic loss

    :param b: numpy ndarray of shape (M,N) of M functions with N samples
    :param X: numpy ndarray of shape (M,N) of M functions with N samples
    :param y: numpy ndarray of shape (1,N) of M functions with N samples
    responses

    :rtype: numpy array
    :return grad: gradient of logistic loss

    """
    z = X.dot(b)
    z = phi(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + lam * b
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
def mlogit_warp_grad(alpha, nu, q, y, max_itr=8000, tol=1e-4,
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
    :param deltag: gradient step size for warping (Default=0.008)
    :param display: display iterations (Default=0)

    :rtype: tuple of numpy array
    :return gam_old: warping function

    """

    alpha = alpha/norm(alpha)
    q, scale = cf.scale_curve(q)  # q/norm(q)
    for ii in range(0, nu.shape[2]):
        nu[:, :, ii], scale = cf.scale_curve(nu[:, :, ii])  # nu/norm(nu)

    gam_old, O_old = mw.ocmlogit_warp(np.ascontiguousarray(alpha),
                                      np.ascontiguousarray(nu),
                                      np.ascontiguousarray(q),
                                      np.ascontiguousarray(y, dtype=np.int32),
                                      max_itr, tol, deltaO, deltag, display)

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
