import numpy as np
import fdasrsf as fs
from scipy.integrate import trapz
from scipy.optimize import fmin_l_bfgs_b
from patsy import bs
import h5py


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


def preproc_open_curve(beta, T=100):
    n, M, k = beta.shape

    q = np.zeros((n, T, k))
    beta2 = np.zeros((n, T, k))
    for i in range(0, k):
        beta1 = fs.resamplecurve(beta[:, :, i], T)
        beta2[:, :, i] = beta1
        q[:, :, i] = fs.curve_to_q(beta1)

    return (q, beta2)


fun = h5py.File('/Users/jdtucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T

a, b, c = C.shape
beta = np.zeros((a, b, 60))
for ii in range(0, 20):
    beta_tmp = np.zeros((a, b+1))
    beta_tmp[:, 0:b] = C[:, :, ii]
    beta_tmp[:, b] = C[:, 0, ii]
    beta[:, :, ii] = fs.resamplecurve(beta_tmp, b)
    beta_tmp[:, 0:b] = C[:, :, ii+20]
    beta_tmp[:, b] = C[:, 0, ii+20]
    beta[:, :, ii+20] = fs.resamplecurve(beta_tmp, b)
    beta_tmp[:, 0:b] = C[:, :, ii+40]
    beta_tmp[:, b] = C[:, 0, ii+40]
    beta[:, :, ii+40] = fs.resamplecurve(beta_tmp, b)

y = np.ones(60, dtype=int)
y[20:40] = 2
y[40:60] = 3

n = beta.shape[0]
N = beta.shape[2]
T = 200
time = np.linspace(0, 1, T)

# Code labels
m = y.max()
Y = np.zeros((N, m), dtype=int)
for ii in range(0, N):
    Y[ii, y[ii] - 1] = 1

B = bs(time, df=60, degree=4, include_intercept=True)
Nb = B.shape[1]

q, beta = preproc_open_curve(beta, T)


Phi = np.ones((N, n * Nb + 1))
for ii in range(0, N):
    for jj in range(0, n):
        for kk in range(1, Nb + 1):
            Phi[ii, jj * Nb + kk] = trapz(q[jj, :, ii] * B[:, kk - 1], time)

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

y = Y[0, :]
q0 = q[:, :, 0]
fun = h5py.File('debug_data_oc.h5', 'w')
fun.create_dataset('q', data=q0)
fun.create_dataset('y', data=y)
fun.create_dataset('alpha', data=alpha)
fun.create_dataset('nu', data=nu)
fun.close()
