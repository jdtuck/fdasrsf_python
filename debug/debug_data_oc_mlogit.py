import numpy as np
import fdasrsf as fs
from scipy.integrate import trapz
from scipy.optimize import fmin_l_bfgs_b
from patsy import bs
import h5py


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
    :return grad: gradient of logisitc loss

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


def preproc_open_curve(beta, T=100):
    n, M, k = beta.shape

    q = np.zeros((n, T, k))
    beta2 = np.zeros((n, T, k))
    for i in range(0, k):
        beta1 = fs.resamplecurve(beta[:, :, i], T)
        beta2[:, :, i] = beta1
        q[:, :, i] = fs.curve_to_q(beta1)

    return (q, beta2)


fun = h5py.File('/Users/jderektucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T

a, b, c = C.shape
beta = np.zeros((a, b, 40))
cnt = 0
for ii in range(40, 60):
    beta_tmp = np.zeros((a, b+1))
    beta_tmp[:, 0:b] = C[:, :, ii]
    beta_tmp[:, b] = C[:, 0, ii]
    beta[:, :, cnt] = fs.resamplecurve(beta_tmp, b)
    beta_tmp = np.zeros((a, b+1))
    beta_tmp[:, 0:b] = C[:, :, ii+1080]
    beta_tmp[:, b] = C[:, 0, ii+1080]
    beta[:, :, cnt+20] = fs.resamplecurve(beta_tmp, b)
    cnt +=1

y = np.ones(40, dtype=int)
y[20:40] = -1

n = beta.shape[0]
N = beta.shape[2]
T = 200
time = np.linspace(0, 1, T)

B = bs(time, df=60, degree=4, include_intercept=True)
Nb = B.shape[1]

q, beta = preproc_open_curve(beta, T)

Phi = np.ones((N, n * Nb + 1))
for ii in range(0, N):
    for jj in range(0, n):
        for kk in range(1, Nb + 1):
            Phi[ii, jj * Nb + kk] = trapz(q[jj, :, ii] * B[:, kk - 1], time)

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

y = y[0]
q0 = q[:, :, 0]
fun = h5py.File('debug_data_oc_logit.h5', 'w')
fun.create_dataset('q', data=q0)
fun.create_dataset('y', data=y)
fun.create_dataset('alpha', data=alpha)
fun.create_dataset('nu', data=nu)
fun.close()
