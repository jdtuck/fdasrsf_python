import numpy as np
import fdasrsf as fs
from numpy.random import normal
from scipy.integrate import trapezoid
from scipy.stats import norm as gauss
from scipy.optimize import fmin_l_bfgs_b
from patsy import bs
import h5py


def mlogit_loss(b, X, Y):
    # multinomial logistic loss (negative log-likelihood)
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
    # gradient of the multinomial logistic loss
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

# Create Original Data
time = np.linspace(0, 1, 101)
M = time.size
N = 30
lam = 0.001
center = np.array([.35, .5, .65])
center2 = np.array([4, 3.7, 4])
sd1 = .05
gam_sd = 8
num_comp = 5
f_orig = np.zeros((M, N * center.size))
omega = 2 * np.pi
cnt = 0
for ii in range(0, center.size):
    tmp = gauss(loc=center[ii], scale=.075)
    for jj in range(0, N):
        f_orig[:, cnt] = normal(center2[ii], sd1) * tmp.pdf(time)
        cnt += 1

q_orig = fs.f_to_srsf(f_orig, time)
y_orig = np.ones(q_orig.shape[1], dtype=int)
y_orig[N:2*N] = 2
y_orig[2*N:3*N] = 3

f = np.zeros((M, f_orig.shape[1]))
q = np.zeros((M, f_orig.shape[1]))
cnt = 0
gam_orig = fs.rgam(M, gam_sd, 3*N)
for ii in range(0, center.size):
    for ii in range(0, N):
        f[:, cnt] = np.interp((time[-1] - time[0]) * gam_orig[:, cnt] + time[0], time, f_orig[:, cnt])
        q[:, cnt] = fs.warp_q_gamma(time, q_orig[:, cnt], gam_orig[:, cnt])
        cnt += 1

y = y_orig
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
B = bs(time, df=20, degree=4, include_intercept=True)
Nb = B.shape[1]

Phi = np.ones((N, Nb+1))
for ii in range(0, N):
    for jj in range(1, Nb+1):
        Phi[ii, jj] = trapezoid(q[:, ii] * B[:, jj-1], time)

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

q0 = q

q = q[:, 0]
y = Y[0, :]


fun = h5py.File('debug_data.h5', 'w')
fun.create_dataset('q', data=q)
fun.create_dataset('y', data=y)
fun.create_dataset('time', data=time)
fun.create_dataset('alpha', data=alpha)
fun.create_dataset('beta', data=beta)
fun.close()
