import numpy as np
import fdasrsf as fs
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.linalg import norm
import h5py

fun = h5py.File('/Users/jdtucker/Documents/Research/fdasrsf/debug/debug_data.h5')
q = fun['q'][:]
y = fun['y'][:]
time = fun['time'][:]
alpha = fun['alpha'][:]
beta = fun['beta'][:]
max_itr = 8000  # 4000
tol = 1e-10
delta = .01
display = 1

TT = time.size
binsize = np.diff(time)
binsize = binsize.mean()
m = beta.shape[1]
alpha = alpha/norm(alpha)
q = q/norm(q)
for i in range(0, m):
    beta[:, i] = beta[:, i]/norm(beta[:, i])
eps = np.finfo(np.double).eps
gam = np.linspace(0, 1, TT)
psi = np.sqrt(np.abs(np.gradient(gam, binsize)) + eps)
gam_old = gam
psi_old = psi

itr = 0
max_val = np.zeros(max_itr+1)
while itr <= max_itr:
    A = np.zeros(m)
    Adiff = np.zeros((TT, m))
    qtmp = np.interp((time[-1] - time[0]) * gam_old + time[0], time, q)
    qtmp_diff = np.interp((time[-1] - time[0]) * gam_old + time[0],
                          time, np.gradient(q, binsize))
    for i in range(0, m):
        A[i] = fs.innerprod_q(time, qtmp * psi_old, beta[:, i])
        tmp1 = trapezoid(qtmp_diff * psi_old * beta[:, i], time)
        tmp2 = cumulative_trapezoid(qtmp_diff * psi_old * beta[:, i], time, initial=0)
        tmp = tmp1 - tmp2
        Adiff[:, i] = 2 * psi_old * tmp + qtmp * beta[:, i]

    tmp1 = np.sum(np.exp(alpha + A))
    tmp2 = np.sum(np.exp(alpha + A) * Adiff, axis=1)
    h = np.sum(y * Adiff, axis=1) - (tmp2 / tmp1)

    tmp = fs.innerprod_q(time, h, psi_old)
    vec = h - tmp*psi_old
    vecnorm = norm(vec) * binsize
    costmp = np.cos(delta * vecnorm) * psi_old
    sintmp = np.sin(delta * vecnorm) * (vec / vecnorm)
    psi_new = costmp + sintmp
    gam_tmp = cumulative_trapezoid(psi_new * psi_new, time, initial=0)
    gam_new = (gam_tmp - gam_tmp[0]) / (gam_tmp[-1] - gam_tmp[0])

    max_val[itr] = np.sum(y * (alpha + A)) - np.log(tmp1)

    if display == 1:
        print("Iteration %d : Cost %f" % (itr+1, max_val[itr]))

    psi_old = psi_new
    gam_old = gam_new

    if itr >= 2:
        max_val_change = max_val[itr] - max_val[itr-1]
        if np.abs(max_val_change) < tol:
            break

    itr += 1
