import numpy as np
import fdasrsf as fs
from scipy.integrate import cumtrapz
from scipy.linalg import norm
import h5py

fun = h5py.File('/Users/jdtucker/Documents/Research/fdasrsf/debug/debug_data.h5')
q = fun['q'][:]
y = fun['y'][:]
alpha = fun['alpha'][:]
nu = fun['nu'][:]
max_itr = 8000  # 4000
tol = 1e-10
deltaO = .01
deltag = .01
display = 1

n = q.shape[0]
TT = q.shape[1]
m = nu.shape[1]
time = np.linspace(0, 1, TT)
binsize = 1. / (TT - 1)

alpha = alpha/norm(alpha)
for i in range(0, m):
    nu[:, :, i] = nu[:, :, i]/np.sqrt(fs.innerprod_q(nu[:, :, i], nu[:, :, i]))

eps = np.finfo(np.double).eps
gam = np.linspace(0, 1, TT)
psi = np.sqrt(np.abs(np.gradient(gam, binsize)) + eps)
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
        A[i] = fs.innerprod_q(qtilde, nu[:, :, i])

    # form gradient for rotation
    B = np.zeros((n, n, m))
    for i in range(0, m):
        B[:, :, i] = fs.innerprod_q(E.dot(qtilde), nu[:, :, i]) * E

    tmp1 = np.sum(np.exp(alpha + A))
    tmp2 = np.sum(tmp1 * B, axis=2)
    hO = np.sum(y * B, axis=1) - (tmp2 / tmp1)
    O_new = O_old.dot(np.exp(deltaO * hO))

    # form gradient for warping
    qtilde_diff = np.gradient(qtilde, binsize)
    qtilde_diff = qtilde_diff[1]
    c = np.zeros((TT, m))
    for i in range(0, m):
        tmp3 = np.zeros((TT, p))
        for j in range(0, p):
            cbar = cumtrapz(f_basis[:, j] * f_basis[:, j], time, initial=0)
            ctmp = 2*qtilde_diff*cbar + qtilde*f_basis[:, j]
            tmp3[:, j] = fs.innerprod_q(ctmp, nu[:, :, i]) * f_basis[:, j]

        c[:, i] = np.sum(tmp3, axis=1)

    tmp2 = np.sum(tmp1 * c, axis=2)
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
    qtilde = fs.group_action_by_gamma(O_old.dot(q), gam_old)

    if itr >= 2:
        max_val_change = max_val[itr] - max_val[itr-1]
        if np.abs(max_val_change) < tol:
            break

    itr += 1
