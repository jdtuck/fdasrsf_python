import numpy as np
import fdasrsf as fs
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import norm, expm
import h5py

fun = h5py.File('/Users/jderektucker/Documents/Research/fdasrsf/debug/debug_data_oc_logit.h5')
q = fun['q'][:]
y = fun['y'].value
alpha = fun['alpha'].value
nu = fun['nu'][:]

max_itr = 8000  # 4000
tol = 1e-4
deltag = .05
deltaO = .1
display = 1

q, scale = fs.scale_curve(q)
nu, scale = fs.scale_curve(nu)

# python code
n = q.shape[0]
TT = q.shape[1]
time = np.linspace(0, 1, TT)
binsize = 1. / (TT - 1)

gam = np.linspace(0, 1, TT)
O = np.eye(n)
O_old = O.copy()
gam_old = gam.copy()
qtilde = q.copy()
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
    A = fs.innerprod_q2(qtilde, nu)

    # form gradient for rotation
    # B = np.zeros((n, n, m))
    # for i in range(0, m):
    #     B[:, :, i] = cf.innerprod_q2(E.dot(qtilde), nu[:, :, i]) * E

    # tmp1 = np.sum(np.exp(alpha + A))
    # tmp2 = np.sum(np.exp(alpha + A) * B, axis=2)
    # hO = np.sum(y * B, axis=2) - (tmp2 / tmp1)
    # O_new = O_old.dot(expm(deltaO * hO))

    theta = np.arccos(O_old[0, 0])
    Ograd = np.array([(-1*np.sin(theta), -1*np.cos(theta)),
                     (np.cos(theta), -1*np.sin(theta))])
    B = fs.innerprod_q2(Ograd.dot(qtilde), nu)
    tmp1 = np.exp((-1*y)*(alpha+A))
    tmp2 = (y*tmp1)/(1+tmp1)
    hO = tmp2*B
    O_new = fs.rot_mat(theta+deltaO*hO)

    # form gradient for warping
    qtilde_diff = np.gradient(qtilde, binsize)
    qtilde_diff = qtilde_diff[1]
    tmp3 = np.zeros((TT, p))
    for i in range(0, p):
        cbar = cumulative_trapezoid(f_basis[:, i], time, initial=0)
        ctmp = 2*qtilde_diff*cbar + qtilde*f_basis[:, i]
        tmp3[:, i] = fs.innerprod_q2(ctmp, nu) * f_basis[:, i]

    c = np.sum(tmp3, axis=1)

    hpsi = tmp2*c

    vecnorm = norm(hpsi)
    costmp = np.cos(deltag * vecnorm) * np.ones(TT)
    sintmp = np.sin(deltag * vecnorm) * (hpsi / vecnorm)
    psi_new = costmp + sintmp
    gam_tmp = cumulative_trapezoid(psi_new * psi_new, time, initial=0)
    gam_tmp = (gam_tmp - gam_tmp[0]) / (gam_tmp[-1] - gam_tmp[0])
    gam_new = np.interp(gam_tmp, time, gam_old)

    max_val[itr] = np.sum(y * (alpha + A)) - np.log(tmp1)

    if display == 1:
        print("Iteration %d : Cost %f" % (itr+1, max_val[itr]))

    gam_old = gam_new.copy()
    O_old = O_new.copy()
    qtilde = fs.group_action_by_gamma(O_old.dot(q), gam_old)

    if vecnorm < tol and hO < tol:
        break

    itr += 1
