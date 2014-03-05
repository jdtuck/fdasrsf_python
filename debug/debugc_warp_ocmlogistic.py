import numpy as np
from scipy.linalg import norm
import fdasrsf as fs
import ocmlogit_warp as mw
import h5py

fun = h5py.File('/home/dtucker/fdasrsf/debug_data_oc.h5')
q = fun['q'][:]
y = fun['y'][:]
alpha = fun['alpha'][:]
nu = fun['nu'][:]

max_itr = 8000  # 4000
tol = 1e-4
deltag = .05
deltaO = .08
display = 1

alpha = alpha/norm(alpha)
q, scale = fs.scale_curve(q)  # q/norm(q)
for ii in range(0, nu.shape[2]):
    nu[:, :, ii], scale = fs.scale_curve(nu[:, :, ii])  # nu/norm(nu)

gam_old, O_old = mw.ocmlogit_warp(np.ascontiguousarray(alpha),
                           np.ascontiguousarray(nu),
                           np.ascontiguousarray(q),
                           np.ascontiguousarray(y, dtype=np.int32), max_itr,
                           tol, deltaO, deltag, display)
