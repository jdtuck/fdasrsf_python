import numpy as np
from scipy.linalg import norm
import fdasrsf as fs
import ocmlogit_warp as mw
import h5py

fun = h5py.File('/Users/jderektucker/Documents/Research/fdasrsf/debug/debug_data_oc_logit.h5')
q = fun['q'][:]
y = fun['y'].value
alpha = fun['alpha'].value
nu = fun['nu'][:]

max_itr = 1  # 4000
tol = 1e-4
deltag = .05
deltaO = .1
display = 1

q, scale = fs.scale_curve(q)  # q/norm(q)
nu, scale = fs.scale_curve(nu)  # nu/norm(nu)

gam_old, O_old = mw.oclogit_warp(np.ascontiguousarray(alpha),
                                 np.ascontiguousarray(nu),
                                 np.ascontiguousarray(q),
                                 np.ascontiguousarray(y, dtype=np.int32),
                                 max_itr, tol, deltaO, deltag, display)
