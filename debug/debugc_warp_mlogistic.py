import numpy as np
import fdasrsf as fs
import mlogit_warp as mw
import h5py

fun = h5py.File('/Users/jdtucker/Documents/Research/fdasrsf/debug/debug_data.h5')
q = fun['q'][:]
y = fun['y'][:]
time = fun['time'][:]
alpha = fun['alpha'][:]
beta = fun['beta'][:]

max_itr = 10000  # 4000
tol = 1e-10
delta = .01
display = 1

gam_old = mw.mlogit_warp(np.ascontiguousarray(alpha),
                         np.ascontiguousarray(beta),
                         time, np.ascontiguousarray(q),
                         np.ascontiguousarray(y, dtype=np.int32), max_itr,
                         tol, delta, display)
