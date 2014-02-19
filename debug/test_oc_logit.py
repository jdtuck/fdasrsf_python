import h5py
import fdasrsf as fs
import numpy as np
fun = h5py.File('/Users/jdtucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T

a, b, c = C.shape
beta = np.zeros((a, b, 40))
for ii in range(0, 20):
    beta_tmp = np.zeros((a, b+1))
    beta_tmp[:, 0:b] = C[:, :, ii]
    beta_tmp[:, b] = C[:, 0, ii]
    beta[:, :, ii] = fs.resamplecurve(beta_tmp, b)
    beta_tmp[:, 0:b] = C[:, :, ii+20]
    beta_tmp[:, b] = C[:, 0, ii+20]
    beta[:, :, ii+20] = fs.resamplecurve(beta_tmp, b)

y = np.ones(40, dtype=int)
y[0:20] = -1

model = fs.oc_elastic_logistic(beta, y)
