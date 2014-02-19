import h5py
import fdasrsf as fs
import numpy as np
import matplotlib.pyplot as plt
fun = h5py.File('/Users/jdtucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T

a, b, c = C.shape
q = np.zeros((a, b, 20))
beta = np.zeros((a, b, 20))
for ii in range(0, 20):
    q[:, :, ii] = fs.curve_to_q(C[:, :, ii])
    beta[:, :, ii] = C[:, :, ii]

mu, betamean, v = fs.curve_karcher_mean(q, beta, mode='O')
K = fs.curve_karcher_cov(betamean, beta, mode='O')
pd = fs.curve_principal_directions(betamean, mu, K, mode='O', no=3, N=3)
samples = fs.sample_shapes(mu, K, mode='O', no=3, numSamp=10)
