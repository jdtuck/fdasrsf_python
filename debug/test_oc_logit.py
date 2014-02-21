import os, sys, inspect
# realpath() with make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/../fdasrsf")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
import h5py
import numpy as np
import curve_functions as cf
import curve_regression as cr
fun = h5py.File('/Users/jderektucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T

a, b, c = C.shape
beta = np.zeros((a, b, 40))
for ii in range(0, 20):
    beta_tmp = np.zeros((a, b+1))
    beta_tmp[:, 0:b] = C[:, :, ii]
    beta_tmp[:, b] = C[:, 0, ii]
    beta[:, :, ii] = cf.resamplecurve(beta_tmp, b)
    beta_tmp[:, 0:b] = C[:, :, ii+20]
    beta_tmp[:, b] = C[:, 0, ii+20]
    beta[:, :, ii+20] = cf.resamplecurve(beta_tmp, b)

y = np.ones(38, dtype=int)
y[0:19] = -1
beta1 = beta[:, :, 0:19]
beta2 = beta[:, :, 20:39]

betatr = np.concatenate((beta1, beta2), axis=2)

beta_tst = np.zeros((a, b, 2))
beta_tst[:, :, 0] = beta[:, :, 19]
beta_tst[:, :, 1] = beta[:, :, 39]
y_test = np.ones(2, dtype=int)
y_test[0] = -1

model = cr.oc_elastic_logistic(betatr, y, cores=5)
out = cr.oc_elastic_prediction(beta_tst, model, y=y_test)
