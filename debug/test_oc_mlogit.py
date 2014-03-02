import os, sys, inspect
# realpath() with make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/../fdasrsf")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
import h5py
import numpy as np
import curve_functions as cf
import curve_regression as cr
fun = h5py.File('/Users/jdtucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T

a, b, c = C.shape
beta = np.zeros((a, b, 60))
for ii in range(0, 20):
    beta_tmp = np.zeros((a, b+1))
    beta_tmp[:, 0:b] = C[:, :, ii]
    beta_tmp[:, b] = C[:, 0, ii]
    beta[:, :, ii] = cf.resamplecurve(beta_tmp, b)
    beta_tmp[:, 0:b] = C[:, :, ii+20]
    beta_tmp[:, b] = C[:, 0, ii+20]
    beta[:, :, ii+20] = cf.resamplecurve(beta_tmp, b)
    beta_tmp[:, 0:b] = C[:, :, ii+40]
    beta_tmp[:, b] = C[:, 0, ii+40]
    beta[:, :, ii+40] = cf.resamplecurve(beta_tmp, b)

y = np.ones(60, dtype=int)
y[20:40] = 2
y[40:60] = 3

model = cr.oc_elastic_mlogistic(beta, y, df=60, T=200, max_itr=40)
out = cr.oc_elastic_prediction(beta, model, y=y)
