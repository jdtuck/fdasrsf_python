# import os, sys, inspect
# # realpath() with make your script run, even if you symlink it :)
# cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/../fdasrsf")
# if cmd_folder not in sys.path:
#     sys.path.insert(0, cmd_folder)
import h5py
import numpy as np
import fdasrsf as fs
# import curve_functions as cf
# import curve_regression as cr
fun = h5py.File('/Users/jderektucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T

a, b, c = C.shape
beta = np.zeros((a, b, 40))
cnt = 0
for ii in range(200, 220):
    beta_tmp = np.zeros((a, b+1))
    beta_tmp[:, 0:b] = C[:, :, ii]
    beta_tmp[:, b] = C[:, 0, ii]
    # beta_tmp, scale = fs.scale_curve(beta_tmp)
    beta[:, :, cnt] = fs.resamplecurve(beta_tmp, b)
    # centroid1 = fs.calculatecentroid(beta_tmp)
    # beta[:, :, cnt] = beta_tmp - np.tile(centroid1, [b, 1]).T
    beta_tmp = np.zeros((a, b+1))
    beta_tmp[:, 0:b] = C[:, :, ii+1080]
    beta_tmp[:, b] = C[:, 0, ii+1080]
    # beta_tmp, scale = fs.scale_curve(beta_tmp)
    beta[:, :, cnt+20] = fs.resamplecurve(beta_tmp, b)
    # centroid1 = fs.calculatecentroid(beta_tmp)
    # beta[:, :, cnt+20] = beta_tmp - np.tile(centroid1, [b, 1]).T
    cnt +=1

y = np.ones(38, dtype=int)
y[19:39] = -1
beta1 = beta[:, :, 0:19]
beta2 = beta[:, :, 20:39]

betatr = np.concatenate((beta1, beta2), axis=2)

beta_tst = np.zeros((a, b, 2))
beta_tst[:, :, 0] = beta[:, :, 19]
beta_tst[:, :, 1] = beta[:, :, 39]
y_test = np.ones(2, dtype=int)
y_test[1] = -1

model = fs.oc_elastic_logistic(betatr, y, T=200, max_itr=40)
out = fs.oc_elastic_prediction(betatr, model, y=y)
out2 = fs.oc_elastic_prediction(beta_tst, model, y=y_test)

