#%%
import fdasrsf as fs
import numpy as np
data = np.load('fdasrsf_python/bin/MPEG7.npz',allow_pickle=True)
Xdata = data['Xdata']
curve = Xdata[0,1]
n,M = curve.shape
K = Xdata.shape[1]

beta = np.zeros((n,M,K))
for i in range(0,K):
    beta[:,:,i] = Xdata[0,i]

obj = fs.fdacurve(beta,N=M,scale=True)
obj.karcher_mean()
obj.srvf_align()
obj.karcher_cov()
obj.shape_pca()
obj.plot_pca()