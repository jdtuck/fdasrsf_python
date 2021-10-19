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

obj = fs.fdacurve(beta,N=M)
obj.karcher_mean(rotation=False)
obj.srvf_align(rotation=False)
obj.karcher_cov()
obj.shape_pca()
obj.plot_pca()