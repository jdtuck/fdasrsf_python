#%%
import fdasrsf as fs
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
data = loadmat("C:/Users/jdtuck/Downloads/MPEG7.mat")

Xdata = data["Xdata"]
curve = Xdata[0,1]
n,M = curve.shape
K = Xdata.shape[1]

beta = np.zeros((n,M,K))
for i in range(0,K):
    beta[:,:,i] = Xdata[0,i]
    
obj = fs.fdacurve(beta,N=M)
obj.karcher_mean()
obj.srvf_align()
obj.karcher_cov()
obj.shape_pca()