#%%
import fdasrsf as fs
import numpy as np
data = np.load('bin/simu_data.npz')
time = data['arr_1']
f = data['arr_0']
out = fs.srsf_align(f,time,omethod="RBFGS",showplot=False)
fs.bootTB(f, time, a=0.5, p=.99, B=500, no=5, parallel=False)
#%%
