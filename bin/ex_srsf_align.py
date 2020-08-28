#%%
import fdasrsf as fs
import numpy as np
data = np.load('bin/simu_data.npz')
time = data['arr_1']
f = data['arr_0']
obj = fs.fdawarp(f,time)
obj.srsf_align(parallel=True)
