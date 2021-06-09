#%%
import fdasrsf as fs
import numpy as np
data = np.load('bin/simu_data.npz')
time = data['arr_1']
f = data['arr_0']
#obj = fs.fdawarp(f,time)
#obj.srsf_align(parallel=True)

fs.pairwise_align_bayes_infHMC(f[:,0], f[:,10], time)
