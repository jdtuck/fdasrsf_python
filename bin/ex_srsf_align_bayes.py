#%%
import fdasrsf as fs
import matplotlib.pyplot as plt
import numpy as np

data = np.load('fdasrsf_python/bin/simu_data.npz')
time = data['arr_1']
f = data['arr_0']
y1i = f[:,0] + np.random.normal(0,np.sqrt(0.01),f.shape[0])
y2i = f[:,7] + np.random.normal(0,np.sqrt(0.01),f.shape[0])

out = fs.pairwise_align_bayes_infHMC(y1i, y2i, time)
