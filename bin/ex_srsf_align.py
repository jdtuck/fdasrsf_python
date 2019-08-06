import fdasrsf as fs
import numpy as np
data = np.load('bin/simu_data.npz')
time = data['arr_1']
f = data['arr_0']
out = fs.srsf_align(f,time,omethod="RBFGS",showplot=True)
