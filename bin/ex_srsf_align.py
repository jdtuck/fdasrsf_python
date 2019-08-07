#%%
import fdasrsf as fs
import numpy as np
data = np.load('bin/simu_data.npz')
time = data['arr_1']
f = data['arr_0']
out = fs.srsf_align(f,time,omethod="RBFGS",showplot=False)
jfpca = fs.jointfPCA(out.fn,time,out.qn,out.q0,out.gam, showplot=False)
samples = fs.joint_gauss_model(out.fn,time,out.qn,out.gam,out.q0)


#%%
