#%%
import fdasrsf as fs
import matplotlib.pyplot as plt
import numpy as np
import GPy
data = np.load('fdasrsf_python/bin/simu_data.npz')
time = data['arr_1']
f = data['arr_0']
y1i = f[:,0] + np.random.normal(0,np.sqrt(0.01),f.shape[0])
y2i = f[:,7] + np.random.normal(0,np.sqrt(0.01),f.shape[0])
M1 = time.shape[0]
kernel1 = GPy.kern.RBF(input_dim=1, variance=y1i.std()/np.sqrt(2), lengthscale=np.mean(time.std()))
kernel2 = GPy.kern.RBF(input_dim=1, variance=y2i.std()/np.sqrt(2), lengthscale=np.mean(time.std()))
model1 = GPy.models.GPRegression(time.reshape((M1,1)),y1i.reshape((M1,1)),kernel1)
model1.optimize()
model2 = GPy.models.GPRegression(time.reshape((M1,1)),y2i.reshape((M1,1)),kernel2)
model2.optimize()

f1_curr, predvar = model1.predict(time.reshape((M1,1)))
f1_curr = f1_curr.reshape(M1)
f2_curr, predvar = model2.predict(time.reshape((M1,1)))
f2_curr = f2_curr.reshape(M1)

out = fs.pairwise_align_bayes_infHMC(y1i, y2i, time)
