#%%
import fdasrsf as fs
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

dat = loadmat('image.mat')
I1 = dat['I1']
I2 = dat['I2']

(m,n) = I1.shape

I1 -= I1.min()
I1 /= I1.max()

I2 -= I2.min()
I2 /= I2.max()

F1 = np.zeros((m,n,2))
F1[:,:,0],F1[:,:,1] = np.gradient(I1, 1/699,1/699)
F2 = np.zeros((m,n,2))
F2[:,:,0],F2[:,:,1] = np.gradient(I2, 1/699,1/699)

F1 -= F1.min()
F1 /= F1.max()

F2 -= F2.min()
F2 /= F2.max()

plt.figure()
plt.imshow(I1)
plt.title('I1')

plt.figure()
plt.imshow(I2)
plt.title('I2')
# %%
M = 10
b = fs.formbasisTid(M, m, n, 't')

out = fs.reparam_image(F1,F2,None,b)
# %%
