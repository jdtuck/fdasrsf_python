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
F1[:,:,1],F1[:,:,0] = np.gradient(I1, 1/699,1/699)
F2 = np.zeros((m,n,2))
F2[:,:,1],F2[:,:,0] = np.gradient(I2, 1/699,1/699)

F1 -= F1.min()
F1 /= F1.max()

F2 -= F2.min()
F2 /= F2.max()

# %%
M = 10
b = fs.formbasisTid(M, m, n, 't')

gamnew,Inew,H,stepsize = fs.reparam_image(F1,F2,None,b,stepsize=1e-2, itermax=1000)

I2_new = fs.apply_gam_imag(I2,gamnew)

plt.figure()
ax1 = plt.subplot(131)
plt.imshow(I1)
plt.title('I1');
ax1 = plt.subplot(132)
plt.imshow(I2)
plt.title('I2');
ax1 = plt.subplot(133)
plt.imshow(I2_new)
plt.title('I2 Warped');

plt.figure()
gamid = fs.makediffeoid(m,n)
plt.quiver(gamid[:,:,0],gamid[:,:,1],gamid[:,:,0]-gamnew[:,:,0],gamid[:,:,1]-gamnew[:,:,1])

# %%
