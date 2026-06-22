import numpy as np
import fdasrsf as fs
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Generate data
np.random.seed(260217)

M = 99
tt = np.linspace(0, 1, M)
n = 101

mu = np.random.normal(0.2, 0.3, size=n)

fns = np.array([
    norm.pdf(tt, mu[i], 0.1) + norm.pdf(tt, 0.8, 0.1)
    for i in range(n)
]).T

plt.plot(tt, fns)


# Error in PPD
fdawarp_obj = fs.fdawarp(fns, tt)
fdawarp_obj.ppd(mu = fns[:, 0],  srvf=False)


# Problem is the number of peaks differs across signals
peaks = [ # find peaks
    find_peaks(fns[:, i])[0]
    for i in range(n)
]
npeaks = np.array([len(p) for p in peaks])
np.unique(npeaks) # show that some signals have 1 peak and others have 2


# Gets further after deleting signals with only 1 peak
# but still get a plotting error
fdawarp_obj = fs.fdawarp(fns[:, npeaks == 2], tt)
fdawarp_obj.ppd(mu = fns[:, 1],  srvf=False)