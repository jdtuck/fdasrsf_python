import numpy as np
import time_warping as tw
data = np.load('simu_data.npz')
time = data['arr_1']
f = data['arr_0']
out = tw.srsf_align(f,time)