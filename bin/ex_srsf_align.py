import fdasrsf as fs
data = np.load('simu_data.npz')
time = data['arr_1']
f = data['arr_0']
out = fs.srsf_align(f,time)
