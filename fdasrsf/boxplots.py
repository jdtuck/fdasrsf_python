import numpy as np



def ampbox(ft, f_median, qt, q_median, time):

    N = ft.shape[1]
    lambda = 0.5

    # compute amplitude distances
    dy = np.zeros(N)
    for i in range(0,N):
        dy[i] = np.sqrt(np.trapz((q_median-qt[:,i])**2,time))

    dy_ordering = dy.argsort()
    CR_50 = dy_ordering[0:np.ceil(N/2).astype('int')]
    tmp = dy[CR_50]
    m = tmp.max()

    # identify amplitude quartiles
    angle = np.zeros((CR_50.shape[0],CR_50.shape[0]))
    energy = np.zeros((CR_50.shape[0],CR_50.shape[0]))
    for i in range(0,CR_50.shape[0]-1):
        for j in range(i+1,CR_50.shape[0]):
            q1 = qt[:,CR_50[i]] - q_median
            q3 = qt[:,CR_50[i]] - q_median
            q1 /= np.sqrt(np.trapz(q1**2,time))
            q3 /= np.sqrt(np.trapz(q3**2,time))
            angle[i,j] = trapz(q1*q3,time)
            energy[i,j] = (1-lambda) * (dy[CR_50[i]]/m+dy[CR_50[j]]/m) - lambda * (angle[i,j] + 1)

    maxloc = energy.argmax()
    maxloc_row,maxloc_col = np.unravel_index(maxloc,energy.shape)

    Q1_index = CR_50[maxloc_row]
    Q3_index = CR_50[maxloc_col]
    Q1_q = qt[:,Q1_index]
    Q3_q = qt[:,Q3_index]
    Q1 = ft[:,Q1_index]
    Q3 = ft[:,Q3_index]

    # indentify amplitude quantiles
