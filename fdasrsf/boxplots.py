"""
Elastic Functional Boxplots

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
import collections


def ampbox(ft, f_median, qt, q_median, time, alpha=.05, k_a=1):
    """
    This function constructs the amplitude boxplot using the elastic
    square-root slope (srsf) framework.

    :param ft: numpy ndarray of shape (M,N) of N functions with M samples
    :param f_median: vector of size M describing the median
    :param qt: numpy ndarray of shape (M,N) of N srsf functions with M samples
    :param q_median: vector of size M describing the srsf median
    :param time: vector of size M describing the time
    :param alpha: quantile value (e.g.,=.05, i.e., 95\%)
    :param k_a: scalar for outlier cutoff (e.g.,=1)

    :rtype: tuple of numpy array
    :return fn: aligned functions - numpy ndarray of shape (M,N) of N
    functions with M samples
    :return Q1: First quartile
    :return Q3: Second quartile
    :return Q1a: First quantile based on alpha
    :return Q3a: Second quantile based on alpha
    :return minn: minimum extreme function
    :return maxx: maximum extreme function
    :return outlier_index: indexes of outlier functions
    :return f_median: median function
    :return q_median: median srsf
    :return plt: surface plot mesh

    """
    N = ft.shape[1]
    lam = 0.5

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
            angle[i,j] = np.trapz(q1*q3,time)
            energy[i,j] = (1-lam) * (dy[CR_50[i]]/m+dy[CR_50[j]]/m) - lam * (angle[i,j] + 1)

    maxloc = energy.argmax()
    maxloc_row,maxloc_col = np.unravel_index(maxloc,energy.shape)

    Q1_index = CR_50[maxloc_row]
    Q3_index = CR_50[maxloc_col]
    Q1_q = qt[:,Q1_index]
    Q3_q = qt[:,Q3_index]
    Q1 = ft[:,Q1_index]
    Q3 = ft[:,Q3_index]

    # identify amplitude quantiles
    dy_ordering = dy.argsort()
    CR_alpha = dy_ordering[0:np.round(N*(1-alpha)).astype('int')]
    tmp = dy[CR_alpha]
    m = tmp.max()
    angle = np.zeros((CR_alpha.shape[0],CR_alpha.shape[0]))
    energy = np.zeros((CR_alpha.shape[0],CR_alpha.shape[0]))
    for i in range(0,CR_alpha.shape[0]-1):
        for j in range(i+1,CR_alpha.shape[0]):
            q1 = qt[:,CR_alpha[i]] - q_median
            q3 = qt[:,CR_alpha[i]] - q_median
            q1 /= np.sqrt(np.trapz(q1**2,time))
            q3 /= np.sqrt(np.trapz(q3**2,time))
            angle[i,j] = np.trapz(q1*q3,time)
            energy[i,j] = (1-lam) * (dy[CR_alpha[i]]/m+dy[CR_alpha[j]]/m) - lam * (angle[i,j] + 1)

    maxloc = energy.argmax()
    maxloc_row,maxloc_col = np.unravel_index(maxloc,energy.shape)

    Q1a_index = CR_alpha[maxloc_row]
    Q3a_index = CR_alpha[maxloc_col]
    Q1a_q = qt[:,Q1a_index]
    Q3a_q = qt[:,Q3a_index]
    Q1a = ft[:,Q1a_index]
    Q3a = ft[:,Q3a_index]

    # compute amplitude whiskers
    IQR = dy[Q1_index] + dy[Q3_index]
    v1 = Q1_q - q_median
    v3 = Q3_q - q_median
    upper_q = Q3_q + k_a * IQR * v3 / np.sqrt(np.trapz(v3**2,time))
    lower_q = Q1_q + k_a * IQR * v1 / np.sqrt(np.trapz(v1**2,time))

    upper_dis = np.sqrt(np.trapz((upper_q - q_median)**2,time))
    lower_dis = np.sqrt(np.trapz((lower_q-q_median)**2,time))
    whisker_dis = max(upper_dis,lower_dis)

    # identify amplitude outliers
    outlier_index = np.array([])
    for i in range(0,N):
        if dy[dy_ordering[N+1-i]] > whisker_dis:
            outlier_index = np.append(outlier_index,dy[dy_ordering[N+1-i]])
    
    # identify amplitude extremes
    distance_to_upper = np.full(N, np.inf)
    distance_to_lower = np.full(N, np.inf)
    out_50_CR = np.setdiff1d(np.arange(0,N), outlier_index)
    for i in range(0,out_50_CR.shape[0]):
        j = out_50_CR[i]
        distance_to_upper[j] = np.sqrt(np.trapz((upper_q-qt[:,j])**2,time))
        distance_to_lower[j] = np.sqrt(np.trapz((lower_q-qt[:,j])**2,time))
    
    max_index = distance_to_upper.argmin()
    min_index = distance_to_lower.argmin()
    min_q = qt[:,min_index]
    max_q = qt[:,max_index]
    minn  ft[:,min_index]
    maxx = ft[:,max_index]

    s = np.linspace(0,1,100)
    Fs2 = np.zeros((length(time),595))
    Fs2[:,0] = (1-s[0]) * minn + s[0] * Q1
    for j in range(1,100):
        Fs2[:,j] = (1-s[j]) * minn + s[j] * Q1a
        Fs2[:,98+j] = (1-s[j]) * Q1a + s[j] * Q1
        Fs2[:,197+j] = (1-s[j]) * Q1 + s[j] * f_median
        Fs2[:,296+j] = (1-s[j]) * f_median + s[j] * Q3
        Fs2[:,395+j] = (1-s[j]) * Q3 + s[j] * Q3a
        Fs2[:,494+j] = (1-s[j]) * Q3a + s[j] * maxx
    
    d1 = np.sqrt(np.trapz((q_median-Q1_q)**2,time))
    d1a = np.sqrt(np.trapz((Q1_q-Q1a_q)**2,time))
    dl = np.sqrt(np.trapz((Q1a_q-min_q)**2,time))
    d3 = np.sqrt(np.trapz((q_median-Q3_q)**2,time))
    d3a = np.sqrt(np.trapz((Q3_q-Q3a_q)**2,time))
    du = np.sqrt(np.trapz((Q3a_q-max_q)**2,time))
    part1=np.linspace(-d1-d1a-dl,-d1-d1a,100)
    part2=np.linspace(-d1-d1a,-d1,100)
    part3=np.linspace(-d1,0,100)
    part4=np.linspace(0,d3,100)
    part5=np.linspace(d3,d3+d3a,100)
    part6=np.linspace(d3+d3a,d3+d3a+du,100)
    allparts = np.array([part1,part2[1:99],part3[1:99],part4[1:99],part5[1:99],part6[1:99])
    U, V = np.meshgrid(time, allparts)
    U = np.transpose(U)
    V = np.transpose(V)

    ampbox = collections.namedtuple('ampbox', ['Q1', 'Q3', 'Q1a', 'Q3a',
                                                     'minn', 'maxx', 'outlier_index',
                                                     'f_median', 'q_median', 'plt'])
    
    plt = collections.namedtuple('plt', ['U', 'V', 'Fs2', 'allparts',
                                                     'd1', 'd1a', 'dl',
                                                     'd3', 'd3a', 'du',
                                                     'Q1q','Q3q'])

    plt_o = plt(U,V,Fs2,allparts,d1,d1a,dl,d3,d3a,du,Q1a_q,Q3a_q)                                           

    out = ampbox(Q1,Q3,Q1a,Q3a,minn,maxx,outlier_index,f_median,q_median,plot_o)

    return (out)

