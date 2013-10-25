"""
Warping Invariant Regression using SRSF

moduleauthor:: Derek Tucker <dtucker@stat.fsu.edu>

"""

import numpy as np
from . import utility_functions as uf
from scipy import dot
from scipy.integrate import trapz
from scipy.linalg import inv, norm
from patsy import bs

def elastic_regression(f, y, time):
    max_itr = 20
    M = f.shape[0]
    N = f.shape[1]

    # Create B-Spline Basis
    B = bs(time, knots=np.linspace(time[0], time[-1], 10), lower_bound=0, upper_bound=time[-1], degree=4,
           include_intercept=True)

    q = uf.f_to_srsf(f, time)

    gamma = np.tile(np.linspace(0, 1, M), (N, 1))
    gamma = gamma.transpose()

    itr = 1
    SSE = np.zeros((max_itr))
    while itr <= max_itr:
        print("Iteration: %d - Fold %d" % itr)
        # align data
        fn = np.zeros((M, N))
        qn = np.zeros((M, N))
        for ii in range(0, N):
            fn[:, ii] = np.interp((time[-1] - time[0]) * gamma[:, ii] + time[0], time, f[:, ii])
            qn[:, ii] = uf.warp_q_gamma(time, q[:, ii], gamma[:, ii])

        # OLS using basis since we have it in this example
        Phi = np.ones((N, 3))
        for ii in range(0, N):
            Phi[ii, 1] = trapz(qn[:, ii] * b1, time)
            Phi[ii, 2] = trapz(qn[:, ii] * b2, time)

        inv_xx = inv(dot(Phi.T, Phi))
        xy = dot(Phi.T, y)
        b = dot(inv_xx, xy)

        alpha = b[0]
        beta = b[1] * b1 + b[2] * b2

        # compute the SSE
        int_X = np.zeros(N)
        for ii in range(0, N):
            int_X[ii] = trapz(qn[:, ii] * beta, time)

        SSE[itr - 1] = sum((y.reshape(N) - alpha - int_X) ** 2)

        # find gamma
        qM = np.zeros((M, N))
        qm = np.zeros((M, N))
        gamma_new = np.zeros((M, N))
        y_M = np.zeros(N)
        y_m = np.zeros(N)
        for ii in range(0, N):
            gam_M = uf.optimum_reparam(beta, time, q[:, ii])
            qM[:, ii] = uf.warp_q_gamma(time, q[:, ii], gam_M)
            y_M[ii] = trapz(qM[:, ii] * beta, time)

            gam_m = uf.optimum_reparam(-1 * beta, time, q[:, ii])
            qm[:, ii] = uf.warp_q_gamma(time, q[:, ii], gam_m)
            y_m[ii] = trapz(qm[:, ii] * beta, time)

            if y[ii] > alpha + y_M[ii]:
                gamma_new[:, ii] = gam_M
            elif y[ii] < alpha + y_m[ii]:
                gamma_new[:, ii] = gam_m
            else:
                gamma_new[:, ii] = uf.zero_crossing(y[ii] - alpha, q[:, ii], beta, time, y_M[ii], y_m[ii], gam_M,
                                                    gam_m)

        if norm(gamma - gamma_new) < 1e-5:
            break
        else:
            gamma = gamma_new

        itr += 1


#def elastic_prediction(alpha, beta, elastic_model)
