import h5py
import fdasrsf as fs
import numpy as np
import matplotlib.pyplot as plt
fun = h5py.File('/Users/jderektucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T

# calculate distance module SO(2) and/or gamma
lam = 0
k = 5
elastic = 1
rotation = 1
returnpath = 1
Disp_geodesic_between_the_curves = 1
Disp_registration_between_curves = 1

beta1 = C[:, :, 0]
beta2 = C[:, :, 2]
n, T = beta1.shape

beta1 = fs.resamplecurve(beta1, T)
beta2 = fs.resamplecurve(beta2, T)

centroid1 = fs.calculatecentroid(beta1)
beta1 = beta1 - np.tile(centroid1, [T, 1]).T
centroid2 = fs.calculatecentroid(beta2)
beta2 = beta2 - np.tile(centroid2, [T, 1]).T

q1 = fs.curve_to_q(beta1)

if rotation:
    beta2, O1, tau = fs.find_rotation_and_seed_coord(beta1, beta2)
    q2 = fs.curve_to_q(beta2)
else:
    O1 = np.eye(2)
    q2 = fs.curve_to_q(beta2)

if elastic:
    # Find the optimal coorespondence
    gam = fs.optimum_reparam_curve(q2, q1, lam)
    gamI = fs.invertGamma(gam)
    # Applying optimal re-parameterization to the second curve
    beta2n = fs.group_action_by_gamma_coord(beta2, gamI)
    q2n = fs.curve_to_q(beta2n)

    if rotation:
        beta2n, O2, tau = fs.find_rotation_and_seed_coord(beta1, beta2n)
        centroid2 = fs.calculatecentroid(beta2n)
        beta2n = beta2n - np.tile(centroid2, [T, 1]).T
        q2n = fs.curve_to_q(beta2n)
        O = O1.dot(O2)
else:
    q2n = q2
    O = O1

# Forming geodesic between the registered curves
dist = np.arccos(fs.innerprod_q(q1, q2n))
if returnpath:
    PsiQ = np.zeros((n, T, k))
    PsiX = PsiQ
    for tau in range(0, k):
        s = dist * tau / (k - 1)
        PsiQ[:, :, tau] = (np.sin(dist-s)*q1+np.sin(s)*q2n)/np.sin(dist)
        PsiX[:, :, tau] = fs.q_to_curve(PsiQ[:, :, tau])

    path = PsiQ
else:
    path = 0

if Disp_registration_between_curves:
    centroid1 = fs.calculatecentroid(beta1)
    beta1 = beta1 - np.tile(centroid1, [T, 1]).T
    centroid2 = fs.calculatecentroid(beta2n)
    beta2n = beta2n - np.tile(centroid2, [T, 1]).T
    beta2n[0, :] = beta2n[0, :] + 1.3
    beta2n[1, :] = beta2n[1, :] - 0.1

    fig, ax = plt.subplots()
    ax.plot(beta1[0, :], beta1[1, :], 'r', linewidth=2)
    fig.hold()
    ax.plot(beta2n[0, :], beta2n[1, :], 'b-o', linewidth=2)

    for j in range(0, np.int(T/5)):
        i = j*5
        ax.plot(np.array([beta1[0, i], beta2n[0, i]]),
                np.array([beta1[1, i], beta2n[1, i]]), 'k', linewidth=1)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.hold()

if Disp_geodesic_between_the_curves:
    fig, ax = plt.subplots()
    fig.hold()
    for tau in range(0, k):
        ax.plot(.35*tau+PsiX[0, :, tau], PsiX[1, :, tau], 'k', linewidth=2)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.hold()
