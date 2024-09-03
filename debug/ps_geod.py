import h5py
import fdasrsf as fs
import matplotlib.pyplot as plt
from numpy import zeros, empty, linspace, sqrt, arange, tile, arccos
from scipy.integrate import trapezoid
from scipy.linalg import norm
import fdasrsf.curve_functions as cf


# path straightening helper functions
def gram_schmidt(basis):
    b1 = basis[0]
    b2 = basis[1]

    basis1 = b1 / sqrt(cf.innerprod_q(b1, b1))
    b2 = b2 - cf.innerprod_q(basis1, b2)*basis1
    basis2 = b2 / sqrt(cf.innerprod_q(b2, b2))

    basis_o = [basis1, basis2]

    return(basis_o)


def find_basis_normal_path(alpha, k=5):
    basis = empty(k, dtype=object)
    for tau in range(0, k):
        q = alpha[:, :, tau]
        b = cf.find_basis_normal(q)
        basis_tmp = gram_schmidt(b)
        basis[tau] = basis_tmp

    return(basis)


def project_tangent(w, q, basis):
    w = w - cf.innerprod_q(w, q) * q
    bo = gram_schmidt(basis)

    wproj = w - cf.innerprod_q(w, bo[0])*bo[0] - cf.innerprod_q(w, bo[1])*bo[1]

    return(wproj)


def calc_alphadot(alpha, basis, T=100, k=5):
    alphadot = zeros((2, T, k))

    for tau in range(0, k):
        if tau == 0:
            v = (k-1)*(alpha[:, :, tau+1] - alpha[:, :, tau])
        elif tau == (k-1):
            v = (k-1)*(alpha[:, :, tau] - alpha[:, :, (tau-1)])
        else:
            v = ((k-1)/2.0)*(alpha[:, :, tau+1] - alpha[:, :, (tau-1)])

        alphadot[:, :, tau] = project_tangent(v, alpha[:, :, tau], basis[tau])

    return(alphadot)


def calculate_energy(alphadot, T=100, k=5):
    integrand1 = zeros((k, T))
    integrand2 = zeros(k)

    for i in range(0, k):
        for j in range(1, T):
            tmp = alphadot[:, j, i].T
            integrand1[i, j] = tmp.dot(alphadot[:, j, i])

        integrand2[i] = trapezoid(integrand1[i, :], linspace(0, 1, T))

    E = 0.5*trapezoid(integrand2, linspace(0, 1, k))

    return(E)


def parallel_translate(w, q1, q2, basis):
    wtilde = w - 2*cf.innerprod_q(w, q2) / cf.innerprod_q(q1+q2, q1+q2)*(q1+q2)
    l = sqrt(cf.innerprod_q(wtilde, wtilde))

    wbar = project_tangent(wtilde, q2, basis)
    normwbar = sqrt(cf.innerprod_q(wbar, wbar))
    if normwbar > 10**(-4):
        wbar = wbar*l/normwbar

    return(wbar)


def cov_integral(alpha, alphadot, basis, T=100, k=5):
    u = zeros((2, T, k))

    for tau in range(1, k):
        w = u[:, :, tau-1]
        q1 = alpha[:, :, tau-1]
        q2 = alpha[:, :, tau]
        b = basis[tau]
        wbar = parallel_translate(w, q1, q2, b)
        u[:, :, tau] = (1/(k-1))*alphadot[:, :, tau]+wbar

    return(u)


def back_parallel_transport(u1, alpha, basis, T=100, k=5):
    utilde = zeros((2, T, k))

    utilde[:, :, k-1] = u1
    for tau in arange(k-2, -1, -1):
        w = utilde[:, :, tau+1]
        q1 = alpha[:, :, tau+1]
        q2 = alpha[:, :, tau]
        b = basis[tau]
        utilde[:, :, tau] = parallel_translate(w, q1, q2, b)

    return(utilde)


def calculate_gradE(u, utilde, T=100, k=5):
    gradE = zeros((2, T, k))
    normgradE = zeros(k)

    for tau in range(2, k+1):
        gradE[:, :, tau-1] = u[:, :, tau-1] - ((tau-1)/(k-1)) * utilde[:, :, tau-1]
        normgradE[tau-1] = sqrt(cf.innerprod_q(gradE[:, :, tau-1], gradE[:, :, tau-1]))

    return(gradE, normgradE)


def update_path(alpha, beta, gradE, delta, T=100, k=5):
    for tau in range(1, k-1):
        alpha_new = alpha[:, :, tau] - delta*gradE[:, :, tau]
        alpha[:, :, tau] = cf.project_curve(alpha_new)
        x = cf.q_to_curve(alpha[:, :, tau])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau] = x + tile(a, [T, 1]).T

    return(alpha, beta)


def geod_dist_path_strt(beta, k=5):
    dist = 0

    for i in range(1, k):
        beta1 = beta[:, :, i-1]
        beta2 = beta[:, :, i]
        q1 = cf.curve_to_q(beta1)
        q2 = cf.curve_to_q(beta2)
        d = arccos(cf.innerprod_q(q1, q2))
        dist += d

    return(dist)


fun = h5py.File('/Users/jdtucker/Documents/Research/SRVF_FDA/Data/Full20shapedata.h5')
C = fun['beta'][:]
C = C.T
shape1 = 0
shapemid = 28
shape2 = 65

beta1 = C[:, :, shape1]
beta2 = C[:, :, shape2]
betamid = C[:, :, shapemid]

init = "geod"
T = 100
k = 7

inits = ["rand", "geod"]
init = [i for i, x in enumerate(inits) if x == init]
init = init[0]
if init != 0 and init != 1:
    init = 0

betanew1, qnew1, A1 = fs.pre_proc_curve(beta1, T)
betanew2, qnew2, A2 = fs.pre_proc_curve(beta2, T)

if init == 0:
    betanewmid, qnewmid, Amid = fs.pre_proc_curve(beta2, T)

if init == 0:
    alpha, beta, O = fs.init_path_rand(betanew1, betanewmid,
                                       betanew2, T, k)
elif init == 1:
    alpha, beta, O = fs.init_path_geod(betanew1, betanew2, T, k)

# path straightening
tol = 1e-2
n = beta.shape[0]
T = beta.shape[1]
maxit = 20
i = 0
g = 1
delta = 0.5
E = zeros(maxit)
gradEnorm = zeros(maxit)
pathsqnc = zeros((n, T, k, maxit+1))

pathsqnc[:, :, :, 0] = beta

while i <= maxit:
    # algorithm 8:
    # compute dalpha/dt along alpha using finite difference appox
    # First calculate basis for normal sapce at each point in alpha
    basis = find_basis_normal_path(alpha, k)
    alphadot = calc_alphadot(alpha, basis, T, k)
    E[i] = calculate_energy(alphadot, T, k)

    # algorithm 9:
    # compute covariant integral of alphadot along alpha. This is the gradient
    # of E in \cal{H}. Later we will project it to the space \cal{H}_{O}
    u1 = cov_integral(alpha, alphadot, basis, T, k)

    # algorithm 10:
    # backward parallel transport of u(1)
    utilde = back_parallel_transport(u1[:, :, -1], alpha, basis, T, k)

    # algorithm 11:
    # compute graident vector field of E in \cal{H}_{O}
    gradE, normgradE = calculate_gradE(u1, utilde, T, k)
    gradEnorm[i] = norm(normgradE)
    g = gradEnorm[i]

    # algorithm 12:
    # update the path along the direction -gradE
    alpha, beta = update_path(alpha, beta, gradE, delta, T, k)

    # path evolution
    pathsqnc[:, :, :, i+1] = beta

    if g < tol:
        break

    i += 1

E = E[0:(i+1)]
gradEnorm = gradEnorm[0:(i+1)]


# plot path evolution
plotidx = arange(0, i+2)
fig, ax = plt.subplots(plotidx.size, k, sharex=True, sharey=True)
for j in plotidx:
    for tau in range(0, k):
        beta_tmp = pathsqnc[:, :, tau, j]
        ax[j, tau].plot(beta_tmp[0, :], beta_tmp[1, :], 'r', linewidth=2)
        ax[j, tau].set_aspect('equal')
        ax[j, tau].axis('off')

fig2, ax2 = plt.subplots()
ax2.plot(E, linewidth=2)


path = beta
dist = geod_dist_path_strt(beta, k)
