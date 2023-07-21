import numpy as np

def kernel(x, y, l2):
    # RBF Kernel
    sqdist = np.sum(x**2,1).reshape(-1,1) + \
        np.sum(y**2,1) - 2*np.dot(x, y.T)
    return np.exp(-.5 * (1/l2) * sqdist)

def gp_posterior(X, y, Xtest, l2=0.1, noise_var=1e-6):
    # compute the mean at our test points.
    N, n = len(X), len(Xtest)
    K = kernel(X, X, l2)
    L = np.linalg.cholesky(K + noise_var*np.eye(N))
    Lk = np.linalg.solve(L, kernel(X, Xtest, l2))
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    # compute the variance at our test points.
    K_ = kernel(Xtest, Xtest, l2)
    sd = np.sqrt(np.diag(K_) - np.sum(Lk**2, axis=0))
    return (mu, sd)
