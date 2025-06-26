import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd


def pcscore2sphere3(n_pc, X_hat, Xs, Tan, V):
    """
    Converts principal component scores to points on the sphere.

    Usage: pcscore2sphere3(n_pc, X_hat, Xs, Tan, V)

    :param n_pc: number of principal components
    :param X_hat: (d x n) matrix of principal component scores
    :param Xs: (d x n) matrix of original data
    :param Tan: tangent space at the origin
    :param V: rotation matrix to align the tangent space with the sphere
    :return: (d x n) matrix of points on the sphere
    """
    d = Tan.shape[0]
    n = Tan.shape[1]
    W = np.zeros((d, n))
    for i in range(n):
        W[:, i] = (
            np.arccos(np.sum(Xs[i, :] * X_hat)) * Tan[:, i] / np.linalg.norm(Tan[:, i])
        )

    lam = np.zeros((n, d))
    for i in range(n):
        for j in range(n_pc):
            lam[i, j] = np.sum(W[:, i] * V[:, j])

    U = np.zeros((n, d))
    for i in range(n):
        for j in range(n_pc):
            U[i, :] = U[i, :] + lam[i, j] * V[:, j]
    S_star = np.zeros((n, n_pc + 1))
    for i in range(n):
        U_norm = np.linalg.norm(U[i, :])
        S_star[i, :] = np.insert(np.sin(U_norm) / U_norm * lam[i, 0:n_pc], 0, np.cos(U_norm))
    return S_star


def rotMat(b):
    """ "
    Returns a rotation matrix that rotates unit vector b to a

    Usage: rot = rotMat(b) returns a d x d rotation matrix that rotate
    unit vector b to the north pole (0,0,...,0,1)

    :param b: (d x 1) vector

    """
    if b.ndim == 1:
        s1 = b.shape[0]
        s2 = 1
    else:
        s1, s2 = b.shape
    d = np.maximum(s1, s2)
    b = b / np.linalg.norm(b)
    if s1 <= s2:
        b = b.T
    a = np.zeros((d, 1))
    a[d - 1] = 1
    alpha = np.arccos(a.T @ b)
    if np.abs(a.T @ b - 1) < 1e-15:
        rot = np.eye(d)
        return rot
    if np.abs(a.T @ b + 1) < 1e-15:
        rot = -1 * np.eye(d)
        return rot

    c = b - a @ (a.T @ b)
    c = c / np.linalg.norm(c)
    A = a @ c.T - c @ a.T
    rot = np.eye(d) + np.sin(alpha) * A + (np.cos(alpha) - 1) * (a @ a.T + c @ c.T)
    return rot


def LogNPd(x):
    """
    Riemannian log map at North pole of S^k

    LogNP(x) returns k x n matrix where each column is a point on tangent
       space at north pole and the input x is (k+1) x n matrix where each
       column is a point on a sphere.
    """
    d, n = x.shape
    scale = np.arccos(x[d - 1, :]) / np.sqrt(1 - x[d - 1, :] ** 2)
    scale[np.isnan(scale)] = 1
    Logpx = np.tile(scale, (d - 1, 1)) * x[0:(d - 1), :]
    return Logpx


def ExpNPd(x):
    """
    Riemannian exponential map at North pole of S^k

    ExpNP(v) returns (k+1) x n matrix where each column is a point on a
                sphere and the input v is k x n matrix where each column
                is a point on tangent  space at north pole.
    """
    d, n = x.shape
    nv = np.sqrt((x**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tmp = np.tile(np.sin(nv) / nv, (d, 1))
    Exppx = np.vstack((tmp * x, np.cos(nv)))
    tmp1 = np.zeros((d + 1, 1))
    tmp1[d] = 1
    Exppx[:, nv < 1e-16] = np.tile(tmp1, (1, (nv < 1e-16).sum()))
    return Exppx


def LMFsphereRes(center, data, greatCircle):
    """
    Auxiliary function for LMFsphereFit.
              -Calculates residuals of circle fit for the given center.
    """
    xmc = data - np.tile(center[:, np.newaxis], (1, data.shape[1]))
    di = np.sqrt(np.sum(xmc**2, axis=0))
    r = np.pi / 2
    if greatCircle == 0:
        r = di.mean()
    rr = (di - r).T

    return rr


def LMFsphereJac(center, data, greatCircle):
    """
    Auxiliary function for LMFsphereFit.
              -Calculates jacobian of circle fit for the given center.
    """
    xmc = data - np.tile(center[:, np.newaxis], (1, data.shape[1]))
    di = np.sqrt(np.sum(xmc**2, axis=0))
    di_vj = -xmc / np.tile(di, (center.shape[0], 1))
    if greatCircle:
        return di_vj.T
    else:
        r_vj = di_vj.mean(axis=1)
        return (di_vj - np.tile(r_vj[:, np.newaxis], (1, data.shape[1]))).T


def LMFsphereFit(data, initialCenter, greatCircle=True):
    """
    The least square estimates of the sphere to the data.
              The non-linear least square estimates are calculated by
              the Levenberg-Marquardt method

    center= LMFsphereFit(A) with d x n data matrix A (any d = 2, 3, ...).
    center, r = LMFsphereFit(A) with d x n data matrix gives the center and
                              the radius.
    center, r = LMFsphereFit(A,1) forces the sphere radius as 1

    @param data: points on sphere
    @param initialCenter: vector of initial guess
    @param greatCircle: if True, fit a great circle; if False, fit a sphere

    Example:
    n = 50
    theta = np.linspace(0,np.pi*1.5,n)
    data = 5*np.vstack((np.cos(theta),np.sin(theta))) + np.random.normal(size=(2,n))
    x, r = LMFsphereFit(data, np.zeros((2,1)))
    Estcirc = r*np.vstack((np.cos(theta), np.sin(theta)))+np.tile(x,(1,data.shape[1]))
    plt.scatter(data[0,:], data[1,:])
    plt.plot(x[0],x[1],'or')
    plt.plot(Estcirc[0,:],Estcirc[1,:])
    """
    result = least_squares(
        LMFsphereRes,
        initialCenter.ravel(),
        jac=LMFsphereJac,
        args=(data, greatCircle),
        method="lm",
        xtol=1e-09,
        max_nfev=1000,
    )

    center = result.x

    center = center[:, np.newaxis]

    xmc = data - np.tile(center, (1, data.shape[1]))
    di = np.sqrt(np.sum(xmc**2, axis=0))

    if greatCircle == 0:
        r = di.mean()
    else:
        r = np.pi / 2

    return center, r


def objfn(center, r, data):
    """
    the objective function that we want to minimize: sum of squared distances
    from the data to the subsphere
    """
    g = np.mean((np.arccos(center.T @ data) - r) ** 2)
    return g


def getSubSphere(data, geodesic=0):
    """
    The least square estimates of the best fitting subsphere
                to the data on the unit hyper-sphere.

    center = getSubSphere(data), with d x n data matrix with
            each column having unit length, returns the center
    center, r = getSubSphere(data), with d x n data matrix with each
               column having unit length, returns the center and the
               geodesic radius.
    center, r= getSubSphere(data, 1) forces the subsphere radius as 1

    """
    # last singular vector
    U, dd, Vh = np.linalg.svd(data)
    initialCenter = U[:, -1]

    c0 = initialCenter[:, np.newaxis]
    TOL = 1e-10
    cnt = 0
    err = 1
    d, n = data.shape
    Gnow = 1e10
    while err > TOL:
        # normalize the new candidate
        c0 /= np.linalg.norm(c0)
        # rotation matrix : c0 -> North Pole
        rot = rotMat(c0)
        # Tangent projection by Log map
        TpData = LogNPd(rot @ data)
        newCenterTp, r = LMFsphereFit(TpData, np.zeros((d - 1, 1)), geodesic)
        if r > np.pi:
            r = np.pi / 2
            U, dd, Vh = np.linalg.svd(TpData)
            newCenterTp = U[:, -1] * np.pi / 2
            newCenterTp = newCenterTp[:, np.newaxis]

        # Bring back to the sphere by Exp map
        newCenter = ExpNPd(newCenterTp)
        # rotate back the newCenter
        center = np.linalg.inv(rot) @ newCenter
        Gnext = objfn(center, r, data)
        err = np.abs(Gnow - Gnext)
        Gnow = Gnext
        c0 = center
        cnt += 1
        if cnt > 30:
            break

    i1save = {"Gnow": Gnow, "center": center, "r": r}

    # last eigenvector of covariance matrix
    K = np.cov(data)
    U, dd, Vh = np.linalg.svd(K, hermitian=True)
    initialCenter = U[:, -1]

    c0 = initialCenter[:, np.newaxis]
    TOL = 1e-10
    cnt = 0
    err = 1
    d, n = data.shape
    Gnow = 1e10
    while err > TOL:
        # normalize the new candidate
        c0 /= np.linalg.norm(c0)
        # rotation matrix : c0 -> North Pole
        rot = rotMat(c0)
        # Tangent projection by Log map
        TpData = LogNPd(rot @ data)
        newCenterTp, r = LMFsphereFit(TpData, np.zeros((d - 1, 1)), geodesic)
        if r > np.pi:
            r = np.pi / 2
            U, dd, Vh = np.linalg.svd(TpData)
            newCenterTp = U[:, -1] * np.pi / 2
            newCenterTp = newCenterTp[:, np.newaxis]

        newCenter = ExpNPd(newCenterTp)
        center = np.linalg.inv(rot) @ newCenter
        Gnext = objfn(center, r, data)
        err = np.abs(Gnow - Gnext)
        Gnow = Gnext
        c0 = center
        cnt += 1
        if cnt > 30:
            break

    if i1save["Gnow"] == np.minimum(Gnow, i1save["Gnow"]):
        center = i1save["center"]
        r = i1save["r"]

    # adjust radius
    if r > np.pi / 2:
        center *= -1
        r = np.pi - r

    return center, r


def geodmeanS1(theta):
    """
    geodesic mean of data on S^1 (Circle) by S. Lu and V. Kulkarni
               method - gives all multiples of geodesic mean set.

    @param theta: a column vector of angles

    @return:
    geodmean: geodesic mean on S^1
    geodvar: geodesic variance on S^2

    """
    n = theta.shape[0]
    meancandi = np.mod(np.mean(theta) + 2 * np.pi * np.arange(0, n) / n, 2 * np.pi)
    theta = np.mod(theta, 2 * np.pi)
    geodvar = np.zeros(n)
    for i in range(n):
        v = meancandi[i]
        tmparray = np.vstack(
            (
                (theta - v) ** 2,
                (theta - v + 2 * np.pi) ** 2,
                (v - theta + 2 * np.pi) ** 2,
            )
        )
        dist2 = tmparray.min(axis=0)
        geodvar[i] = dist2.sum()

    ind = np.argmin(geodvar)
    geodmean = np.mod(meancandi[ind], 2 * np.pi)
    geodvar = geodvar[ind] / n

    return geodmean, geodvar


def PNSs2e(spheredata, PNS):
    """
    PNS Sphere to Euclidean-type representation

    Usage: EuclidData = PNSs2e(A,PNS)

      with d x n matrix A consists of column
        vectors that are on the (d-1) sphere, and PNS is the output from
    """
    kk, n = spheredata.shape
    Res = np.zeros((kk - 1, n))
    currentSphere = spheredata.copy()

    for i in range(kk - 1):
        v = PNS["orthaxis"][i]
        r = PNS["dist"][i]
        res = np.arccos(v.T @ currentSphere) - r
        Res[i, :] = res
        NestedSphere = rotMat(v) @ currentSphere
        currentSphere = NestedSphere[0:(kk - i), :] / np.tile(
            np.sqrt(1 - NestedSphere[-1, :] ** 2), (kk - i, 1)
        )

    S1toRadian = np.arctan2(currentSphere[1, :], currentSphere[0, :])
    devS1 = np.mod(S1toRadian - PNS["orthaxis"][-1] + np.pi, 2 * np.pi) - np.pi
    Res[kk - 2, :] = devS1

    EuclidData = np.flipud(np.tile(np.flipud(PNS["radii"]), (1, n)) * Res)

    return EuclidData


def PNSe2s(resmat, PNS):
    """
    PNS coordinate transform from Euclidean-type residual matrix to Sphere

    Usage: Spheredata = PNSe2s(data,PNS)

      where 'data' is d x m data matrix in PNS coordinate system (for any
      m >= 1), 'PNS' is the structural array
    """
    dm, n = resmat.shape
    # dm is the intrinsic dimension of the sphere
    #    or the reduced sphere in HDLSS case.
    # n  is the sample size.
    NSOrthaxis = PNS["orthaxis"][:-1]
    NSOrthaxis.reverse()
    NSradius = np.flipud(PNS["dist"])
    geodmean = PNS["orthaxis"][-1]

    # standardize components
    res = resmat / np.tile(np.flipud(PNS["radii"][0:dm]), (1, n))

    # iteratviely mapping back to S^d
    # S^1 to S^2
    tmpmat = np.tile(np.sin(NSradius[0] + res[1, :]), (2, 1)) * np.vstack(
        (np.cos(geodmean + res[0, :]), np.sin(geodmean + res[0, :]))
    )
    tmpmat = np.vstack((tmpmat, np.cos(NSradius[0] + res[1, :])))
    T = rotMat(NSOrthaxis[0]).T @ tmpmat
    # S^2 to S^d
    for i in range(dm - 2):
        tmpmat = np.vstack(
            (
                np.tile(np.sin(NSradius[i + 1] + res[i + 2, :]), (3 + i, 1)) * T,
                np.cos(NSradius[i + 1] + res[i + 2, :]),
            )
        )
        T = rotMat(NSOrthaxis[i + 1]).T @ tmpmat

    if PNS["basisu"].shape[0] != 0:
        T = PNS["basisu"][:, 0:T.shape[0]] @ T

    return T


def fastPNSe2s(res, PNS):
    """
    Fast PNS coordinate transform from Euclidean-type residual matrix to Sphere

    Usage: Spheredata = fastPNSe2s(res, PNS)

      where 'res' is d x m data matrix in PNS coordinate system (for any
      m >= 1), 'PNS' is the structural array
    """
    GG = PNSe2s(res, PNS)
    n = GG.shape[1]
    muhat = PNS["muhat"]
    n_pc = PNS["n_pc"]
    s = np.arccos(GG[0, :])
    ones = np.ones((n, 1))
    approx1 = GG[1:(n_pc + 1), :].T @ PNS["pca"][:, 0:n_pc].T + np.diag(
        np.cos(s)
    ) @ ones @ muhat[:, np.newaxis].T / np.linalg.norm(muhat)

    return approx1


def PNSmainHDLSS(data, itype="small", a=0.05, R=100, thresh=1e-15):
    """
    Analysis of Principal Nested Spheres for data on hyperspheres

    Usage: resmat, PNS = PNSmain(data)

    @param data: (d+1) x n data matrix where each column is a unit vector.
    @param itype:  'seq_test' : (default) ordinary Principal Nested Sphere
                                with sequential tests (not implemented)
                   'small'    : Principal Nested SMALL Sphere
                   'great'    : Principal Nested GREAT Sphere (radius pi/2)
    @param alpha: 0.05 (default) : size of Type I error allowed for each test,
                 could be any number between 0 and 1.
    @param R:100 (default) : number of bootsrap samples to be evaluated for
                             the sequential test.
    @param thresh: 1e-15 (default): eigenvalue threshold

    @return:
    resmat: The commensurate residual matrix (X_PNS). Each entry in row k
            works like the kth principal component score.
    PNS: Dictionary with the following fields
        mean     : location of the PNSmean
        radii    : size (radius) of PNS
        orthoaxis: orthogonal axis 'v_i' of subspheres
        dist     : distance 'r_i' of subspheres
        pvalues  : p-values of LRT and parametric boostrap tests (if any)
        itype    : type of methods for fitting subspheres

    """
    if itype == "small":
        itype = 1
    elif itype == "great":
        itype = 2
    else:
        raise Exception("itype not implemented")

    # data on (k-1)-sphere in Real^k
    k, n = data.shape
    U, s, Vh = np.linalg.svd(data, full_matrices=False)

    maxd = np.where(s < thresh)[0]
    if maxd.shape[0] == 0 or k > n:
        maxd = np.minimum(k, n) + 1
    else:
        maxd = maxd[0]
    # dimension of subspace that contains no data
    nullspdim = k - maxd + 1

    d = k - 1  # intrinsic dimension of sphere
    print("dataset is on %d-sphere" % d)

    dm = maxd - 2  # intrinsic dimension of the smallest nested sphere that
    # contains variation. I.e. this is the dimension to which
    # can be trivially reduced.
    resmat = np.zeros((dm, n))  # d dimensional residual matrix
    # there will be dm -1 subspheres
    orthaxis = []
    dist = np.zeros((dm - 1, 1))

    if nullspdim > 0:
        print("..found null space of dimension %d to be trivally reduced" % nullspdim)
        print(".. then narrow down to %d-sphere" % dm)
        # (HDLSS case) fit nested great spheres for dimension reduction
        # where no residual is present.
        currentSphere = U[:, 0:(dm + 1)].T @ data
    else:
        print(".. Check that the following holds: %d = %d" % (d, dm))
        currentSphere = data

    pvalues = np.nan
    for i in range(dm - 1):
        # estimate the best fitting subsphere
        # with small sphere if itype = 1
        # with great sphere if itype = 2
        center, r = getSubSphere(currentSphere, itype - 1)
        res = np.arccos(center.T @ currentSphere) - r
        dist[i] = r
        # save subsphere parameters
        orthaxis.append(center)
        # save residuals
        resmat[i, :] = res
        # projection to subsphere and transformation to isomorphic
        # sphere
        NestedSphere = rotMat(center) @ currentSphere
        currentSphere = NestedSphere[0:(dm - i), :] / np.tile(
            np.sqrt(1 - NestedSphere[dm - i, :] ** 2), (dm - i, 1)
        )

    # currentSphere has intrinsic dimension 1
    # compute PNSmean and deviations.

    # parametrize 1-sphere to angles
    S1toRadian = np.arctan2(currentSphere[1, :], currentSphere[0, :])
    # geodesic mean of angles
    meantheta, vartheta = geodmeanS1(S1toRadian.T)
    orthaxis.append(meantheta)
    # save deviations from PNSmean
    resmat[dm - 1, :] = np.mod(S1toRadian - meantheta + np.pi, 2 * np.pi) - np.pi

    radii = [1]
    for i in range(dm - 1):
        radii.append(np.prod(np.sin(dist[0:i])))

    radii = np.array(radii)
    radii = radii[:, np.newaxis]
    resmat = np.flipud(np.tile(radii, (1, n)) * resmat)

    if nullspdim > 0:
        basisu = U[:, 0:(dm + 1)]
    else:
        basisu = np.empty((0))

    PNS = {
        "radii": radii,  # size (radius) of nested spheres from largest to smallest
        "orthaxis": orthaxis,  # orthogonal axis of (d-1) subspheres and the anglemean for PNSmean
        "dist": dist,  # distances for (d-1) subspheres
        "pvalues": pvalues,  # d-1 pvalues from sequential tests
        "basisu": basisu,
        "type": itype,
    }

    pnsmean = PNSe2s(np.zeros((dm, 1)), PNS)  # PNSmean of the data

    PNS["mean"] = pnsmean

    return resmat, PNS


def fastpns(x, n_pc="Full", itype="small", a=0.05, R=100, thresh=1e-15):
    """
    Fast Principal Nested Spheres (PNS) for data on hyperspheres.

    Usage: resmat, PNS = fastpns(x)

    @param x: (d+1) x n data matrix where each column is a unit vector.
    @param n_pc: number of principal components to use, or "Full" for
                 all and "Approx" for an approximate number based on
                 99% explained variance
    @param itype: 'small' or 'great' for small or great spheres.
    @param a: significance level for tests.
    @param R: number of bootstrap samples for tests.
    @param thresh: threshold for eigenvalues.

    @return:
    resmat: Commensurate residual matrix (X_PNS).
    PNS: Dictionary with PNS parameters.
    """
    n = x.shape[1]
    pdim = x.shape[0]
    if n_pc == "Full":
        n_pc = np.minimum(pdim - 1, n - 1)
    if n_pc == "Approx" or n_pc == pdim - 1:
        K = np.cov(x)
        U, s, V = svd(K)
        cumm_coef = np.cumsum(s) / np.sum(s)
        n_pc = int(np.argwhere(cumm_coef <= 0.99)[-1])

    Xs = x.T
    for i in range(n):
        Xs[i, :] = Xs[i, :] / np.linalg.norm(Xs[i, :])

    muhat = np.mean(Xs, axis=0)
    muhat = muhat / np.linalg.norm(muhat)

    TT = Xs.copy()
    for i in range(n):
        TT[i, :] = Xs[i, :] - np.sum(Xs[i, :] * muhat) * muhat

    TT = TT.T
    K = np.cov(TT)
    U, s, V = svd(K)
    pcapercent = np.sum(s[0:n_pc] ** 2 / np.sum(s**2))
    print("Initial PNS subsphere dimension: %d\n" % (n_pc + 1))
    print(
        "Percentage of variability in PNS sequence %f" % np.round(pcapercent * 100, 2)
    )

    ans = pcscore2sphere3(n_pc, muhat, Xs, TT, U)
    Xssubsphere = ans.T

    resmat, PNS = PNSmainHDLSS(Xssubsphere, itype, a, R, thresh)

    PNS["spehredata"] = Xssubsphere
    PNS["pca"] = U
    PNS["muhat"] = muhat
    PNS["n_pc"] = n_pc

    return resmat, PNS
