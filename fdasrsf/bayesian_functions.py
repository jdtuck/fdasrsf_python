import numpy as np
from numpy.random import rand, randn
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre
import fdasrsf.geometry as geo
import fdasrsf.utility_functions as uf
import cbayesian as bay


def f_SSEg_pw(g, q1, q2):
    obs_domain = np.linspace(0, 1, g.shape[0])
    exp1g_temp = uf.f_predictfunction(uf.f_exp1(g), obs_domain, 0)
    pt = np.insert(bay.bcuL2norm2(obs_domain, exp1g_temp), 0, 0)
    tmp = uf.f_predictfunction(q2, pt, 0)
    vec = (q1 - tmp * exp1g_temp) ** 2
    out = vec.sum()
    return out


def f_logl_pw(g, q1, q2, var1, SSEg):
    if SSEg == 0:
        SSEg = f_SSEg_pw(g, q1, q2)

    n = q1.shape[0]
    out = (
        n * np.log(1 / np.sqrt(2 * np.pi))
        - n * np.log(np.sqrt(var1))
        - SSEg / (2 * var1)
    )

    return out


def f_updateg_pw(g_coef_curr, g_basis, var1_curr, q1, q2, SSE_curr, propose_g_coef):
    g_coef_prop = propose_g_coef(g_coef_curr)

    tst = uf.f_exp1(uf.f_basistofunction(g_basis["x"], 0, g_coef_prop["prop"], g_basis))

    while tst.min() < 0:
        g_coef_prop = propose_g_coef(g_coef_curr)
        tst = uf.f_exp1(
            uf.f_basistofunction(g_basis["x"], 0, g_coef_prop["prop"], g_basis)
        )

    if SSE_curr == 0:
        SSE_curr = f_SSEg_pw(
            uf.f_basistofunction(g_basis["x"], 0, g_coef_curr, g_basis), q1, q2
        )

    SSE_prop = f_SSEg_pw(
        uf.f_basistofunction(g_basis["x"], 0, g_coef_prop["prop"], g_basis), q1, q2
    )

    logl_curr = f_logl_pw(
        uf.f_basistofunction(g_basis["x"], 0, g_coef_curr, g_basis),
        q1,
        q2,
        var1_curr,
        SSE_curr,
    )

    logl_prop = f_logl_pw(
        uf.f_basistofunction(g_basis["x"], 0, g_coef_prop["prop"], g_basis),
        q1,
        q2,
        var1_curr,
        SSE_prop,
    )

    ratio = np.exp(np.minimum(0, logl_prop - logl_curr))

    u = np.random.rand()
    if u <= ratio:
        g_coef = g_coef_prop["prop"]
        logl = logl_prop
        SSE = SSE_prop
        accept = True
        zpcnInd = g_coef_prop["ind"]

    if u > ratio:
        g_coef = g_coef_curr
        logl = logl_curr
        SSE = SSE_curr
        accept = False
        zpcnInd = g_coef_prop["ind"]

    return g_coef, logl, SSE, accept, zpcnInd


# ---------------------------------------------------------------------------
# Bayesian HMC alignment: warping basis, exact warp geometry and the
# infinite-dimensional (preconditioned) HMC transition.  Ported from
# fdasrvf_MATLAB (bayes_warp_basis, bayes_coef_to_gam, bayes_gam_to_coef,
# bayes_warp_geom, infhmc_step, infhmc_step_batch).
# ---------------------------------------------------------------------------


def bayes_warp_basis(t, ncoef, basis="fourier"):
    r"""
    Zero-mean orthonormal basis for warping coefficients

    Returns ncoef basis functions on t in [0,1] that integrate to zero (so
    they lie in the tangent space at the identity / give a centered h) and
    are orthonormal in :math:`L^2[0,1]`.

    :param t: grid on [0,1]
    :param ncoef: number of basis functions (even for 'fourier')
    :param basis: 'fourier' (sqrt(2) sin(2 pi j t), sqrt(2) cos(2 pi j t)) or
                  'legendre' (shifted Legendre polynomials of degree 1..ncoef)

    :rtype: numpy ndarray
    :return Phi: (len(t), ncoef) basis matrix
    """
    t = np.asarray(t, dtype=float).ravel()
    ncoef = int(ncoef)
    Phi = np.zeros((t.shape[0], ncoef))
    if basis.lower() == "fourier":
        for i in range(1, ncoef + 1):
            j = int(np.ceil(i / 2))
            if i % 2 == 1:
                Phi[:, i - 1] = np.sqrt(2) * np.sin(2 * j * np.pi * t)
            else:
                Phi[:, i - 1] = np.sqrt(2) * np.cos(2 * j * np.pi * t)
    elif basis.lower() == "legendre":
        x = 2 * t - 1
        for i in range(1, ncoef + 1):
            Phi[:, i - 1] = np.sqrt(2 * i + 1) * eval_legendre(i, x)
    else:
        raise Exception("Incorrect Basis Specified")

    return Phi


def _trapz_weights(T):
    """trapezoid quadrature weights on linspace(0,1,T) and the spacing"""
    dt = 1.0 / (T - 1)
    w = np.full(T, dt)
    w[0] = dt / 2
    w[-1] = dt / 2
    return w, dt


def bayes_coef_to_gam(c, Phi, rep):
    r"""
    Warping functions from basis coefficients

    :param c: (D,) or (D,K) coefficients, one column per warping function
    :param Phi: (T,D) basis on the uniform grid linspace(0,1,T)
    :param rep: 0 = shooting vector (v-transform), 1 = clr (h-transform)

    :rtype: tuple of numpy ndarray
    :return gam: (T,K) warping functions
    :return psi: (T,K) SRSFs sqrt(gam') of the warping functions (unit L2 norm)

    See :func:`bayes_warp_geom` for the two representations.
    """
    c = np.asarray(c, dtype=float)
    onedim = c.ndim == 1
    if onedim:
        c = c.reshape((-1, 1))

    T = Phi.shape[0]
    w, dt = _trapz_weights(T)
    V = Phi @ c
    K = c.shape[1]
    psi = np.zeros((T, K))
    for k in range(0, K):
        v = V[:, k]
        if rep == 0:
            r = np.sqrt(np.sum(w * v**2))
            if r < 1e-10:
                p = 1.0 + v
            else:
                p = np.cos(r) + np.sin(r) / r * v
            psi[:, k] = np.abs(p)
        else:
            psi[:, k] = np.exp((v - v.max()) / 2)

    psi2 = psi**2
    G = np.vstack(
        (np.zeros((1, K)), np.cumsum(dt / 2 * (psi2[0:-1, :] + psi2[1:, :]), axis=0))
    )
    GT = G[-1, :]
    gam = G / GT
    gam[-1, :] = 1.0
    psi = psi / np.sqrt(GT)

    if onedim:
        return gam[:, 0], psi[:, 0]

    return gam, psi


def bayes_gam_to_coef(gam, Phi, rep):
    r"""
    Project warping functions onto warping basis coefficients

    Inverse of :func:`bayes_coef_to_gam` up to basis truncation.

    :param gam: (T,) or (T,K) warping functions on linspace(0,1,T)
    :param Phi: (T,D) orthonormal zero-mean basis (:func:`bayes_warp_basis`)
    :param rep: 0 = shooting vector (v-transform), 1 = clr (h-transform)

    :rtype: numpy ndarray
    :return c: (D,K) coefficients
    """
    gam = np.asarray(gam, dtype=float)
    onedim = gam.ndim == 1
    if onedim:
        gam = gam.reshape((-1, 1))

    T, K = gam.shape
    w, dt = _trapz_weights(T)
    c = np.zeros((Phi.shape[1], K))
    for k in range(0, K):
        g = (gam[:, k] - gam[0, k]) / (gam[-1, k] - gam[0, k])
        gd = np.maximum(np.gradient(g, dt), 1e-8)
        if rep == 0:
            psi = np.sqrt(gd)
            psi = psi / np.sqrt(np.sum(w * psi**2))
            v, theta = geo.inv_exp_map(np.ones(T), psi)
            r = np.sqrt(np.sum(w * v**2))
            if r >= 0.95 * np.pi:
                v = v * (0.95 * np.pi / r)
        else:
            v = np.log(gd)
            v = v - np.sum(w * v)

        c[:, k] = Phi.T @ (w * v)

    if onedim:
        return c[:, 0]

    return c


def bayes_warp_geom(c, Phi, rep, q1, q2, var1, grad=True):
    r"""
    Negative log-likelihood and exact gradient of a warp

    Model (SRVF space): :math:`q_1(t_k) = (q_2 \circ \gamma)(t_k)
    \sqrt{\gamma'(t_k)} + e_k`, :math:`e_k \sim N(0, var1)`, with the warping
    function represented by basis coefficients c on linspace(0,1,T)

    rep = 0 (shooting vector / v-transform): v = Phi c is a tangent vector at
    the identity of the Hilbert sphere, psi = cos|v| + sin|v|/|v| v,
    gam = int psi^2.  Requires |v| < pi (nll = inf otherwise).

    rep = 1 (centered log-ratio / h-transform): h = Phi c,
    gam' = exp(h) / int exp(h); unconstrained.

    :param c: (D,) or (D,N) basis coefficients
    :param Phi: (T,D) basis matrix (columns integrate to zero)
    :param rep: 0 = shooting vector, 1 = clr (h-transform)
    :param q1: (T,) target SRVF, or (T,N) one per column of c
    :param q2: (T,) or (T,N) SRVF to be warped
    :param var1: noise variance
    :param grad: compute the gradient (default = T)

    :rtype: tuple
    :return nll: negative log-likelihood SSE/(2 var1) + T/2 log(var1)
    :return grad: gradient of nll with respect to c
    :return SSE: sum of squared residuals
    :return gam: warping function on the grid
    :return q2w: warped SRVF (q2 o gam) sqrt(gam')

    Batched use: c (D,N), q2 (T,N) and q1 (T,) shared or (T,N) evaluate N
    independent warps; nll and SSE are (N,), grad (D,N), gam and q2w (T,N).
    """
    c = np.asarray(c, dtype=float)
    onedim = c.ndim == 1
    if onedim:
        c = c.reshape((-1, 1))

    T, D = Phi.shape
    N = c.shape[1]
    w, dt = _trapz_weights(T)
    w = w.reshape((T, 1))

    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1m = q1.reshape((T, 1)) if q1.ndim == 1 else q1
    q2m = q2.reshape((T, 1)) if q2.ndim == 1 else q2
    if q1m.shape[1] == 1 and N > 1:
        q1m = np.broadcast_to(q1m, (T, N))
    if q2m.shape[1] == 1 and N > 1:
        q2m = np.broadcast_to(q2m, (T, N))

    V = Phi @ c

    bad = np.zeros(N, dtype=bool)
    if rep == 0:
        r = np.sqrt(np.sum(w * V**2, axis=0))
        bad = r >= np.pi
        small = r < 1e-10
        rs = np.where(small, 1.0, r)
        s = np.where(small, 1.0, np.sin(rs) / rs)
        ds = np.where(small, 0.0, (rs * np.cos(rs) - np.sin(rs)) / rs**2)
        sr = np.where(small, 0.0, np.sin(rs))
        cr = np.where(small, 1.0, np.cos(rs))
        psi = cr + s * V
        sgn = np.where(psi < 0, -1.0, 1.0)
        apsi = np.abs(psi)
    else:
        psi = np.exp((V - V.max(axis=0)) / 2)
        sgn = np.ones((T, N))
        apsi = psi

    # gam = G / G(T), G = cumtrapz(psi^2)
    psi2 = psi**2
    G = np.vstack(
        (np.zeros((1, N)), np.cumsum(dt / 2 * (psi2[0:-1, :] + psi2[1:, :]), axis=0))
    )
    GT = G[-1, :]
    gam = G / GT
    gam[-1, :] = 1.0

    # linear interpolation of q2 at gam on the uniform grid; q2dg is the slope
    # of the interpolant, so the gradient is exact for the discretized model
    pos = np.clip(gam * (T - 1), 0, T - 1)
    k = np.minimum(np.floor(pos).astype(int), T - 2)
    fr = pos - k
    q2k = np.take_along_axis(q2m, k, axis=0)
    q2k1 = np.take_along_axis(q2m, k + 1, axis=0)
    q2g = q2k * (1 - fr) + q2k1 * fr
    q2dg = (q2k1 - q2k) * (T - 1)

    a = 1 / np.sqrt(GT)
    q2w = q2g * apsi * a
    res = q1m - q2w
    SSE = np.sum(res**2, axis=0)
    nll = SSE / (2 * var1) + T / 2 * np.log(var1)

    g_coef = np.zeros((D, N))
    if grad:
        # d(sum res.*q2w)/dc written as beta' * dpsi/dc: the cumulative
        # trapezoid in gam is reverse-accumulated so no (T,D) arrays are formed
        wk = res * a
        A = wk * q2dg * apsi
        Bv = wk * q2g * sgn
        kappa = np.sum(res * q2w, axis=0) / (2 * GT)
        S = np.cumsum(A[::-1, :], axis=0)[::-1, :]  # suffix sums of A
        coefP = np.zeros((T, N))
        coefP[0 : T - 1, :] = dt / 2 * S[1:T, :]
        coefP[1:T, :] = coefP[1:T, :] + dt / 2 * S[1:T, :]
        alpha = coefP / GT - (np.sum(A * gam, axis=0) / GT + kappa) * w
        beta = 2 * alpha * psi + Bv
        if rep == 0:
            gvec = s * (Phi.T @ beta)
            dr = (Phi.T @ (w * V)) / rs
            fac = np.where(small, 0.0, -sr * np.sum(beta, axis=0) + ds * np.sum(beta * V, axis=0))
            gvec = gvec + fac * dr
        else:
            gvec = 0.5 * (Phi.T @ (beta * psi))

        g_coef = -gvec / var1

    if bad.any():
        # outside the injectivity radius of the shooting representation
        nll = np.where(bad, np.inf, nll)
        SSE = np.where(bad, np.inf, SSE)
        g_coef[:, bad] = 0.0
        gam = np.where(bad, np.linspace(0, 1, T).reshape((T, 1)), gam)
        q2w = np.where(bad, q2m, q2w)

    if onedim:
        return nll[0], g_coef[:, 0], SSE[0], gam[:, 0], q2w[:, 0]

    return nll, g_coef, SSE, gam, q2w


def infhmc_step(c, nll, g, aux, geom, C, epsilon, L):
    r"""
    One preconditioned (infinite-dimensional) HMC transition

    Samples pi(c) ~ exp(-nll(c)) N(c; 0, diag(C)) with the splitting
    integrator of Beskos et al. (2011): the Gaussian prior is integrated
    exactly by a rotation and the likelihood enters through half kicks
    v <- v - epsilon/2 C grad(nll).  The integrator is reversible and volume
    preserving, so acceptance uses the exact change in
    H = nll(c) + c'C^{-1}c/2 + v'C^{-1}v/2.

    :param c: (D,) current state
    :param nll: negative log-likelihood at c
    :param g: gradient of nll at c
    :param aux: any extra output of geom at c (may be None)
    :param geom: function returning (nll, g, aux) at c
    :param C: (D,) prior variances
    :param epsilon: step size (rotation angle)
    :param L: maximum number of leapfrog steps (uniform on 1..L)

    :rtype: tuple
    :return c, nll, g, aux: new state and its geometry
    :return accept: True if the proposal was accepted
    :return nsteps: number of leapfrog steps used
    :return alpha: acceptance probability min(1, exp(-dH)) (0 if the
                   trajectory left the support), for step-size adaptation
    """
    C = np.asarray(C, dtype=float).ravel()
    c = np.asarray(c, dtype=float)
    v = np.sqrt(C) * randn(c.shape[0])
    H0 = nll + 0.5 * np.sum(c**2 / C) + 0.5 * np.sum(v**2 / C)

    nsteps = int(np.ceil(rand() * L))
    q = c.copy()
    gq = np.asarray(g, dtype=float).copy()
    cs = np.cos(epsilon)
    sn = np.sin(epsilon)
    nll_q = nll
    aux_q = aux
    for l in range(0, nsteps):
        v = v - epsilon / 2 * C * gq
        q_new = q * cs + v * sn
        v = v * cs - q * sn
        q = q_new
        nll_q, gq, aux_q = geom(q)
        if not np.isfinite(nll_q):
            break
        v = v - epsilon / 2 * C * gq

    H1 = nll_q + 0.5 * np.sum(q**2 / C) + 0.5 * np.sum(v**2 / C)
    logRatio = H0 - H1
    if np.isfinite(logRatio):
        alpha = np.minimum(1.0, np.exp(logRatio))
    else:
        alpha = 0.0
    accept = bool(np.isfinite(logRatio) and (np.log(rand()) < logRatio))
    if accept:
        c = q
        nll = nll_q
        g = gq
        aux = aux_q

    return c, nll, g, aux, accept, nsteps, alpha


def infhmc_step_batch(c, nll, g, aux, geom, C, epsilon, L):
    r"""
    Preconditioned HMC transitions for independent blocks

    Runs :func:`infhmc_step` on N conditionally independent blocks at once
    (one column of c per block), sharing a target of the form
    prod_i exp(-nll_i(c_i)) N(c_i; 0, diag(C)).  All blocks integrate the same
    (random) number of leapfrog steps, with their own step sizes, and each is
    accepted or rejected on its own energy change; since the blocks do not
    interact, every column is a valid HMC transition for its own conditional
    posterior.

    :param c: (D,N) current states
    :param nll: (N,) negative log-likelihoods at c
    :param g: (D,N) gradients of nll at c
    :param aux: (P,N) extra output of geom at c (may be None)
    :param geom: function returning (nll, g, aux) for all N columns
    :param C: (D,) prior variances
    :param epsilon: (N,) or scalar step sizes
    :param L: maximum number of leapfrog steps (uniform on 1..L)

    :rtype: tuple
    :return c, nll, g, aux: new states and their geometry
    :return accept: (N,) bool, True where the proposal was accepted
    :return alpha: (N,) acceptance probabilities min(1, exp(-dH)) (0 where
                   the trajectory left the support), for step-size adaptation
    """
    C = np.asarray(C, dtype=float).reshape((-1, 1))
    c = np.asarray(c, dtype=float)
    D, N = c.shape
    epsilon = np.broadcast_to(np.asarray(epsilon, dtype=float).reshape((1, -1)), (1, N))
    v = np.sqrt(C) * randn(D, N)
    H0 = nll + 0.5 * np.sum(c**2 / C, axis=0) + 0.5 * np.sum(v**2 / C, axis=0)

    nsteps = int(np.ceil(rand() * L))
    q = c.copy()
    gq = np.asarray(g, dtype=float).copy()
    cs = np.cos(epsilon)
    sn = np.sin(epsilon)
    nll_q = nll
    aux_q = aux
    failed = np.zeros(N, dtype=bool)
    for l in range(0, nsteps):
        v = v - (epsilon / 2) * C * gq
        q_new = q * cs + v * sn
        v = v * cs - q * sn
        q = q_new
        nll_q, gq, aux_q = geom(q)
        # a block whose trajectory leaves the support (nll = inf) is rejected,
        # as in infhmc_step; zero its force so it stays finite
        bad = ~np.isfinite(nll_q)
        failed = failed | bad
        gq = gq.copy()
        gq[:, bad] = 0.0
        v = v - (epsilon / 2) * C * gq

    H1 = nll_q + 0.5 * np.sum(q**2 / C, axis=0) + 0.5 * np.sum(v**2 / C, axis=0)
    logRatio = H0 - H1
    ok = ~failed & np.isfinite(logRatio)
    alpha = np.zeros(N)
    alpha[ok] = np.minimum(1.0, np.exp(logRatio[ok]))
    accept = ok & (np.log(rand(N)) < logRatio)

    c = c.copy()
    nll = np.array(nll, dtype=float, copy=True)
    g = np.asarray(g, dtype=float).copy()
    c[:, accept] = q[:, accept]
    nll[accept] = nll_q[accept]
    g[:, accept] = gq[:, accept]
    if aux is not None:
        aux = np.asarray(aux, dtype=float).copy()
        aux[:, accept] = aux_q[:, accept]

    return c, nll, g, aux, accept, alpha


def dual_avg_step(state, alpha=None, target=None):
    r"""
    Dual-averaging step-size adaptation for HMC

    Nesterov dual averaging as in Hoffman and Gelman (2014), "The No-U-Turn
    Sampler", Algorithm 5.  Call once per burn-in iteration with the
    acceptance probability of the transition; at the end of burn-in use
    exp(state["log_eps_bar"]), the averaged step size, which is much less
    noisy than the last iterate.  Works elementwise for a vector of step
    sizes.

    Usage: state = dual_avg_step(epsilon0)                 # initialize
           epsilon, state = dual_avg_step(state, alpha, target)

    :param state: initial step size(s) when called with one argument,
                  otherwise the adaptation state
    :param alpha: acceptance probability(ies) of the last transition, in [0,1]
    :param target: target acceptance probability

    :rtype: dict, or tuple of (numpy ndarray, dict)
    :return epsilon: step size(s) for the next iteration (capped at pi/2)
    :return state: updated state; state["log_eps_bar"] holds the averaged
                   log step size
    """
    if alpha is None:
        eps0 = np.asarray(state, dtype=float)
        return {
            "mu": np.log(10 * eps0),
            "Hbar": np.zeros(eps0.shape),
            "log_eps_bar": np.log(eps0),
            "n": 0,
        }

    t0 = 10
    gamma = 0.05
    kappa = 0.75

    state = dict(state)
    state["n"] = state["n"] + 1
    n = state["n"]
    alpha = np.asarray(alpha, dtype=float).reshape(np.shape(state["Hbar"]))
    state["Hbar"] = (1 - 1 / (n + t0)) * state["Hbar"] + (target - alpha) / (n + t0)
    log_eps = np.minimum(
        state["mu"] - np.sqrt(n) / gamma * state["Hbar"], np.log(np.pi / 2)
    )
    eta = n ** (-kappa)
    state["log_eps_bar"] = eta * log_eps + (1 - eta) * state["log_eps_bar"]
    epsilon = np.exp(log_eps)

    return epsilon, state


def spline_deriv_matrix(t):
    r"""
    (T,T) matrix mapping f to the derivative, at the grid, of its interpolating
    (not-a-knot) cubic spline, as computed in :func:`fdasrsf.utility_functions.f_to_srsf`

    :param t: vector of sample points

    :rtype: numpy ndarray
    :return Dspl: (T,T) derivative matrix
    """
    t = np.asarray(t, dtype=float).ravel()
    T = t.shape[0]
    cs = CubicSpline(t, np.eye(T), axis=0)
    return cs(t, 1)


def srvf_from_deriv(fy):
    """srvf of a function from its derivative"""
    eps = np.finfo(float).eps
    return fy / np.sqrt(np.abs(fy) + eps)


# ---------------------------------------------------------------------------
# Group-wise Bayesian alignment (Z-mixture pCN), ported from
# fdasrvf_MATLAB (fdawarp.time_warping_bayes and its local functions)
# ---------------------------------------------------------------------------


def f_SSEg_group(g, q, q_star):
    r"""
    SSE of each warped function against the template

    :param g: (T,) or (T,N) shooting vectors of the warping functions
    :param q: (T,N) srsfs of the functions
    :param q_star: (T,) srsf of the template

    :rtype: numpy ndarray
    :return out: (N,) sum of squared residuals of each curve
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape((-1, 1))
    g = np.asarray(g, dtype=float)
    if g.ndim == 1:
        g = np.tile(g.reshape((-1, 1)), (1, q.shape[1]))

    N = q.shape[1]
    q_star = np.ascontiguousarray(q_star, dtype=float)
    out = np.zeros(N)
    for ii in range(0, N):
        out[ii] = f_SSEg_pw(
            np.ascontiguousarray(g[:, ii]), q_star, np.ascontiguousarray(q[:, ii])
        )

    return out


def f_logl_group(g, q, q_star, var1, SSEg):
    r"""
    log-likelihood of the warping functions given the functions and template

    :param g: (T,N) shooting vectors of the warping functions
    :param q: (T,N) srsfs of the functions
    :param q_star: (T,) srsf of the template
    :param var1: model variance
    :param SSEg: (N,) sum of squared residuals, recalculated when 0

    :rtype: double
    :return out: log-likelihood
    """
    if np.all(SSEg == 0):
        SSEg = f_SSEg_group(g, q, q_star)

    n = q.shape[0]
    out = (
        n * np.log(1 / np.sqrt(2 * np.pi))
        - n * np.log(np.sqrt(var1))
        - SSEg / (2 * var1)
    )

    return np.sum(out)


def f_updateg_group(g_coef_curr, g_basis, var1_curr, q, q_star, propose_g_coef):
    r"""
    pCN update of the warping coefficients of one function given the template

    :param g_coef_curr: (D,) current coefficients
    :param g_basis: basis dict of the warping functions
    :param var1_curr: model variance
    :param q: (T,) srsf of the function
    :param q_star: (T,) srsf of the template
    :param propose_g_coef: proposal function

    :rtype: tuple
    :return g_coef: new coefficients
    :return accept: True if the proposal was accepted
    :return zpcnInd: index of the pCN mixture component used
    """
    g_coef_prop = propose_g_coef(g_coef_curr)

    tst = uf.f_exp1(
        uf.f_basistofunction(g_basis["x"], 0, g_coef_prop["prop"], g_basis)
    )
    while tst.min() < 0:
        g_coef_prop = propose_g_coef(g_coef_curr)
        tst = uf.f_exp1(
            uf.f_basistofunction(g_basis["x"], 0, g_coef_prop["prop"], g_basis)
        )

    SSE_curr = f_SSEg_pw(
        uf.f_basistofunction(g_basis["x"], 0, g_coef_curr, g_basis), q_star, q
    )
    SSE_prop = f_SSEg_pw(
        uf.f_basistofunction(g_basis["x"], 0, g_coef_prop["prop"], g_basis),
        q_star,
        q,
    )

    logl_ratio = -(SSE_prop - SSE_curr) / (2 * var1_curr)
    # min(1, exp(logl_ratio)), without overflowing for large ratios
    ratio = np.exp(np.minimum(0, logl_ratio))

    u = rand()
    if u <= ratio:
        return g_coef_prop["prop"], True, g_coef_prop["ind"]

    return g_coef_curr, False, g_coef_prop["ind"]


def f_updateq(
    g_coef_curr,
    g_basis,
    var1_curr,
    q,
    q_star_coef_curr,
    q_basis,
    SSE_curr,
    propose_q_star,
):
    r"""
    pCN update of the template coefficients given the warping functions

    :param g_coef_curr: (D,N) current warping coefficients
    :param g_basis: basis dict of the warping functions
    :param var1_curr: model variance
    :param q: (T,N) srsfs of the functions
    :param q_star_coef_curr: (Dq,) current template coefficients
    :param q_basis: basis dict of the template
    :param SSE_curr: (N,) sum of squared residuals, recalculated when 0
    :param propose_q_star: proposal function

    :rtype: tuple
    :return q_star: new template coefficients
    :return accept: True if the proposal was accepted
    :return zpcnInd: index of the pCN mixture component used
    """
    q_star_coef_prop = propose_q_star(q_star_coef_curr)
    ind = q_star_coef_prop["ind"]
    q_star_prop = uf.f_basistofunction(
        q_basis["x"], 0, q_star_coef_prop["prop"], q_basis
    )
    q_star_curr = uf.f_basistofunction(q_basis["x"], 0, q_star_coef_curr, q_basis)

    g_curr = _g_coef_to_g(g_coef_curr, g_basis)
    if np.all(SSE_curr == 0):
        SSE_curr = f_SSEg_group(g_curr, q, q_star_curr)

    SSE_prop = f_SSEg_group(g_curr, q, q_star_prop)

    logl_curr = f_logl_group(g_curr, q, q_star_curr, var1_curr, SSE_curr)
    logl_prop = f_logl_group(g_curr, q, q_star_prop, var1_curr, SSE_prop)

    ratio = np.minimum(1, np.exp(logl_prop - logl_curr))

    u = rand()
    if u <= ratio:
        return q_star_coef_prop["prop"], True, ind

    return q_star_coef_curr, False, ind


def _g_coef_to_g(g_coef, g_basis):
    """(T,N) shooting vectors from (D,N) coefficients, one column at a time"""
    g_coef = np.asarray(g_coef, dtype=float)
    if g_coef.ndim == 1:
        return uf.f_basistofunction(g_basis["x"], 0, g_coef, g_basis)

    N = g_coef.shape[1]
    T = g_basis["x"].shape[0]
    g = np.zeros((T, N))
    for ii in range(0, N):
        g[:, ii] = uf.f_basistofunction(g_basis["x"], 0, g_coef[:, ii], g_basis)

    return g
